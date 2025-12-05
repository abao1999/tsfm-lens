import math
from itertools import combinations
from typing import Dict, Tuple

import torch

EPS = 1e-12


def _one_hot_smooth_batched(
    vocab_size: int, y_true: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Batched smoothed one-hot observation distributions.

    Args
    ----
    vocab_size : int
    y_true     : LongTensor [B]  (indices in [0, V))
    eps        : float  (small smoothing)

    Returns
    -------
    o : FloatTensor [B, V]
        o[b] = (1 - eps) * e_{y_true[b]} + eps * Uniform(V)
    """
    B = y_true.shape[0]
    o = torch.full(
        (B, vocab_size), eps / vocab_size, dtype=torch.float32, device=y_true.device
    )
    o.scatter_(1, y_true.view(-1, 1), 1.0 - eps)
    return o


def _project_simplex_clip_renorm_batched(
    P: torch.Tensor, minval: float = 1e-12
) -> torch.Tensor:
    """
    Clamp to [minval, +inf) then renormalize along the last dimension.
    Works with any leading batch shape; only assumes last dim = V.

    Args
    ----
    P : FloatTensor [..., V]

    Returns
    -------
    P_proj : FloatTensor [..., V]  (row-wise on last dim sums to 1)
    """
    P = torch.clamp(P, min=minval)
    Z = P.sum(dim=-1, keepdim=True)
    return P / Z


@torch.no_grad()
def enkf_soft_assimilate(
    logits: torch.Tensor,  # [B, N, V]
    y_true: torch.Tensor,  # [B]
    *,
    obs_smooth: float = 1e-6,  # ε for smoothed one-hot observation
    R: float = 1e-2,  # observation noise (larger => weaker update)
    jitter: float = 1e-6,  # numerical stability for matrix inverse
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Matrix-based EnKF assimilation for categorical next-token distributions

    State (per batch b):
        - N rollout probability vectors p_{b,i} ∈ Δ^{V-1} (rows of probs).
        - Observation is ground-truth token y_true[b], encoded as smoothed one-hot o_b.

    EnKF analogy (H = I):
        Prior mean μ_b = mean_i p_{b,i}, prior covariance Σ_b = Cov[p_{b,i}] across i.
        K_b = Σ_b (Σ_b + R I)^{-1}

    Update:
        p_{b,i}^a = p_{b,i}^f + K_b (o_b - p_{b,i}^f), then project to simplex.

    Args
    ----
    logits : FloatTensor [B, N, V]
    y_true : LongTensor [B]
        Observed ground-truth token indices in [0, V).
    obs_smooth : float
        Smoothing for the observed categorical to avoid degeneracy.
    R : float
        Observation noise; higher R => smaller gain (softer nudging).
    jitter : float
        Added to the diagonal of (Σ_b + R I) for numerical stability.

    Returns
    -------
    probs_assimilated : FloatTensor [B, N, V]
        Updated probabilities for each batch and rollout after matrix-based assimilation.
    info : dict of FloatTensors
        Diagnostics per batch:
          - 'gain'         : [B]  Effective gain (trace of Kalman gain matrix)
          - 'trace_Sigma'  : [B]  trace of Σ_b (sum of per-dimension variances across rollouts)
    """
    # Convert logits to probability distributions
    P = torch.softmax(logits.float(), dim=-1)  # [B, N, V]

    B, N, V = P.shape
    assert B >= 1 and N >= 1 and V >= 2, "Expected [B, N, V] with N>=1, V>=2."

    # Create smoothed one-hot observation distributions for ground truth tokens
    o = _one_hot_smooth_batched(V, y_true, eps=obs_smooth).to(P.device)  # [B, V]

    # Compute prior statistics across rollouts for each batch
    mu = P.mean(dim=1)  # [B, V] - prior mean across rollouts
    X = P - mu.unsqueeze(1)  # [B, N, V] - deviations from mean

    # Compute covariance matrix and its trace for each batch
    if N > 1:
        var_dim = X.var(dim=1, unbiased=True)  # [B, V]
        trace_Sigma = var_dim.sum(dim=-1)  # [B]
        Sigma = torch.matmul(X.transpose(1, 2), X) / (N - 1)  # [B, V, V]
    else:
        var_dim = torch.zeros((B, V), device=P.device)
        trace_Sigma = torch.zeros(B, device=P.device)
        Sigma = torch.zeros((B, V, V), device=P.device)

    # Prepare for Kalman gain computation
    I = torch.eye(V, device=P.device).expand(B, V, V)  # [B, V, V]
    A = Sigma + (R + jitter) * I  # [B, V, V]

    # Compute Kalman gain matrix K = Σ @ A^{-1}
    # This is the standard EnKF formula for linear observation operator H = I
    K = torch.linalg.solve(A, Sigma.transpose(-2, -1)).transpose(-2, -1)

    # Compute innovation: difference between observation and prior predictions
    innov = o.unsqueeze(1) - P  # [B, N, V]

    # Apply Kalman gain to innovation and update probabilities
    # p_{b,i}^a = p_{b,i}^f + K @ (o_b - p_{b,i}^f)
    # Pa = P + torch.matmul(innov, K)  # [B, N, V]
    Pa = P + torch.matmul(innov, K)  # [B, N, V]

    # Project updated probabilities back to probability simplex
    Pa = _project_simplex_clip_renorm_batched(Pa)

    # Compute effective gain diagnostic: trace of Kalman gain matrix
    # Higher values indicate stronger updates, lower values indicate weaker updates
    eff_gain = torch.einsum("bii->b", K)  # [B]
    # # A_inv_Sigma = torch.linalg.solve(A, Sigma)  # [B, V, V]
    # eff_gain = torch.einsum("bii->b", torch.matmul(Sigma, A_inv_Sigma))  # [B]

    trace_Sigma_after = Pa.var(dim=1, unbiased=True).sum(
        dim=-1
    )  # [B]  # posterior covariance trace
    entropy_before = (
        -(P * P.log()).sum(dim=-1).mean(dim=1)
    )  # [B]  # mean entropy before assimilation
    entropy_after = (
        -(Pa * Pa.log()).sum(dim=-1).mean(dim=1)
    )  # [B]  # mean entropy after assimilation
    entropy_change = entropy_before - entropy_after  # [B]  # reduction in entropy

    # Return updated probabilities and diagnostic information
    info = {
        "gain": eff_gain,
        "trace_Sigma": trace_Sigma,
        "trace_Sigma_after": trace_Sigma_after,
        "entropy_before": entropy_before,
        "entropy_after": entropy_after,
        "entropy_change": entropy_change,
    }
    return Pa, info


def uncertainty_triggered_intervention(
    logits_batched: torch.Tensor,
    conf_thresh: float = 0.2,
    dis_thresh: float = 0.9,
    k: int = 3,  # number of tokens to consider for soft consensus (should be odd)
    verbose: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute a (optionally time-aware) batched intervention decision for uncertainty-based data assimilation.

    This function determines, for each batch, whether to intervene (i.e., inject ground truth)
    at a given rollout timestep based on two uncertainty metrics:
      - Soft consensus confidence: sum of probabilities of k tokens closest in value to the most probable token
        (k//2 below, the most probable, and k//2 above, in token index space).
      - Soft variation ratio: fraction of probability mass outside these k tokens.

    The intervention rule is:
        Intervene for batch b if:
            (soft_pmax_mean[b] < conf_thresh_t) OR (soft_VR[b] > dis_thresh_t)
    where conf_thresh_t and dis_thresh_t are the (optionally time-adjusted) thresholds.

    Parameters
    ----------
    logits_batched : torch.Tensor
        Logits for the next token, shape (B, N, V), where:
            B = batch size,
            N = number of rollouts/samples,
            V = vocabulary size.
    conf_thresh : float, optional
        Base threshold for mean per-rollout max probability (confidence).
    dis_thresh : float, optional
        Base threshold for variation ratio (disagreement).
    k : int, optional
        Number of tokens to consider for soft consensus (should be odd).

    Returns
    -------
    intervene : torch.Tensor
        Boolean tensor of shape (B,), True if intervention is triggered for each batch.
    metrics : dict
        Dictionary containing:
            - 'soft_pmax_mean': torch.Tensor, shape (B,), mean sum of centered-k probs per batch.
            - 'soft_VR': torch.Tensor, shape (B,), soft variation ratio per batch.
    """
    B, N, V = logits_batched.shape
    probs = torch.softmax(logits_batched, dim=-1)  # (B, N, V)

    # For each rollout, find the most probable token (by value, not by probability)
    pmax_indices = probs.argmax(
        dim=2
    )  # (B, N) -- token indices (by value, not by probability)

    # For each rollout, select k//2 tokens below, the most probable, and k//2 tokens above (in token index space)
    k = min(k, V)
    half_k = k // 2

    # Prepare indices for all rollouts

    # We'll build a mask of shape (B, N, V) where True for the k closest tokens in value
    consensus_mask = torch.zeros(
        (B, N, V), dtype=torch.bool, device=logits_batched.device
    )

    for offset in range(-half_k, half_k + 1):
        idx = pmax_indices + offset  # (B, N)
        idx = idx.clamp(0, V - 1)
        # For each (b, n), set consensus_mask[b, n, idx[b, n]] = True
        batch_idx = (
            torch.arange(B, device=logits_batched.device).unsqueeze(1).expand(B, N)
        )
        rollout_idx = (
            torch.arange(N, device=logits_batched.device).unsqueeze(0).expand(B, N)
        )
        consensus_mask[batch_idx, rollout_idx, idx] = True

    # Now, for each (b, n), sum the probabilities of the consensus tokens
    consensus_probs = probs * consensus_mask.float()  # (B, N, V)
    soft_pmax = consensus_probs.sum(dim=2)  # (B, N)
    soft_pmax_mean = soft_pmax.mean(dim=1)  # (B,)

    # Soft variation ratio: for each rollout, sum of probabilities outside consensus tokens
    soft_vr_per_rollout = 1.0 - soft_pmax  # (B, N)
    soft_VR = soft_vr_per_rollout.mean(dim=1)  # (B,)

    conf_thresh_t = torch.full((B,), conf_thresh, device=logits_batched.device)
    dis_thresh_t = torch.full((B,), dis_thresh, device=logits_batched.device)

    intervene = (soft_pmax_mean < conf_thresh_t) | (soft_VR > dis_thresh_t)

    if verbose:
        print(
            f"pmax_mean: {soft_pmax_mean.float().cpu().numpy()}, VR: {soft_VR.float().cpu().numpy()}"
        )
    return intervene, {
        "soft_pmax_mean": soft_pmax_mean,
        "soft_VR": soft_VR,
    }


def composite_feature_triggered_intervention(
    logits_batched: torch.Tensor,
    sampled_tokens_batched: torch.LongTensor,
    V: int,
    *,
    k: int = 0,
    lookahead_seqs: list[list[int]] | None = None,
    weights: dict | None = None,
    tau: float = 0.45,
    vr_only_thresh: float = 0.60,
    pmax_mean_thresh: float = 0.40,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Essentially a more complex version of uncertainty_triggered_intervention.
    Compute rollout-based uncertainty features and make a time-aware intervention decision.

    Computes a composite score S and intervenes if S >= tau or VR >= vr_only_thresh or pmax_mean < pmax_mean_thresh

    Composite score S is a weighted sum of the following features:
        VR (Variation Ratio): Measures vote disagreement (higher = more uncertainty).
        H_vote (Vote Entropy): Captures diversity in votes (higher = more uncertainty).
        1 - pmax_mean: Penalizes low per-rollout confidence.
        TD (Trajectory Divergence): Measures short-horizon disagreement in rollouts.
        conf_vote_gap: Captures lack of confidence even among the majority.


    Parameters
    ----------
    logits_batched : torch.Tensor, shape [B, N, V]
        Next-token logits for each of B batches, each with N rollouts (each rollout has its own history).
    sampled_tokens_batched : torch.LongTensor, shape [B, N]
        Tokens actually sampled by each rollout at this step for each batch (argmax/greedy or stochastic).
    V : int
        Vocabulary size (for entropy normalization).
    k : int, default=0
        If >0 and lookahead_seqs provided, compute short-horizon trajectory divergence TD_{t,k}.
    lookahead_seqs : list[list[int] or 1D tensors], optional
        lookahead_seqs[i] contains the next k tokens from rollout i (starting at current step).
    weights : dict or None
        Weights for composite score S (defaults below).
    tau : float
        Base score threshold (applied to the boosted score S * w(t)).
    vr_only_thresh, pmax_mean_thresh : float
        Base simple-trigger thresholds (VR high, pmax_mean low).
    verbose : bool
        If True, print diagnostic information.

    Returns
    -------
    intervene : torch.Tensor, shape (B,)
        Final decision for each batch.
    features : dict
        Uncertainty features.
    S : torch.Tensor, shape (B,)
        Composite score.
    """

    B, N, V_ = logits_batched.shape
    assert V_ == V, f"V mismatch: logits_batched.shape[-1]={V_} vs V={V}"

    p = torch.softmax(logits_batched.float(), dim=2)  # [B, N, V]

    # per-rollout confidence
    pmax_vals, _ = p.max(dim=2)  # [B, N]
    pmax_mean = pmax_vals.mean(dim=1)  # [B]

    # votes (argmax tokens actually chosen at this step)
    VR = torch.zeros(B, device=logits_batched.device)
    H_vote = torch.zeros(B, device=logits_batched.device)
    conf_vote_gap = torch.zeros(B, device=logits_batched.device)

    for b in range(B):
        # More efficient counting since V >> N
        counts = torch.zeros(V, device=sampled_tokens_batched.device)
        idxs, vals = torch.unique(sampled_tokens_batched[b], return_counts=True)
        counts[idxs] = vals.float()
        f = counts / float(N)  # empirical vote distribution over tokens

        VR[b] = 1.0 - f.max()

        f_clamped = f.clamp(min=EPS)
        H_vote[b] = (-f_clamped * f_clamped.log()).sum()
        # f_sorted, _ = torch.sort(f, descending=True)
        # C2[b] = f_sorted[:2].sum()

        # confidence among the voters of the top-voted token
        top_v = int(torch.argmax(f).item())
        mask = sampled_tokens_batched[b] == top_v
        if mask.any():
            conf_vote_gap[b] = (1.0 - p[b, mask, top_v]).mean()
        else:
            conf_vote_gap[b] = 0.0

    # optional short-horizon disagreement
    TD = torch.zeros(B, device=logits_batched.device)
    if lookahead_seqs is not None and len(lookahead_seqs) == N and k > 0:
        for b in range(B):
            total = 0.0
            pairs = 0
            for i, j in combinations(range(N), 2):
                seq_i, seq_j = lookahead_seqs[i], lookahead_seqs[j]
                assert len(seq_i) >= k and len(seq_j) >= k
                diff = sum(1 for s in range(k) if int(seq_i[s]) != int(seq_j[s]))
                total += diff / float(k)
                pairs += 1
            TD[b] = total / pairs if pairs > 0 else 0.0

    features = {
        "VR": VR,
        "H_vote": H_vote,
        "pmax_mean": pmax_mean,
        "conf_vote_gap": conf_vote_gap,
        "TD": TD,
    }

    if verbose:
        print(
            f"VR: {VR.float().cpu().numpy()}, H_vote: {H_vote.float().cpu().numpy()}, pmax_mean: {pmax_mean.float().cpu().numpy()}, conf_vote_gap: {conf_vote_gap.float().cpu().numpy()}, TD: {TD.float().cpu().numpy()}"
        )

    # --- Decision logic ---
    if weights is None:
        weights = {
            "VR": 0.4,
            "H_vote": 0.2,
            "pmax_mean": 0.2,
            "TD": 0.15,
            "conf_gap": 0.05,
        }
    assert all(w >= 0 and w <= 1 for w in weights.values()), (
        "Weights must be between 0 and 1"
    )
    assert sum(weights.values()) == 1.0, "Weights must sum to 1.0"

    # normalize vote entropy
    H_norm = features["H_vote"] / math.log(max(2, V))

    # base composite score S in [roughly 0..1] for each batch
    S = (
        weights["VR"] * features["VR"]
        + weights["H_vote"] * H_norm
        + weights["pmax_mean"] * (1.0 - features["pmax_mean"])
        + weights["TD"] * features["TD"]
        + weights["conf_gap"] * features["conf_vote_gap"]
    )

    # final decision: boosted score or relaxed triggers for each batch
    intervene = (
        (S >= tau)
        | (features["VR"] >= vr_only_thresh)
        | (features["pmax_mean"] < pmax_mean_thresh)
    )

    return intervene, S
