"""
This file contains a suite of custom implementations of metrics to compare two time-series.
For the GIFT-Eval benchmark evaluations, we use the implementations from GluonTS instead, which have nothing to do with this file
"""

from functools import partial
from typing import Literal

import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy import signal, stats
from scipy.ndimage import gaussian_filter1d
from scipy.stats import (
    pearsonr,
    spearmanr,
)


def _ssim_1d_univariate(
    x: np.ndarray,
    y: np.ndarray,
    sigma: float = 1.5,
    K1: float = 0.01,
    K2: float = 0.03,
    L: float | None = None,
    return_map: bool = False,
    eps: float = 1e-12,
) -> tuple[float, np.ndarray] | float:
    """
    Helper function to compute SSIM for univariate 1D signals.
    Internal use only - see ssim_1d for full documentation.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    # Local means and second moments via Gaussian smoothing
    mu_x = gaussian_filter1d(x, sigma=sigma, mode="reflect")
    mu_y = gaussian_filter1d(y, sigma=sigma, mode="reflect")
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    s_x2 = gaussian_filter1d(x * x, sigma=sigma, mode="reflect") - mu_x2
    s_y2 = gaussian_filter1d(y * y, sigma=sigma, mode="reflect") - mu_y2
    s_xy = gaussian_filter1d(x * y, sigma=sigma, mode="reflect") - mu_xy

    if L is None:
        # Heuristic if dynamic range not known
        L = 6.0 * np.maximum(np.std(x), np.std(y)) + eps

    C1 = (K1 * L) ** 2  # type: ignore
    C2 = (K2 * L) ** 2  # type: ignore

    num = (2 * mu_xy + C1) * (2 * s_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (s_x2 + s_y2 + C2)
    ssim_map = num / (den + eps)
    ssim_mean = float(np.mean(ssim_map))
    return (ssim_mean, ssim_map) if return_map else ssim_mean


def _ssim_1d_univariate_batched(
    x: np.ndarray,
    y: np.ndarray,
    sigma: float = 1.5,
    K1: float = 0.01,
    K2: float = 0.03,
    L: float | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Vectorized helper function to compute SSIM for a batch of univariate 1D signals.

    Args:
        x: Batch of first signals of shape (batch_size, T)
        y: Batch of second signals of shape (batch_size, T)
        sigma: Standard deviation of the Gaussian window
        K1: First stability constant
        K2: Second stability constant
        L: Dynamic range of the signals. If None, computed per batch element
        eps: Small constant to prevent division by zero

    Returns:
        Array of SSIM values of shape (batch_size,)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    if x.ndim != 2:
        raise ValueError(f"Expected 2D input (batch_size, T), got shape {x.shape}")

    # Local means and second moments via Gaussian smoothing along axis=1 (time dimension)
    mu_x = gaussian_filter1d(x, sigma=sigma, mode="reflect", axis=1)
    mu_y = gaussian_filter1d(y, sigma=sigma, mode="reflect", axis=1)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    s_x2 = gaussian_filter1d(x * x, sigma=sigma, mode="reflect", axis=1) - mu_x2
    s_y2 = gaussian_filter1d(y * y, sigma=sigma, mode="reflect", axis=1) - mu_y2
    s_xy = gaussian_filter1d(x * y, sigma=sigma, mode="reflect", axis=1) - mu_xy

    if L is None:
        # Compute per-batch dynamic range
        std_x = np.std(x, axis=1, keepdims=True)
        std_y = np.std(y, axis=1, keepdims=True)
        L = 6.0 * np.maximum(std_x, std_y) + eps

    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    num = (2 * mu_xy + C1) * (2 * s_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (s_x2 + s_y2 + C2)
    ssim_map = num / (den + eps)

    # Mean over time dimension for each batch
    return np.mean(ssim_map, axis=1)


def ssim_1d(
    x: np.ndarray,
    y: np.ndarray,
    sigma: float = 1.5,
    K1: float = 0.01,
    K2: float = 0.03,
    L: float | None = None,
    return_map: bool = False,
    eps: float = 1e-12,
) -> tuple[float, np.ndarray] | float:
    """
    Calculate the 1D Structural Similarity Index (SSIM) between two aligned time series using Gaussian smoothing.

    In its typical formulation, SSIM compares local patterns of pixel intensities that have been normalized for luminance and contrast.
    The measure was originally designed to improve on traditional methods like mean squared error which have been proven to be inconsistent with human visual perception.
    In our case, we compute SSIM on various windows of the univariate signals using a separable 1D Gaussian filter.

    For multivariate time series (2D arrays with shape (T, D)), the metric is computed for each dimension
    separately and then averaged across dimensions.

    The SSIM score is calculated as:
    SSIM(x,y) = [2μxμy + C1][2σxy + C2] / [μx² + μy² + C1][σx² + σy² + C2]
    where:
    - μx, μy are local means
    - σx, σy are local standard deviations
    - σxy is the local covariance
    - C1, C2 are constants for stability

    Args:
        x: First input signal. Can be 1D array of shape (T,) or 2D array of shape (T, D).
        y: Second input signal. Must have the same shape as x.
        sigma: Standard deviation of the Gaussian window in samples. Controls the size
              of the local neighborhood. Larger values analyze coarser signal structure.
              Default is 1.5.
        K1: First stability constant for the luminance term. Default is 0.01.
        K2: Second stability constant for the contrast term. Default is 0.03.
        L: Dynamic range of the signals. If None, estimated as 6*max(std(x),std(y)).
           For normalized signals, L=1 is appropriate.
        return_map: If True, returns both mean SSIM and the pointwise SSIM values.
                   Note: Only supported for 1D inputs. Raises ValueError for 2D inputs.
        eps: Small constant to prevent division by zero. Default is 1e-12.

    Returns:
        If return_map is False:
            float: The mean SSIM value between -1 and 1, where 1 indicates perfect
                  structural similarity. For multivariate inputs, this is the mean across dimensions.
        If return_map is True (only for 1D inputs):
            tuple: (mean_ssim, ssim_map) where ssim_map is an array of pointwise
                  SSIM values the same length as the input signals.

    Notes:
        - Input signals should be aligned in time/space before comparison
        - The Gaussian window makes this implementation translation-invariant
        - SSIM is symmetric: ssim_1d(x,y) = ssim_1d(y,x)
        - For best results, signals should have similar dynamic ranges
        - For multivariate time series, computes dimension-wise mean
    """
    x = np.asarray(x).squeeze()
    y = np.asarray(y).squeeze()

    if x.ndim != y.ndim:
        raise ValueError("x and y must have the same number of dimensions")

    # Handle 1D case
    if x.ndim == 1:
        return _ssim_1d_univariate(x, y, sigma, K1, K2, L, return_map, eps)

    # Handle 2D case (multivariate time series)
    elif x.ndim == 2:
        if return_map:
            raise ValueError("return_map=True is only supported for 1D inputs")

        if x.shape[1] != y.shape[1]:
            raise ValueError(
                f"x and y must have the same number of dimensions. Got x.shape={x.shape}, y.shape={y.shape}"
            )

        scores = []
        for i in range(x.shape[1]):
            s = _ssim_1d_univariate(x[:, i], y[:, i], sigma, K1, K2, L, False, eps)
            scores.append(s)

        return float(np.mean(scores))

    else:
        raise ValueError(f"Input arrays must be 1D or 2D. Got x.ndim={x.ndim}, y.ndim={y.ndim}")


def _ms_ssim_1d_univariate(
    x: np.ndarray,
    y: np.ndarray,
    levels: int = 5,
    sigma: float = 1.5,
    K1: float = 0.01,
    K2: float = 0.03,
    L: float | None = None,
    downsample: int = 2,
    weights: np.ndarray | None = None,
    eps: float = 1e-12,
) -> float:
    """
    Helper function to compute MS-SSIM for univariate 1D signals.
    Internal use only - see ms_ssim_1d for full documentation.

    Numerical Stability Improvements:
    ---------------------------------
    This implementation includes several safeguards against numerical instabilities
    that can cause NaN or invalid results:

    1. **Identical Signal Detection** (Early Return):
       - Detects when x and y are identical and returns 1.0 immediately
       - Avoids unnecessary computation and potential numerical issues

    2. **Signal Length Validation**:
       - Verifies signal is long enough for multi-scale decomposition
       - Falls back to single-scale SSIM if signal too short after downsampling
       - Prevents decimation errors on very short signals

    3. **Non-negative Variance Enforcement**:
       - Ensures computed variances (s_x2, s_y2) are non-negative
       - Fixes floating-point precision issues: variance = E[X²] - E[X]² can be
         slightly negative due to rounding errors

    4. **Constant Signal Handling**:
       - Detects when both signals are constant (std < eps)
       - Returns 1.0 if equal constants, 0.0 if different constants
       - Avoids division by near-zero dynamic range values
       - Prevents instability in the normalization constants C1 and C2

    5. **Value Clamping Before Exponentiation** (Critical Fix):
       - Clips luminance (l) and contrast-structure (cs) to [0, 1+eps] range
       - **Key issue**: Numerical errors can push l or cs slightly negative
       - Raising negative values to fractional powers (e.g., (-0.01)^0.2856) produces NaN
       - The weighted product uses fractional exponents from the weights array

    6. **Final Safety Check**:
       - Verifies the final MS-SSIM result is finite (not NaN or inf)
       - Returns 0.0 if result is invalid
       - Last line of defense against any remaining edge cases

    Common Sources of NaN Values (Now Fixed):
    -----------------------------------------
    - Constant or near-constant signals → handled by constant signal detection
    - Negative l/cs values raised to fractional powers → handled by clamping
    - Division by zero in dynamic range → handled by constant signal checks
    - Signals too short for decomposition levels → handled by length validation
    - Floating-point variance errors → handled by non-negative enforcement
    """
    if weights is None:
        # Standard MS-SSIM weights from Wang et al. (2003)
        weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if len(weights) != levels:
        raise ValueError("weights length must equal 'levels'.")

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    # Check if signals are identical (perfect match)
    if np.allclose(x, y, rtol=eps, atol=eps):
        return 1.0

    # Check minimum signal length for multi-scale decomposition
    min_length = downsample ** (levels - 1) * 4  # Need at least 4 samples at coarsest scale
    if len(x) < min_length:
        # Fall back to single-scale SSIM if signal too short
        levels = 1
        weights = np.array([1.0])

    def _l_cs(xl, yl):
        # Return luminance and contrast-structure components (Wang et al.)
        mu_x = gaussian_filter1d(xl, sigma=sigma, mode="reflect")
        mu_y = gaussian_filter1d(yl, sigma=sigma, mode="reflect")
        mu_x2, mu_y2 = mu_x * mu_x, mu_y * mu_y
        mu_xy = mu_x * mu_y
        s_x2 = gaussian_filter1d(xl * xl, sigma=sigma, mode="reflect") - mu_x2
        s_y2 = gaussian_filter1d(yl * yl, sigma=sigma, mode="reflect") - mu_y2
        s_xy = gaussian_filter1d(xl * yl, sigma=sigma, mode="reflect") - mu_xy

        # Ensure variances are non-negative (floating point precision can make them slightly negative)
        s_x2 = np.maximum(s_x2, 0.0)
        s_y2 = np.maximum(s_y2, 0.0)

        if L is None:
            # Use a more robust estimate of dynamic range
            std_x, std_y = np.std(xl), np.std(yl)
            if std_x < eps and std_y < eps:
                # Both signals are constant - check if they're equal
                if np.allclose(np.mean(xl), np.mean(yl), rtol=eps, atol=eps):
                    return 1.0, 1.0  # Identical constant signals
                else:
                    return 0.0, 0.0  # Different constant signals
            L_loc = 6.0 * np.maximum(std_x, std_y)
        else:
            L_loc = L
        C1 = (K1 * L_loc) ** 2
        C2 = (K2 * L_loc) ** 2

        # luminance and contrast-structure (cs)
        l_vals = (2 * mu_x * mu_y + C1) / (mu_x2 + mu_y2 + C1 + eps)
        cs = (2 * s_xy + C2) / (s_x2 + s_y2 + C2 + eps)

        # Clamp to valid range to prevent NaN from negative values raised to fractional powers
        # In theory, l and cs should be in [0, 1], but numerical issues can push them slightly outside
        l_mean = float(np.mean(np.clip(l_vals, 0.0, 1.0 + eps)))
        cs_mean = float(np.mean(np.clip(cs, 0.0, 1.0 + eps)))

        return l_mean, cs_mean

    xs, ys = x.copy(), y.copy()
    cs_list = []
    for s in range(levels - 1):
        _, cs = _l_cs(xs, ys)
        cs_list.append(cs)
        # downsample by averaging then decimating (anti-alias)
        xs = signal.decimate(xs, downsample, ftype="fir", zero_phase=True)
        ys = signal.decimate(ys, downsample, ftype="fir", zero_phase=True)

    l_M, _ = _l_cs(xs, ys)

    # Aggregate with weights - all values are now safely non-negative
    cs_array = np.array(cs_list)
    cs_prod = np.prod(cs_array ** weights[:-1])
    ms = cs_prod * (l_M ** weights[-1])

    # Final safety check
    if not np.isfinite(ms):
        return 0.0

    return float(ms)


def ms_ssim_1d(
    x: np.ndarray,
    y: np.ndarray,
    levels: int = 5,
    sigma: float = 1.5,
    K1: float = 0.01,
    K2: float = 0.03,
    L: float | None = None,
    downsample: int = 2,
    weights: np.ndarray | None = None,
    eps: float = 1e-12,
) -> float:
    """
    1D Multi-Scale Structural Similarity Index (MS-SSIM) between two signals.

    This function computes the MS-SSIM metric which assesses the perceptual similarity between two 1D signals
    across multiple scales. At each scale, it measures both the structural/contrast similarity and luminance
    similarity, with the final metric being a weighted combination across scales.

    For multivariate time series (2D arrays with shape (T, D)), the metric is computed for each dimension
    separately and then averaged across dimensions.

    The implementation follows Wang et al. (2003), decomposing the comparison into:
    - Contrast and structure (cs) components computed at each scale except the coarsest
    - Luminance (l) component computed only at the coarsest scale
    - Final score: MS-SSIM = (prod_i cs_i^{w_i}) * l^{w_M}

    Args:
        x: First input signal. Can be 1D array of shape (T,) or 2D array of shape (T, D).
        y: Second input signal. Must have the same shape as x.
        levels: Number of scales to analyze (default 5). Each level downsamples the signals.
        sigma: Standard deviation of Gaussian window used for local statistics (default 1.5).
               Unlike 2D MS-SSIM, this can remain fixed across scales for 1D signals.
        K1: First stability constant for SSIM calculation (default 0.01)
        K2: Second stability constant for SSIM calculation (default 0.03)
        L: Dynamic range of the signals. If None, estimated as 6*max(std(x),std(y)) at each scale.
           For normalized signals, L=1 is appropriate.
        downsample: Integer factor for downsampling between scales (default 2).
                   Uses anti-aliased decimation.
        weights: Array of weights for each scale, length must equal levels.
                If None, uses weights from Wang et al.: [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        eps: Small constant to prevent division by zero (default 1e-12)

    Returns:
        float: MS-SSIM score typically in range [0, 1] where 1 indicates perfect similarity.
              For multivariate inputs, this is the mean across dimensions.
              Values slightly outside this range are possible but rare.

    Notes:
        - Input signals should be aligned in time before comparison
        - The implementation is symmetric: ms_ssim_1d(x,y) = ms_ssim_1d(y,x)
        - Anti-aliased downsampling helps prevent artifacts between scales
        - For best results, signals should have similar dynamic ranges
        - The metric is more robust than single-scale SSIM to variations in scale
        - For multivariate time series, computes dimension-wise mean

    References:
        Wang, Z., Simoncelli, E. P., & Bovik, A. C. (2003). Multiscale structural similarity
        for image quality assessment. IEEE Asilomar Conference on Signals, Systems and
        Computers, 2, 1398-1402.
    """
    x = np.asarray(x).squeeze()
    y = np.asarray(y).squeeze()

    if x.ndim != y.ndim:
        raise ValueError("x and y must have the same number of dimensions")

    # Handle 1D case
    if x.ndim == 1:
        return _ms_ssim_1d_univariate(x, y, levels, sigma, K1, K2, L, downsample, weights, eps)

    # Handle 2D case (multivariate time series)
    elif x.ndim == 2:
        if x.shape[1] != y.shape[1]:
            raise ValueError(
                f"x and y must have the same number of dimensions. Got x.shape={x.shape}, y.shape={y.shape}"
            )

        scores = []
        for i in range(x.shape[1]):
            s = _ms_ssim_1d_univariate(x[:, i], y[:, i], levels, sigma, K1, K2, L, downsample, weights, eps)
            scores.append(s)

        return float(np.mean(scores))

    else:
        raise ValueError(f"Input arrays must be 1D or 2D. Got x.ndim={x.ndim}, y.ndim={y.ndim}")


# --------------------------------------------------------
# 3) MMD (RBF kernel) over values or patch-embeddings
# --------------------------------------------------------


def _extract_patches_1d(x: np.ndarray, patch_len: int, stride: int) -> np.ndarray:
    """
    Extract overlapping fixed-length patches from a 1D array.

    This function slides a window of length `patch_len` over the 1D input and
    returns overlapping patches separated by `stride` samples. Trailing incomplete
    windows are dropped. If the input is shorter than `patch_len`, a single
    zero-padded patch is returned.

    Args:
        x (np.ndarray): 1D input array.
        patch_len (int): Length of each patch (window size). Must be > 0.
        stride (int): Step size between consecutive patch starts. Must be > 0.

    Returns:
        np.ndarray: Array of shape (n_patches, patch_len) containing the patches.
            - If len(x) >= patch_len: n_patches = (len(x) - patch_len) // stride + 1
            - If len(x) < patch_len: shape is (1, patch_len) with zero-padding

    Raises:
        ValueError: If `patch_len` <= 0 or `stride` <= 0.

    Notes:
        - The input is coerced to float64 and flattened to 1D.
        - Uses numpy stride tricks to build a view, then returns a contiguous copy.
        - Falls back to a safe Python loop if stride-trick construction fails.

    Examples:
        >>> _extract_patches_1d(np.array([1, 2, 3, 4, 5]), patch_len=3, stride=2)
        array([[1., 2., 3.],
            [3., 4., 5.]])

        >>> _extract_patches_1d(np.array([1, 2]), patch_len=4, stride=1)
        array([[1., 2., 0., 0.]])
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if patch_len <= 0 or stride <= 0:
        raise ValueError("patch_len and stride must be positive integers")
    if x.size < patch_len:
        pad = np.zeros(patch_len, dtype=np.float64)
        pad[: x.size] = x
        return pad[None, :]
    n = (x.size - patch_len) // stride + 1
    try:
        shape = (n, patch_len)
        strides = (x.strides[0] * stride, x.strides[0])
        patches = as_strided(x, shape=shape, strides=strides)
        return patches.copy()
    except Exception:
        out = np.empty((n, patch_len), dtype=np.float64)
        for i in range(n):
            s = i * stride
            out[i] = x[s : s + patch_len]
        return out


def _patch_features_1d(
    P: np.ndarray,
    patch_feature: Literal["raw", "zscore", "diff", "dct", "fft_mag", "concat_tf"] = "raw",
    n_coeffs: int | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    P = np.asarray(P, dtype=np.float64)
    N, L = P.shape
    pf = patch_feature.lower()

    if pf == "raw":
        return P
    if pf == "zscore":
        mu = P.mean(axis=1, keepdims=True)
        sd = P.std(axis=1, keepdims=True)
        return (P - mu) / (sd + eps)
    if pf == "diff":
        D = np.diff(P, axis=1)
        mu = D.mean(axis=1, keepdims=True)
        sd = D.std(axis=1, keepdims=True)
        return (D - mu) / (sd + eps)

    if pf in {"dct", "concat_tf"}:
        from scipy.fft import dct
    if pf in {"fft_mag", "concat_tf"}:
        pass  # numpy.fft.rfft is always available

    if pf == "dct":
        from scipy.fft import dct

        k = n_coeffs if (n_coeffs is not None) else min(8, L)
        C = dct(P, type=2, norm="ortho", axis=1)[:, :k]  # type: ignore
        mu = C.mean(axis=0, keepdims=True)
        sd = C.std(axis=0, keepdims=True)
        return (C - mu) / (sd + eps)

    if pf == "fft_mag":
        kmax = L // 2 + 1
        k = n_coeffs if (n_coeffs is not None) else min(8, kmax)
        F = np.abs(np.fft.rfft(P, axis=1))[:, :k]
        F = np.log1p(F)
        mu = F.mean(axis=0, keepdims=True)
        sd = F.std(axis=0, keepdims=True)
        return (F - mu) / (sd + eps)

    if pf == "concat_tf":
        Z = (P - P.mean(axis=1, keepdims=True)) / (P.std(axis=1, keepdims=True) + eps)
        from scipy.fft import dct

        k_dct = n_coeffs if (n_coeffs is not None) else min(4, L)
        C = dct(P, type=2, norm="ortho", axis=1)[:, :k_dct]  # type: ignore
        C = (C - C.mean(0, keepdims=True)) / (C.std(0, keepdims=True) + eps)
        kmax = L // 2 + 1
        k_fft = n_coeffs if (n_coeffs is not None) else min(4, kmax)
        F = np.abs(np.fft.rfft(P, axis=1))[:, :k_fft]
        F = np.log1p(F)
        F = (F - F.mean(0, keepdims=True)) / (F.std(0, keepdims=True) + eps)
        return np.concatenate([Z, C, F], axis=1)

    raise ValueError("patch_feature must be one of {'raw','zscore','diff','dct','fft_mag','concat_tf'}")


# ---------- NEW: Random Fourier Features for RBF ----------
def _rff_features(X: np.ndarray, sigma: float, m: int, rng: np.random.Generator) -> np.ndarray:
    """
    Random Fourier Features for RBF kernel.
    phi(x) = sqrt(2/m) * cos(W x + b), with W ~ N(0, 1/sigma^2 I), b ~ Uniform[0, 2π].
    X: (n, d)
    """
    n, d = X.shape
    W = rng.normal(loc=0.0, scale=1.0 / (sigma + 1e-12), size=(d, m))
    b = rng.uniform(0.0, 2.0 * np.pi, size=(m,))
    Z = X @ W + b
    return np.sqrt(2.0 / m) * np.cos(Z)


def _median_bandwidth(X: np.ndarray, Y: np.ndarray, eps: float) -> float:
    XX = np.sum(X * X, axis=1, keepdims=True)
    YY = np.sum(Y * Y, axis=1, keepdims=True)
    Dxy = XX + YY.T - 2.0 * (X @ Y.T)
    Dxx = XX + XX.T - 2.0 * (X @ X.T)
    Dyy = YY + YY.T - 2.0 * (Y @ Y.T)
    pool = np.concatenate([Dxy.ravel(), Dxx[np.triu_indices_from(Dxx, 1)], Dyy[np.triu_indices_from(Dyy, 1)]])
    pool = pool[pool > 0]
    if pool.size == 0:
        return 1.0
    med = float(np.median(pool))
    sigma = float(np.sqrt(0.5 * med + eps))
    if not np.isfinite(sigma) or sigma < eps:
        sigma = 1.0
    return sigma


# ---------- CORE: MMD with RBF/COSINE + RFF option ----------
def _mmd_1d(
    x: np.ndarray,
    y: np.ndarray,
    use_patches: bool = False,
    patch_len: int = 16,
    stride: int = 8,
    bandwidth: float | None = None,
    eps: float = 1e-12,
    *,
    patch_feature: Literal["raw", "zscore", "diff", "dct", "fft_mag", "concat_tf"] = "raw",
    n_coeffs: int | None = None,
    kernel: Literal["rbf", "cosine"] = "rbf",
    rff_n_components: int | None = None,
    random_state: int | None = None,
) -> float:
    """
    Compute Maximum Mean Discrepancy (MMD) between two univariate time series.

    MMD is a kernel-based distance measure between distributions. This implementation
    supports both exact computation and fast approximation via Random Fourier Features,
    with additional options for patch-based comparison to capture local temporal patterns.

    Parameters
    ----------
    x : np.ndarray
        First time series (will be flattened to 1D).
    y : np.ndarray
        Second time series (will be flattened to 1D).
    use_patches : bool, default=False
        If True, extract overlapping patches and compute MMD on patch features.
        This captures local temporal patterns rather than point-wise distributions.
    patch_len : int, default=16
        Length of each patch when use_patches=True.
    stride : int, default=8
        Stride between consecutive patches when use_patches=True.
    bandwidth : float or None, default=None
        Kernel bandwidth (sigma) for RBF kernel. If None, uses median heuristic
        (median of pairwise distances between samples from x and y).
    eps : float, default=1e-12
        Small constant for numerical stability in normalization.
    patch_feature : {'raw', 'zscore', 'diff', 'dct', 'fft_mag', 'concat_tf'}, default='raw'
        Feature transformation applied to patches when use_patches=True:
        - 'raw': Use raw patch values
        - 'zscore': Normalize each patch to zero mean, unit variance
        - 'diff': First-order differences within patch
        - 'dct': Discrete Cosine Transform coefficients
        - 'fft_mag': FFT magnitude spectrum
        - 'concat_tf': Concatenate time-domain (raw) and frequency-domain (fft_mag) features
    n_coeffs : int or None, default=None
        Number of frequency coefficients to keep for 'dct' or 'fft_mag' features.
        If None, keeps all coefficients.
    kernel : {'rbf', 'cosine'}, default='rbf'
        Kernel function to use:
        - 'rbf': Radial Basis Function (Gaussian) kernel
        - 'cosine': Cosine similarity kernel
    rff_n_components : int or None, default=None
        Number of Random Fourier Features for fast RBF approximation.
        If provided (and > 0), uses RFF-based biased estimator instead of exact computation.
        Enables O(n·D) complexity instead of O(n²) where D is rff_n_components.
        Only applies to RBF kernel.
    random_state : int or None, default=None
        Random seed for reproducibility of RFF sampling.

    Returns
    -------
    float
        MMD² estimate (non-negative). Returns 0.0 if either input is empty.

    Notes
    -----
    Design Choices Beyond Basic MMD+RBF:

    1. **Unbiased Estimator**: Uses the unbiased U-statistic form with diagonal removal
       for exact computation, which has lower bias than the biased V-statistic.

    2. **Patch-Based Comparison**: Optional patch extraction allows comparing local
       temporal patterns rather than marginal distributions. Useful for capturing
       autocorrelation structure and dynamic patterns in time series.

    3. **Multiple Feature Spaces**: Supports various patch feature transformations
       (temporal, spectral, hybrid) to capture different aspects of time series similarity.

    4. **Cosine Kernel**: Alternative to RBF that is invariant to magnitude scaling,
       useful when comparing normalized patterns.

    5. **RFF Approximation**: Scalable approximation for large datasets using Random
       Fourier Features (Rahimi & Recht, 2007). Provides O(n·D) complexity with
       controllable accuracy via rff_n_components. Note: RFF uses biased estimator
       (feature-mean difference) for computational efficiency.

    6. **Automatic Bandwidth Selection**: Median heuristic for RBF bandwidth when
       not specified, providing reasonable defaults without manual tuning.

    The exact RBF/cosine computation uses the unbiased form:
        MMD² = (1/(n(n-1))) Σᵢ≠ⱼ k(xᵢ,xⱼ) + (1/(m(m-1))) Σᵢ≠ⱼ k(yᵢ,yⱼ) - (2/nm) Σᵢⱼ k(xᵢ,yⱼ)

    The RFF approximation uses the biased form:
        MMD² ≈ ‖mean(φ(x)) - mean(φ(y))‖² where φ is the RFF transformation

    References
    ----------
    Gretton, A., et al. (2012). A kernel two-sample test. JMLR.
    Rahimi, A., & Recht, B. (2007). Random features for large-scale kernel machines. NeurIPS.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    if use_patches:
        Px = _extract_patches_1d(x, patch_len, stride)
        Py = _extract_patches_1d(y, patch_len, stride)
        X = _patch_features_1d(Px, patch_feature=patch_feature, n_coeffs=n_coeffs, eps=eps)
        Y = _patch_features_1d(Py, patch_feature=patch_feature, n_coeffs=n_coeffs, eps=eps)
    else:
        X = x.reshape(-1, 1)
        Y = y.reshape(-1, 1)

    n, m = X.shape[0], Y.shape[0]
    if n == 0 or m == 0:
        return 0.0

    k = kernel.lower()
    if k not in {"rbf", "cosine"}:
        raise ValueError("kernel must be one of {'rbf','cosine'}")

    # ---- Cosine kernel (exact, unbiased form) ----
    if k == "cosine":
        # Normalize rows; zero rows stay zero.
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)
        Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + eps)

        Kxx = Xn @ Xn.T
        Kyy = Yn @ Yn.T
        Kxy = Xn @ Yn.T

        np.fill_diagonal(Kxx, 0.0)
        np.fill_diagonal(Kyy, 0.0)
        mmd2 = Kxx.sum() / (max(n * (n - 1), 1)) + Kyy.sum() / (max(m * (m - 1), 1)) - 2.0 * Kxy.mean()
        return float(max(mmd2, 0.0))

    # ---- RBF kernel: exact or RFF ----
    if rff_n_components is not None and rff_n_components > 0:
        # Fast MMD via feature-mean difference (biased but consistent)
        sigma = _median_bandwidth(X, Y, eps) if bandwidth is None else float(bandwidth)
        rng = np.random.default_rng(random_state)
        ZX = _rff_features(X, sigma, int(rff_n_components), rng)
        ZY = _rff_features(Y, sigma, int(rff_n_components), rng)
        muX = ZX.mean(axis=0)
        muY = ZY.mean(axis=0)
        mmd2 = float(np.sum((muX - muY) ** 2))
        return max(mmd2, 0.0)

    # Exact RBF (unbiased with diagonals dropped)
    XX = np.sum(X * X, axis=1, keepdims=True)
    YY = np.sum(Y * Y, axis=1, keepdims=True)
    Dxx = XX + XX.T - 2.0 * (X @ X.T)
    Dyy = YY + YY.T - 2.0 * (Y @ Y.T)
    Dxy = XX + YY.T - 2.0 * (X @ Y.T)

    if bandwidth is None:
        bandwidth = _median_bandwidth(X, Y, eps)
    sigma = float(bandwidth)
    gamma = 1.0 / (2.0 * sigma * sigma)

    Kxx = np.exp(-gamma * np.clip(Dxx, 0.0, None))
    Kyy = np.exp(-gamma * np.clip(Dyy, 0.0, None))
    Kxy = np.exp(-gamma * np.clip(Dxy, 0.0, None))

    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    mmd2 = Kxx.sum() / (max(n * (n - 1), 1)) + Kyy.sum() / (max(m * (m - 1), 1)) - 2.0 * Kxy.mean()
    return float(max(mmd2, 0.0))


def _mmd_batched(
    x: np.ndarray,
    y: np.ndarray,
    use_patches: bool = False,
    patch_len: int = 16,
    stride: int = 8,
    bandwidth: float | None = None,
    eps: float = 1e-12,
    patch_feature: Literal["raw", "zscore", "diff", "dct", "fft_mag", "concat_tf"] = "raw",
    n_coeffs: int | None = None,
    kernel: Literal["rbf", "cosine"] = "rbf",
    rff_n_components: int | None = None,
    random_state: int | None = None,
) -> np.ndarray:
    """
    Vectorized MMD for a batch of univariate 1D signals.

    Args:
        x: Batch of first signals of shape (batch_size, T)
        y: Batch of second signals of shape (batch_size, T)
        use_patches: If True, extract patches and compare patch distributions
        patch_len: Length of each patch
        stride: Stride for patch extraction
        bandwidth: Kernel bandwidth (sigma) for RBF kernel. If None, uses median heuristic
        eps: Numerical stability constant
        patch_feature: Feature transformation for patches
        n_coeffs: Number of frequency coefficients for spectral features
        kernel: Kernel function ('rbf' or 'cosine')
        rff_n_components: Number of Random Fourier Features for fast approximation
        random_state: Random seed for RFF sampling

    Returns:
        Array of MMD² values of shape (batch_size,)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    if x.ndim != 2:
        raise ValueError(f"Expected 2D input (batch_size, T), got shape {x.shape}")

    batch_size, T = x.shape
    mmd2_values = np.empty(batch_size, dtype=np.float64)

    k = kernel.lower()
    if k not in {"rbf", "cosine"}:
        raise ValueError("kernel must be one of {'rbf','cosine'}")

    # Process each sample in the batch
    for i in range(batch_size):
        try:
            xi = x[i]
            yi = y[i]

            if use_patches:
                # Extract and transform patches
                Px = _extract_patches_1d(xi, patch_len, stride)
                Py = _extract_patches_1d(yi, patch_len, stride)
                X = _patch_features_1d(Px, patch_feature=patch_feature, n_coeffs=n_coeffs, eps=eps)
                Y = _patch_features_1d(Py, patch_feature=patch_feature, n_coeffs=n_coeffs, eps=eps)
            else:
                X = xi.reshape(-1, 1)
                Y = yi.reshape(-1, 1)

            n, m = X.shape[0], Y.shape[0]
            if n == 0 or m == 0:
                mmd2_values[i] = 0.0
                continue

            # ---- Cosine kernel (exact, unbiased form) ----
            if k == "cosine":
                # Normalize rows
                Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)
                Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + eps)

                Kxx = Xn @ Xn.T
                Kyy = Yn @ Yn.T
                Kxy = Xn @ Yn.T

                np.fill_diagonal(Kxx, 0.0)
                np.fill_diagonal(Kyy, 0.0)
                mmd2 = Kxx.sum() / (max(n * (n - 1), 1)) + Kyy.sum() / (max(m * (m - 1), 1)) - 2.0 * Kxy.mean()
                mmd2_values[i] = float(max(mmd2, 0.0))
                continue

            # ---- RBF kernel: exact or RFF ----
            if rff_n_components is not None and rff_n_components > 0:
                # Fast MMD via RFF
                sigma = _median_bandwidth(X, Y, eps) if bandwidth is None else float(bandwidth)
                rng = np.random.default_rng(random_state)
                ZX = _rff_features(X, sigma, int(rff_n_components), rng)
                ZY = _rff_features(Y, sigma, int(rff_n_components), rng)
                muX = ZX.mean(axis=0)
                muY = ZY.mean(axis=0)
                mmd2 = float(np.sum((muX - muY) ** 2))
                mmd2_values[i] = max(mmd2, 0.0)
                continue

            # Exact RBF (unbiased with diagonals dropped)
            XX = np.sum(X * X, axis=1, keepdims=True)
            YY = np.sum(Y * Y, axis=1, keepdims=True)
            Dxx = XX + XX.T - 2.0 * (X @ X.T)
            Dyy = YY + YY.T - 2.0 * (Y @ Y.T)
            Dxy = XX + YY.T - 2.0 * (X @ Y.T)

            if bandwidth is None:
                sigma = _median_bandwidth(X, Y, eps)
            else:
                sigma = float(bandwidth)
            gamma = 1.0 / (2.0 * sigma * sigma)

            Kxx = np.exp(-gamma * np.clip(Dxx, 0.0, None))
            Kyy = np.exp(-gamma * np.clip(Dyy, 0.0, None))
            Kxy = np.exp(-gamma * np.clip(Dxy, 0.0, None))

            np.fill_diagonal(Kxx, 0.0)
            np.fill_diagonal(Kyy, 0.0)
            mmd2 = Kxx.sum() / (max(n * (n - 1), 1)) + Kyy.sum() / (max(m * (m - 1), 1)) - 2.0 * Kxy.mean()
            mmd2_values[i] = float(max(mmd2, 0.0))

        except Exception:
            mmd2_values[i] = np.nan

    return mmd2_values


def mmd(
    x: np.ndarray,
    y: np.ndarray,
    use_patches: bool = False,
    patch_len: int = 16,
    stride: int = 8,
    bandwidth: float | None = None,
    eps: float = 1e-12,
    *,
    patch_feature: Literal["raw", "zscore", "diff", "dct", "fft_mag", "concat_tf"] = "raw",
    n_coeffs: int | None = None,
    kernel: Literal["rbf", "cosine"] = "rbf",
    rff_n_components: int | None = None,
    random_state: int | None = None,
) -> float:
    """
    MMD between time series with:
      - optional local patch features (shape-aware),
      - kernel={'rbf','cosine'},
      - optional Random Fourier Features for fast RBF.

    Notes:
      * RFF path uses the feature-mean estimator (biased but consistent).
      * Cosine kernel L2-normalizes features per sample.
    """
    x = np.asarray(x).squeeze()
    y = np.asarray(y).squeeze()

    if x.ndim != y.ndim:
        raise ValueError("x and y must have the same number of dimensions")

    if x.ndim == 1:
        mmd2_val = _mmd_1d(
            x,
            y,
            use_patches=use_patches,
            patch_len=patch_len,
            stride=stride,
            bandwidth=bandwidth,
            eps=eps,
            patch_feature=patch_feature,
            n_coeffs=n_coeffs,
            kernel=kernel,
            rff_n_components=rff_n_components,
            random_state=random_state,
        )
        return np.sqrt(mmd2_val)
    elif x.ndim == 2:
        if x.shape[1] != y.shape[1]:
            raise ValueError(
                f"x and y must have the same number of dimensions. Got x.shape={x.shape}, y.shape={y.shape}"
            )
        ds = []
        # Take average over dimensions
        for d in range(x.shape[1]):
            mmd2_val = _mmd_1d(
                x[:, d],
                y[:, d],
                use_patches=use_patches,
                patch_len=patch_len,
                stride=stride,
                bandwidth=bandwidth,
                eps=eps,
                patch_feature=patch_feature,
                n_coeffs=n_coeffs,
                kernel=kernel,
                rff_n_components=rff_n_components,
                random_state=random_state,
            )
            ds.append(np.sqrt(mmd2_val))
        return np.sqrt(float(np.mean(ds)))

    else:
        raise ValueError(f"Input arrays must be 1D or 2D. Got x.ndim={x.ndim}, y.ndim={y.ndim}")


def energy_distance(
    x: np.ndarray,
    y: np.ndarray,
    use_patches: bool = False,
    patch_len: int = 16,
    stride: int = 8,
    eps: float = 1e-12,
    *,
    patch_feature: Literal["raw", "zscore", "diff", "dct", "fft_mag", "concat_tf"] = "raw",
    n_coeffs: int | None = None,
) -> float:
    """
    Energy Distance between two time series - an alternative to MMD with better dynamic range.

    Energy distance is defined as:
        E(X, Y) = 2·E[||X - Y||] - E[||X - X'||] - E[||Y - Y'||]

    where X, X' are i.i.d. samples from distribution P and Y, Y' from Q.

    Advantages over MMD:
    - No kernel bandwidth to tune (kernel-free)
    - Generally larger values, better dynamic range
    - Direct interpretation as expected distance
    - Always positive, no clipping needed

    Parameters
    ----------
    x, y : np.ndarray
        Time series to compare (1D arrays)
    use_patches : bool, default=False
        If True, extract patches and compare patch distributions
    patch_len, stride : int
        Patch extraction parameters
    eps : float
        Numerical stability constant
    patch_feature : str
        Feature transformation for patches
    n_coeffs : int or None
        Number of frequency coefficients for spectral features

    Returns
    -------
    float
        Energy distance (non-negative)

    Notes
    -----
    Energy distance is equivalent to MMD with a specific kernel (the distance kernel),
    but it's computed directly without exponential kernel evaluations, which can help
    with numerical stability and dynamic range.

    References
    ----------
    Székely, G. J., & Rizzo, M. L. (2013). Energy statistics: A class of statistics
    based on distances. Journal of Statistical Planning and Inference.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    if use_patches:
        Px = _extract_patches_1d(x, patch_len, stride)
        Py = _extract_patches_1d(y, patch_len, stride)
        X = _patch_features_1d(Px, patch_feature=patch_feature, n_coeffs=n_coeffs, eps=eps)
        Y = _patch_features_1d(Py, patch_feature=patch_feature, n_coeffs=n_coeffs, eps=eps)
    else:
        X = x.reshape(-1, 1)
        Y = y.reshape(-1, 1)

    n, m = X.shape[0], Y.shape[0]
    if n == 0 or m == 0:
        return 0.0

    # Precompute squared norms for efficient pairwise distance computation
    XX = np.sum(X * X, axis=1, keepdims=True)
    YY = np.sum(Y * Y, axis=1, keepdims=True)

    # E[||X - Y||] - compute pairwise L2 distances efficiently
    Dxy = XX + YY.T - 2.0 * (X @ Y.T)
    Dxy = np.sqrt(np.clip(Dxy, 0.0, None))
    term1 = 2.0 * np.mean(Dxy)

    # E[||X - X'||]
    if n > 1:
        Dxx = XX + XX.T - 2.0 * (X @ X.T)
        Dxx = np.sqrt(np.clip(Dxx, 0.0, None))
        np.fill_diagonal(Dxx, 0.0)
        term2 = Dxx.sum() / (n * (n - 1))
    else:
        term2 = 0.0

    # E[||Y - Y'||]
    if m > 1:
        Dyy = YY + YY.T - 2.0 * (Y @ Y.T)
        Dyy = np.sqrt(np.clip(Dyy, 0.0, None))
        np.fill_diagonal(Dyy, 0.0)
        term3 = Dyy.sum() / (m * (m - 1))
    else:
        term3 = 0.0

    energy = term1 - term2 - term3
    return float(max(energy, 0.0))


def _energy_distance_batched(
    x: np.ndarray,
    y: np.ndarray,
    use_patches: bool = False,
    patch_len: int = 16,
    stride: int = 8,
    eps: float = 1e-12,
    patch_feature: Literal["raw", "zscore", "diff", "dct", "fft_mag", "concat_tf"] = "raw",
    n_coeffs: int | None = None,
) -> np.ndarray:
    """
    Vectorized energy distance for a batch of univariate 1D signals.

    Args:
        x: Batch of first signals of shape (batch_size, T)
        y: Batch of second signals of shape (batch_size, T)
        use_patches: If True, extract patches and compare patch distributions
        patch_len: Length of each patch
        stride: Stride for patch extraction
        eps: Numerical stability constant
        patch_feature: Feature transformation for patches
        n_coeffs: Number of frequency coefficients for spectral features

    Returns:
        Array of energy distances of shape (batch_size,)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    if x.ndim != 2:
        raise ValueError(f"Expected 2D input (batch_size, T), got shape {x.shape}")

    batch_size, T = x.shape
    distances = np.empty(batch_size, dtype=np.float64)

    # Process each sample in the batch
    for i in range(batch_size):
        try:
            xi = x[i]
            yi = y[i]

            if use_patches:
                # Extract and transform patches
                Px = _extract_patches_1d(xi, patch_len, stride)
                Py = _extract_patches_1d(yi, patch_len, stride)
                X = _patch_features_1d(Px, patch_feature=patch_feature, n_coeffs=n_coeffs, eps=eps)
                Y = _patch_features_1d(Py, patch_feature=patch_feature, n_coeffs=n_coeffs, eps=eps)
            else:
                X = xi.reshape(-1, 1)
                Y = yi.reshape(-1, 1)

            n, m = X.shape[0], Y.shape[0]
            if n == 0 or m == 0:
                distances[i] = 0.0
                continue

            # Precompute squared norms for efficient pairwise distance computation
            XX = np.sum(X * X, axis=1, keepdims=True)
            YY = np.sum(Y * Y, axis=1, keepdims=True)

            # E[||X - Y||]
            Dxy = XX + YY.T - 2.0 * (X @ Y.T)
            Dxy = np.sqrt(np.clip(Dxy, 0.0, None))
            term1 = 2.0 * np.mean(Dxy)

            # E[||X - X'||]
            if n > 1:
                Dxx = XX + XX.T - 2.0 * (X @ X.T)
                Dxx = np.sqrt(np.clip(Dxx, 0.0, None))
                np.fill_diagonal(Dxx, 0.0)
                term2 = Dxx.sum() / (n * (n - 1))
            else:
                term2 = 0.0

            # E[||Y - Y'||]
            if m > 1:
                Dyy = YY + YY.T - 2.0 * (Y @ Y.T)
                Dyy = np.sqrt(np.clip(Dyy, 0.0, None))
                np.fill_diagonal(Dyy, 0.0)
                term3 = Dyy.sum() / (m * (m - 1))
            else:
                term3 = 0.0

            energy = term1 - term2 - term3
            distances[i] = float(max(energy, 0.0))

        except Exception:
            distances[i] = np.nan

    return distances


def _welch_psd(
    x: np.ndarray,
    fs: float = 1.0,
    nperseg: int | None = None,
    noverlap: int | None = None,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate power spectral density using Welch's method.

    Computes the power spectral density estimate using Welch's averaged periodogram method.
    The signal is divided into overlapping segments, each segment is windowed, and the
    periodograms are computed and averaged.

    Args:
        x: 1D input array containing the time series data.
        fs: Sampling frequency of the time series. Defaults to 1.0.
        nperseg: Length of each segment. If None, defaults to 256 samples.
        noverlap: Number of points to overlap between segments. If None, defaults to nperseg//2.
        window: Type of window function to apply. Default is "hann". Common options include
               "hamming", "blackman", and "bartlett".

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - f: Array of sample frequencies
            - Pxx: Power spectral density estimate at each frequency point.
                  Values are clipped to be non-negative.

    Note:
        The function uses scipy.signal.welch internally with detrend="constant" and
        scaling="density". The output is one-sided (frequencies from 0 to fs/2).
    """
    f, Pxx = signal.welch(
        x,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
        return_onesided=True,
    )
    Pxx = np.clip(Pxx, 0.0, None)
    return f, Pxx


def _mmd2_rbf_from_weights_on_grid(
    u: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    bandwidth: float | None = None,
    eps: float = 1e-12,
) -> float:
    """
    Compute MMD^2 between two discrete distributions over the same grid u,
    with probability weights p and q (sum to 1), using an RBF kernel on u.

    Args:
        u: Grid locations (e.g., frequencies), shape (M,)
        p: Weights over u for distribution P (>=0, sum ~ 1), shape (M,)
        q: Weights over u for distribution Q (>=0, sum ~ 1), shape (M,)
        bandwidth: RBF bandwidth over u; if None, uses median heuristic on pairwise |u_i - u_j|
        eps: Small epsilon for numerical stability

    Returns:
        MMD^2 as a non-negative float.
    """
    u = np.asarray(u).reshape(-1)
    p = np.asarray(p).reshape(-1)
    q = np.asarray(q).reshape(-1)

    # Normalize weights defensively
    ps = p.sum()
    qs = q.sum()
    if ps <= eps or qs <= eps:
        # Degenerate spectra → distance is just difference in total mass (both normalized to avoid NaNs)
        p = np.ones_like(u) / len(u)
        q = np.ones_like(u) / len(u)
    else:
        p = p / (ps + eps)
        q = q / (qs + eps)

    # Choose coordinate on which to compare
    diffs = u[:, None] - u[None, :]
    D2 = diffs**2

    if bandwidth is None:
        # Median heuristic on |u_i - u_j|, ignoring zeros
        # Fallback: span/20 if everything is equal
        iu = np.triu_indices(len(u), k=1)
        pairwise = np.sqrt(D2[iu])
        if pairwise.size == 0 or float(np.median(pairwise)) < eps:
            span = float(np.max(u) - np.min(u))
            bandwidth = max(span / 20.0, eps)
        else:
            bandwidth = float(np.median(pairwise)) + eps

    K = np.exp(-D2 / (2.0 * bandwidth**2))
    # Quadratic form with weights
    term_pp = float(p @ K @ p)
    term_qq = float(q @ K @ q)
    term_pq = float(p @ K @ q)
    return term_pp + term_qq - 2.0 * term_pq


def _spectral_mmd_1d(
    x: np.ndarray,
    y: np.ndarray,
    fs: float = 1.0,
    nperseg: int | None = None,
    noverlap: int | None = None,
    window: str = "hann",
    representation: str = "log_psd",
    bandwidth: float | None = None,
    return_spectral_information: bool = False,
    eps: float = 1e-12,
    *,
    mode: str = "values",  # NEW: {"values", "freq", "logfreq"}
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray] | float:
    """
    Helper function to compute spectral MMD-RBF for 1D signals.

    mode="values": original behavior — MMD on PSD *values* after `representation`.
    mode="freq":   shape-aware — MMD on frequency grid using normalized PSD as weights.
    mode="logfreq": same as "freq" but on log-frequency grid (scale-invariant).
    """
    f, Sx = _welch_psd(x, fs, nperseg, noverlap, window)
    _, Sy = _welch_psd(y, fs, nperseg, noverlap, window)

    mode = mode.lower()
    if mode == "values":
        # Your original pipeline
        if representation == "psd":
            Xv = Sx
            Yv = Sy
        elif representation == "norm_psd":
            Xv = Sx / (Sx.sum() + eps)
            Yv = Sy / (Sy.sum() + eps)
        elif representation == "mag":
            Xv = np.sqrt(Sx)
            Yv = np.sqrt(Sy)
        elif representation == "log_psd":
            Xv = np.log(Sx + eps)
            Yv = np.log(Sy + eps)
        else:
            raise ValueError("representation must be one of {'psd','norm_psd','mag','log_psd'}")

        mmd2 = _mmd_1d(Xv, Yv, bandwidth=bandwidth)
        if return_spectral_information:
            return mmd2, f, Xv, Yv
        return mmd2

    elif mode in {"freq", "logfreq"}:
        # Shape-aware spectral MMD on (log-)frequency axis with PSD as weights
        u = np.log(f + 1e-9) if mode == "logfreq" else f
        px = Sx / (Sx.sum() + eps)
        qy = Sy / (Sy.sum() + eps)

        mmd2 = _mmd2_rbf_from_weights_on_grid(u, px, qy, bandwidth=bandwidth, eps=eps)
        if return_spectral_information:
            # Return the weights we actually used (probabilities over frequency)
            return mmd2, f, px, qy
        return mmd2

    else:
        raise ValueError("mode must be one of {'values','freq','logfreq'}")


def spectral_mmd(
    x: np.ndarray,
    y: np.ndarray,
    fs: float = 1.0,
    nperseg: int | None = None,
    noverlap: int | None = None,
    window: str = "hann",
    representation: str = "log_psd",
    bandwidth: float | None = None,
    return_spectral_information: bool = False,
    eps: float = 1e-12,
    *,
    mode: str = "values",  # NEW: {"values", "freq", "logfreq"}
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray] | float:
    """
    Spectral MMD^2 between time series via Welch spectra + RBF MMD.

    mode="values": MMD over PSD value vectors (your original definition).
    mode="freq": MMD over frequency locations with PSD-normalized weights (shape-aware).
    mode="logfreq": same as "freq" but compares on log-frequency.

    For multivariate time series (2D arrays (T, D)), the metric is computed per dimension
    and averaged.

    Args:
        x, y: 1D (T,) or 2D (T, D) arrays.
        fs, nperseg, noverlap, window: Welch parameters.
        representation: Only used when mode="values".
        bandwidth: RBF bandwidth. If None:
            - mode="values": median heuristic on PSD values (handled by _mmd_1d).
            - mode in {"freq","logfreq"}: median heuristic on (log-)frequency grid.
        return_spectral_information: If True (and inputs are 1D), returns (mmd2, freqs, X_repr, Y_repr)
            - For mode="values": X_repr, Y_repr are the transformed PSD vectors.
            - For mode in {"freq","logfreq"}: X_repr, Y_repr are the frequency weights (px, qy).
        eps: Numerical stability.
        mode: "values" | "freq" | "logfreq" (see above).

    Returns:
        float or tuple depending on return_spectral_information.
    """
    x = np.asarray(x).squeeze()
    y = np.asarray(y).squeeze()

    if x.ndim != y.ndim:
        raise ValueError("x and y must have the same number of dimensions")

    if x.ndim == 1:
        return _spectral_mmd_1d(
            x, y, fs, nperseg, noverlap, window, representation, bandwidth, return_spectral_information, eps, mode=mode
        )

    elif x.ndim == 2:
        if return_spectral_information:
            raise ValueError("return_spectral_information=True is only supported for 1D inputs")

        if x.shape[1] != y.shape[1]:
            raise ValueError(
                f"x and y must have the same number of dimensions. Got x.shape={x.shape}, y.shape={y.shape}"
            )

        distances = []
        for i in range(x.shape[1]):
            d = _spectral_mmd_1d(
                x[:, i], y[:, i], fs, nperseg, noverlap, window, representation, bandwidth, False, eps, mode=mode
            )
            distances.append(float(d))

        return float(np.mean(distances))

    else:
        raise ValueError(f"Input arrays must be 1D or 2D. Got x.ndim={x.ndim}")


def _spectral_wasserstein_1d(
    x: np.ndarray,
    y: np.ndarray,
    fs: float = 1.0,
    nperseg: int | None = None,
    noverlap: int | None = None,
    p: int = 1,
    return_spectral_information: bool = False,
    eps: float = 1e-12,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray] | float:
    """
    Helper function to compute spectral Wasserstein distance for 1D signals.
    Internal use only - see spectral_wasserstein for full documentation.
    """
    # If nperseg is None, set it to avoid inconsistent frequency grids
    if nperseg is None:
        # Use the smaller of 256 or the minimum signal length to ensure both signals can be processed
        nperseg = min(256, len(x), len(y))

    fx, Sx = _welch_psd(x, fs, nperseg, noverlap)
    fy, Sy = _welch_psd(y, fs, nperseg, noverlap)

    # Align to a common frequency grid if needed (prefer higher-resolution grid)
    if len(fx) != len(fy) or not np.allclose(fx, fy):
        if len(fx) >= len(fy):
            f = fx
            Sy = np.interp(fx, fy, Sy)
        else:
            f = fy
            Sx = np.interp(fy, fx, Sx)
    else:
        f = fx

    # Non-negativity, finite checks, and power validation
    Sx = np.clip(Sx, 0.0, None)
    Sy = np.clip(Sy, 0.0, None)

    if not (np.isfinite(Sx).all() and np.isfinite(Sy).all()):
        return (float("nan"), f, Sx, Sy) if return_spectral_information else float("nan")

    Sx_sum = float(Sx.sum())
    Sy_sum = float(Sy.sum())
    if not (np.isfinite(Sx_sum) and np.isfinite(Sy_sum)):
        return (float("nan"), f, Sx, Sy) if return_spectral_information else float("nan")

    # Edge cases: degenerate spectra
    if Sx_sum < eps and Sy_sum < eps:
        return (0.0, f, Sx, Sy) if return_spectral_information else 0.0
    if (Sx_sum < eps) != (Sy_sum < eps):  # only one has power
        return (float("nan"), f, Sx, Sy) if return_spectral_information else float("nan")

    # Normalize to probability weights over frequency
    Sx /= Sx_sum
    Sy /= Sy_sum

    # Early exit: identical spectra after normalization
    if np.allclose(Sx, Sy, rtol=0.0, atol=eps):
        return (0.0, f, Sx, Sy) if return_spectral_information else 0.0

    # Ensure arrays have matching sizes
    assert len(f) == len(Sx) == len(Sy), f"Size mismatch: f={len(f)}, Sx={len(Sx)}, Sy={len(Sy)}"

    # Trivial grids: distance is zero if only one frequency bin
    if len(f) <= 1:
        return (0.0, f, Sx, Sy) if return_spectral_information else 0.0

    if p == 1:
        try:
            d = stats.wasserstein_distance(f, f, u_weights=Sx, v_weights=Sy)
        except (ValueError, RuntimeError):
            d = float("nan")
        return (float(d), f, Sx, Sy) if return_spectral_information else float(d)

    if p == 2:
        try:
            Fx = np.cumsum(Sx)
            Fy = np.cumsum(Sy)
            Fx /= Fx[-1]
            Fy /= Fy[-1]

            # Build strictly increasing CDF grids to avoid interpolation issues on flat segments
            Fx_unique, idx_x = np.unique(Fx, return_index=True)
            Fy_unique, idx_y = np.unique(Fy, return_index=True)

            # Pad CDFs to cover full [0,1] range
            xp_x = np.concatenate(([0.0], Fx_unique, [1.0]))
            fp_x = np.concatenate(([f[0]], f[idx_x], [f[-1]]))

            xp_y = np.concatenate(([0.0], Fy_unique, [1.0]))
            fp_y = np.concatenate(([f[0]], f[idx_y], [f[-1]]))

            # uniform quantiles
            u = np.linspace(0.0, 1.0, num=len(f), endpoint=True)
            fx_i = np.interp(u, xp_x, fp_x)
            fy_i = np.interp(u, xp_y, fp_y)
            d2 = np.mean((fx_i - fy_i) ** 2)
            d2 = float(d2) if np.isfinite(d2) and d2 >= 0 else float("nan")
            d = float(np.sqrt(d2)) if np.isfinite(d2) else float("nan")
        except (ValueError, RuntimeError, FloatingPointError):
            d = float("nan")
        return (d, f, Sx, Sy) if return_spectral_information else d

    raise ValueError("Only p=1 or p=2 supported.")


def _spectral_wasserstein_batched(
    x: np.ndarray,
    y: np.ndarray,
    fs: float = 1.0,
    nperseg: int | None = None,
    noverlap: int | None = None,
    p: int = 1,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Vectorized spectral Wasserstein distance for a batch of univariate 1D signals.

    Args:
        x: Batch of first signals of shape (batch_size, T)
        y: Batch of second signals of shape (batch_size, T)
        fs: Sampling frequency
        nperseg: Length of each segment for Welch's method
        noverlap: Number of points to overlap between segments
        p: Order of Wasserstein distance (1 or 2)
        eps: Small constant to prevent division by zero

    Returns:
        Array of Wasserstein distances of shape (batch_size,)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    if x.ndim != 2:
        raise ValueError(f"Expected 2D input (batch_size, T), got shape {x.shape}")

    batch_size, T = x.shape

    # Set nperseg
    if nperseg is None:
        nperseg = min(256, T)

    # For batched Welch, we need to call it per sample since scipy doesn't support batching
    # But we can still benefit from vectorized post-processing
    distances = np.empty(batch_size, dtype=np.float64)

    # Compute Welch PSD for all samples (unfortunately scipy.signal.welch doesn't batch)
    # So we still loop but vectorize the distance computation
    for i in range(batch_size):
        try:
            fx, Sx = _welch_psd(x[i], fs, nperseg, noverlap)
            fy, Sy = _welch_psd(y[i], fs, nperseg, noverlap)

            # Align to common frequency grid
            if len(fx) != len(fy) or not np.allclose(fx, fy):
                if len(fx) >= len(fy):
                    f = fx
                    Sy = np.interp(fx, fy, Sy)
                else:
                    f = fy
                    Sx = np.interp(fy, fx, Sx)
            else:
                f = fx

            # Non-negativity and finite checks
            Sx = np.clip(Sx, 0.0, None)
            Sy = np.clip(Sy, 0.0, None)

            if not (np.isfinite(Sx).all() and np.isfinite(Sy).all()):
                distances[i] = np.nan
                continue

            Sx_sum = float(Sx.sum())
            Sy_sum = float(Sy.sum())

            if not (np.isfinite(Sx_sum) and np.isfinite(Sy_sum)):
                distances[i] = np.nan
                continue

            # Edge cases
            if Sx_sum < eps and Sy_sum < eps:
                distances[i] = 0.0
                continue
            if (Sx_sum < eps) != (Sy_sum < eps):
                distances[i] = np.nan
                continue

            # Normalize
            Sx /= Sx_sum
            Sy /= Sy_sum

            # Check if identical
            if np.allclose(Sx, Sy, rtol=0.0, atol=eps):
                distances[i] = 0.0
                continue

            if len(f) <= 1:
                distances[i] = 0.0
                continue

            # Compute distance
            if p == 1:
                distances[i] = stats.wasserstein_distance(f, f, u_weights=Sx, v_weights=Sy)
            elif p == 2:
                # Compute CDFs
                Fx = np.cumsum(Sx)
                Fy = np.cumsum(Sy)
                Fx /= Fx[-1]
                Fy /= Fy[-1]

                # Build strictly increasing CDF grids
                Fx_unique, idx_x = np.unique(Fx, return_index=True)
                Fy_unique, idx_y = np.unique(Fy, return_index=True)

                # Pad CDFs
                xp_x = np.concatenate(([0.0], Fx_unique, [1.0]))
                fp_x = np.concatenate(([f[0]], f[idx_x], [f[-1]]))

                xp_y = np.concatenate(([0.0], Fy_unique, [1.0]))
                fp_y = np.concatenate(([f[0]], f[idx_y], [f[-1]]))

                # Interpolate
                u = np.linspace(0.0, 1.0, num=len(f), endpoint=True)
                fx_i = np.interp(u, xp_x, fp_x)
                fy_i = np.interp(u, xp_y, fp_y)
                d2 = np.mean((fx_i - fy_i) ** 2)

                if np.isfinite(d2) and d2 >= 0:
                    distances[i] = np.sqrt(d2)
                else:
                    distances[i] = np.nan
            else:
                raise ValueError("Only p=1 or p=2 supported")

        except Exception:
            distances[i] = np.nan

    return distances


def spectral_wasserstein(
    x: np.ndarray,
    y: np.ndarray,
    fs: float = 1.0,
    nperseg: int | None = None,
    noverlap: int | None = None,
    p: int = 1,
    return_spectral_information: bool = False,
    eps: float = 1e-12,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray] | float:
    """
    Calculate the Wasserstein distance between the normalized power spectra of two signals using Welch's method.

    This function computes the p-Wasserstein distance (p=1 or p=2) between the normalized power spectral densities
    of two input signals. The power spectra are estimated using Welch's method and then normalized to sum to 1,
    effectively treating them as probability distributions over frequencies.

    For multivariate time series (2D arrays with shape (T, D)), the metric is computed for each dimension
    separately and then averaged across dimensions.

    Args:
        x: First input signal. Can be 1D array of shape (T,) or 2D array of shape (T, D).
        y: Second input signal. Must have the same shape as x.
        fs: Sampling frequency of the signals in Hz. Defaults to 1.0.
        nperseg: Length of each segment for Welch's method. If None, defaults to 256 samples.
        noverlap: Number of samples to overlap between segments. If None, defaults to nperseg//2.
        p: Order of the Wasserstein distance. Must be either 1 or 2.
            p=1 uses scipy's exact implementation.
            p=2 uses a discrete approximation based on inverse CDF interpolation.
        return_spectral_information: If True, returns additional spectral information along with the distance.
            Note: Only supported for 1D inputs. Raises ValueError for 2D inputs.
        eps: Small epsilon to add to denominators to prevent division by zero. Defaults to 1e-12.

    Returns:
        If return_spectral_information is False:
            float: The p-Wasserstein distance between the normalized power spectra.
                   For multivariate inputs, this is the mean across dimensions.
        If return_spectral_information is True (only for 1D inputs):
            tuple containing:
            - float: The p-Wasserstein distance
            - np.ndarray: Frequency grid used for spectral estimation
            - np.ndarray: Normalized power spectrum of signal x
            - np.ndarray: Normalized power spectrum of signal y

    Notes:
        - For p=1, the exact Wasserstein distance is computed using scipy.stats.wasserstein_distance
        - For p=2, the distance is approximated by interpolating the inverse CDFs on a discrete grid
        - The power spectra are normalized to sum to 1 to treat them as probability distributions
        - Small epsilon (1e-12) is added to denominators to prevent division by zero
        - The distance will be larger when the spectral shapes differ more significantly
        - For multivariate time series, computes dimension-wise mean
    """
    x = np.asarray(x).squeeze()
    y = np.asarray(y).squeeze()

    if x.ndim != y.ndim:
        raise ValueError("x and y must have the same number of dimensions")

    # Handle 1D case
    if x.ndim == 1:
        return _spectral_wasserstein_1d(x, y, fs, nperseg, noverlap, p, return_spectral_information, eps)

    # Handle 2D case (multivariate time series)
    elif x.ndim == 2:
        if return_spectral_information:
            raise ValueError("return_spectral_information=True is only supported for 1D inputs")

        if x.shape[1] != y.shape[1]:
            raise ValueError(
                f"x and y must have the same number of dimensions. Got x.shape={x.shape}, y.shape={y.shape}"
            )

        distances = []
        for i in range(x.shape[1]):
            d = _spectral_wasserstein_1d(x[:, i], y[:, i], fs, nperseg, noverlap, p, False, eps)
            distances.append(d)

        return float(np.mean(distances))

    else:
        raise ValueError(f"Input arrays must be 1D or 2D. Got x.ndim={x.ndim}, y.ndim={y.ndim}")


def _cross_spectral_phase_similarity_1d(
    x: np.ndarray,
    y: np.ndarray,
    fs: float = 1.0,
    nperseg: int | None = None,
    noverlap: int | None = None,
    window: str = "hann",
    return_spectral_information: bool = False,
    eps: float = 1e-12,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray] | float:
    """
    Helper function to compute cross-spectral phase similarity for 1D signals.
    Internal use only - see cross_spectral_phase_similarity for full documentation.
    """
    f, Cxy = signal.csd(
        x,
        y,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        return_onesided=True,
        scaling="density",
    )
    phase = np.angle(Cxy)
    w = np.abs(Cxy)
    w_sum = w.sum() + eps
    score = float(np.sum(w * np.cos(phase)) / w_sum)
    if return_spectral_information:
        return score, f, phase, w
    return score


def cross_spectral_phase_similarity(
    x: np.ndarray,
    y: np.ndarray,
    fs: float = 1.0,
    nperseg: int | None = None,
    noverlap: int | None = None,
    window: str = "hann",
    return_spectral_information: bool = False,
    eps: float = 1e-12,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray] | float:
    """
    Compute a phase alignment score between two signals based on their cross-spectral phase.

    This function calculates a similarity score between -1 and 1 that measures how well
    the phases of two signals align across frequencies. A score of 1 indicates perfect
    phase alignment, 0 indicates no consistent phase relationship, and -1 indicates
    anti-phase alignment.

    The score is computed by taking the cosine of the cross-spectral phase at each
    frequency and weighting it by the magnitude of the cross-spectral density |Cxy|.
    This weighting ensures that frequencies with stronger coupling between the signals
    contribute more to the final score.

    For multivariate time series (2D arrays with shape (T, D)), the metric is computed for each dimension
    separately and then averaged across dimensions.

    Args:
        x: First input signal. Can be 1D array of shape (T,) or 2D array of shape (T, D).
        y: Second input signal. Must have the same dimensionality as x.
        fs (float): Sampling frequency of the signals. Defaults to 1.0.
        nperseg (int, optional): Length of each segment for spectral estimation.
            If None, defaults to signal length.
        noverlap (int, optional): Number of points to overlap between segments.
            If None, defaults to nperseg // 2.
        window (str): Window function to use. Defaults to "hann".
        return_spectral_information (bool): If True, returns additional spectral
            information along with the similarity score. Note: Only supported for 1D inputs.
        eps (float): Small epsilon to add to denominators to prevent division by zero. Defaults to 1e-12.

    Returns:
        If return_spectral_information is False:
            float: Phase similarity score in [-1, 1]. For multivariate inputs, this is the mean across dimensions.
        If return_spectral_information is True (only for 1D inputs):
            tuple containing:
            - float: Phase similarity score
            - np.ndarray: Frequency grid used for spectral estimation
            - np.ndarray: Phase angle at each frequency
            - np.ndarray: Magnitude weights |Cxy| at each frequency

    Notes:
        - The score is weighted by the magnitude of the cross-spectral density,
          giving more importance to frequencies where both signals have significant
          shared power
        - A small epsilon (1e-12) is added to denominators to prevent division by zero
        - The function uses scipy.signal.csd for cross-spectral density estimation
        - For multivariate time series, computes dimension-wise mean
    """
    x = np.asarray(x).squeeze()
    y = np.asarray(y).squeeze()

    if x.ndim != y.ndim:
        raise ValueError("x and y must have the same number of dimensions")

    # Handle 1D case
    if x.ndim == 1:
        return _cross_spectral_phase_similarity_1d(
            x, y, fs, nperseg, noverlap, window, return_spectral_information, eps
        )

    # Handle 2D case (multivariate time series)
    elif x.ndim == 2:
        if return_spectral_information:
            raise ValueError("return_spectral_information=True is only supported for 1D inputs")

        if x.shape[1] != y.shape[1]:
            raise ValueError(
                f"x and y must have the same number of dimensions. Got x.shape={x.shape}, y.shape={y.shape}"
            )

        scores = []
        for i in range(x.shape[1]):
            s = _cross_spectral_phase_similarity_1d(x[:, i], y[:, i], fs, nperseg, noverlap, window, False, eps)
            scores.append(s)

        return float(np.mean(scores))

    else:
        raise ValueError(f"Input arrays must be 1D or 2D. Got x.ndim={x.ndim}, y.ndim={y.ndim}")


def _mean_coherence_1d(
    x: np.ndarray,
    y: np.ndarray,
    fs: float = 1.0,
    nperseg: int | None = None,
    noverlap: int | None = None,
    window: str = "hann",
    weighted: bool = False,
    eps: float = 1e-12,
) -> float:
    """
    Helper function to compute mean coherence for 1D signals.
    Internal use only - see mean_coherence for full documentation.
    """
    _, coh = signal.coherence(x, y, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, detrend="constant")
    if not weighted:
        return float(np.mean(coh))

    # Weight by shared power estimate from Welch
    _, Sx = _welch_psd(x, fs, nperseg, noverlap)
    _, Sy = _welch_psd(y, fs, nperseg, noverlap)
    w = np.minimum(Sx, Sy)
    denom = w.sum()
    return float(np.sum(w * coh) / (denom if denom > 0 else eps))


def mean_coherence(
    x: np.ndarray,
    y: np.ndarray,
    fs: float = 1.0,
    nperseg: int | None = None,
    noverlap: int | None = None,
    window: str = "hann",
    weighted: bool = False,
) -> float:
    """
    Calculate the mean magnitude-squared coherence across frequencies between two signals.

    The magnitude-squared coherence estimates the correlation between two signals
    as a function of frequency. This function computes the coherence at each frequency
    and then takes the mean across all frequencies to produce a single similarity score.

    For multivariate time series (2D arrays with shape (T, D)), the metric is computed for each dimension
    separately and then averaged across dimensions.

    Args:
        x: First input signal. Can be 1D array of shape (T,) or 2D array of shape (T, D).
        y: Second input signal. Must have the same dimensionality as x.
        fs (float): Sampling frequency of the signals. Defaults to 1.0.
        nperseg (int, optional): Length of each segment for spectral estimation.
            If None, defaults to signal length.
        noverlap (int, optional): Number of points to overlap between segments.
            If None, defaults to nperseg // 2.
        window (str): Window function to use. Defaults to "hann".
        weighted (bool): If True, weights the coherence by the shared power estimate from Welch.

    Returns:
        float: Mean coherence score in [0, 1], where 1 indicates perfect coherence.
               For multivariate inputs, this is the mean across dimensions.

    Notes:
        - Uses scipy.signal.coherence for magnitude-squared coherence estimation
        - For multivariate time series, computes dimension-wise mean
    """
    x = np.asarray(x).squeeze()
    y = np.asarray(y).squeeze()

    if x.ndim != y.ndim:
        raise ValueError("x and y must have the same number of dimensions")

    # Handle 1D case
    if x.ndim == 1:
        return _mean_coherence_1d(x, y, fs, nperseg, noverlap, window, weighted)

    # Handle 2D case (multivariate time series)
    if x.ndim == 2:
        if x.shape[1] != y.shape[1]:
            raise ValueError(
                f"x and y must have the same number of dimensions. Got x.shape={x.shape}, y.shape={y.shape}"
            )

        return float(
            np.mean(
                [
                    _mean_coherence_1d(x[:, i], y[:, i], fs, nperseg, noverlap, window, weighted)
                    for i in range(x.shape[1])
                ]
            )
        )

    else:
        raise ValueError(f"Input arrays must be 1D or 2D. Got x.ndim={x.ndim}, y.ndim={y.ndim}")


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Squared Error
    """
    return np.mean(np.square(y_true - y_pred))  # type: ignore


def rmse(x: np.ndarray, y: np.ndarray) -> float:
    """
    Root Mean Squared Error
    """
    return np.sqrt(mse(x, y))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error
    """
    return np.mean(np.abs(y_true - y_pred))  # type: ignore


def smape(x, y, eps=1e-10, scaled=True):
    """Symmetric mean absolute percentage error"""
    scale = 0.5 if scaled else 1.0
    return scale * 200 * np.mean(np.abs(x - y) / (np.abs(x) + np.abs(y) + eps))


def _calculate_season_error(y_past, m, time_dim=-1):
    """
    Calculate the mean absolute error between the forward and backward slices of the
    past data.

    Args:
        y_past (np.ndarray): The past data
        m (int): The season length
        time_dim (int): The dimension of the time series

    Returns:
        float: The mean absolute error

    Examples:
        >>> calculate_season_error(y_past, m)
    """
    assert 0 < m < y_past.shape[time_dim], "Season length must be less than the length of the training data"
    yt_forward = np.take(y_past, range(m, y_past.shape[time_dim]), axis=time_dim)
    yt_backward = np.take(y_past, range(y_past.shape[time_dim] - m), axis=time_dim)
    return np.mean(np.abs(yt_forward - yt_backward))


def mase(y, yhat, y_train=None, m=1, time_dim=-1, eps=1e-10):
    """
    The mean absolute scaled error.

    Args:
        y (ndarray): The true values.
        yhat (ndarray): The predicted values.
        y_train (ndarray): The training values.
        m (int): The season length, which is the number of time steps that are
            skipped when computing the denominator. Default is 1.
        time_dim (int): The dimension of the time series. Default is -1.

    Returns:
        mase_val (float): The MASE error
    """
    if y_train is None:
        y_train = y.copy()

    assert _are_broadcastable(yhat.shape, y_train.shape)
    assert _are_broadcastable(y.shape, y_train.shape)

    season_error = _calculate_season_error(y_train, m, time_dim)
    return np.mean(np.abs(y - yhat)) / (season_error + eps)


def wape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-10) -> float:
    """
    Weighted Absolute Percentage Error
    """
    return 100 * np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + eps)


def wql(y_true: np.ndarray, y_pred: np.ndarray, quantile: float = 0.5, eps: float = 1e-10) -> float:
    """
    Weighted quantile loss (pinball loss)

    The loss is asymmetric - it penalizes over/under-prediction differently based on the quantile level.

    Args:
        y_true (np.ndarray): Actual observed values (ground truth), shape (N,) or (N, D)
        y_pred (np.ndarray): Predicted quantile values for the specified quantile level, shape (N,) or (N, D)
        quantile (float): Quantile level being predicted (0 < quantile < 1).
                         0.5 = median, 0.9 = 90th percentile, 0.1 = 10th percentile
        eps (float): Small value to avoid division by zero
    """
    if not (0 < quantile < 1):
        raise ValueError("quantile must be between 0 and 1")

    error = y_true - y_pred
    quantile_loss = np.maximum(quantile * error, (quantile - 1) * error)
    total_loss = np.sum(quantile_loss)
    normalization = np.sum(np.abs(y_true)) + eps

    return total_loss / normalization


def _mse_batched(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Vectorized MSE for a batch of flattened signals.

    Args:
        y_true: Batch of true signals of shape (batch_size, -1)
        y_pred: Batch of predicted signals of shape (batch_size, -1)

    Returns:
        Array of MSE values of shape (batch_size,)
    """
    return np.mean(np.square(y_true - y_pred), axis=1)


def _mae_batched(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Vectorized MAE for a batch of flattened signals.

    Args:
        y_true: Batch of true signals of shape (batch_size, -1)
        y_pred: Batch of predicted signals of shape (batch_size, -1)

    Returns:
        Array of MAE values of shape (batch_size,)
    """
    return np.mean(np.abs(y_true - y_pred), axis=1)


def _rmse_batched(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Vectorized RMSE for a batch of flattened signals.

    Args:
        y_true: Batch of true signals of shape (batch_size, -1)
        y_pred: Batch of predicted signals of shape (batch_size, -1)

    Returns:
        Array of RMSE values of shape (batch_size,)
    """
    return np.sqrt(np.mean(np.square(y_true - y_pred), axis=1))


def _smape_batched(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-10, scaled: bool = True) -> np.ndarray:
    """
    Vectorized SMAPE for a batch of flattened signals.

    Args:
        y_true: Batch of true signals of shape (batch_size, -1)
        y_pred: Batch of predicted signals of shape (batch_size, -1)
        eps: Small constant to prevent division by zero
        scaled: Whether to scale by 0.5

    Returns:
        Array of SMAPE values of shape (batch_size,)
    """
    scale = 0.5 if scaled else 1.0
    return (
        scale
        * 200
        * np.mean(
            np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + eps),
            axis=1,
        )
    )


def _calculate_season_error_batched(y_past: np.ndarray, m: int, eps: float = 1e-10) -> np.ndarray:
    """
    Calculate the mean absolute error between the forward and backward slices of the
    past data for a batch of signals.

    Args:
        y_past: Batch of past data of shape (batch_size, T)
        m: The season length
        eps: Small constant to prevent issues with edge cases

    Returns:
        Array of season errors of shape (batch_size,)
    """
    if y_past.ndim != 2:
        raise ValueError(f"Expected 2D input (batch_size, T), got shape {y_past.shape}")

    batch_size, T = y_past.shape

    if m <= 0 or m >= T:
        raise ValueError(f"Season length must be 0 < m < T, got m={m}, T={T}")

    # Forward slice: from m to end
    yt_forward = y_past[:, m:]
    # Backward slice: from 0 to T-m
    yt_backward = y_past[:, : T - m]

    # Compute mean absolute error per batch element
    return np.mean(np.abs(yt_forward - yt_backward), axis=1)


def _mase_batched(
    y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray | None = None, m: int = 1, eps: float = 1e-10
) -> np.ndarray:
    """
    Vectorized MASE (Mean Absolute Scaled Error) for a batch of flattened signals.

    Args:
        y_true: Batch of true signals of shape (batch_size, -1)
        y_pred: Batch of predicted signals of shape (batch_size, -1)
        y_train: Batch of training signals of shape (batch_size, -1). If None, uses y_true
        m: The season length for computing the scaling factor (default: 1)
        eps: Small constant to prevent division by zero

    Returns:
        Array of MASE values of shape (batch_size,)
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    if y_true.ndim != 2:
        raise ValueError(f"Expected 2D input (batch_size, -1), got shape {y_true.shape}")

    if y_train is None:
        y_train = y_true.copy()
    else:
        y_train = np.asarray(y_train, dtype=np.float64)

    # Compute season error for each batch element
    season_errors = _calculate_season_error_batched(y_train, m, eps)

    # Compute MAE per batch element
    mae_values = np.mean(np.abs(y_true - y_pred), axis=1)

    # Return scaled error
    return mae_values / (season_errors + eps)


def _wape_batched(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Vectorized WAPE (Weighted Absolute Percentage Error) for a batch of flattened signals.

    Args:
        y_true: Batch of true signals of shape (batch_size, -1)
        y_pred: Batch of predicted signals of shape (batch_size, -1)
        eps: Small constant to prevent division by zero

    Returns:
        Array of WAPE values of shape (batch_size,)
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    if y_true.ndim != 2:
        raise ValueError(f"Expected 2D input (batch_size, -1), got shape {y_true.shape}")

    # Compute sum of absolute errors per batch element
    numerator = np.sum(np.abs(y_true - y_pred), axis=1)
    # Compute sum of absolute true values per batch element
    denominator = np.sum(np.abs(y_true), axis=1)

    return 100 * numerator / (denominator + eps)


def _wql_batched(y_true: np.ndarray, y_pred: np.ndarray, quantile: float = 0.5, eps: float = 1e-10) -> np.ndarray:
    """
    Vectorized WQL (Weighted Quantile Loss) for a batch of flattened signals.

    Args:
        y_true: Batch of true signals of shape (batch_size, -1)
        y_pred: Batch of predicted signals of shape (batch_size, -1)
        quantile: Quantile level being predicted (0 < quantile < 1)
        eps: Small constant to prevent division by zero

    Returns:
        Array of WQL values of shape (batch_size,)
    """
    if not (0 < quantile < 1):
        raise ValueError("quantile must be between 0 and 1")

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    if y_true.ndim != 2:
        raise ValueError(f"Expected 2D input (batch_size, -1), got shape {y_true.shape}")

    # Compute error
    error = y_true - y_pred

    # Compute quantile loss per element
    quantile_loss = np.maximum(quantile * error, (quantile - 1) * error)

    # Sum quantile loss per batch element
    total_loss = np.sum(quantile_loss, axis=1)

    # Compute normalization per batch element
    normalization = np.sum(np.abs(y_true), axis=1)

    return total_loss / (normalization + eps)


def _pearson_batched(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Vectorized Pearson correlation for a batch of univariate 1D signals.

    Args:
        y_true: Batch of true signals of shape (batch_size, T)
        y_pred: Batch of predicted signals of shape (batch_size, T)
        eps: Small constant to prevent division by zero

    Returns:
        Array of Pearson correlation values of shape (batch_size,)
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    if y_true.ndim != 2:
        raise ValueError(f"Expected 2D input (batch_size, T), got shape {y_true.shape}")

    # Compute variances per batch element
    var_true = np.var(y_true, axis=1)
    var_pred = np.var(y_pred, axis=1)

    # Initialize result array
    batch_size = y_true.shape[0]
    correlations = np.zeros(batch_size)

    # Identify valid indices (non-zero variance)
    valid_mask = (var_true > eps) & (var_pred > eps)

    if np.any(valid_mask):
        # Vectorized Pearson correlation for valid elements
        # r = cov(X,Y) / (std(X) * std(Y))
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]

        # Center the data
        y_true_centered = y_true_valid - np.mean(y_true_valid, axis=1, keepdims=True)
        y_pred_centered = y_pred_valid - np.mean(y_pred_valid, axis=1, keepdims=True)

        # Compute correlation
        numerator = np.sum(y_true_centered * y_pred_centered, axis=1)
        denominator = np.sqrt(np.sum(y_true_centered**2, axis=1) * np.sum(y_pred_centered**2, axis=1))

        correlations[valid_mask] = numerator / (denominator + eps)

    return correlations


def _spearman_batched(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Vectorized Spearman correlation for a batch of univariate 1D signals.

    Args:
        y_true: Batch of true signals of shape (batch_size, T)
        y_pred: Batch of predicted signals of shape (batch_size, T)
        eps: Small constant to prevent division by zero

    Returns:
        Array of Spearman correlation values of shape (batch_size,)
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    if y_true.ndim != 2:
        raise ValueError(f"Expected 2D input (batch_size, T), got shape {y_true.shape}")

    # Compute variances per batch element
    var_true = np.var(y_true, axis=1)
    var_pred = np.var(y_pred, axis=1)

    # Initialize result array
    batch_size = y_true.shape[0]
    correlations = np.zeros(batch_size)

    # Identify valid indices (non-zero variance)
    valid_mask = (var_true > eps) & (var_pred > eps)

    if np.any(valid_mask):
        # Convert to ranks for valid elements
        # argsort twice gives ranks
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]

        # Compute ranks: argsort of argsort gives ranks
        ranks_true = np.argsort(np.argsort(y_true_valid, axis=1), axis=1).astype(np.float64)
        ranks_pred = np.argsort(np.argsort(y_pred_valid, axis=1), axis=1).astype(np.float64)

        # Now compute Pearson correlation on ranks
        ranks_true_centered = ranks_true - np.mean(ranks_true, axis=1, keepdims=True)
        ranks_pred_centered = ranks_pred - np.mean(ranks_pred, axis=1, keepdims=True)

        numerator = np.sum(ranks_true_centered * ranks_pred_centered, axis=1)
        denominator = np.sqrt(np.sum(ranks_true_centered**2, axis=1) * np.sum(ranks_pred_centered**2, axis=1))

        correlations[valid_mask] = numerator / (denominator + eps)

    return correlations


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Spearman Correlation. Returns dimensionwise mean for multivariate time series of
    shape (T, D)

    Args:
        y_true (np.ndarray): The true values
        y_pred (np.ndarray): The predicted values

    Returns:
        float: The Spearman Correlation
    """
    y_true, y_pred = np.array(y_true).squeeze(), np.array(y_pred).squeeze()
    if y_true.ndim != y_pred.ndim:
        raise ValueError("y_true and y_pred must have the same number of dimensions")

    var_true = np.var(y_true, axis=0) if y_true.ndim > 1 else np.var(y_true)
    var_pred = np.var(y_pred, axis=0) if y_pred.ndim > 1 else np.var(y_pred)

    if y_true.ndim == 1:
        if np.isclose(var_true, 0) or np.isclose(var_pred, 0):
            return 0.0
        return spearmanr(y_true, y_pred)[0]  # type: ignore

    else:
        all_vals = []
        for i in range(y_true.shape[1]):
            if np.isclose(var_true[i], 0) or np.isclose(var_pred[i], 0):  # type: ignore
                all_vals.append(0.0)
            else:
                all_vals.append(spearmanr(y_true[:, i], y_pred[:, i])[0])
        return np.mean(all_vals)  # type: ignore


def pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Pearson Correlation. Returns dimensionwise mean for multivariate time series of
    shape (T, D)

    Args:
        y_true (np.ndarray): The true values
        y_pred (np.ndarray): The predicted values

    Returns:
        float: The Pearson Correlation
    """
    y_true, y_pred = np.array(y_true).squeeze(), np.array(y_pred).squeeze()
    if y_true.ndim != y_pred.ndim:
        raise ValueError("y_true and y_pred must have the same number of dimensions")

    var_true = np.var(y_true, axis=0) if y_true.ndim > 1 else np.var(y_true)
    var_pred = np.var(y_pred, axis=0) if y_pred.ndim > 1 else np.var(y_pred)

    if y_true.ndim == 1:
        if np.isclose(var_true, 0) or np.isclose(var_pred, 0):
            return 0.0
        return pearsonr(y_true, y_pred)[0]  # type: ignore

    else:
        all_vals = []
        for i in range(y_true.shape[1]):
            if np.isclose(var_true[i], 0) or np.isclose(var_pred[i], 0):  # type: ignore
                all_vals.append(0.0)
            else:
                all_vals.append(pearsonr(y_true[:, i], y_pred[:, i])[0])
        return np.mean(all_vals)  # type: ignore


def _spectral_hellinger_distance_1d(p: np.ndarray, q: np.ndarray, axis: int = 0) -> float:
    """
    Compute the Hellinger distance between two distributions.

    Args:
        p (np.ndarray): The first distribution
        q (np.ndarray): The second distribution
        axis (int): The axis to sum over

    Returns:
        float: The Hellinger distance
    """
    assert np.allclose(1.0, [p.sum(), q.sum()]), "p and q must be normalized"
    return np.sqrt(1 - np.sum(np.sqrt(p * q), axis=axis))


def _spectral_hellinger_batched(
    ts_true: np.ndarray,
    ts_gen: np.ndarray,
    num_freq_bins: int = 100,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Vectorized spectral Hellinger distance for a batch of univariate 1D signals.

    Args:
        ts_true: Batch of true signals of shape (batch_size, T)
        ts_gen: Batch of generated signals of shape (batch_size, T)
        num_freq_bins: Number of frequency bins to use in FFT
        eps: Small value to prevent division by zero

    Returns:
        Array of Hellinger distances of shape (batch_size,)
    """
    ts_true = np.asarray(ts_true, dtype=np.float64)
    ts_gen = np.asarray(ts_gen, dtype=np.float64)

    if ts_true.shape != ts_gen.shape:
        raise ValueError("ts_true and ts_gen must have the same shape")

    if ts_true.ndim != 2:
        raise ValueError(f"Expected 2D input (batch_size, T), got shape {ts_true.shape}")

    # Vectorized FFT across batch dimension
    # FFT is computed along axis=1 (time dimension)
    P_true = (np.abs(np.fft.fft(ts_true, axis=1)) ** 2)[:, :num_freq_bins]
    P_gen = (np.abs(np.fft.fft(ts_gen, axis=1)) ** 2)[:, :num_freq_bins]

    # Normalize per batch element
    sum_true = P_true.sum(axis=1, keepdims=True)
    sum_gen = P_gen.sum(axis=1, keepdims=True)

    # Prevent division by zero
    sum_true = np.maximum(sum_true, eps)
    sum_gen = np.maximum(sum_gen, eps)

    P_true_norm = P_true / sum_true
    P_gen_norm = P_gen / sum_gen

    # Handle near-zero power cases with uniform distribution
    zero_power_true = sum_true.squeeze() <= eps
    zero_power_gen = sum_gen.squeeze() <= eps

    if np.any(zero_power_true):
        P_true_norm[zero_power_true] = 1.0 / num_freq_bins
    if np.any(zero_power_gen):
        P_gen_norm[zero_power_gen] = 1.0 / num_freq_bins

    # Compute Hellinger distance for each batch element
    hellinger_dists = np.sqrt(1.0 - np.sum(np.sqrt(P_true_norm * P_gen_norm), axis=1))

    return hellinger_dists


def spectral_hellinger(
    ts_true: np.ndarray,
    ts_gen: np.ndarray,
    num_freq_bins: int = 100,
    eps: float = 1e-10,
) -> float:
    """
    Compute the average Hellinger distance between power spectra of two multivariate
    time series.

    Args:
        ts_true (np.ndarray): True time series, shape (T, D) or (T,).
        ts_gen (np.ndarray): Generated time series, shape (T, D) or (T,).
        num_freq_bins (int): Number of frequency bins to use in FFT for power spectrum.
        eps (float): Small value to prevent division by zero. Default is 1e-10.

    Returns:
        avg_dh: Average Hellinger distance across all dimensions.

    References:
        Mikhaeil et al. Advances in Neural Information Processing Systems, 35:
            11297–11312, December 2022.
    """
    ts_true = np.asarray(ts_true).squeeze()
    ts_gen = np.asarray(ts_gen).squeeze()
    ndims = ts_true.ndim
    if ndims > 2:
        raise ValueError("ts_true and ts_gen must both be either shape (T, D) or (T,)")
    if ts_gen.ndim != ndims:
        raise ValueError("ts_true and ts_gen must have the same number of dimensions")

    if ndims == 1:
        ts_true = ts_true[:, None]
        ts_gen = ts_gen[:, None]

    # data dimension
    d = ts_true.shape[1]
    if d != ts_gen.shape[1]:
        raise ValueError("ts_true and ts_gen must have the same number of dimensions")

    # Vectorized FFT and power spectra across all dimensions
    P_true = (np.abs(np.fft.fft(ts_true, axis=0)) ** 2)[:num_freq_bins, :]
    P_gen = (np.abs(np.fft.fft(ts_gen, axis=0)) ** 2)[:num_freq_bins, :]

    # Normalize safely per-dimension; fallback to uniform if total power too small
    sum_true = P_true.sum(axis=0)
    sum_gen = P_gen.sum(axis=0)

    denom_true = np.maximum(sum_true, eps)[None, :]
    denom_gen = np.maximum(sum_gen, eps)[None, :]

    P_true_norm = P_true / denom_true
    P_gen_norm = P_gen / denom_gen

    # Override columns with near-zero total power to uniform distributions
    if np.any(sum_true <= eps):
        P_true_norm[:, sum_true <= eps] = 1.0 / num_freq_bins
    if np.any(sum_gen <= eps):
        P_gen_norm[:, sum_gen <= eps] = 1.0 / num_freq_bins

    # Hellinger distance per dimension and mean across dimensions
    hellinger_per_dim = np.sqrt(1.0 - np.sum(np.sqrt(P_true_norm * P_gen_norm), axis=0))
    return float(np.mean(hellinger_per_dim))


def _are_broadcastable(shape1: tuple[int, ...], shape2: tuple[int, ...]) -> bool:
    """
    Check if two numpy arrays are broadcastable.

    Args:
        shape1 (tuple[int, ...]): The shape of the first array
        shape2 (tuple[int, ...]): The shape of the second array

    Returns:
        bool: True if the arrays are broadcastable, False otherwise

    Examples:
        >>> are_broadcastable((1, 2, 3), (4, 5, 6))
        False
        >>> are_broadcastable((1, 2, 3), (3,))
        True
    """
    # reverse the shapes to align dimensions from the end
    shape1, shape2 = shape1[::-1], shape2[::-1]
    # iterate over the dimensions
    for dim1, dim2 in zip(shape1, shape2):
        if dim1 != dim2 and dim1 != 1 and dim2 != 1:
            return False
    return True


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    verbose: bool = False,
    include: list[str] | None = None,
    batch_axis: int | None = None,
    batch_aggregation: Literal["mean", "median"] | None = None,
) -> dict[str, float]:
    """
    Compute multiple time series metrics between true and predicted values.

    Args:
        y_true (np.ndarray): The true values of shape (..., T, ...), where T is the time dimension
        y_pred (np.ndarray): The predicted values of shape (..., T, ...), must be broadcastable with y_true
        verbose (bool): Whether to print the computed metrics. Default is False.
        include (optional, list[str]): List of metrics to compute. If None, computes all available metrics.
            Available metrics are: "mse", "mae", "rmse", "smape", "spearman", "pearson", "hellinger_distance"
        batch_axis (optional, int): The axis to treat as the batch dimension. If None, adds a singleton
            batch dimension at axis 0. The batch dimension must be the same for y_true and y_pred.

    Returns:
        dict[str, float]: Dictionary mapping metric names to their computed values, averaged across
            the batch dimension. Each metric is cast to float.

    Raises:
        ValueError: If the batch dimension sizes don't match between y_true and y_pred
        ValueError: If the shapes of y_true and y_pred are not broadcastable
        ValueError: If any requested metric in 'include' is not a valid metric name

    Examples:
        >>> # Basic usage with default settings
        >>> metrics = compute_metrics(y_true, y_pred)
        >>>
        >>> # Specify batch axis and compute only MSE and MAE
        >>> metrics = compute_metrics(y_true, y_pred, include=["mse", "mae"], batch_axis=0)
        >>>
        >>> # Print metrics while computing
        >>> metrics = compute_metrics(y_true, y_pred, verbose=True)
    """
    # Create a single batch dimension as dimension 0
    if batch_axis is None:
        batch_axis = 0
        y_true = y_true[None, ...]
        y_pred = y_pred[None, ...]

    assert y_pred.shape[batch_axis] == y_true.shape[batch_axis], (
        f"specified batch_dim {batch_axis} must be the same for y_true and y_pred"
    )
    assert _are_broadcastable(y_true.shape, y_pred.shape), "y_true and y_pred must have broadcastable shapes"

    metric_functions = {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "smape": smape,
        "mase": mase,
        "wape": wape,
        "wql": wql,
        "spearman": spearman,
        "pearson": pearson,
        "ssim": ssim_1d,
        "ms_ssim": ms_ssim_1d,
        "mmd": partial(mmd, kernel="rbf", use_patches=False, patch_feature="raw", patch_len=16, stride=8),
        "energy_distance": partial(energy_distance, use_patches=False, patch_feature="raw", patch_len=16, stride=8),
        "spectral_mmd": spectral_mmd,
        "spectral_hellinger": spectral_hellinger,
        "spectral_wasserstein-1": partial(spectral_wasserstein, p=1),
        "spectral_wasserstein-2": partial(spectral_wasserstein, p=2),
        "cross_spectral_phase_similarity": cross_spectral_phase_similarity,
        "mean_coherence": mean_coherence,
    }

    if include is None:
        include = list(metric_functions.keys())

    assert all(metric in metric_functions for metric in include), (
        f"Invalid metrics specified. Must be one of {list(metric_functions.keys())}"
    )

    # Mapping of metrics to their batched functions
    # Flat metrics that work on reshaped (batch_size, -1) arrays
    flat_batched_metrics = {
        "mse": _mse_batched,
        "mae": _mae_batched,
        "rmse": _rmse_batched,
        "smape": _smape_batched,
        "mase": _mase_batched,
        "wql": _wql_batched,
        "wape": _wape_batched,
    }

    # Non-flat metrics that work on (batch_size, T) or (batch_size, T, D) arrays
    batched_metric_map = {
        "ssim": _ssim_1d_univariate_batched,
        "pearson": _pearson_batched,
        "spearman": _spearman_batched,
        "spectral_hellinger": _spectral_hellinger_batched,
        "spectral_wasserstein-1": lambda y_true, y_pred: _spectral_wasserstein_batched(y_true, y_pred, p=1),
        "spectral_wasserstein-2": lambda y_true, y_pred: _spectral_wasserstein_batched(y_true, y_pred, p=2),
        "energy_distance": lambda y_true, y_pred: _energy_distance_batched(
            y_true, y_pred, use_patches=False, patch_feature="raw", patch_len=16, stride=8
        ),
        "mmd": lambda y_true, y_pred: np.sqrt(
            _mmd_batched(y_true, y_pred, kernel="rbf", use_patches=False, patch_feature="raw", patch_len=16, stride=8)
        ),
    }

    batch_size = y_true.shape[batch_axis]
    metrics = {}

    # Move batch axis to position 0 once for all vectorizable metrics
    if batch_axis != 0:
        y_true_batch0 = np.moveaxis(y_true, batch_axis, 0)
        y_pred_batch0 = np.moveaxis(y_pred, batch_axis, 0)
    else:
        y_true_batch0 = y_true
        y_pred_batch0 = y_pred

    for metric in include:
        # Fast path for vectorizable metrics
        if metric in flat_batched_metrics or metric in batched_metric_map:
            try:
                # Simple flat metrics (mse, mae, rmse, smape)
                if metric in flat_batched_metrics:
                    y_true_flat = y_true_batch0.reshape(batch_size, -1)
                    y_pred_flat = y_pred_batch0.reshape(batch_size, -1)
                    values = flat_batched_metrics[metric](y_true_flat, y_pred_flat)

                # Complex metrics with specialized batched functions
                else:
                    if verbose:
                        print(f"Computing {metric} with batched function")
                    batched_fn = batched_metric_map[metric]

                    if y_true_batch0.ndim == 2:
                        # Univariate case: (batch_size, T)
                        values = batched_fn(y_true_batch0, y_pred_batch0)
                    elif y_true_batch0.ndim == 3:
                        # Multivariate case: (batch_size, T, D) - average over dimensions
                        num_dims = y_true_batch0.shape[2]
                        values = np.mean(
                            [batched_fn(y_true_batch0[:, :, d], y_pred_batch0[:, :, d]) for d in range(num_dims)],
                            axis=0,
                        )
                    else:
                        raise ValueError(f"{metric} expects 2D or 3D input, got {y_true_batch0.ndim}D")

            except Exception as e:
                if verbose:
                    print(f"Warning: Vectorized {metric} computation failed, falling back to loop: {e}")
                # Fallback to loop
                values = np.empty(batch_size, dtype=np.float64)
                for i in range(batch_size):
                    try:
                        values[i] = metric_functions[metric](
                            np.take(y_true, i, axis=batch_axis),
                            np.take(y_pred, i, axis=batch_axis),
                        )
                    except Exception:
                        values[i] = np.nan
                        if verbose:
                            print(f"Warning: {metric} computation failed for batch {i}")

        else:
            # Loop-based path for non-vectorizable metrics
            values = np.empty(batch_size, dtype=np.float64)
            for i in range(batch_size):
                try:
                    values[i] = metric_functions[metric](
                        np.take(y_true, i, axis=batch_axis),
                        np.take(y_pred, i, axis=batch_axis),
                    )
                except Exception as e:
                    values[i] = np.nan
                    if verbose:
                        print(f"Warning: {metric} computation failed for batch {i}: {e}")

        # Aggregate results
        if batch_aggregation == "mean":
            metrics[metric] = float(np.nanmean(values))
        elif batch_aggregation == "median":
            metrics[metric] = float(np.nanmedian(values))
        else:  # None case
            metrics[metric] = values.tolist() if isinstance(values, np.ndarray) else values

    if verbose:
        for key, value in metrics.items():
            if isinstance(value, (list, np.ndarray)):
                print(f"{key}: {value}")
            else:
                print(f"{key}: {value:.4f}")

    return metrics
