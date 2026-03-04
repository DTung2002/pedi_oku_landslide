import numpy as np
from scipy.ndimage import gaussian_filter

def smooth_gaussian(arr: np.ndarray, sigma_px: float) -> np.ndarray:
    """
    Gaussian smoothing in pixel units.
    - arr: 2D float32/float64 array
    - sigma_px: standard deviation in pixels (e.g. 1.0 ~ nhẹ, 2.0 ~ vừa, 3.0+ ~ mạnh)
    """
    if sigma_px <= 0:
        return arr
    # giữ NaN: tạm thời thay bằng median để tránh rách, sau đó phục hồi NaN
    nan_mask = ~np.isfinite(arr)
    med = float(np.nanmedian(arr)) if np.isfinite(np.nanmedian(arr)) else 0.0
    work = np.where(nan_mask, med, arr)
    # kernel size ~ 6*sigma + 1 (truncate=3.0 => radius=3*sigma)
    out = gaussian_filter(work, sigma=sigma_px, truncate=3.0)
    # khôi phục NaN
    out[nan_mask] = np.nan
    return out.astype(arr.dtype, copy=False)
