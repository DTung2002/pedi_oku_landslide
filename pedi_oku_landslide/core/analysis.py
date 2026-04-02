from typing import Sequence, Union

import numpy as np
from scipy.ndimage import convolve, uniform_filter

def smooth_mean(arr: np.ndarray, radius_px: Union[float, Sequence[float]]) -> np.ndarray:
    """
    Mean smoothing (average) in pixel units.
    - arr: 2D float32/float64 array
    - radius_px:
        * scalar: isotropic radius in pixels
        * sequence(ry, rx): anisotropic radius in pixels for (row, col)
    """
    if isinstance(radius_px, (tuple, list)):
        if len(radius_px) != 2:
            raise ValueError("radius_px sequence must have length 2: (ry, rx)")
        ry = float(radius_px[0])
        rx = float(radius_px[1])
        if ry <= 0 and rx <= 0:
            return arr
        size = (
            max(1, int(round(ry * 2 + 1))),
            max(1, int(round(rx * 2 + 1))),
        )
    else:
        r = float(radius_px)
        if r <= 0:
            return arr
        s = max(1, int(round(r * 2 + 1)))
        size = (s, s)

    # giữ NaN: tạm thời thay bằng median để tránh rách, sau đó phục hồi NaN
    nan_mask = ~np.isfinite(arr)
    med = float(np.nanmedian(arr)) if np.isfinite(np.nanmedian(arr)) else 0.0
    work = np.where(nan_mask, med, arr)

    # uniform_filter calculates the multidimensional uniform filter.
    # size = 2 * radius + 1 for each axis
    out = uniform_filter(work, size=size)

    # khôi phục NaN
    out[nan_mask] = np.nan
    return out.astype(arr.dtype, copy=False)


def _radius_pair_px(radius_px: Union[float, Sequence[float]]) -> tuple[float, float]:
    if isinstance(radius_px, (tuple, list)):
        if len(radius_px) != 2:
            raise ValueError("radius_px sequence must have length 2: (ry, rx)")
        ry = max(0.0, float(radius_px[0]))
        rx = max(0.0, float(radius_px[1]))
        return ry, rx
    r = max(0.0, float(radius_px))
    return r, r


def _gaussian_circle_kernel(
    radius_px: Union[float, Sequence[float]],
    sigma_percent: float = 50.0,
) -> np.ndarray:
    ry, rx = _radius_pair_px(radius_px)
    if ry <= 0.0 and rx <= 0.0:
        return np.ones((1, 1), dtype=float)

    ry_i = max(1, int(np.ceil(ry)))
    rx_i = max(1, int(np.ceil(rx)))
    yy, xx = np.mgrid[-ry_i:ry_i + 1, -rx_i:rx_i + 1]

    if ry > 0.0:
        yy_norm = yy / float(ry)
    else:
        yy_norm = np.zeros_like(yy, dtype=float)
    if rx > 0.0:
        xx_norm = xx / float(rx)
    else:
        xx_norm = np.zeros_like(xx, dtype=float)

    circle_mask = (yy_norm * yy_norm + xx_norm * xx_norm) <= 1.0
    if not np.any(circle_mask):
        circle_mask[ry_i, rx_i] = True

    sigma_factor = max(1e-6, float(sigma_percent) / 100.0)
    sigma_y = max(1e-6, float(ry) * sigma_factor) if ry > 0.0 else 1e-6
    sigma_x = max(1e-6, float(rx) * sigma_factor) if rx > 0.0 else 1e-6

    kernel = np.exp(-0.5 * ((yy / sigma_y) ** 2 + (xx / sigma_x) ** 2))
    kernel = np.where(circle_mask, kernel, 0.0)
    denom = float(np.sum(kernel))
    if denom <= 0.0 or not np.isfinite(denom):
        kernel = np.zeros_like(kernel, dtype=float)
        kernel[ry_i, rx_i] = 1.0
        return kernel
    return kernel / denom


def smooth_gaussian_qgis(
    arr: np.ndarray,
    radius_px: Union[float, Sequence[float]],
    sigma_percent: float = 50.0,
) -> np.ndarray:
    """
    Approximate QGIS/SAGA Gaussian Filter with:
    - kernel radius in pixels
    - sigma expressed as percentage of the kernel radius
    - circular kernel support
    """
    ry, rx = _radius_pair_px(radius_px)
    if ry <= 0.0 and rx <= 0.0:
        return arr

    kernel = _gaussian_circle_kernel((ry, rx), sigma_percent=float(sigma_percent))
    arr_f = np.asarray(arr, dtype=float)
    nan_mask = ~np.isfinite(arr_f)
    valid = (~nan_mask).astype(float)
    work = np.where(nan_mask, 0.0, arr_f)

    weighted = convolve(work * valid, kernel, mode="constant", cval=0.0)
    weights = convolve(valid, kernel, mode="constant", cval=0.0)

    out = np.full(arr_f.shape, np.nan, dtype=float)
    keep = weights > 1e-12
    out[keep] = weighted[keep] / weights[keep]
    out[nan_mask] = np.nan
    return out.astype(arr.dtype, copy=False)
