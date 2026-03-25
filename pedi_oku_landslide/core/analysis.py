from typing import Sequence, Union

import numpy as np
from scipy.ndimage import uniform_filter

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
