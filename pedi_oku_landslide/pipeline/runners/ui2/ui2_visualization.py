from typing import Optional

import numpy as np


def hillshade(z: np.ndarray, cell: float) -> np.ndarray:
    zf = z.astype("float32")
    mask = ~np.isfinite(zf)
    if mask.any():
        zf[mask] = np.nanmean(zf)
    gy, gx = np.gradient(zf, cell, cell)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gy, gx)
    az = np.deg2rad(315.0)
    alt = np.deg2rad(45.0)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    hs = np.clip(hs, 0, 1)
    return (hs * 255).astype(np.uint8)


def rgba_from_scalar(
    values: np.ndarray,
    cm: str = "turbo",
    alpha: float = 0.75,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> np.ndarray:
    import matplotlib.cm as cm_mod

    a = values.astype("float32")
    nan_mask = ~np.isfinite(a)
    if vmin is None:
        vmin = np.nanpercentile(a, 2)
    if vmax is None:
        vmax = np.nanpercentile(a, 98)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = np.nanmin(a), np.nanmax(a)
    rng = float(max(1e-9, vmax - vmin))
    x = np.clip((a - vmin) / rng, 0.0, 1.0)
    rgba = cm_mod.get_cmap(cm)(x, bytes=True).astype(np.uint8)
    a_byte = int(max(0, min(255, round(alpha * 255.0))))
    rgba[..., 3] = a_byte
    if nan_mask.any():
        rgba[..., 3][nan_mask] = 0
    return rgba
