import numpy as np
from matplotlib.colors import LightSource

def hillshade(dem: np.ndarray, azdeg: float = 315, altdeg: float = 45) -> np.ndarray:
    """Tạo hillshade uint8 từ DEM (mảng float)."""
    ls = LightSource(azdeg=azdeg, altdeg=altdeg)
    shaded = ls.hillshade(dem, vert_exag=1, dx=1, dy=1)
    return (np.clip(shaded, 0, 1) * 255).astype("uint8")
