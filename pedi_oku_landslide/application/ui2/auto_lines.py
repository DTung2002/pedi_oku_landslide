from typing import Dict, List, Optional

import rasterio

from pedi_oku_landslide.pipeline.runners.ui2.ui2_auto_lines import generate_auto_lines_from_arrays


def generate_auto_lines_from_slipzone(
    dem_path: str,
    dx_path: str,
    dy_path: str,
    slip_mask_path: str,
    main_num_even: int,
    main_offset_m: float,
    cross_num_even: int,
    cross_offset_m: float,
    base_length_m: Optional[float] = None,
    min_mag_thresh: float = 1e-4,
) -> Dict[str, List[Dict]]:
    with rasterio.open(dem_path) as src:
        transform = src.transform
    with rasterio.open(dx_path) as src_dx:
        dx = src_dx.read(1).astype(float)
    with rasterio.open(dy_path) as src_dy:
        dy = src_dy.read(1).astype(float)
    with rasterio.open(slip_mask_path) as src_mask:
        mask = (src_mask.read(1) > 0)
    return generate_auto_lines_from_arrays(
        dx=dx,
        dy=dy,
        mask=mask,
        transform=transform,
        main_num_even=main_num_even,
        main_offset_m=main_offset_m,
        cross_num_even=cross_num_even,
        cross_offset_m=cross_offset_m,
        base_length_m=base_length_m,
        min_mag_thresh=min_mag_thresh,
    )
