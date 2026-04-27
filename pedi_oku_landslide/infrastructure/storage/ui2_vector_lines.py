import json
import os
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
from rasterio.crs import CRS
from shapely.geometry import LineString, MultiLineString


DEFAULT_CRS = CRS.from_epsg(6678)


def save_selected_lines_gpkg(
    lines: List[LineString],
    crs,
    output_path: str = "output/UI2/step2_selected_lines/selected_lines.gpkg",
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if crs is None:
        crs = DEFAULT_CRS
    elif isinstance(crs, str):
        try:
            crs = CRS.from_string(crs)
        except Exception:
            crs = DEFAULT_CRS
    gdf = gpd.GeoDataFrame(geometry=lines, crs=crs)
    gdf.to_file(output_path, driver="GPKG")
    return f"[✓] Saved {len(lines)} lines → {output_path}", {"gpkg_path": output_path}


def read_vector_lines(file_path: str) -> Tuple[List[LineString], Optional[object]]:
    gdf = gpd.read_file(file_path)
    crs = gdf.crs
    lines: List[LineString] = []
    for geom in gdf.geometry:
        if isinstance(geom, MultiLineString):
            for g in geom.geoms:
                if isinstance(g, LineString) and len(g.coords) >= 2:
                    lines.append(LineString(g.coords))
        elif isinstance(geom, LineString) and len(geom.coords) >= 2:
            lines.append(LineString(geom.coords))
    return lines, crs


def write_shared_json(before_file: str, after_file: str, workspace: str = "output/UI2", **extras):
    os.makedirs(workspace, exist_ok=True)
    json_path = os.path.join(workspace, "ui_shared_data.json")
    data = {"before_file": before_file, "after_file": after_file, "workspace": workspace}
    for k, v in (extras or {}).items():
        if v is not None:
            data[k] = v
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return f"[✓] Shared data saved to {json_path}", {"json_path": json_path}
