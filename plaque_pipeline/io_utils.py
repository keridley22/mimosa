from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import tifffile

from .config import Config, CFG


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_name(text: str) -> str:
    text = str(text).strip().replace(" ", "_")
    text = re.sub(r"[^A-Za-z0-9_+\-.]", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_") or "unnamed"


def strip_tiff_suffix(path: Path) -> str:
    name = path.name
    lower = name.lower()
    for suffix in (".ome.tiff", ".ome.tif", ".tiff", ".tif"):
        if lower.endswith(suffix):
            return name[:-len(suffix)]
    return path.stem


def save_ome_zyx(path: Path, arr_zyx: np.ndarray, cfg: Config = CFG) -> None:
    ensure_dir(path.parent)
    arr_zyx = np.asarray(arr_zyx)
    tifffile.imwrite(
        path,
        arr_zyx,
        ome=True,
        metadata={
            "axes": "ZYX",
            "PhysicalSizeX": cfg.voxel_xy_um,
            "PhysicalSizeY": cfg.voxel_xy_um,
            "PhysicalSizeZ": cfg.voxel_z_um,
        },
        compression="zlib",
    )
