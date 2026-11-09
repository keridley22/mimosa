"""Reject channel or spatial calibration mismatches before analysing images."""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
import tifffile


def validate_pixels(pixels, cfg):
    spacing = {}
    for axis, expected in (("X", cfg.voxel_xy_um), ("Y", cfg.voxel_xy_um), ("Z", cfg.voxel_z_um)):
        key = f"PhysicalSize{axis}"
        value = float(pixels.get(key, "nan"))
        unit = pixels.get(key + "Unit", "µm")
        factors = {"µm": 1, "μm": 1, "um": 1, "nm": .001, "mm": 1000, "m": 1e6}
        if unit not in factors:
            raise ValueError(f"Unrecognised unit {unit!r} for {key}")
        value *= factors[unit]
        allowed = (cfg.allowed_input_voxel_xy_um if cfg.resample_input_to_nominal and axis in ("X", "Y")
                   else (expected,))
        if not math.isfinite(value) or value <= 0 or not any(math.isclose(
                value, candidate, rel_tol=cfg.voxel_spacing_relative_tolerance) for candidate in allowed):
            raise ValueError(f"{key}={value} µm differs from configured {expected} µm")
        spacing[axis] = value
    if not math.isclose(spacing["X"], spacing["Y"], rel_tol=1e-6):
        raise ValueError("This pipeline requires equal X/Y voxel spacing")
    channels = [x.get("Name", "") for x in pixels if x.tag.endswith("}Channel")]
    # Verify the known acquisition layout, without deriving marker identities
    # from Leica's misleading nominal channel names.
    if tuple(channels) != cfg.expected_ome_channel_names:
        raise ValueError(f"Unexpected acquisition channel names/order: {channels}")
    indices = (cfg.mx04_channel_index, cfg.oc_channel_index)
    if len(set(indices)) != 2 or any(i < 0 or i >= len(channels) for i in indices):
        raise ValueError(f"Invalid marker channel indices: {indices}")
    return spacing


def validate_ometiff(path, cfg):
    with tifffile.TiffFile(path) as tif:
        if not tif.ome_metadata:
            raise ValueError(f"No OME metadata in {path}")
        root = ET.fromstring(tif.ome_metadata)
        pixels = next(x for x in root.iter() if x.tag.endswith("}Pixels"))
        return validate_pixels(pixels, cfg)
