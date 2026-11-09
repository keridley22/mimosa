from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from skimage.restoration import rolling_ball
from skimage.transform import resize

from .config import Config, CFG
from .io_utils import ensure_dir, safe_name, save_ome_zyx, strip_tiff_suffix


def normalise_to_uint16(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    arr = arr - np.nanmin(arr)
    maxv = np.nanmax(arr)
    if maxv > 0:
        arr = arr / maxv
    return np.clip(arr * 65535, 0, 65535).astype(np.uint16)


def fast_rolling_ball_plane(plane: np.ndarray, cfg: Config = CFG) -> np.ndarray:
    """Estimate rolling-ball background at reduced XY resolution, then resize back."""
    plane = np.asarray(plane, dtype=np.float32)
    scale = cfg.rolling_ball_downsample_scale
    if not (0 < scale <= 1):
        raise ValueError("rolling_ball_downsample_scale must be > 0 and <= 1")

    if scale == 1:
        background = rolling_ball(plane, radius=cfg.rolling_ball_radius_px)
    else:
        small_shape = (
            max(1, round(plane.shape[0] * scale)),
            max(1, round(plane.shape[1] * scale)),
        )
        small = resize(
            plane,
            small_shape,
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.float32)
        scaled_radius = max(1, round(cfg.rolling_ball_radius_px * scale))
        small_background = rolling_ball(small, radius=scaled_radius)
        background = resize(
            small_background,
            plane.shape,
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.float32)

    corrected = plane - background
    corrected[corrected < 0] = 0
    return corrected


def preprocess_stack_zyx(stack: np.ndarray, cfg: Config = CFG) -> np.ndarray:
    """Fast rolling-ball subtraction + Gaussian blur on a ZYX stack."""
    stack = np.asarray(stack)
    if stack.ndim != 3:
        raise ValueError(f"Expected ZYX stack, got {stack.shape}")

    processed = np.empty(stack.shape, dtype=np.float32)
    for z in range(stack.shape[0]):
        corrected = fast_rolling_ball_plane(stack[z], cfg)
        if cfg.gaussian_sigma_px > 0:
            corrected = ndi.gaussian_filter(corrected, sigma=cfg.gaussian_sigma_px)
        processed[z] = corrected

    if cfg.preprocess_dtype == "uint16":
        return normalise_to_uint16(processed)
    if cfg.preprocess_dtype == "float32":
        return processed.astype(np.float32)
    raise ValueError(f"Unsupported preprocess_dtype: {cfg.preprocess_dtype}")


def read_ometiff_as_zcyx(path: Path) -> tuple[np.ndarray, str]:
    """Read an OME-TIFF and standardise common axis layouts to ZCYX."""
    with tifffile.TiffFile(path) as tif:
        if not tif.series:
            raise ValueError(f"No TIFF series found in {path.name}")
        series = tif.series[0]
        arr = series.asarray()
        axes = str(getattr(series, "axes", ""))

    original_axes = axes
    original_shape = arr.shape

    # Remove singleton axes while keeping metadata aligned.
    if axes and len(axes) == arr.ndim:
        keep_axes = []
        squeeze_axes = []
        for i, (ax, size) in enumerate(zip(axes, arr.shape)):
            if size == 1:
                squeeze_axes.append(i)
            else:
                keep_axes.append(ax)
        if squeeze_axes:
            arr = np.squeeze(arr, axis=tuple(squeeze_axes))
            axes = "".join(keep_axes)
    else:
        arr = np.squeeze(arr)

    if axes == "ZCYX":
        return arr, original_axes
    if axes == "CZYX":
        return np.moveaxis(arr, 0, 1), original_axes
    if axes == "ZYX":
        return arr[:, np.newaxis, :, :], original_axes
    if axes == "CYX":
        return arr[np.newaxis, :, :, :], original_axes
    if axes == "YX":
        return arr[np.newaxis, np.newaxis, :, :], original_axes

    if "T" in axes:
        t_axis = axes.index("T")
        if arr.shape[t_axis] != 1:
            raise ValueError(f"{path.name} has >1 timepoint; expected T=1")
        arr = np.take(arr, 0, axis=t_axis)
        axes = axes.replace("T", "")
        if axes == "ZCYX":
            return arr, original_axes
        if axes == "CZYX":
            return np.moveaxis(arr, 0, 1), original_axes

    # Fallback for known Bio-Formats exports.
    if arr.ndim == 4 and arr.shape[1] <= 6:
        print(f"  WARNING: assuming ZCYX from shape {original_shape}, axes={original_axes!r}")
        return arr, original_axes
    if arr.ndim == 4 and arr.shape[0] <= 6:
        print(f"  WARNING: assuming CZYX from shape {original_shape}, axes={original_axes!r}")
        return np.moveaxis(arr, 0, 1), original_axes
    if arr.ndim == 3:
        return arr[:, np.newaxis, :, :], original_axes

    raise ValueError(
        f"Could not convert {path.name} to ZCYX. "
        f"Original shape={original_shape}, axes={original_axes!r}, current shape={arr.shape}"
    )


def collect_input_ometiffs(cfg: Config = CFG) -> list[Path]:
    files: list[Path] = []
    for suffix in cfg.input_suffixes:
        files.extend(cfg.raw_input_dir.glob(f"*{suffix}"))
    return sorted(set(files), key=lambda p: p.name)


def should_process_file(path: Path, arr_zcyx: np.ndarray, cfg: Config) -> tuple[bool, str]:
    name = path.name
    if cfg.filename_must_contain is not None and cfg.filename_must_contain not in name:
        return False, f"filename does not contain {cfg.filename_must_contain!r}"
    for bad in cfg.filename_skip_contains:
        if bad and bad in name:
            return False, f"filename contains skipped term {bad!r}"
    if arr_zcyx.ndim != 4:
        return False, f"not ZCYX after conversion: {arr_zcyx.shape}"
    z, c, _, _ = arr_zcyx.shape
    if z < cfg.min_z_planes:
        return False, f"too few Z planes: Z={z}"
    if cfg.require_two_channels:
        needed = max(cfg.mx04_channel_index, cfg.oc_channel_index) + 1
        if c < needed:
            return False, f"not enough channels: C={c}; need {needed}"
    return True, "included"


def copy_training_subset(cfg: Config = CFG) -> None:
    for channel in (cfg.mx04_tag, cfg.oc_tag):
        src_files = sorted((cfg.preprocessed_dir / channel).glob("*.ome.tif*"))
        dst = cfg.training_subset_dir / channel
        ensure_dir(dst)
        if not src_files:
            continue
        n = min(cfg.n_training_subset, len(src_files))
        idx = np.linspace(0, len(src_files) - 1, n).round().astype(int)
        for i in sorted(set(idx.tolist())):
            shutil.copy2(src_files[i], dst / src_files[i].name)


def preprocess_ometiff_folder(cfg: Config = CFG) -> None:
    """Preprocess all converted multi-channel OME-TIFFs."""
    ensure_dir(cfg.output_root)
    ensure_dir(cfg.mx04_preprocessed_dir)
    ensure_dir(cfg.oc_preprocessed_dir)

    image_files = collect_input_ometiffs(cfg)
    if not image_files:
        raise FileNotFoundError(f"No OME-TIFFs found in {cfg.raw_input_dir}")

    print(f"Found {len(image_files)} OME-TIFFs")
    print(f"Preprocessed images will be saved under:\n  {cfg.preprocessed_dir}")
    print(f"  MX04 -> {cfg.mx04_preprocessed_dir}")
    print(f"  OC   -> {cfg.oc_preprocessed_dir}")

    log_rows: list[dict] = []
    skipped_rows: list[dict] = []

    for i, img_path in enumerate(image_files, start=1):
        print(f"\n[{i}/{len(image_files)}] {img_path.name}")
        try:
            arr_zcyx, axes = read_ometiff_as_zcyx(img_path)
            include, reason = should_process_file(img_path, arr_zcyx, cfg)
            if not include:
                print(f"  SKIPPED: {reason}")
                skipped_rows.append({"source_file": img_path.name, "reason": reason})
                continue

            z, c, y, x = arr_zcyx.shape
            sample_base = safe_name(strip_tiff_suffix(img_path))
            mx04_raw = arr_zcyx[:, cfg.mx04_channel_index, :, :]
            oc_raw = arr_zcyx[:, cfg.oc_channel_index, :, :]

            if mx04_raw.shape != oc_raw.shape:
                raise ValueError(f"MX04/OC shape mismatch: {mx04_raw.shape} vs {oc_raw.shape}")

            print(f"  shape={arr_zcyx.shape}; preprocessing MX04...")
            mx04 = preprocess_stack_zyx(mx04_raw, cfg)
            print("  preprocessing OC...")
            oc = preprocess_stack_zyx(oc_raw, cfg)

            mx04_out = cfg.mx04_preprocessed_dir / f"{sample_base}{cfg.mx04_suffix}"
            oc_out = cfg.oc_preprocessed_dir / f"{sample_base}{cfg.oc_suffix}"
            save_ome_zyx(mx04_out, mx04, cfg)
            save_ome_zyx(oc_out, oc, cfg)

            print(f"  saved MX04: {mx04_out}")
            print(f"  saved OC:   {oc_out}")

            log_rows.append({
                "source_file": img_path.name,
                "sample_base": sample_base,
                "original_axes": axes,
                "shape_zcyx": str(arr_zcyx.shape),
                "Z": z,
                "C": c,
                "Y": y,
                "X": x,
                "mx04_channel_index": cfg.mx04_channel_index,
                "oc_channel_index": cfg.oc_channel_index,
                "rolling_ball_radius_px": cfg.rolling_ball_radius_px,
                "rolling_ball_downsample_scale": cfg.rolling_ball_downsample_scale,
                "gaussian_sigma_px": cfg.gaussian_sigma_px,
                "mx04_path": str(mx04_out),
                "oc_path": str(oc_out),
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            skipped_rows.append({"source_file": img_path.name, "reason": f"ERROR: {e}"})

    pd.DataFrame(log_rows).to_csv(cfg.output_root / "preprocessing_log.csv", index=False)
    pd.DataFrame(skipped_rows).to_csv(cfg.output_root / "preprocessing_skipped_files.csv", index=False)
    copy_training_subset(cfg)

    print("\nPreprocessing complete.")
    print(f"Preprocessed images: {cfg.preprocessed_dir}")
    print(f"Training subset:     {cfg.training_subset_dir}")
