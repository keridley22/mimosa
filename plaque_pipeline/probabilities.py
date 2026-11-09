from __future__ import annotations

import re
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .config import Config, CFG
from .io_utils import ensure_dir, save_ome_zyx


def find_h5_probability_dataset(h5: h5py.File) -> str:
    candidates: list[str] = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.ndim >= 3:
            candidates.append(name)

    h5.visititems(visitor)
    preferred = [c for c in candidates if "exported_data" in c.lower() or "prob" in c.lower()]
    if preferred:
        return preferred[0]
    if candidates:
        return candidates[0]
    raise ValueError("No suitable probability dataset found in H5 file")


def probability_to_zyx(prob: np.ndarray, foreground_index: int, axes: str | None = None) -> np.ndarray:
    prob = np.asarray(prob)
    print(f"      Raw probability shape: {prob.shape}")
    if axes is None:
        if prob.ndim == 4 and prob.shape[-1] == 2:
            axes = "zyxc"
        elif prob.ndim == 5 and prob.shape[0] == 1 and prob.shape[-1] == 2:
            axes = "tzyxc"
        else:
            raise ValueError("Export both probability classes with axes zyxc, or include HDF5 axistags")
    axes = axes.lower()
    if len(axes) != prob.ndim or len(set(axes)) != len(axes):
        raise ValueError(f"Invalid axes {axes!r} for {prob.shape}")
    if "t" in axes:
        i = axes.index("t")
        if prob.shape[i] != 1:
            raise ValueError("Multiple timepoints are not supported")
        prob = np.take(prob, 0, axis=i)
        axes = axes.replace("t", "")
    if set(axes) != set("zyxc") or prob.shape[axes.index("c")] != 2:
        raise ValueError(f"Expected 3D, two-class probabilities, got {axes} {prob.shape}")
    if not np.isfinite(prob).all() or prob.min() < 0 or prob.max() > 1:
        raise ValueError("Expected finite probabilities in [0, 1]; export float32 without rescaling")
    prob = np.take(prob, foreground_index, axis=axes.index("c"))
    axes = axes.replace("c", "")
    prob = np.transpose(prob, [axes.index(a) for a in "zyx"])
    print(f"      Foreground ZYX shape: {prob.shape}")
    return prob.astype(np.float32)


def _clean_h5_base(h5_path: Path, channel: str) -> str:
    """
    Convert an ilastik probability filename back to the original image basename.

    Examples:
        sample_MX04.h5
            -> sample

        sample_MX04_Probabilities.h5
            -> sample

        sample_MX04.ome.tiff_Probabilities.h5
            -> sample
    """
    base = h5_path.name

    # Remove ilastik probability/H5 endings
    base = re.sub(
        r"_Probabilities\.(?:h5|hdf5)$",
        "",
        base,
        flags=re.IGNORECASE,
    )

    base = re.sub(
        r"\.(?:h5|hdf5)$",
        "",
        base,
        flags=re.IGNORECASE,
    )

    # Remove any TIFF extension that ilastik retained
    base = re.sub(
        r"\.ome\.tiff?$",
        "",
        base,
        flags=re.IGNORECASE,
    )

    base = re.sub(
        r"\.tiff?$",
        "",
        base,
        flags=re.IGNORECASE,
    )
    base = re.sub(r"\.ome$", "", base, flags=re.IGNORECASE)

    # Remove the existing channel suffix so it is added exactly once later
    if base.lower().endswith(f"_{channel.lower()}"):
        base = base[: -(len(channel) + 1)]

    return base


def h5_to_masks(channel: str, cfg: Config = CFG) -> None:
    in_dir = cfg.probability_dir / channel
    out_dir = cfg.masks_dir / channel
    ensure_dir(out_dir)

    h5_files = sorted(list(in_dir.glob("*.h5")) + list(in_dir.glob("*.hdf5")))
    if not h5_files:
        raise FileNotFoundError(f"No .h5/.hdf5 probability maps found in {in_dir}")

    suffix = cfg.mx04_suffix if channel == cfg.mx04_tag else cfg.oc_suffix
    rows = []
    for h5_path in h5_files:
        print(f"\nConverting {channel}: {h5_path.name}")
        with h5py.File(h5_path, "r") as h5:
            dataset_name = find_h5_probability_dataset(h5)
            dataset = h5[dataset_name]
            prob = dataset[()]
            tags = dataset.attrs.get("axistags")
            axes = "".join(a["key"] for a in json.loads(tags)["axes"]) if tags else None

        fg = probability_to_zyx(prob, cfg.foreground_class_index, axes)
        mask = (fg >= cfg.probability_threshold).astype(np.uint8) * 255
        base = _clean_h5_base(h5_path, channel)
        out_path = out_dir / f"{base}{suffix}"
        save_ome_zyx(out_path, mask, cfg)
        voxels_on = int((mask > 0).sum())
        print(f"      Saved mask: {out_path.name}; voxels on: {voxels_on}")
        rows.append({
            "channel": channel,
            "h5_file": h5_path.name,
            "dataset": dataset_name,
            "mask_file": out_path.name,
            "threshold": cfg.probability_threshold,
            "foreground_class_index": cfg.foreground_class_index,
            "shape_zyx": str(mask.shape),
            "voxels_on": voxels_on,
        })

    pd.DataFrame(rows).to_csv(out_dir / f"{channel}_h5_to_mask_log.csv", index=False)


def h5_to_all_masks(cfg: Config = CFG) -> None:
    # Fail before producing partial results when a batch export is incomplete.
    from .io_utils import strip_tiff_suffix
    import tifffile
    for channel in (cfg.mx04_tag, cfg.oc_tag):
        expected = {strip_tiff_suffix(p).removesuffix("_" + channel): p
                    for p in (cfg.preprocessed_dir / channel).glob("*.ome.tif*")}
        files = list((cfg.probability_dir / channel).glob("*.h5")) + list((cfg.probability_dir / channel).glob("*.hdf5"))
        actual = {_clean_h5_base(p, channel): p for p in files}
        if len(actual) != len(files) or not expected or set(actual) != set(expected):
            raise ValueError(f"Incomplete/duplicate {channel} export. Missing: {sorted(set(expected)-set(actual))}; "
                             f"unexpected: {sorted(set(actual)-set(expected))}")
        for name, path in actual.items():
            with h5py.File(path, "r") as h:
                d = h[find_h5_probability_dataset(h)]
                tags = d.attrs.get("axistags")
                axes = "".join(a["key"] for a in json.loads(tags)["axes"]) if tags else None
                fg = probability_to_zyx(d[()], cfg.foreground_class_index, axes)
            with tifffile.TiffFile(expected[name]) as t:
                if fg.shape != t.series[0].shape:
                    raise ValueError(f"Probability/input shape mismatch for {name}")
    h5_to_masks(cfg.mx04_tag, cfg)
    h5_to_masks(cfg.oc_tag, cfg)
