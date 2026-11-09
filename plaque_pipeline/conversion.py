"""Bio-Formats conversion with a manifest and explicit series selection."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import pandas as pd

from .calibration import validate_pixels
from .io_utils import safe_name
from .metadata import load_metadata, resolve_metadata


def check_free_space(path, minimum_gib=2):
    import shutil
    probe = Path(path)
    while not probe.exists():
        probe = probe.parent
    free = shutil.disk_usage(probe).free / 2**30
    if free < minimum_gib:
        raise RuntimeError(f"Only {free:.2f} GiB free at {probe}; free space before resuming")


def inventory_lifs(cfg):
    files = sorted(cfg.raw_lif_dir.glob("*.lif"))
    if not files:
        raise FileNotFoundError(f"No LIFs in {cfg.raw_lif_dir}")
    lookup = load_metadata(cfg)
    rows = []
    env = dict(os.environ)
    env["JAVA_TOOL_OPTIONS"] = env.get("JAVA_TOOL_OPTIONS", "") + " -Dfile.encoding=UTF-8"
    audit = cfg.output_root / "input_audit"
    audit.mkdir(parents=True, exist_ok=True)
    for lif in files:
        result = subprocess.run([str(cfg.bftools_dir / "showinf"), "-nopix", "-omexml-only",
                                 "-novalid", str(lif)], capture_output=True, check=True, env=env)
        text = result.stdout.decode("utf-8")
        xml = text[text.index("<OME"):text.rindex("</OME>") + 6]
        (audit / (lif.stem + ".ome.xml")).write_text(xml)
        root = ET.fromstring(xml)
        ns = {"o": root.tag.split("}")[0][1:]}
        for index, image in enumerate(root.findall("o:Image", ns)):
            name = image.get("Name", f"series_{index}")
            pixels = image.find("o:Pixels", ns)
            z, c, t = (int(pixels.get("Size" + a)) for a in ("Z", "C", "T"))
            reason = "included"
            if z < cfg.min_z_planes or t != 1 or c != 2:
                reason = "not a two-channel 3D stack with one timepoint"
            elif cfg.filename_must_contain and cfg.filename_must_contain not in name:
                reason = f"name lacks {cfg.filename_must_contain}"
            elif any(x and x in name for x in cfg.filename_skip_contains):
                reason = "configured filename exclusion"
            row = {"lif_file": str(lif), "series_index": index, "series_name": name,
                   "Z": z, "C": c, "T": t, "included": reason == "included", "reason": reason}
            if row["included"]:
                row.update(resolve_metadata(name, lookup))
                spacing = validate_pixels(pixels, cfg)
                row.update({f"physical_size_{k.lower()}_um": v for k, v in spacing.items()})
                row["analysis_voxel_xy_um"] = cfg.voxel_xy_um
                row["analysis_voxel_z_um"] = cfg.voxel_z_um
                row["resample_xy"] = cfg.resample_input_to_nominal and abs(spacing["X"] / cfg.voxel_xy_um - 1) > cfg.voxel_spacing_relative_tolerance
                row["output_file"] = str(cfg.raw_input_dir / f"{safe_name(lif.stem)}__s{index:03d}_{safe_name(name)}.ome.tiff")
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(audit / "lif_series_manifest.csv", index=False)
    included = df.loc[df["included"]]
    if included.empty:
        raise ValueError("No LIF series passed selection")
    counts = included.groupby(["sample_id", "animal_id", "Genotype"]).size().rename("n_stacks").reset_index()
    counts.to_csv(audit / "available_animals.csv", index=False)
    missing = pd.DataFrame([r for key, r in lookup.items() if key not in set(included["sample_id"])],
                           columns=["sample_id", "animal_id", "Genotype"])
    missing.to_csv(audit / "metadata_without_selected_images.csv", index=False)
    print(counts.to_string(index=False))
    print(f"Selected {len(included)} / {len(df)} series. Audit: {audit}")
    return df


def convert_lifs(cfg):
    df = inventory_lifs(cfg)
    cfg.raw_input_dir.mkdir(parents=True, exist_ok=True)
    for row in df.loc[df["included"]].to_dict("records"):
        check_free_space(cfg.output_root)
        out = Path(row["output_file"])
        if out.exists():
            print(f"Already converted: {out.name}")
            continue
        partial = out.with_name(out.name.replace(".ome.tiff", ".partial.ome.tiff"))
        compression = ["-compression", cfg.conversion_compression] if cfg.conversion_compression else []
        subprocess.run([str(cfg.bftools_dir / "bfconvert"), *compression, "-overwrite", "-series",
                        str(row["series_index"]), row["lif_file"], str(partial)], check=True)
        partial.replace(out)
