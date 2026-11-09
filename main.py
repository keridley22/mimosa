#!/usr/bin/env python3
"""Run the six-month cohort with the corrected transplanted image-processing method."""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path

from plaque_pipeline.young_config import YOUNG_CFG


def main():
    parser = argparse.ArgumentParser(description="Young (6m) plaque analysis")
    parser.add_argument("stage", choices=("setup", "inventory", "convert", "preprocess", "segment",
                                         "h5_to_masks", "morphology", "stats", "translatecsvs", "halo_stats",
                                         "post_ilastik", "analysis", "profiles", "run"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--input-dir", type=Path)
    args = parser.parse_args()
    cfg = YOUNG_CFG
    if args.output_root:
        cfg = replace(cfg, output_root=args.output_root.resolve())
    if args.input_dir:
        cfg = replace(cfg, raw_input_dir=args.input_dir.resolve())
    from plaque_pipeline.setup import copy_ilastik_projects, validate_output_root
    from plaque_pipeline.conversion import inventory_lifs, convert_lifs
    from plaque_pipeline.preprocess import preprocess_ometiff_folder
    from plaque_pipeline.segmentation import segment_original_projects
    from plaque_pipeline.probabilities import h5_to_all_masks
    from plaque_pipeline.morphology import process_morphology
    from plaque_pipeline.stats import run_default_ttests
    from plaque_pipeline.halo_stats import run_halo_stats
    from plaque_pipeline.young_profiles import run_young_profiles
    validate_output_root(cfg)
    functions = {"setup": copy_ilastik_projects, "inventory": inventory_lifs, "convert": convert_lifs,
                 "preprocess": preprocess_ometiff_folder, "segment": segment_original_projects,
                 "h5_to_masks": h5_to_all_masks, "morphology": process_morphology,
                 "stats": run_default_ttests, "halo_stats": run_halo_stats, "profiles": run_young_profiles}
    sequences = {
        "run": ("setup", "convert", "preprocess", "segment", "h5_to_masks", "morphology", "stats", "halo_stats", "profiles"),
        "post_ilastik": ("h5_to_masks", "morphology", "stats", "halo_stats"),
        "analysis": ("morphology", "stats", "halo_stats"),
        "translatecsvs": ("stats", "halo_stats"),
    }
    if args.stage == "run" and any(cfg.probability_dir.glob("*/*.h5")):
        raise ValueError("Probabilities exist. Resume with 'bash run.sh segment', then 'bash run.sh post_ilastik'.")
    audit = cfg.output_root / "input_audit"
    audit.mkdir(parents=True, exist_ok=True)
    stages = sequences.get(args.stage, (args.stage,))
    for stage in stages:
        status = dict(command=args.stage, stage=stage, state="running", updated_utc=datetime.now(timezone.utc).isoformat())
        (audit / "run_status.json").write_text(json.dumps(status, indent=2))
        print(f"\n=== {stage} ===", flush=True)
        try:
            functions[stage](cfg)
        except Exception as error:
            status.update(state="failed", error=str(error))
            (audit / "run_status.json").write_text(json.dumps(status, indent=2))
            raise
    status.update(state="complete", updated_utc=datetime.now(timezone.utc).isoformat())
    (audit / "run_status.json").write_text(json.dumps(status, indent=2))
    print(f"Completed {args.stage}: {cfg.output_root}", flush=True)


if __name__ == "__main__":
    main()
