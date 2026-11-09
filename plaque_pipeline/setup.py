from __future__ import annotations

import json
from dataclasses import asdict
import shutil
import h5py


def validate_output_root(cfg):
    legacy = cfg.previous_output_root.parent / 'transplanted_plaque_pipeline_outputs'
    if cfg.output_root.resolve() == legacy.resolve():
        raise ValueError('This is the old output directory with mismatched marker/model results. '
                         'Use the new corrected output directory; do not reuse old probabilities.')
    snapshot = cfg.output_root / 'config_snapshot.json'
    if snapshot.exists():
        saved = json.loads(snapshot.read_text())
        for key in ('mx04_channel_index', 'oc_channel_index'):
            if saved.get(key) != getattr(cfg, key):
                raise ValueError('This output directory used different channel routing. Use a fresh output root.')
        for key in ('voxel_xy_um', 'voxel_z_um', 'resample_input_to_nominal', 'metadata_genotype_map'):
            if key in saved and saved[key] != getattr(cfg, key):
                raise ValueError(f'This output directory used different {key}. Use a fresh output root.')


def setup_directories(cfg):
    validate_output_root(cfg)
    directories = [cfg.output_root, cfg.raw_input_dir, cfg.ilastik_project_dir,
                   cfg.morphology_combined_dir, cfg.halo_mask_dir, cfg.output_root / "input_audit"]
    for channel in (cfg.mx04_tag, cfg.oc_tag):
        directories += [cfg.preprocessed_dir / channel, cfg.training_subset_dir / channel,
                        cfg.probability_dir / channel, cfg.mask_dir / channel,
                        cfg.cleaned_mask_dir / channel, cfg.central_mask_dir / channel,
                        cfg.ttest_dir / channel / "individual_plaque_values",
                        cfg.ttest_dir / channel / "animal_means"]
    for path in directories:
        path.mkdir(parents=True, exist_ok=True)
    (cfg.output_root / "config_snapshot.json").write_text(json.dumps(asdict(cfg), default=str, indent=2))
    print(f"Output folders ready: {cfg.output_root}")


def copy_ilastik_projects(cfg):
    """Copy trained projects together with relative training-image dependencies."""
    setup_directories(cfg)
    for channel in (cfg.mx04_tag, cfg.oc_tag):
        src, dst = cfg.source_project(channel), cfg.working_project(channel)
        if dst.exists():
            print(f"Keeping existing working project: {dst}")
            continue
        with h5py.File(src, "r") as h:
            labels = [x.decode() for x in h["PixelClassification/LabelNames"][()]]
            if labels != ["Plaque", "Background"] or "PixelClassification/ClassifierForests" not in h:
                raise ValueError(f"Unexpected class order or missing trained classifier in {src}")
            for lane in h["Input Data/infos"].values():
                rel = lane["Raw Data/filePath"][()].decode()
                from pathlib import Path
                source = src.parent / rel
                target = dst.parent / rel
                if Path(rel).is_absolute() or ".." in Path(rel).parts:
                    raise ValueError(f"Unexpected nonlocal training path: {rel}")
                if not source.is_file():
                    raise FileNotFoundError(source)
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
        shutil.copy2(src, dst)
        with h5py.File(dst, "r+") as h:
            export = h["Prediction Export"]
            export["OutputFilenameFormat"][()] = str(
                cfg.probability_dir / channel / "{nickname}_Probabilities.h5")
            export["OutputInternalPath"][()] = "exported_data"
        print(f"Copied {channel} working project from {src.name}: {dst}")
