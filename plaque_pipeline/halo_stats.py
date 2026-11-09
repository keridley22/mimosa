"""Audited OC halo / OC-to-MX04 ratios, with animals as statistical replicates."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from .stats import analyse_dataset

DATASET = "OC_halo_and_ratios"
OUTCOMES = (
    "oc_halo_volume_um3", "oc_halo_fraction_of_oc",
    "volume_ratio_oc_m04", "surface_area_ratio_oc_m04",
)
MEASUREMENTS = (
    "area_voxels_m04", "volume_um3_m04", "surface_area_m04",
    "area_voxels_oc", "volume_um3_oc", "surface_area_oc", "oc_halo_voxels",
)


def divide_or_nan(numerator, denominator):
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0:
        return np.nan
    return float(numerator / denominator)


def calculate_halo(mx04, oc, voxel_volume):
    """Voxelwise set difference; MX04 need not be entirely inside OC."""
    if mx04.shape != oc.shape or mx04.ndim != 3:
        raise ValueError("Central masks must share the same 3D grid")
    halo = (oc > 0) & ~(mx04 > 0)
    count = int(np.count_nonzero(halo))
    return halo, {
        "oc_halo_voxels": count,
        "oc_halo_volume_um3": count * voxel_volume,
        "oc_halo_fraction_of_oc": divide_or_nan(count, np.count_nonzero(oc)),
    }


def require_match(actual, expected, description):
    if not np.allclose(actual, expected, rtol=1e-8, atol=1e-8, equal_nan=True):
        raise ValueError(f"Calculation audit failed: {description}: {actual} != {expected}")


def load_cohort(root, cohort, cfg):
    source = root / "morphology_results/combined/decision_tree_features.csv"
    df = pd.read_csv(source)
    identity = "animal_id" if "animal_id" in df else "sample_id"
    if cohort != "historical" and identity != "animal_id":
        raise ValueError("Current cohort needs explicit animal_id metadata")
    required = {"image_name", "sample_id", identity, "Genotype", *MEASUREMENTS, *OUTCOMES}
    if not required.issubset(df):
        raise ValueError(f"{source}: missing {sorted(required - set(df))}")
    if df[["image_name", identity, "Genotype"]].isna().any().any() or df.image_name.duplicated().any():
        raise ValueError(f"Missing or duplicate identities in {source}")
    df["Genotype"] = df.Genotype.replace(cfg.genotype_map)
    if cohort == "historical":
        eligible = df.Genotype.isin(cfg.train_labels)
        excluded = df.image_name.isin(cfg.historical_excluded_images)
    else:
        eligible = (df.Genotype.isin(cfg.current_comparison_genotypes) if cfg.include_historical_controls
                    else df.Genotype.isin(cfg.analysis_genotypes))
        excluded = df.image_name.isin(cfg.excluded_images)
    df["cohort"] = cohort
    df["source_table"] = str(source)
    df["source_image_name"] = df.image_name
    df["original_animal_id"] = df[identity].astype(str)
    df["animal_id"] = cohort + ":" + df.original_animal_id
    omitted = df.loc[~eligible | excluded].copy()
    omitted["omission_reason"] = np.where(excluded.loc[omitted.index], "configured_image_exclusion",
                                         "cohort_genotype_not_pooled")
    included = df.loc[eligible & ~excluded].copy()
    if included.empty:
        raise ValueError(f"No eligible plaques in {source}")
    included[list(MEASUREMENTS + OUTCOMES)] = included[list(MEASUREMENTS + OUTCOMES)].apply(
        pd.to_numeric, errors="raise").replace([np.inf, -np.inf], np.nan)
    return included, omitted, source


def audit_cohort(df, root, cfg):
    """Check source marker tables, central masks, halo masks and saved calculations."""
    morphology = root / "morphology_results"
    source_paths = []
    for channel, suffix in (("MX04", "m04"), ("OC", "oc")):
        path = morphology / "combined" / f"overall_{channel}_central.csv"
        source_paths.append(path)
        table = pd.read_csv(path).set_index("image_name", verify_integrity=True)
        aligned = table.loc[df.image_name]
        if not np.array_equal(aligned.Genotype.replace(cfg.genotype_map), df.Genotype):
            raise ValueError(f"Genotype mismatch against {path}")
        identity = "animal_id" if "animal_id" in aligned else "sample_id"
        if not np.array_equal(aligned[identity].astype(str), df.original_animal_id):
            raise ValueError(f"Animal identity mismatch against {path}")
        for source_col, feature_col in (("area_voxels", f"area_voxels_{suffix}"),
                                        ("volume_um3", f"volume_um3_{suffix}"),
                                        ("surface_area_um2", f"surface_area_{suffix}")):
            require_match(df[feature_col].to_numpy(), aligned[source_col].to_numpy(), str(path / source_col))

    voxel_volume = cfg.voxel_xy_um ** 2 * cfg.voxel_z_um
    rows = []
    for record in df.to_dict("records"):
        name = record["image_name"]
        masks = {}
        for channel, suffix in (("MX04", "m04"), ("OC", "oc")):
            path = morphology / "central_masks" / channel / f"{name}_{channel}_central.ome.tiff"
            masks[channel] = tifffile.imread(path) > 0
            count = int(np.count_nonzero(masks[channel]))
            require_match(count, record[f"area_voxels_{suffix}"], f"{name}: {channel} mask voxels")
            require_match(count * voxel_volume, record[f"volume_um3_{suffix}"],
                          f"{name}: {channel} voxel calibration")
            record[f"source_{suffix}_central_mask"] = str(path)
        halo, calculated = calculate_halo(masks["MX04"], masks["OC"], voxel_volume)
        halo_path = morphology / "halo_masks" / f"{name}_OC_halo.ome.tiff"
        saved_halo = tifffile.imread(halo_path) > 0
        if not np.array_equal(halo, saved_halo):
            raise ValueError(f"Saved halo mask differs from OC minus MX04 mask: {name}")
        calculated["volume_ratio_oc_m04"] = divide_or_nan(record["volume_um3_oc"], record["volume_um3_m04"])
        calculated["surface_area_ratio_oc_m04"] = divide_or_nan(record["surface_area_oc"], record["surface_area_m04"])
        for field, value in calculated.items():
            require_match(value, record[field], f"{name}: {field}")
            record[field] = value
        record["oc_mx04_overlap_voxels"] = int(np.count_nonzero(masks["OC"] & masks["MX04"]))
        record["source_halo_mask"] = str(halo_path)
        record["voxel_volume_um3"] = voxel_volume
        record["calculation_audit"] = "passed"
        record["undefined_outcomes"] = ";".join(f for f in OUTCOMES if not np.isfinite(record[f]))
        record["image_name"] = record["cohort"] + ":" + name
        rows.append(record)
    return pd.DataFrame(rows), source_paths


def run_halo_stats(cfg, output_dir=None):
    frames, omitted, sources = [], [], []
    cohorts = [(cfg.current_cohort_name if cfg.include_historical_controls else "current", cfg.output_root)]
    if cfg.include_historical_controls:
        cohorts.insert(0, ("historical", cfg.previous_output_root))
    for cohort, root in cohorts:
        selected, excluded, source = load_cohort(root, cohort, cfg)
        audited, marker_sources = audit_cohort(selected, root, cfg)
        frames.append(audited)
        omitted.append(excluded)
        sources.extend([source, *marker_sources])
        print(f"{cohort}: verified central masks, halo masks and ratios for {len(audited)} plaques", flush=True)
    plaques = pd.concat(frames, ignore_index=True)
    columns = ["image_name", "source_image_name", "sample_id", "animal_id", "original_animal_id",
               "Genotype", "cohort", *MEASUREMENTS, *OUTCOMES, "oc_mx04_overlap_voxels",
               "voxel_volume_um3", "calculation_audit", "undefined_outcomes", "source_table",
               "source_m04_central_mask", "source_oc_central_mask", "source_halo_mask"]
    plaques = plaques[columns]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(output_dir) if output_dir else cfg.ttest_dir / DATASET / f"run_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    input_path = output / "plaque_calculations.csv"
    plaques.to_csv(input_path, index=False)
    pd.concat(omitted, ignore_index=True).to_csv(output / "omitted_source_records.csv", index=False)
    analysis_cfg = replace(cfg, excluded_images=())
    if cfg.include_historical_controls:
        analysis_cfg = replace(analysis_cfg, analysis_genotypes=(*cfg.train_labels, *cfg.current_comparison_genotypes))
    results = analyse_dataset(input_path, DATASET, analysis_cfg, feature_names=OUTCOMES, output_dir=output)

    # Counts expose per-outcome missingness. Each outcome mean uses its valid plaques.
    counts = plaques.groupby(["Genotype", "animal_id"], sort=False)[list(OUTCOMES)].count()
    counts.rename(columns={f: f + "_n_plaques" for f in OUTCOMES}).reset_index().to_csv(
        output / "animal_valid_plaque_counts.csv", index=False)
    means = pd.read_csv(output / f"{DATASET}_all_animal_means.csv")
    summary = []
    for genotype in analysis_cfg.analysis_genotypes:
        subset = means.loc[means.Genotype == genotype]
        plaque_subset = plaques.loc[plaques.Genotype == genotype]
        for feature in OUTCOMES:
            values = subset[feature].dropna()
            summary.append(dict(Genotype=genotype, feature=feature, n_animals=len(values),
                                n_plaques=plaque_subset[feature].count(), mean=values.mean(),
                                SD=values.std(), SEM=values.sem()))
    pd.DataFrame(summary).to_csv(output / "group_summary_animal_means.csv", index=False)
    pd.DataFrame([
        ("oc_halo_voxels", "voxels", "count(OC central mask AND NOT MX04 central mask)"),
        ("oc_halo_volume_um3", "um^3", "oc_halo_voxels * voxel_volume_um3"),
        ("oc_halo_fraction_of_oc", "fraction (0-1)", "oc_halo_voxels / area_voxels_oc"),
        ("volume_ratio_oc_m04", "dimensionless", "volume_um3_oc / volume_um3_m04"),
        ("surface_area_ratio_oc_m04", "dimensionless", "surface_area_oc / surface_area_m04"),
    ], columns=["feature", "unit", "calculation"]).to_csv(output / "metric_definitions.csv", index=False)
    source_records = [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources]
    manifest = dict(created_utc=stamp, sources=source_records, n_plaques=len(plaques),
                    n_animals=plaques.animal_id.nunique(), outcomes=OUTCOMES,
                    voxel_volume_um3=cfg.voxel_xy_um**2*cfg.voxel_z_um,
                    comparisons=cfg.ttest_comparisons, test="two-sided Welch independent t-test on animal means",
                    multiple_testing=f"Benjamini-Hochberg across all {len(results)} outcome/comparison tests; valid p-values only",
                    historical_excluded_images=cfg.historical_excluded_images,
                    current_excluded_images=cfg.excluded_images,
                    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "README.md").write_text(f"""# OC halo and OC/MX04 ratios

{len(plaques)} central plaques from {plaques.animal_id.nunique()} animals. Historical
APP-WT and APP-FIRE controls plus the {cfg.current_cohort_name} cohort are used when
include_historical_controls=True. Configured exclusions are retained; new cohort
controls are not pooled with the historical controls in that mode.

OC halo means OC-positive voxels outside the MX04 central mask. It is a mask set
difference, not OC total volume minus MX04 total volume. Halo fraction is the
fraction of OC volume outside MX04 (0-1). Volume and surface area ratios divide OC
by MX04; ratios are dimensionless. Surface areas come from the existing central
morphology tables (um^2). Nominal voxel volume is {cfg.voxel_xy_um**2*cfg.voxel_z_um:.4f} um^3.
All included central mask counts and saved halo masks were checked, as were
the source marker tables, calibration, and saved derived metrics. This checks
calculation consistency; it does not validate biological segmentation quality.

Each animal mean is the arithmetic mean of its individual plaque metric values.
For ratios this is the mean of plaque ratios, not the ratio of summed volumes or
areas. T-tests use one mean per animal, two-sided Welch tests with unequal
variances. Group SD/SEM describe animal means. The signed mean difference is
group2 minus group1; the t statistic uses group1 minus group2. Raw p-values and
Benjamini-Hochberg FDR values are both reported across all {len(results)} tests
(four outcomes and {len(cfg.ttest_comparisons)} comparisons). Halo voxel
counts are saved for auditing but not tested again because they duplicate the
halo volume outcome after constant rescaling. Missing values or nonpositive
denominators yield blanks, not zero; valid counts are supplied per outcome.

Files:
- plaque_calculations.csv: one row per plaque, formulas' inputs, values and source paths.
- {DATASET}_included_plaques.csv: the same included records, sorted by group/animal/image.
- {DATASET}_all_animal_means.csv: one row per animal, four outcome means.
- {DATASET}_ttest_results.csv: all pairwise animal-level tests, raw and FDR p-values.
- individual_plaque_values/: one CSV per outcome, one animal per column.
- animal_means/: one CSV per outcome, one genotype per column, one animal mean per row.
  Rows in different genotype columns are unpaired; animal_column_key.csv maps the ordering.
- animal_valid_plaque_counts.csv: valid plaque counts for each animal/outcome.
- group_summary_animal_means.csv: group mean, SD and SEM with animal/plaque counts.
- metric_definitions.csv, manifest.json and omitted_source_records.csv: definitions and provenance.

Reproduce from the installed pipeline with `bash run.sh halo_stats`.
Each run writes a new timestamped folder and leaves source masks/tables unchanged.
""")
    print(f"Saved OC halo and ratio results: {output}", flush=True)
    return output
