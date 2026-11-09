"""Animal-level Welch tests and all CSV layouts from translatecsvs.py."""
from __future__ import annotations

from pathlib import Path
from dataclasses import replace
import warnings
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

FEATURES = (
    "area_voxels", "volume_um3", "extent", "surface_area_um2",
    "complexity_volume_surface", "sphericity", "major_axis_length_um",
    "minor_axis_length_um", "elongation", "solidity", "occupied_z_slices",
)
RESULT_COLUMNS = (
    "feature", "group1", "group2", "n1_animals", "n2_animals", "mean1", "SD1",
    "mean2", "SD2", "difference_group2_minus_group1", "t_statistic", "p_value",
    "status", "FDR_BH_p_value", "significant_raw_p", "significant_FDR",
)


def bh_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    adjusted = np.full(p.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid])]
    n = len(order)
    if n:
        adjusted[order] = np.clip(np.minimum.accumulate(
            (p[order] * n / np.arange(1, n + 1))[::-1])[::-1], 0, 1)
    return adjusted


def analyse_dataset(csv_path: Path, dataset_name: str, cfg, *, feature_names=None, output_dir=None):
    df = pd.read_csv(csv_path)
    required = {"image_name", "animal_id", "Genotype"}
    if not required.issubset(df.columns):
        raise ValueError(f"{csv_path}: missing {sorted(required - set(df.columns))}")
    if df[list(required)].isna().any().any():
        raise ValueError(f"{csv_path}: missing image, animal, or genotype identity")
    if df["image_name"].duplicated().any():
        raise ValueError("Expected one central-plaque record per image")
    unknown = set(df["Genotype"]) - set(cfg.analysis_genotypes)
    if unknown:
        raise ValueError(f"Unconfigured genotypes: {sorted(unknown)}")
    if df.groupby("animal_id")["Genotype"].nunique().gt(1).any():
        raise ValueError("An animal is assigned to more than one genotype")
    df = df.loc[~df["image_name"].isin(cfg.excluded_images)].copy()
    requested = FEATURES if feature_names is None else tuple(feature_names)
    if feature_names is not None and not set(requested).issubset(df.columns):
        raise ValueError(f"Missing requested features: {sorted(set(requested) - set(df.columns))}")
    features = [f for f in requested if f in df.columns]
    if not features:
        raise ValueError(f"No morphology features in {csv_path}")
    df[features] = df[features].apply(pd.to_numeric, errors="raise").replace([np.inf, -np.inf], np.nan)
    rank = {g: i for i, g in enumerate(cfg.analysis_genotypes)}
    df = df.assign(_order=df["Genotype"].map(rank)).sort_values(
        ["_order", "animal_id", "image_name"]).drop(columns="_order")
    animal_means = df.groupby(["Genotype", "animal_id"], sort=False, as_index=False)[features].mean()
    out = Path(output_dir) if output_dir is not None else cfg.ttest_dir / dataset_name
    plaque_dir, mean_dir = out / "individual_plaque_values", out / "animal_means"
    plaque_dir.mkdir(parents=True, exist_ok=True)
    mean_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{dataset_name}_included_plaques.csv", index=False)
    animal_means.to_csv(out / f"{dataset_name}_all_animal_means.csv", index=False)
    key_columns = [c for c in ("animal_id", "Genotype", "cohort", "original_animal_id") if c in df]
    animal_order = df[key_columns].drop_duplicates()
    animal_order.to_csv(out / "animal_column_key.csv", index=False)
    df.groupby(["Genotype", "animal_id"], sort=False).size().rename("n_images").reset_index().to_csv(
        out / "animal_image_counts.csv", index=False)
    rows = []
    for feature in features:
        # Layout 1: one animal per column, individual plaque values down rows.
        pd.DataFrame({animal: df.loc[df["animal_id"] == animal, feature].dropna().reset_index(drop=True)
                      for animal in animal_order["animal_id"]}).to_csv(plaque_dir / f"{feature}.csv", index=False)
        # Layout 2: genotype columns, one animal mean per row (rows are not paired).
        pd.DataFrame({g: animal_means.loc[animal_means["Genotype"] == g, feature].reset_index(drop=True)
                      for g in cfg.analysis_genotypes}).to_csv(mean_dir / f"{feature}.csv", index=False)
        for g1, g2 in cfg.ttest_comparisons:
            x = animal_means.loc[animal_means["Genotype"] == g1, feature].dropna()
            y = animal_means.loc[animal_means["Genotype"] == g2, feature].dropna()
            stat = p = np.nan
            status = "insufficient_animals" if min(len(x), len(y)) < 2 else "ok"
            if status == "ok":
                if x.var() == 0 and y.var() == 0:
                    status = "zero_variance_both_groups"
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        result = ttest_ind(x, y, equal_var=False, alternative="two-sided")
                    stat, p = float(result.statistic), float(result.pvalue)
                    if not np.isfinite(p):
                        status = "undefined_test"
            rows.append(dict(feature=feature, group1=g1, group2=g2,
                             n1_animals=len(x), n2_animals=len(y), mean1=x.mean(), SD1=x.std(),
                             mean2=y.mean(), SD2=y.std(), difference_group2_minus_group1=y.mean()-x.mean(),
                             t_statistic=stat, p_value=p, status=status))
    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    # One correction family per dataset: all configured features and comparisons.
    results["FDR_BH_p_value"] = bh_adjust(results["p_value"])
    results["significant_raw_p"] = results["p_value"] < .05
    results["significant_FDR"] = results["FDR_BH_p_value"] < .05
    results.sort_values("p_value", na_position="last").to_csv(out / f"{dataset_name}_ttest_results.csv", index=False)
    print(f"{dataset_name}: {len(df)} images, {len(animal_means)} animals; {sum(results.status == 'ok')} tests computed")
    return results


def historical_comparison_input(current_path, channel, cfg):
    """Use prior controls and new transplant animals without pooling new controls."""
    current = pd.read_csv(current_path)
    source = cfg.historical_control_table(channel)
    historical = pd.read_csv(source)
    required = {"image_name", "sample_id", "Genotype", "area_voxels", "volume_um3"}
    if not required.issubset(historical):
        raise ValueError(f"Missing historical columns in {source}: {required - set(historical)}")
    historical["Genotype"] = historical["Genotype"].replace(cfg.genotype_map)
    historical = historical.loc[
        historical.Genotype.isin(["APP-WT", "APP-FIRE"])
        & ~historical.image_name.isin(cfg.historical_excluded_images)].copy()
    expected_volume = cfg.voxel_xy_um ** 2 * cfg.voxel_z_um
    calibration = (historical.volume_um3 / historical.area_voxels).replace([np.inf, -np.inf], np.nan).dropna()
    if calibration.empty or not np.allclose(calibration, expected_volume, rtol=1e-6):
        raise ValueError("Historical table voxel calibration does not match the new nominal calibration")
    # The historical metadata has no PF animal IDs; sample_id is its recorded
    # animal unit. Namespace those IDs to avoid collisions with reused slides.
    identity = "animal_id" if "animal_id" in historical else "sample_id"
    if historical[identity].isna().any():
        raise ValueError("Missing historical animal identifiers")
    historical["original_animal_id"] = historical[identity].astype(str)
    historical["animal_id"] = "historical:" + historical["original_animal_id"]
    historical["cohort"] = "historical"
    historical["source_table"] = str(source)
    historical["source_marker_label"] = channel
    audit = cfg.ttest_dir / "comparison_inputs"
    audit.mkdir(parents=True, exist_ok=True)
    current.loc[~current.Genotype.isin(cfg.current_comparison_genotypes)].to_csv(
        audit / f"{channel}_new_cohort_controls_not_pooled.csv", index=False)
    current = current.loc[current.Genotype.isin(cfg.current_comparison_genotypes)
                          & ~current.image_name.isin(cfg.excluded_images)].copy()
    if current.empty:
        raise ValueError("No configured comparison animals in the current morphology table")
    if current.animal_id.isna().any():
        raise ValueError("Missing current animal identifiers")
    current["original_animal_id"] = current.animal_id.astype(str)
    current["animal_id"] = cfg.current_cohort_name + ":" + current["original_animal_id"]
    current["cohort"] = cfg.current_cohort_name
    current["source_table"] = str(current_path)
    current["source_marker_label"] = channel
    combined = pd.concat([historical, current], ignore_index=True)
    combined["source_image_name"] = combined["image_name"]
    combined["image_name"] = combined["cohort"] + ":" + combined["image_name"]
    combined["marker"] = channel
    path = audit / f"{channel}_historical_controls_plus_{cfg.current_cohort_name}.csv"
    combined.to_csv(path, index=False)
    return path


def run_default_ttests(cfg):
    paths = {ch: cfg.morphology_combined_dir / f"overall_{ch}_central.csv" for ch in ("MX04", "OC")}
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(f"Run morphology first: {path}")
    for ch, path in paths.items():
        analysis_cfg = cfg
        if cfg.include_historical_controls:
            path = historical_comparison_input(path, ch, cfg)
            analysis_cfg = replace(cfg, analysis_genotypes=(*cfg.train_labels, *cfg.current_comparison_genotypes))
        analyse_dataset(path, ch, analysis_cfg)
