from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from .io_utils import ensure_dir


def run_welch_ttests(df: pd.DataFrame, group_col: str, features: list[str], out_csv: Path) -> None:
    if df is None or df.empty or group_col not in df.columns:
        return
    d = df.copy()
    d = d[d[group_col].notna()]
    d = d[d[group_col].astype(str) != "Unknown"]
    groups = sorted(d[group_col].astype(str).unique())
    if len(groups) < 2:
        print(f"Skipping t-tests for {out_csv.name}: fewer than 2 groups")
        return

    rows = []
    for feat in features:
        if feat not in d.columns:
            continue
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                g1, g2 = groups[i], groups[j]
                x = pd.to_numeric(d.loc[d[group_col].astype(str) == g1, feat], errors="coerce").dropna().values
                y = pd.to_numeric(d.loc[d[group_col].astype(str) == g2, feat], errors="coerce").dropna().values
                if len(x) < 2 or len(y) < 2:
                    continue
                stat, p = ttest_ind(x, y, equal_var=False)
                rows.append({
                    "feature": feat, "group1": g1, "group2": g2,
                    "n1": len(x), "n2": len(y),
                    "mean1": float(np.mean(x)), "mean2": float(np.mean(y)),
                    "t_stat": float(stat), "p_value": float(p),
                })
    if rows:
        ensure_dir(out_csv.parent)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"Saved t-tests: {out_csv}")


def run_default_ttests(combined_dir: Path, out_dir: Path) -> None:
    ensure_dir(out_dir)
    central_features = [
        "area_voxels", "volume_um3", "extent", "surface_area_um2",
        "complexity_volume_surface", "sphericity", "major_axis_length_um",
        "minor_axis_length_um", "solidity", "volume_ratio_mx04_over_oc",
    ]
    summary_features = [
        "number_of_objects", "total_volume_um3", "mean_sphericity", "mean_solidity",
        "mean_complexity_volume_surface", "volume_ratio_mx04_over_oc",
    ]
    for fname, features in [
        ("overall_MX04_central.csv", central_features),
        ("overall_OC_central.csv", central_features),
        ("overall_MX04_summary.csv", summary_features),
        ("overall_OC_summary.csv", summary_features),
    ]:
        path = combined_dir / fname
        if path.exists():
            df = pd.read_csv(path)
            group_col = "Genotype" if "Genotype" in df.columns else "genotype"
            run_welch_ttests(df, group_col, features, out_dir / f"ttest_{fname}")
