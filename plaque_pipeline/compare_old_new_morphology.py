from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


# ============================================================
# PATHS
# ============================================================

OLD_OC = Path(
    "/Users/katherineridley/Projects/PlaqueStack/Transplanted/"
    "combined results 2026/Masks/overall_OC2_central.csv"
)

OLD_M04 = Path(
    "/Users/katherineridley/Projects/PlaqueStack/Transplanted/"
    "combined results 2026/Masks/overall_M04_central.csv"
)

NEW_ROOT = Path(
    "/Users/katherineridley/Projects/PlaqueStack/"
    "ilastik_protocol/plaque_pipeline_outputs/morphology_results"
)

NEW_OC = NEW_ROOT / "combined" / "overall_OC_central.csv"
NEW_M04 = NEW_ROOT / "combined" / "overall_MX04_central.csv"

OUTPUT = Path(
    "/Users/katherineridley/Projects/PlaqueStack/"
    "ilastik_protocol/plaque_pipeline_outputs/"
    "old_vs_new_morphology_comparison"
)

OUTPUT.mkdir(parents=True, exist_ok=True)
(OUTPUT / "whole_dataset").mkdir(exist_ok=True)
(OUTPUT / "matched_subset").mkdir(exist_ok=True)
(OUTPUT / "by_genotype").mkdir(exist_ok=True)
(OUTPUT / "matched_scatterplots").mkdir(exist_ok=True)


# ============================================================
# COLUMN HARMONISATION
# ============================================================

OLD_TO_NEW_FEATURES = {
    "surface_area": "surface_area_um2",
    "complexity": "complexity_volume_surface",
    "major_axis_length": "major_axis_length_um",
    "minor_axis_length": "minor_axis_length_um",
}

DIRECT_FEATURES = [
    "area_voxels",
    "volume_um3",
    "extent",
    "sphericity",
    "solidity",
]


# ============================================================
# HELPERS
# ============================================================

def safe_name(x):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x))


def load_csv(path, label):
    if not path.exists():
        raise FileNotFoundError(f"Could not find {label}:\n{path}")

    df = pd.read_csv(path)

    print(f"{label}: {len(df)} rows")
    print(f"  columns: {list(df.columns)}")

    return df


def canonical_genotype(value):
    if pd.isna(value):
        return np.nan

    x = str(value).strip()

    mapping = {
        "APPPS1": "APP-WT",
        "APPPS1xFIRE": "APP-FIRE",
        "APP-WT": "APP-WT",
        "APP-FIRE": "APP-FIRE",
    }

    return mapping.get(x, x)


def extract_old_image_id(value):
    """
    Convert legacy image_name values such as:

    C_1_M04_OC_C_1_M04_OC_DGG_P+_151123_1 - C_1_M04_OC.lif_

    into:

    C_1_M04_OC_DGG_P+_151123_1
    """

    x = str(value)

    # Strip path just in case
    x = Path(x).name

    # Remove trailing Leica file descriptor
    x = re.sub(
        r"\s*-\s*[A-Z]_\d+_M04_OC\.lif_?$",
        "",
        x,
        flags=re.IGNORECASE,
    )

    # Legacy names often repeat the animal prefix twice:
    # C_1_M04_OC_C_1_M04_OC_DGG...
    duplicate_prefix = re.match(
        r"^([A-Z]_\d+_M04_OC)_\1_(.+)$",
        x,
        flags=re.IGNORECASE,
    )

    if duplicate_prefix:
        prefix = duplicate_prefix.group(1)
        rest = duplicate_prefix.group(2)
        x = f"{prefix}_{rest}"

    # Remove any remaining .lif suffix
    x = re.sub(
        r"\.lif_?$",
        "",
        x,
        flags=re.IGNORECASE,
    )

    # Remove trailing underscores/spaces
    x = x.rstrip("_ ").strip()

    return x


def extract_new_image_id(value):
    """
    Convert new filenames such as:

    C_1_M04_OC_DGG_P+_151123_1_MX04.ome.tiff

    or:

    C_1_M04_OC_DGG_P+_151123_1_OC.ome.tiff

    into:

    C_1_M04_OC_DGG_P+_151123_1
    """

    x = Path(str(value)).name

    # Remove TIFF/H5 extensions repeatedly
    patterns = [
        r"\.ome\.tiff?$",
        r"\.tiff?$",
        r"\.h5$",
        r"\.hdf5$",
    ]

    changed = True

    while changed:
        changed = False

        for pattern in patterns:
            new_x = re.sub(
                pattern,
                "",
                x,
                flags=re.IGNORECASE,
            )

            if new_x != x:
                x = new_x
                changed = True

    # Remove ilastik suffix if still present
    x = re.sub(
        r"_Probabilities$",
        "",
        x,
        flags=re.IGNORECASE,
    )

    # Remove final channel suffix
    x = re.sub(
        r"_(MX04|M04|OC2|OC)$",
        "",
        x,
        flags=re.IGNORECASE,
    )

    return x.rstrip("_ ").strip()


def harmonise_columns(df, dataset_type):
    """
    Rename old morphology columns onto the new naming scheme.
    """

    df = df.copy()

    if dataset_type == "old":
        df = df.rename(columns=OLD_TO_NEW_FEATURES)

    return df


def prepare_old(df, channel):
    df = harmonise_columns(df, "old")

    df["comparison_id"] = df["image_name"].map(
        extract_old_image_id
    )

    df["comparison_genotype"] = df["Genotype"].map(
        canonical_genotype
    )

    df["dataset_version"] = "Old"
    df["comparison_channel"] = channel

    return df


def prepare_new(df, channel):
    df = harmonise_columns(df, "new")

    df["comparison_id"] = df["image_name"].map(
        extract_new_image_id
    )

    df["comparison_genotype"] = df["Genotype"].map(
        canonical_genotype
    )

    df["dataset_version"] = "New"
    df["comparison_channel"] = channel

    return df


def get_comparable_features(old, new):
    wanted = DIRECT_FEATURES + list(
        OLD_TO_NEW_FEATURES.values()
    )

    features = []

    for feature in wanted:

        if feature not in old.columns:
            continue

        if feature not in new.columns:
            continue

        if not pd.api.types.is_numeric_dtype(old[feature]):
            continue

        if not pd.api.types.is_numeric_dtype(new[feature]):
            continue

        features.append(feature)

    # New pipeline may also contain elongation
    if (
        "elongation" in old.columns
        and "elongation" in new.columns
    ):
        features.append("elongation")

    return features


# ============================================================
# PLOTTING
# ============================================================

def plot_distribution(
    old_values,
    new_values,
    feature,
    channel,
    title_suffix,
    output_dir,
):

    old_values = pd.to_numeric(
        old_values,
        errors="coerce",
    ).dropna()

    new_values = pd.to_numeric(
        new_values,
        errors="coerce",
    ).dropna()

    if len(old_values) == 0 or len(new_values) == 0:
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.hist(
        old_values,
        bins=30,
        density=True,
        histtype="step",
        linewidth=2,
        label=f"Old (n={len(old_values)})",
    )

    ax.hist(
        new_values,
        bins=30,
        density=True,
        histtype="step",
        linewidth=2,
        label=f"New (n={len(new_values)})",
    )

    ax.set_xlabel(feature)
    ax.set_ylabel("Density")
    ax.set_title(
        f"{channel}: {feature}\n{title_suffix}"
    )
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir
        / f"{channel}_{safe_name(title_suffix)}_{safe_name(feature)}.png",
        dpi=200,
    )

    plt.close(fig)


def scatter_plot(matched, feature, channel):

    old_col = f"{feature}_old"
    new_col = f"{feature}_new"

    temp = matched[
        ["comparison_id", old_col, new_col]
    ].dropna()

    if len(temp) < 2:
        return

    x = temp[old_col].astype(float)
    y = temp[new_col].astype(float)

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(
        x,
        y,
        alpha=0.7,
    )

    low = min(x.min(), y.min())
    high = max(x.max(), y.max())

    ax.plot(
        [low, high],
        [low, high],
        linestyle="--",
    )

    ax.set_xlabel("Old analysis")
    ax.set_ylabel("New analysis")
    ax.set_title(
        f"{channel}: {feature}\nMatched images n={len(temp)}"
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT
        / "matched_scatterplots"
        / f"{channel}_{safe_name(feature)}.png",
        dpi=200,
    )

    plt.close(fig)


# ============================================================
# COMPARISON
# ============================================================

def compare_channel(old, new, channel):

    print()
    print("=" * 70)
    print(channel)
    print("=" * 70)

    features = get_comparable_features(old, new)

    print(f"Comparable features: {len(features)}")
    print(features)

    # --------------------------------------------------------
    # MATCHING REPORT
    # --------------------------------------------------------

    old_ids = set(old["comparison_id"])
    new_ids = set(new["comparison_id"])

    all_ids = sorted(old_ids | new_ids)

    match_report = pd.DataFrame({
        "comparison_id": all_ids
    })

    match_report["in_old"] = match_report[
        "comparison_id"
    ].isin(old_ids)

    match_report["in_new"] = match_report[
        "comparison_id"
    ].isin(new_ids)

    match_report["matched"] = (
        match_report["in_old"]
        & match_report["in_new"]
    )

    match_report.to_csv(
        OUTPUT / f"{channel}_image_matching.csv",
        index=False,
    )

    print(f"Images in old: {len(old_ids)}")
    print(f"Images in new: {len(new_ids)}")
    print(
        "Matched:",
        int(match_report["matched"].sum()),
    )
    print(
        "Old only:",
        int(
            (
                match_report["in_old"]
                & ~match_report["in_new"]
            ).sum()
        ),
    )
    print(
        "New only:",
        int(
            (
                ~match_report["in_old"]
                & match_report["in_new"]
            ).sum()
        ),
    )

    print("\nExample normalised IDs:")

    print("\nOLD:")
    print(
        old[
            ["image_name", "comparison_id"]
        ].head(5).to_string(index=False)
    )

    print("\nNEW:")
    print(
        new[
            ["image_name", "comparison_id"]
        ].head(5).to_string(index=False)
    )

    # --------------------------------------------------------
    # BUILD MATCHED TABLE
    # --------------------------------------------------------

    if old["comparison_id"].duplicated().any():
        warnings.warn(
            f"{channel}: duplicate old comparison IDs detected."
        )

    if new["comparison_id"].duplicated().any():
        warnings.warn(
            f"{channel}: duplicate new comparison IDs detected."
        )

    old_match = (
        old
        .drop_duplicates("comparison_id")
        [
            [
                "comparison_id",
                "comparison_genotype",
            ]
            + features
        ]
    )

    new_match = (
        new
        .drop_duplicates("comparison_id")
        [
            [
                "comparison_id",
                "comparison_genotype",
            ]
            + features
        ]
    )

    matched = old_match.merge(
        new_match,
        on="comparison_id",
        how="inner",
        suffixes=("_old", "_new"),
    )

    matched.to_csv(
        OUTPUT / f"{channel}_matched_values.csv",
        index=False,
    )

    # --------------------------------------------------------
    # WHOLE DATASET DISTRIBUTIONS
    # --------------------------------------------------------

    whole_rows = []

    for feature in features:

        old_values = pd.to_numeric(
            old[feature],
            errors="coerce",
        )

        new_values = pd.to_numeric(
            new[feature],
            errors="coerce",
        )

        whole_rows.append({
            "channel": channel,
            "feature": feature,

            "old_n": old_values.notna().sum(),
            "new_n": new_values.notna().sum(),

            "old_mean": old_values.mean(),
            "new_mean": new_values.mean(),

            "old_median": old_values.median(),
            "new_median": new_values.median(),

            "old_sd": old_values.std(),
            "new_sd": new_values.std(),

            "old_q25": old_values.quantile(0.25),
            "new_q25": new_values.quantile(0.25),

            "old_q75": old_values.quantile(0.75),
            "new_q75": new_values.quantile(0.75),

            "mean_difference_new_minus_old":
                new_values.mean()
                - old_values.mean(),
        })

        plot_distribution(
            old_values,
            new_values,
            feature,
            channel,
            "Whole dataset",
            OUTPUT / "whole_dataset",
        )

    pd.DataFrame(whole_rows).to_csv(
        OUTPUT / f"{channel}_whole_dataset_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # MATCHED SUBSET DISTRIBUTIONS
    # --------------------------------------------------------

    matched_rows = []

    for feature in features:

        old_col = f"{feature}_old"
        new_col = f"{feature}_new"

        old_values = pd.to_numeric(
            matched[old_col],
            errors="coerce",
        )

        new_values = pd.to_numeric(
            matched[new_col],
            errors="coerce",
        )

        matched_rows.append({
            "channel": channel,
            "feature": feature,

            "n_matched": len(matched),

            "old_mean": old_values.mean(),
            "new_mean": new_values.mean(),

            "old_median": old_values.median(),
            "new_median": new_values.median(),

            "old_sd": old_values.std(),
            "new_sd": new_values.std(),

            "mean_difference_new_minus_old":
                new_values.mean()
                - old_values.mean(),
        })

        plot_distribution(
            old_values,
            new_values,
            feature,
            channel,
            "Matched subset",
            OUTPUT / "matched_subset",
        )

    pd.DataFrame(matched_rows).to_csv(
        OUTPUT / f"{channel}_matched_subset_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # BY GENOTYPE
    # --------------------------------------------------------

    genotypes = sorted(
        set(old["comparison_genotype"].dropna())
        | set(new["comparison_genotype"].dropna())
    )

    genotype_rows = []

    for genotype in genotypes:

        old_g = old[
            old["comparison_genotype"] == genotype
        ]

        new_g = new[
            new["comparison_genotype"] == genotype
        ]

        for feature in features:

            old_values = pd.to_numeric(
                old_g[feature],
                errors="coerce",
            )

            new_values = pd.to_numeric(
                new_g[feature],
                errors="coerce",
            )

            genotype_rows.append({
                "channel": channel,
                "genotype": genotype,
                "feature": feature,

                "old_n": old_values.notna().sum(),
                "new_n": new_values.notna().sum(),

                "old_mean": old_values.mean(),
                "new_mean": new_values.mean(),

                "old_median": old_values.median(),
                "new_median": new_values.median(),

                "old_sd": old_values.std(),
                "new_sd": new_values.std(),
            })

            plot_distribution(
                old_values,
                new_values,
                feature,
                channel,
                genotype,
                OUTPUT / "by_genotype",
            )

    pd.DataFrame(genotype_rows).to_csv(
        OUTPUT / f"{channel}_by_genotype_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # MATCHED AGREEMENT METRICS
    # --------------------------------------------------------

    agreement_rows = []

    for feature in features:

        old_col = f"{feature}_old"
        new_col = f"{feature}_new"

        temp = matched[
            ["comparison_id", old_col, new_col]
        ].dropna()

        if len(temp) == 0:
            continue

        x = temp[old_col].astype(float).values
        y = temp[new_col].astype(float).values

        diff = y - x

        pearson = np.nan
        spearman = np.nan

        if len(temp) >= 3:

            try:
                pearson = pearsonr(x, y).statistic
            except Exception:
                pass

            try:
                spearman = spearmanr(x, y).statistic
            except Exception:
                pass

        mae = np.mean(np.abs(diff))
        rmse = np.sqrt(np.mean(diff ** 2))

        mean_old = np.mean(x)

        relative_mae = (
            mae / abs(mean_old)
            if mean_old != 0
            else np.nan
        )

        agreement_rows.append({
            "channel": channel,
            "feature": feature,

            "n_matched": len(temp),

            "pearson_r": pearson,
            "spearman_r": spearman,

            "MAE": mae,
            "RMSE": rmse,
            "relative_MAE": relative_mae,

            "mean_old": np.mean(x),
            "mean_new": np.mean(y),

            "mean_difference_new_minus_old":
                np.mean(diff),

            "median_difference_new_minus_old":
                np.median(diff),
        })

        scatter_plot(
            matched,
            feature,
            channel,
        )

    agreement = pd.DataFrame(agreement_rows)

    agreement.to_csv(
        OUTPUT / f"{channel}_matched_agreement.csv",
        index=False,
    )

    return agreement


# ============================================================
# MAIN
# ============================================================

def main():

    print("\nLoading morphology results...\n")

    old_oc = prepare_old(
        load_csv(OLD_OC, "Old OC"),
        "OC",
    )

    old_m04 = prepare_old(
        load_csv(OLD_M04, "Old M04"),
        "MX04",
    )

    new_oc = prepare_new(
        load_csv(NEW_OC, "New OC"),
        "OC",
    )

    new_m04 = prepare_new(
        load_csv(NEW_M04, "New MX04"),
        "MX04",
    )

    oc_agreement = compare_channel(
        old_oc,
        new_oc,
        "OC",
    )

    mx_agreement = compare_channel(
        old_m04,
        new_m04,
        "MX04",
    )

    pd.concat(
        [
            oc_agreement,
            mx_agreement,
        ],
        ignore_index=True,
    ).to_csv(
        OUTPUT / "ALL_FEATURES_matched_agreement.csv",
        index=False,
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Results saved to:\n{OUTPUT}")
    print()


if __name__ == "__main__":
    main()