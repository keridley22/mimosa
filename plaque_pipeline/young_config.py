"""Six-month cohort; processing inherits the corrected transplanted settings."""
from dataclasses import replace
from itertools import combinations
from pathlib import Path

from .config import Config

PROJECT = Path("/Users/katherineridley/Projects/PlaqueStack/ilastik_protocol")
GROUPS = ("APP-WT", "APP-FIRE", "APP-WT young", "APP-FIRE young")
YOUNG_CFG = replace(
    Config(),
    raw_lif_dir=PROJECT / "plaque_pipeline_modular_fast/young_lifs",
    genotype_csv=PROJECT / "plaque_pipeline_modular_fast/young_lifs/metadata.csv",
    raw_input_dir=PROJECT / "plaque_pipeline_modular_fast/young_converted_ometiffs",
    output_root=PROJECT / "young_plaque_pipeline_outputs_20260929",
    metadata_genotype_map={"APP-WT": "APP-WT young", "APP-FIRE": "APP-FIRE young"},
    current_cohort_name="young",
    current_comparison_genotypes=("APP-WT young", "APP-FIRE young"),
    analysis_genotypes=GROUPS,
    ttest_comparisons=tuple(combinations(GROUPS, 2)),
    conversion_compression="LZW",
    link_training_subset=True,
    resample_input_to_nominal=True,
    allowed_input_voxel_xy_um=(0.18, 0.13875024437927663),
)
