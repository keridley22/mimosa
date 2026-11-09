from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:

    # =========================================================================
    # INPUT / OUTPUT
    # =========================================================================

    raw_input_dir: Path = Path(
        "/Users/katherineridley/Projects/PlaqueStack/"
        "ilastik_protocol/plaque_pipeline_modular_fast/"
        "transplanted_converted_ometiffs"
    )

    output_root: Path = Path(
        "/Users/katherineridley/Projects/PlaqueStack/"
        "ilastik_protocol/transplanted_plaque_pipeline_outputs_corrected_20260922"
    )

    genotype_csv: Path = Path(
        "/Users/katherineridley/Projects/PlaqueStack/"
        "ilastik_protocol/plaque_pipeline_modular_fast/"
        "transplanted_lifs/metadata.csv"
    )

    # =========================================================================
    # CHANNELS
    # =========================================================================

    mx04_tag: str = "MX04"
    oc_tag: str = "OC"

    # Biological identities confirmed by the user and original training inputs.
    # The nominal OME names are misleading; they do not identify the markers.
    mx04_channel_index: int = 0
    oc_channel_index: int = 1
    expected_ome_channel_names: tuple[str, str] = ("Leica/ALEXA 594", "Leica/DAPI")

    mx04_suffix: str = "_MX04.ome.tiff"
    oc_suffix: str = "_OC.ome.tiff"

    # =========================================================================
    # PREPROCESSING
    # =========================================================================

    rolling_ball_radius_px: int = 50

    # Fast rolling-ball approximation
    rolling_ball_downsample_scale: float = 0.25

    gaussian_sigma_px: float = 1.0

    preprocess_dtype: str = "uint16"

    n_training_subset: int = 10

    # =========================================================================
    # ILASTIK / PROBABILITY MASKS
    # =========================================================================

    probability_threshold: float = 0.60

    # Label 1 = plaque
    # Label 2 = background
    foreground_class_index: int = 0

    ilastik_executable: Path = Path(
        "/Users/katherineridley/Downloads/ilastik-1.4.2-arm64-OSX.app/Contents/MacOS/ilastik"
    )
    ilastik_threads: int = 2
    ilastik_ram_mb: int = 2048

    # =========================================================================
    # PHYSICAL VOXEL SIZE
    # =========================================================================
    #
    # Nominal spacing shared with the previous APP-WT / APP-FIRE dataset
    #
    # Transplanted LIF stacks were checked: approximately 0.18 / 0.18 / 1 µm.
    # The OME spacing must agree within 1%; other acquisitions fail validation.
    # =========================================================================

        # =========================================================================
    # PHYSICAL VOXEL SIZE
    # =========================================================================

    voxel_xy_um: float = 0.18
    voxel_z_um: float = 1.0

    # =========================================================================
    # MORPHOLOGY CLEANING
    # =========================================================================

    # Match historical workflow: remove 30% from each XY edge,
    # leaving the central 40% x 40% of the field.
    crop_frac: float = 0.30

    # Slightly stronger than historical radius 1 because
    # the new probability masks are more pixelated.
    morphology_closing_radius: int = 2

    # Fill internal gaps in the fragmented plaque masks.
    morphology_fill_holes: bool = True

    # Object inclusion thresholds.
    per_object_min_area_voxels: int = 550
    central_min_size_voxels: int = 1000

    # Do not merge disconnected components after labelling.
    merge_distance_voxels: int = 0

    # Light final cleanup of the selected central plaque.
    central_closing_radius: int = 1

    # Compatibility with older code.
    label_closing_radius: int = 2

    # QC representative Z slice.
    qc_z: int = 13

    # =========================================================================
    # GENOTYPE LABELS
    # =========================================================================

    train_labels: tuple[str, str] = (
        "APP-WT",
        "APP-FIRE",
    )

    genotype_map: dict[str, str] = field(
        default_factory=lambda: {
            "APPPS1": "APP-WT",
            "APPPS1xFIRE": "APP-FIRE",
            "APP-WT": "APP-WT",
            "APP-FIRE": "APP-FIRE",
            "APP-FIRE+transplant": "APP-FIRE+transplant",
            "APP-FIRE+transplanted": "APP-FIRE+transplant",
            "APP-NLRP3-/-": "APP-NLRP3-/-",
            "APP-NLRP3−/−": "APP-NLRP3-/-",
        }
    )

    # Exclusions are specific to this cohort; historical exclusions are not inherited.
    excluded_images: tuple[str, ...] = ()

    raw_lif_dir: Path = Path(
        "/Users/katherineridley/Projects/PlaqueStack/ilastik_protocol/"
        "plaque_pipeline_modular_fast/transplanted_lifs"
    )
    bftools_dir: Path = Path("/Users/katherineridley/Downloads/bftools")
    previous_output_root: Path = Path(
        "/Users/katherineridley/Projects/PlaqueStack/ilastik_protocol/plaque_pipeline_outputs"
    )
    input_suffixes: tuple[str, ...] = (".ome.tiff", ".ome.tif")
    filename_must_contain: str | None = "_P+_"
    filename_skip_contains: tuple[str, ...] = ()
    min_z_planes: int = 2
    require_two_channels: bool = True
    # Check OME calibration against the historical nominal spacing before processing.
    voxel_spacing_relative_tolerance: float = 0.01
    analysis_genotypes: tuple[str, ...] = (
        "APP-WT", "APP-FIRE", "APP-FIRE+transplant", "WT",
    )
    ttest_comparisons: tuple[tuple[str, str], ...] = (
        ("APP-WT", "APP-FIRE+transplant"),
        ("APP-FIRE", "APP-FIRE+transplant"),
        ("APP-WT", "APP-FIRE"),
    )
    include_historical_controls: bool = True
    current_cohort_name: str = "transplanted"
    current_comparison_genotypes: tuple[str, ...] = ("APP-FIRE+transplant",)
    metadata_genotype_map: dict[str, str] = field(default_factory=dict)
    resample_input_to_nominal: bool = False
    allowed_input_voxel_xy_um: tuple[float, ...] = ()
    conversion_compression: str | None = None
    link_training_subset: bool = False
    # Union of the historical pipeline and translatecsvs.py exclusion lists.
    # These apply only to historical tables, never to new image names.
    historical_excluded_images: tuple[str, ...] = (
        "D_4_M04_OC_L4_P+_221123_1", "D_3_M04_OC_L4_P+_221123_1",
        "H_1_M04_OC_L1_P+_110124_1", "B_1_M04_OC_L23_P+_021123_2",
        "C_2_M04_OC_L1_P+_161123_2", "F_3_M04_OC_L1_P+_231123_1",
        "C_2_M04_OC_L1_P+_161123_1", "C_2_M04_OC_L4_P+_161123_2",
        "H_1_M04_OC_L1_P+_110124_2", "C_2_M04_OC_L23_P+_161123_2",
        "A_4_L1_P+_241023_A_4_M04_OC_P+_DGG_011123_1",
        "B_4_M04_OC_DGM_P+_131123_1", "B_4_M04_OC_DGG_P+_151123_1",
        "C_1_M04_OC_L23_P+_151123_2", "H_3_M04_OC_DGM_P+_210224_1",
    )

    def historical_control_table(self, channel: str) -> Path:
        source_name = {"MX04": "MX04", "OC": "OC"}[channel]
        return (self.previous_output_root / "morphology_results" / "combined"
                / f"overall_{source_name}_central.csv")

    @property
    def mx04_preprocessed_dir(self) -> Path:
        return self.preprocessed_mx04_dir

    @property
    def oc_preprocessed_dir(self) -> Path:
        return self.preprocessed_oc_dir

    @property
    def ttest_dir(self) -> Path:
        return self.morphology_dir / "ttests"

    @property
    def ilastik_project_dir(self) -> Path:
        return self.output_root / "ilastik_projects"

    def source_project(self, channel: str) -> Path:
        # Preserve the original biological marker identities end to end.
        source_name = {"MX04": "MX04", "OC": "OC"}[channel]
        return (self.previous_output_root / "ilastik_training_subset" / source_name
                / f"{source_name}_pixel_classifier.ilp")

    def working_project(self, channel: str) -> Path:
        return self.ilastik_project_dir / f"{channel}_{self.current_cohort_name}.ilp"

    # =========================================================================
    # DECISION TREE
    # =========================================================================

    test_size: float = 0.20

    # Historical model used random_state = 0
    random_state: int = 0

    decision_tree_cv_folds: int = 5
    decision_tree_scoring: str = "accuracy"

    # -------------------------------------------------------------------------
    # Historical fixed decision-tree parameters
    # -------------------------------------------------------------------------

    decision_tree_criterion: str = "entropy"
    decision_tree_max_depth: int = 8
    decision_tree_min_samples_leaf: int = 4
    decision_tree_min_samples_split: int = 10

    # =========================================================================
    # DECISION TREE FEATURES
    # =========================================================================
    #
    # Keep this exact set unless deliberately changing the model.
    # =========================================================================

    model_features: tuple[str, ...] = (

        "solidity_oc",
        "complexity_oc",
        "major_axis_length_m04",
        "solidity_m04",
        "elongation_oc",
        "minor_axis_length_m04",
        "surface_area_oc",
        "area_voxels_m04",
        "sphericity_m04",
        "volume_um3_m04",
        "volume_ratio_oc_m04",
        "surface_area_ratio_oc_m04",
        "surface_area_m04",
        "sphericity_oc",
        "minor_axis_length_oc",
        "area_voxels_oc",
        "major_axis_length_oc",
        "extent_oc",
        "extent_m04",
        "elongation_m04",
        "complexity_m04",
        "volume_um3_oc",
        "oc_halo_volume_um3",
    )

    # =========================================================================
    # DERIVED DIRECTORIES
    # =========================================================================

    # -------------------------------------------------------------------------
    # PREPROCESSING OUTPUT
    # -------------------------------------------------------------------------

    @property
    def preprocessed_dir(self) -> Path:
        return (
            self.output_root
            / "preprocessed_ome_tiffs"
        )

    @property
    def preprocessed_mx04_dir(self) -> Path:
        return (
            self.preprocessed_dir
            / "MX04"
        )

    @property
    def preprocessed_oc_dir(self) -> Path:
        return (
            self.preprocessed_dir
            / "OC"
        )

    # -------------------------------------------------------------------------
    # ILASTIK TRAINING SUBSET
    # -------------------------------------------------------------------------

    @property
    def ilastik_training_subset_dir(self) -> Path:
        return (
            self.output_root
            / "ilastik_training_subset"
        )

    # Compatibility alias
    @property
    def training_subset_dir(self) -> Path:
        return self.ilastik_training_subset_dir

    # -------------------------------------------------------------------------
    # ILASTIK PROBABILITY OUTPUTS
    # -------------------------------------------------------------------------

    @property
    def ilastik_probability_dir(self) -> Path:
        return (
            self.output_root
            / "ilastik_probabilities"
        )

    # IMPORTANT:
    # probabilities.py currently expects cfg.probability_dir
    @property
    def probability_dir(self) -> Path:
        return self.ilastik_probability_dir

    # Compatibility alias
    @property
    def probabilities_dir(self) -> Path:
        return self.ilastik_probability_dir

    # -------------------------------------------------------------------------
    # BINARY MASKS
    # -------------------------------------------------------------------------

    @property
    def mask_dir(self) -> Path:

        threshold = int(
            round(
                self.probability_threshold
                * 100
            )
        )

        return (
            self.output_root
            / f"masks_threshold_{threshold:03d}"
        )

    # Compatibility alias
    @property
    def masks_dir(self) -> Path:
        return self.mask_dir

    @property
    def mx04_mask_dir(self) -> Path:
        return (
            self.mask_dir
            / "MX04"
        )

    @property
    def oc_mask_dir(self) -> Path:
        return (
            self.mask_dir
            / "OC"
        )

    # -------------------------------------------------------------------------
    # MORPHOLOGY ROOT
    # -------------------------------------------------------------------------

    @property
    def morphology_dir(self) -> Path:
        return (
            self.output_root
            / "morphology_results"
        )

    # Compatibility alias
    @property
    def morphology_output_dir(self) -> Path:
        return self.morphology_dir

    # -------------------------------------------------------------------------
    # MORPHOLOGY CSV OUTPUT
    # -------------------------------------------------------------------------

    @property
    def morphology_combined_dir(self) -> Path:
        return (
            self.morphology_dir
            / "combined"
        )

    # Compatibility alias
    @property
    def combined_morphology_dir(self) -> Path:
        return self.morphology_combined_dir

    # -------------------------------------------------------------------------
    # CLEANED MASKS
    # -------------------------------------------------------------------------

    @property
    def cleaned_mask_dir(self) -> Path:
        return (
            self.morphology_dir
            / "cleaned_masks"
        )

    @property
    def cleaned_mx04_mask_dir(self) -> Path:
        return (
            self.cleaned_mask_dir
            / "MX04"
        )

    @property
    def cleaned_oc_mask_dir(self) -> Path:
        return (
            self.cleaned_mask_dir
            / "OC"
        )

    # -------------------------------------------------------------------------
    # CENTRAL PLAQUE MASKS
    # -------------------------------------------------------------------------

    @property
    def central_mask_dir(self) -> Path:
        return (
            self.morphology_dir
            / "central_masks"
        )

    @property
    def central_mx04_mask_dir(self) -> Path:
        return (
            self.central_mask_dir
            / "MX04"
        )

    @property
    def central_oc_mask_dir(self) -> Path:
        return (
            self.central_mask_dir
            / "OC"
        )

    # -------------------------------------------------------------------------
    # OC HALO MASKS
    # -------------------------------------------------------------------------

    @property
    def halo_mask_dir(self) -> Path:
        return (
            self.morphology_dir
            / "halo_masks"
        )

    # Compatibility alias
    @property
    def oc_halo_mask_dir(self) -> Path:
        return self.halo_mask_dir

    # -------------------------------------------------------------------------
    # DECISION TREE FEATURE TABLE
    # -------------------------------------------------------------------------

    @property
    def feature_table_path(self) -> Path:
        return (
            self.morphology_combined_dir
            / "decision_tree_features.csv"
        )

    # Compatibility aliases
    @property
    def combined_feature_csv(self) -> Path:
        return self.feature_table_path

    @property
    def decision_tree_feature_csv(self) -> Path:
        return self.feature_table_path

    # -------------------------------------------------------------------------
    # DECISION TREE OUTPUT
    # -------------------------------------------------------------------------

    @property
    def decision_tree_dir(self) -> Path:
        return (
            self.output_root
            / "decision_tree_results"
        )

    # Compatibility alias
    @property
    def decision_tree_output_dir(self) -> Path:
        return self.decision_tree_dir


# =============================================================================
# GLOBAL CONFIG OBJECT
# =============================================================================

CFG = Config()
