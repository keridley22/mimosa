from __future__ import annotations

import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile

from scipy.ndimage import binary_fill_holes

from skimage.measure import (
    label,
    marching_cubes,
    mesh_surface_area,
    regionprops,
)

from skimage.morphology import (
    ball,
    closing,
    remove_small_objects,
)

from .config import Config, CFG
from .io_utils import ensure_dir
from .metadata import load_metadata, resolve_metadata


# =============================================================================
# BASIC PHYSICAL HELPERS
# =============================================================================


def spacing_zyx(
    cfg: Config = CFG,
) -> tuple[float, float, float]:
    """
    Physical voxel spacing in Z, Y, X order.
    """

    return (
        cfg.voxel_z_um,
        cfg.voxel_xy_um,
        cfg.voxel_xy_um,
    )


def voxel_volume_um3(
    cfg: Config = CFG,
) -> float:
    """
    Physical volume of one voxel.
    """

    return (
        cfg.voxel_z_um
        * cfg.voxel_xy_um
        * cfg.voxel_xy_um
    )


# =============================================================================
# IMAGE LOADING
# =============================================================================


def read_binary_mask(
    path: Path,
) -> np.ndarray:
    """
    Load a TIFF mask as a boolean ZYX stack.
    """

    arr = tifffile.imread(path)

    arr = np.asarray(arr)

    arr = np.squeeze(arr)

    if arr.ndim != 3:
        raise ValueError(
            f"Expected 3D ZYX mask for {path.name}, "
            f"found shape {arr.shape}"
        )

    return arr > 0


# =============================================================================
# CENTRAL XY CROP
# =============================================================================


def crop_xy(
    img: np.ndarray,
    crop_frac: float,
) -> np.ndarray:
    """
    Crop equally from each XY edge while retaining all Z planes.

    Historical behaviour:

        crop_frac = 0.30

    retains:

        Y = 30% -> 70%
        X = 30% -> 70%

    i.e. the central 40% x 40% of the field.
    """

    if crop_frac <= 0:
        return img

    if crop_frac >= 0.5:
        raise ValueError(
            "crop_frac must be < 0.5"
        )

    _, height, width = img.shape

    y0 = int(
        crop_frac * height
    )

    y1 = int(
        (1.0 - crop_frac) * height
    )

    x0 = int(
        crop_frac * width
    )

    x1 = int(
        (1.0 - crop_frac) * width
    )

    return img[
        :,
        y0:y1,
        x0:x1,
    ]


# =============================================================================
# REMOVE SMALL OBJECTS
# =============================================================================


def remove_small_objects_compat(
    mask: np.ndarray,
    minimum_size: int,
) -> np.ndarray:
    """
    Remove connected components containing fewer than minimum_size voxels.

    Handles both older and newer scikit-image APIs.
    """

    binary = mask.astype(bool)

    if minimum_size <= 1:
        return binary

    try:

        # New scikit-image behaviour:
        # removes objects <= max_size
        return remove_small_objects(
            binary,
            max_size=minimum_size - 1,
        )

    except TypeError:

        # Older scikit-image
        return remove_small_objects(
            binary,
            min_size=minimum_size,
        )


# =============================================================================
# HOLE FILLING
# =============================================================================


def fill_holes_xy(
    mask: np.ndarray,
) -> np.ndarray:
    """
    Fill holes independently within each XY plane.

    This is useful for pixelated plaque masks where an apparent hole in
    one section may remain connected to background somewhere else in Z
    and therefore would not be filled by 3D hole filling alone.
    """

    out = np.zeros_like(
        mask,
        dtype=bool,
    )

    for z in range(
        mask.shape[0]
    ):

        out[z] = binary_fill_holes(
            mask[z]
        )

    return out


def fill_holes_3d(
    mask: np.ndarray,
) -> np.ndarray:
    """
    Fill fully enclosed holes in XYZ.
    """

    return binary_fill_holes(
        mask
    ).astype(bool)


# =============================================================================
# MASK CLEANING
# =============================================================================


def clean_plaque_mask(
    mask: np.ndarray,
    cfg: Config = CFG,
    minimum_size: int | None = None,
) -> np.ndarray:
    """
    Clean a fragmented 3D plaque mask.

    Order is important:

        binary
          ↓
        3D closing
          ↓
        XY hole filling
          ↓
        3D hole filling
          ↓
        second light closing
          ↓
        optional small-object filtering

    The fragment joining happens BEFORE small objects are removed so nearby
    pieces have a chance to become a single biological plaque.
    """

    binary = mask.astype(bool)

    # -------------------------------------------------------------------------
    # FIRST 3D CLOSING
    # -------------------------------------------------------------------------

    radius = int(
        cfg.morphology_closing_radius
    )

    if radius > 0:

        binary = closing(
            binary,
            footprint=ball(radius),
        )

    # -------------------------------------------------------------------------
    # HOLE FILLING
    # -------------------------------------------------------------------------

    if cfg.morphology_fill_holes:

        # Fill holes within individual optical sections.
        binary = fill_holes_xy(
            binary
        )

        # Then fill enclosed 3D cavities.
        binary = fill_holes_3d(
            binary
        )

    # -------------------------------------------------------------------------
    # SECOND LIGHT CLOSING
    # -------------------------------------------------------------------------
    #
    # Use radius 1 here rather than repeating radius 2.
    #
    # This catches one-voxel discontinuities left after hole filling without
    # aggressively expanding/merging nearby plaques.
    # -------------------------------------------------------------------------

    if radius > 0:

        binary = closing(
            binary,
            footprint=ball(1),
        )

    # -------------------------------------------------------------------------
    # REMOVE SMALL OBJECTS ONLY AFTER JOINING / FILLING
    # -------------------------------------------------------------------------

    if (
        minimum_size is not None
        and minimum_size > 0
    ):

        binary = remove_small_objects_compat(
            binary,
            minimum_size,
        )

    return binary.astype(bool)


# =============================================================================
# CONNECTED COMPONENT HELPERS
# =============================================================================


def find_largest_component(
    mask: np.ndarray,
) -> np.ndarray:
    """
    Return only the largest 3D connected component.
    """

    labels = label(
        mask.astype(bool),
        connectivity=3,
    )

    if labels.max() == 0:

        return np.zeros_like(
            mask,
            dtype=bool,
        )

    counts = np.bincount(
        labels.ravel()
    )

    counts[0] = 0

    largest = int(
        np.argmax(counts)
    )

    return (
        labels == largest
    )


# =============================================================================
# CENTRAL PLAQUE EXTRACTION
# =============================================================================


def extract_central_plaque(
    cleaned_mask: np.ndarray,
    cfg: Config = CFG,
) -> np.ndarray:
    """
    Identify the dominant plaque after cleaning.

    The full image has already been cropped around the centre, matching the
    logic of the historical workflow.

    Small-object filtering happens AFTER the mask has been joined and filled.
    """

    binary = cleaned_mask.astype(bool)

    # Remove components that remain too small after cleaning.
    binary = remove_small_objects_compat(
        binary,
        cfg.central_min_size_voxels,
    )

    if not np.any(binary):

        return np.zeros_like(
            binary,
            dtype=bool,
        )

    # Take dominant coherent component.
    central = find_largest_component(
        binary
    )

    # Optional final central-object closing.
    radius = int(
        cfg.central_closing_radius
    )

    if radius > 0:

        central = closing(
            central,
            footprint=ball(radius),
        )

        if cfg.morphology_fill_holes:

            central = fill_holes_xy(
                central
            )

            central = fill_holes_3d(
                central
            )

    # Critical:
    # guarantee that we return ONE connected object.
    central = find_largest_component(
        central
    )

    return central.astype(bool)


# =============================================================================
# SURFACE AREA
# =============================================================================


def calculate_surface_area_from_mask(
    mask: np.ndarray,
    cfg: Config = CFG,
) -> float:
    """
    Calculate physical 3D surface area in µm² using marching cubes.
    """

    binary = mask.astype(bool)

    if not np.any(binary):

        return np.nan

    # Padding is important if the object touches its local boundary.
    padded = np.pad(
        binary.astype(np.uint8),
        pad_width=1,
        mode="constant",
        constant_values=0,
    )

    try:

        verts, faces, _, _ = marching_cubes(
            padded,
            level=0.5,
            spacing=spacing_zyx(cfg),
        )

        area = mesh_surface_area(
            verts,
            faces,
        )

        return float(
            area
        )

    except (
        ValueError,
        RuntimeError,
    ):

        return np.nan


# =============================================================================
# SAFE SOLIDITY
# =============================================================================


def safe_solidity(
    region,
) -> float:
    """
    Calculate solidity while safely handling failed 3D convex hulls.

    A plaque is NOT discarded if convex hull generation fails.
    Only solidity becomes NaN.
    """

    try:

        with warnings.catch_warnings():

            warnings.simplefilter(
                "ignore"
            )

            value = float(
                region.solidity
            )

    except Exception:

        return np.nan

    if not np.isfinite(value):

        return np.nan

    return value


# =============================================================================
# SINGLE PLAQUE MORPHOLOGY
# =============================================================================


def empty_properties() -> dict:
    """
    Standard empty morphology result.
    """

    return {
        "area_voxels": np.nan,
        "volume_um3": np.nan,
        "extent": np.nan,
        "surface_area_um2": np.nan,
        "complexity_volume_surface": np.nan,
        "sphericity": np.nan,
        "solidity": np.nan,
        "major_axis_length_um": np.nan,
        "minor_axis_length_um": np.nan,
        "elongation": np.nan,
        "centroid_z_um": np.nan,
        "centroid_y_um": np.nan,
        "centroid_x_um": np.nan,
        "occupied_z_slices": 0,
    }


def properties_from_single_mask(
    mask: np.ndarray,
    cfg: Config = CFG,
) -> dict:
    """
    Measure one coherent 3D plaque.

    IMPORTANT:
    - area_voxels is true voxel count.
    - volume_um3 is voxel count × physical voxel volume.
    - axis dimensions use physical spacing.
    """

    binary = mask.astype(bool)

    if not np.any(binary):

        return empty_properties()

    # Ensure exactly one connected component.
    binary = find_largest_component(
        binary
    )

    labelled = label(
        binary,
        connectivity=3,
    )

    raw_regions = regionprops(
        labelled
    )

    physical_regions = regionprops(
        labelled,
        spacing=spacing_zyx(cfg),
    )

    if (
        len(raw_regions) == 0
        or len(physical_regions) == 0
    ):

        return empty_properties()

    r_raw = raw_regions[0]
    r_phys = physical_regions[0]

    # -------------------------------------------------------------------------
    # TRUE VOXEL COUNT
    # -------------------------------------------------------------------------

    area_voxels = int(
        r_raw.area
    )

    # -------------------------------------------------------------------------
    # PHYSICAL VOLUME
    # -------------------------------------------------------------------------

    volume_um3 = (
        area_voxels
        * voxel_volume_um3(cfg)
    )

    # -------------------------------------------------------------------------
    # SURFACE AREA
    # -------------------------------------------------------------------------

    surface_area_um2 = (
        calculate_surface_area_from_mask(
            binary,
            cfg,
        )
    )

    # -------------------------------------------------------------------------
    # COMPLEXITY = V / A
    # -------------------------------------------------------------------------

    if (
        np.isfinite(surface_area_um2)
        and surface_area_um2 > 0
    ):

        complexity = (
            volume_um3
            / surface_area_um2
        )

    else:

        complexity = np.nan

    # -------------------------------------------------------------------------
    # SPHERICITY
    # -------------------------------------------------------------------------

    if (
        np.isfinite(surface_area_um2)
        and surface_area_um2 > 0
        and volume_um3 > 0
    ):

        sphericity = (
            np.pi ** (1.0 / 3.0)
            * (
                6.0 * volume_um3
            ) ** (2.0 / 3.0)
            / surface_area_um2
        )

    else:

        sphericity = np.nan

    # -------------------------------------------------------------------------
    # AXIS LENGTHS
    # -------------------------------------------------------------------------

    try:

        major_axis = float(
            r_phys.axis_major_length
        )

    except Exception:

        major_axis = np.nan

    try:

        minor_axis = float(
            r_phys.axis_minor_length
        )

    except Exception:

        minor_axis = np.nan

    # -------------------------------------------------------------------------
    # ELONGATION
    # -------------------------------------------------------------------------

    if (
        np.isfinite(major_axis)
        and np.isfinite(minor_axis)
        and minor_axis > 0
    ):

        elongation = (
            major_axis
            / minor_axis
        )

    else:

        elongation = np.nan

    # -------------------------------------------------------------------------
    # SOLIDITY
    # -------------------------------------------------------------------------

    solidity = safe_solidity(
        r_raw
    )

    # -------------------------------------------------------------------------
    # Z EXTENT QC
    # -------------------------------------------------------------------------

    occupied_z_slices = int(
        np.sum(
            np.any(
                binary,
                axis=(1, 2),
            )
        )
    )

    # -------------------------------------------------------------------------
    # CENTROID
    # -------------------------------------------------------------------------

    try:

        centroid = tuple(
            float(v)
            for v in r_phys.centroid
        )

    except Exception:

        centroid = (
            np.nan,
            np.nan,
            np.nan,
        )

    return {

        "area_voxels":
            area_voxels,

        "volume_um3":
            float(volume_um3),

        "extent":
            float(r_raw.extent),

        "surface_area_um2":
            float(surface_area_um2)
            if np.isfinite(surface_area_um2)
            else np.nan,

        "complexity_volume_surface":
            float(complexity)
            if np.isfinite(complexity)
            else np.nan,

        "sphericity":
            float(sphericity)
            if np.isfinite(sphericity)
            else np.nan,

        "solidity":
            solidity,

        "major_axis_length_um":
            major_axis,

        "minor_axis_length_um":
            minor_axis,

        "elongation":
            elongation,

        "centroid_z_um":
            centroid[0],

        "centroid_y_um":
            centroid[1],

        "centroid_x_um":
            centroid[2],

        "occupied_z_slices":
            occupied_z_slices,
    }


# =============================================================================
# ALL PLAQUE OBJECTS
# =============================================================================


def calculate_morphological_properties(
    cleaned_mask: np.ndarray,
    min_area: int,
    cfg: Config = CFG,
) -> list[dict]:
    """
    Calculate morphology for every sufficiently large connected object.

    Size filtering is applied after mask cleaning / fragment joining.
    """

    labelled = label(
        cleaned_mask.astype(bool),
        connectivity=3,
    )

    regions = regionprops(
        labelled
    )

    output: list[dict] = []

    for region in regions:

        # Region.area here is true voxel count because no spacing was passed.
        if region.area < min_area:
            continue

        object_mask = (
            labelled == region.label
        )

        props = properties_from_single_mask(
            object_mask,
            cfg,
        )

        props[
            "label"
        ] = int(
            region.label
        )

        output.append(
            props
        )

    return output


# =============================================================================
# FILE NAME MATCHING
# =============================================================================


def mask_base_name(
    path: Path,
    channel: str,
) -> str:
    """
    Remove channel suffixes from mask filenames.
    """

    name = path.name

    name = re.sub(
        r"\.ome\.tiff?$",
        "",
        name,
        flags=re.IGNORECASE,
    )

    name = re.sub(
        r"\.tiff?$",
        "",
        name,
        flags=re.IGNORECASE,
    )

    name = re.sub(
        rf"_{re.escape(channel)}$",
        "",
        name,
        flags=re.IGNORECASE,
    )

    return name


# =============================================================================
# SAMPLE / GENOTYPE METADATA
# =============================================================================


SAMPLE_RE = re.compile(
    r"([A-Z]_\d+)"
)


def extract_sample_id(
    value: str,
) -> str | None:

    match = SAMPLE_RE.search(
        str(value)
    )

    if match:
        return match.group(1)

    return None


def load_genotype_lookup(
    cfg: Config = CFG,
) -> dict[str, str]:
    """
    Read metadata.csv and create sample -> genotype lookup.
    """

    if not cfg.genotype_csv.exists():

        print(
            f"WARNING: genotype file not found: "
            f"{cfg.genotype_csv}"
        )

        return {}

    metadata = pd.read_csv(
        cfg.genotype_csv
    )

    if "Genotype" not in metadata.columns:

        raise ValueError(
            "metadata.csv requires a 'Genotype' column"
        )

    lookup = {}

    for _, row in metadata.iterrows():

        sample_id = None

        for column in (
            "filename",
            "slide",
            "section",
        ):

            if column not in metadata.columns:
                continue

            sample_id = extract_sample_id(
                row[column]
            )

            if sample_id is not None:
                break

        if sample_id is None:
            continue

        genotype = str(
            row["Genotype"]
        ).strip()

        genotype = cfg.genotype_map.get(
            genotype,
            genotype,
        )

        lookup[
            sample_id
        ] = genotype

    return lookup


# =============================================================================
# FEATURE TABLE NAMING
# =============================================================================


def add_channel_features(
    output: dict,
    props: dict,
    suffix: str,
) -> None:
    """
    Convert morphology names to the feature names expected by the existing
    decision-tree workflow.
    """

    output[
        f"area_voxels_{suffix}"
    ] = props["area_voxels"]

    output[
        f"volume_um3_{suffix}"
    ] = props["volume_um3"]

    output[
        f"extent_{suffix}"
    ] = props["extent"]

    output[
        f"surface_area_{suffix}"
    ] = props["surface_area_um2"]

    output[
        f"complexity_{suffix}"
    ] = props["complexity_volume_surface"]

    output[
        f"sphericity_{suffix}"
    ] = props["sphericity"]

    output[
        f"solidity_{suffix}"
    ] = props["solidity"]

    output[
        f"major_axis_length_{suffix}"
    ] = props["major_axis_length_um"]

    output[
        f"minor_axis_length_{suffix}"
    ] = props["minor_axis_length_um"]

    output[
        f"elongation_{suffix}"
    ] = props["elongation"]

    output[
        f"occupied_z_slices_{suffix}"
    ] = props["occupied_z_slices"]


# =============================================================================
# QC IMAGE
# =============================================================================


def save_qc(
    mask: np.ndarray,
    outpath: Path,
    z: int = 13,
) -> None:
    """
    Historical-style QC:
    maximum projection + representative Z slice.
    """

    ensure_dir(
        outpath.parent
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
    )

    axes[0].imshow(
        mask.max(axis=0),
        cmap="gray",
    )

    axes[0].set_title(
        "Max projection"
    )

    axes[0].axis(
        "off"
    )

    z = min(
        z,
        mask.shape[0] - 1,
    )

    axes[1].imshow(
        mask[z],
        cmap="gray",
    )

    axes[1].set_title(
        f"Z slice {z}"
    )

    axes[1].axis(
        "off"
    )

    plt.tight_layout()

    plt.savefig(
        outpath,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =============================================================================
# SAVE BINARY MASK
# =============================================================================


def save_binary_mask(
    path: Path,
    mask: np.ndarray,
) -> None:
    """
    Save mask using 0/255 so it displays properly in Fiji/ImageJ.
    """

    ensure_dir(
        path.parent
    )

    tifffile.imwrite(
        path,
        (
            mask.astype(np.uint8)
            * 255
        ),
        photometric="minisblack",
        compression="zlib",
        metadata={"axes": "ZYX"},
    )


# =============================================================================
# MAIN MORPHOLOGY PIPELINE
# =============================================================================


def process_morphology(
    cfg: Config = CFG,
) -> None:

    # =========================================================================
    # DIRECTORIES
    # =========================================================================

    ensure_dir(
        cfg.morphology_dir
    )

    ensure_dir(
        cfg.morphology_combined_dir
    )

    ensure_dir(
        cfg.cleaned_mask_dir
        / "MX04"
    )

    ensure_dir(
        cfg.cleaned_mask_dir
        / "OC"
    )

    ensure_dir(
        cfg.central_mask_dir
        / "MX04"
    )

    ensure_dir(
        cfg.central_mask_dir
        / "OC"
    )

    ensure_dir(
        cfg.halo_mask_dir
    )

    qc_dir = (
        cfg.morphology_dir
        / "qc"
    )

    ensure_dir(
        qc_dir
    )

    # =========================================================================
    # FIND INPUT MASKS
    # =========================================================================

    mx04_files = sorted(
        cfg.mx04_mask_dir.glob(
            "*.tif*"
        )
    )

    oc_files = sorted(
        cfg.oc_mask_dir.glob(
            "*.tif*"
        )
    )

    if not mx04_files:

        raise FileNotFoundError(
            f"No MX04 masks found in "
            f"{cfg.mx04_mask_dir}"
        )

    if not oc_files:

        raise FileNotFoundError(
            f"No OC masks found in "
            f"{cfg.oc_mask_dir}"
        )

    mx04_lookup = {

        mask_base_name(
            path,
            "MX04",
        ):
        path

        for path in mx04_files
    }

    oc_lookup = {

        mask_base_name(
            path,
            "OC",
        ):
        path

        for path in oc_files
    }

    shared_names = sorted(
        set(mx04_lookup)
        & set(oc_lookup)
    )

    all_shared_names = sorted(
    set(mx04_lookup)
    & set(oc_lookup)
    )

    excluded_present = [
        name
        for name in all_shared_names
        if name in cfg.excluded_images
    ]

    shared_names = [
        name
        for name in all_shared_names
        if name not in cfg.excluded_images
    ]

    print(
        f"Excluded images present: {len(excluded_present)}"
    )

    for name in excluded_present:
        print(
            f"  EXCLUDED: {name}"
        )

    print()
    print(
        f"MX04 masks: {len(mx04_files)}"
    )

    print(
        f"OC masks: {len(oc_files)}"
    )

    print(
        f"Matched pairs: {len(shared_names)}"
    )

    print(
        f"Crop fraction: {cfg.crop_frac}"
    )

    print(
        f"Closing radius: "
        f"{cfg.morphology_closing_radius}"
    )

    print(
        f"Hole filling: "
        f"{cfg.morphology_fill_holes}"
    )

    print()

    if not shared_names:

        raise FileNotFoundError(
            "No matching MX04/OC pairs."
        )

    metadata_lookup = load_metadata(cfg)
    for name in shared_names:
        resolve_metadata(name, metadata_lookup)
    if set(mx04_lookup) != set(oc_lookup):
        raise ValueError("Unpaired MX04/OC masks; complete both probability exports first")

    # =========================================================================
    # OUTPUT COLLECTIONS
    # =========================================================================

    mx04_central_rows = []
    oc_central_rows = []

    mx04_object_rows = []
    oc_object_rows = []

    feature_rows = []

    # =========================================================================
    # PROCESS STACKS
    # =========================================================================

    for image_name in shared_names:

        print(
            f"Processing pair: {image_name}"
        )

        # ---------------------------------------------------------------------
        # LOAD
        # ---------------------------------------------------------------------

        mx04_raw = read_binary_mask(
            mx04_lookup[
                image_name
            ]
        )

        oc_raw = read_binary_mask(
            oc_lookup[
                image_name
            ]
        )

        if mx04_raw.shape != oc_raw.shape:

            print(
                "  WARNING: shape mismatch "
                f"{mx04_raw.shape} vs "
                f"{oc_raw.shape}"
            )

            continue

        original_shape = (
            mx04_raw.shape
        )

        # ---------------------------------------------------------------------
        # CENTRAL XY CROP
        # ---------------------------------------------------------------------

        mx04_raw = crop_xy(
            mx04_raw,
            cfg.crop_frac,
        )

        oc_raw = crop_xy(
            oc_raw,
            cfg.crop_frac,
        )

        print(
            f"  crop: {original_shape} "
            f"-> {mx04_raw.shape}"
        )

        # ---------------------------------------------------------------------
        # CLEAN / CONNECT PIXELATED PLAQUE MASKS
        #
        # Deliberately NO small-object filter here.
        #
        # First give fragmented pieces the chance to connect.
        # ---------------------------------------------------------------------

        mx04_clean = clean_plaque_mask(
            mx04_raw,
            cfg=cfg,
            minimum_size=None,
        )

        oc_clean = clean_plaque_mask(
            oc_raw,
            cfg=cfg,
            minimum_size=None,
        )

        # ---------------------------------------------------------------------
        # SAVE CLEANED MASKS
        # ---------------------------------------------------------------------

        save_binary_mask(
            cfg.cleaned_mask_dir
            / "MX04"
            / (
                f"{image_name}"
                "_MX04_cleaned.ome.tiff"
            ),
            mx04_clean,
        )

        save_binary_mask(
            cfg.cleaned_mask_dir
            / "OC"
            / (
                f"{image_name}"
                "_OC_cleaned.ome.tiff"
            ),
            oc_clean,
        )

        # ---------------------------------------------------------------------
        # QC CLEANED MASKS
        # ---------------------------------------------------------------------

        save_qc(
            mx04_clean,
            qc_dir
            / (
                f"{image_name}"
                "_MX04_cleaned.png"
            ),
            cfg.qc_z,
        )

        save_qc(
            oc_clean,
            qc_dir
            / (
                f"{image_name}"
                "_OC_cleaned.png"
            ),
            cfg.qc_z,
        )

        # ---------------------------------------------------------------------
        # ALL OBJECT MORPHOLOGY
        #
        # Filtering is applied HERE, after cleaning.
        # ---------------------------------------------------------------------

        mx04_objects = (
            calculate_morphological_properties(
                mx04_clean,
                cfg.per_object_min_area_voxels,
                cfg,
            )
        )

        for row in mx04_objects:

            r = row.copy()

            r[
                "image_name"
            ] = image_name

            mx04_object_rows.append(
                r
            )

        oc_objects = (
            calculate_morphological_properties(
                oc_clean,
                cfg.per_object_min_area_voxels,
                cfg,
            )
        )

        for row in oc_objects:

            r = row.copy()

            r[
                "image_name"
            ] = image_name

            oc_object_rows.append(
                r
            )

        # ---------------------------------------------------------------------
        # CENTRAL PLAQUE
        # ---------------------------------------------------------------------

        mx04_central = (
            extract_central_plaque(
                mx04_clean,
                cfg,
            )
        )

        oc_central = (
            extract_central_plaque(
                oc_clean,
                cfg,
            )
        )

        # ---------------------------------------------------------------------
        # SAVE CENTRAL PLAQUES
        # ---------------------------------------------------------------------

        save_binary_mask(
            cfg.central_mask_dir
            / "MX04"
            / (
                f"{image_name}"
                "_MX04_central.ome.tiff"
            ),
            mx04_central,
        )

        save_binary_mask(
            cfg.central_mask_dir
            / "OC"
            / (
                f"{image_name}"
                "_OC_central.ome.tiff"
            ),
            oc_central,
        )

        # ---------------------------------------------------------------------
        # CENTRAL QC
        # ---------------------------------------------------------------------

        save_qc(
            mx04_central,
            qc_dir
            / (
                f"{image_name}"
                "_MX04_central.png"
            ),
            cfg.qc_z,
        )

        save_qc(
            oc_central,
            qc_dir
            / (
                f"{image_name}"
                "_OC_central.png"
            ),
            cfg.qc_z,
        )

        # ---------------------------------------------------------------------
        # CENTRAL MORPHOLOGY
        # ---------------------------------------------------------------------

        mx04_props = (
            properties_from_single_mask(
                mx04_central,
                cfg,
            )
        )

        oc_props = (
            properties_from_single_mask(
                oc_central,
                cfg,
            )
        )

        # ---------------------------------------------------------------------
        # METADATA
        # ---------------------------------------------------------------------

        record = resolve_metadata(image_name, metadata_lookup)
        sample_id = record["sample_id"]
        animal_id = record["animal_id"]
        genotype = record["Genotype"]

        # ---------------------------------------------------------------------
        # OCCUPANCY QC
        # ---------------------------------------------------------------------

        mx04_raw_fraction = float(
            np.mean(
                mx04_raw
            )
        )

        mx04_clean_fraction = float(
            np.mean(
                mx04_clean
            )
        )

        mx04_final_fraction = float(
            np.mean(
                mx04_central
            )
        )

        oc_raw_fraction = float(
            np.mean(
                oc_raw
            )
        )

        oc_clean_fraction = float(
            np.mean(
                oc_clean
            )
        )

        oc_final_fraction = float(
            np.mean(
                oc_central
            )
        )

        # ---------------------------------------------------------------------
        # CENTRAL TABLES
        # ---------------------------------------------------------------------

        mx04_row = {

            "image_name":
                image_name,

            "sample_id":
                sample_id,

            "animal_id": animal_id,

            "Genotype":
                genotype,

            **mx04_props,

            "qc_raw_fraction_on":
                mx04_raw_fraction,

            "qc_clean_fraction_on":
                mx04_clean_fraction,

            "qc_final_fraction_on":
                mx04_final_fraction,
        }

        oc_row = {

            "image_name":
                image_name,

            "sample_id":
                sample_id,

            "animal_id": animal_id,

            "Genotype":
                genotype,

            **oc_props,

            "qc_raw_fraction_on":
                oc_raw_fraction,

            "qc_clean_fraction_on":
                oc_clean_fraction,

            "qc_final_fraction_on":
                oc_final_fraction,
        }

        mx04_central_rows.append(
            mx04_row
        )

        oc_central_rows.append(
            oc_row
        )

        # =====================================================================
        # OC HALO
        # =====================================================================

        oc_halo = (
            oc_central
            & ~mx04_central
        )

        save_binary_mask(
            cfg.halo_mask_dir
            / (
                f"{image_name}"
                "_OC_halo.ome.tiff"
            ),
            oc_halo,
        )

        oc_halo_voxels = int(
            np.sum(
                oc_halo
            )
        )

        oc_halo_volume_um3 = (
            oc_halo_voxels
            * voxel_volume_um3(cfg)
        )

        oc_voxels = int(
            np.sum(
                oc_central
            )
        )

        if oc_voxels > 0:

            oc_halo_fraction = (
                oc_halo_voxels
                / oc_voxels
            )

        else:

            oc_halo_fraction = np.nan

        # =====================================================================
        # DECISION-TREE FEATURE ROW
        # =====================================================================

        feature_row = {

            "image_name":
                image_name,

            "sample_id":
                sample_id,

            "animal_id": animal_id,

            "Genotype":
                genotype,
        }

        add_channel_features(
            feature_row,
            mx04_props,
            "m04",
        )

        add_channel_features(
            feature_row,
            oc_props,
            "oc",
        )

        # ---------------------------------------------------------------------
        # VOLUME RATIO
        # ---------------------------------------------------------------------

        mx04_volume = (
            mx04_props[
                "volume_um3"
            ]
        )

        oc_volume = (
            oc_props[
                "volume_um3"
            ]
        )

        if (
            np.isfinite(mx04_volume)
            and mx04_volume > 0
            and np.isfinite(oc_volume)
        ):

            feature_row[
                "volume_ratio_oc_m04"
            ] = (
                oc_volume
                / mx04_volume
            )

        else:

            feature_row[
                "volume_ratio_oc_m04"
            ] = np.nan

        # ---------------------------------------------------------------------
        # SURFACE AREA RATIO
        # ---------------------------------------------------------------------

        mx04_surface = (
            mx04_props[
                "surface_area_um2"
            ]
        )

        oc_surface = (
            oc_props[
                "surface_area_um2"
            ]
        )

        if (
            np.isfinite(mx04_surface)
            and mx04_surface > 0
            and np.isfinite(oc_surface)
        ):

            feature_row[
                "surface_area_ratio_oc_m04"
            ] = (
                oc_surface
                / mx04_surface
            )

        else:

            feature_row[
                "surface_area_ratio_oc_m04"
            ] = np.nan

        # ---------------------------------------------------------------------
        # HALO
        # ---------------------------------------------------------------------

        feature_row[
            "oc_halo_voxels"
        ] = (
            oc_halo_voxels
        )

        feature_row[
            "oc_halo_volume_um3"
        ] = float(
            oc_halo_volume_um3
        )

        feature_row[
            "oc_halo_fraction_of_oc"
        ] = (
            oc_halo_fraction
        )

        # ---------------------------------------------------------------------
        # QC
        # ---------------------------------------------------------------------

        feature_row[
            "qc_raw_fraction_on_m04"
        ] = (
            mx04_raw_fraction
        )

        feature_row[
            "qc_clean_fraction_on_m04"
        ] = (
            mx04_clean_fraction
        )

        feature_row[
            "qc_final_fraction_on_m04"
        ] = (
            mx04_final_fraction
        )

        feature_row[
            "qc_raw_fraction_on_oc"
        ] = (
            oc_raw_fraction
        )

        feature_row[
            "qc_clean_fraction_on_oc"
        ] = (
            oc_clean_fraction
        )

        feature_row[
            "qc_final_fraction_on_oc"
        ] = (
            oc_final_fraction
        )

        feature_rows.append(
            feature_row
        )

    # =========================================================================
    # SAVE CSV OUTPUTS
    # =========================================================================

    for row in mx04_object_rows + oc_object_rows:
        row.update(resolve_metadata(row["image_name"], metadata_lookup))

    pd.DataFrame(
        mx04_object_rows
    ).to_csv(
        cfg.morphology_combined_dir
        / "overall_MX04_all_objects.csv",
        index=False,
    )

    pd.DataFrame(
        oc_object_rows
    ).to_csv(
        cfg.morphology_combined_dir
        / "overall_OC_all_objects.csv",
        index=False,
    )

    pd.DataFrame(
        mx04_central_rows
    ).to_csv(
        cfg.morphology_combined_dir
        / "overall_MX04_central.csv",
        index=False,
    )

    pd.DataFrame(
        oc_central_rows
    ).to_csv(
        cfg.morphology_combined_dir
        / "overall_OC_central.csv",
        index=False,
    )

    feature_df = pd.DataFrame(
        feature_rows
    )

    feature_df.to_csv(
        cfg.feature_table_path,
        index=False,
    )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print()
    print(
        "Morphology complete"
    )

    print(
        f"Pairs analysed: "
        f"{len(feature_df)}"
    )

    print(
        f"Crop fraction: "
        f"{cfg.crop_frac}"
    )

    print(
        f"Closing radius: "
        f"{cfg.morphology_closing_radius}"
    )

    print(
        f"Hole filling: "
        f"{cfg.morphology_fill_holes}"
    )

    print(
        f"Voxel spacing: "
        f"{cfg.voxel_xy_um} x "
        f"{cfg.voxel_xy_um} x "
        f"{cfg.voxel_z_um} µm"
    )

    print(
        f"Feature table: "
        f"{cfg.feature_table_path}"
    )

    if not feature_df.empty:

        for column in (
            "solidity_m04",
            "solidity_oc",
        ):

            if column not in feature_df.columns:
                continue

            n_missing = int(
                feature_df[
                    column
                ].isna().sum()
            )

            print(
                f"Missing {column}: "
                f"{n_missing} / "
                f"{len(feature_df)}"
            )


# =============================================================================
# OPTIONAL ALIAS
# =============================================================================


def run_morphology(
    cfg: Config = CFG,
) -> None:
    """
    Compatibility alias.
    """

    return process_morphology(
        cfg
    )
