# Six-month young plaque analysis

This separate copy uses the corrected transplanted pipeline's original same-marker
ilastik classifiers, raw channel 0 as MX04 and channel 1 as OC, 0.60 threshold,
rolling-ball/Gaussian preprocessing, central 40% XY crop, and identical morphology
cleaning, object thresholds and feature definitions. It does not overwrite the
historical or transplanted analyses.

## Run

```bash
cd /Users/katherineridley/Projects/PlaqueStack/ilastik_protocol/plaque_pipeline_young
bash run.sh run
```

This converts the LIF, preprocesses, segments both markers, measures morphology,
and exports animal-level Welch tests plus OC halo/ratio CSVs, followed by PCA and
saved historical decision-tree projections. Individual stages
are available with `bash run.sh --help`. To resume after partial segmentation:

```bash
bash run.sh segment
bash run.sh post_ilastik
```

Configuration: `plaque_pipeline/young_config.py` overrides `Config` defaults.
Raw LIF: `plaque_pipeline_modular_fast/young_lifs/20250611_04002.lif`.
Original metadata: `plaque_pipeline_modular_fast/young_lifs/metadata.csv`.
Converted originals: `plaque_pipeline_modular_fast/young_converted_ometiffs/`.
Results: `young_plaque_pipeline_outputs_20260929/` under `ilastik_protocol`.
`input_audit/run_status.json` records stage completion or failure.

## Acquisition calibration

The LIF has 31 series. The existing P+ selection includes 29 stacks and excludes
two P- stacks. Nineteen included stacks have 0.180375 um XY pixels; ten have
0.138750 um pixels. Z spacing is approximately 1.000715 um throughout.

As selected by the user, the ten finer stacks are downsampled in XY to the prior
0.18 um analysis grid before preprocessing and classification. Each XY plane is
resized separately using linear interpolation and anti-aliasing; channels and Z
planes are never mixed. A 1024 x 1024 fine field becomes 789 x 789 pixels.
The rounded field width differs by under 0.1%; the field is not expanded to match
other acquisitions. Source converted images retain the original resolution and
OME calibration. Source and analysis spacing, shapes, and resampling flags are
recorded in the inventory and preprocessing logs. Stacks already within the
existing 1% calibration tolerance retain the same processing as the previous
analysis. All final measurements use the same nominal 0.18 x 0.18 x 1 um voxels.
Implementation reference: https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.resize

The metadata's APP-WT/APP-FIRE labels map to APP-WT young/APP-FIRE young only when
reading the young metadata. Historical genotype labels are not relabelled.
A_4/PF1006, B_1/PF1025, C_2/PF1068 are young APP-WT; A_1/PF1021 and C_4/PF1136
are young APP-FIRE. C_3/PF1129 has no matching series in this LIF and is recorded
as missing, never included as a zero-valued animal.

## Statistics and CSVs

As requested, historical APP-WT and APP-FIRE controls are compared with the two
young groups. Existing historical exclusions remain; there are no new image
exclusions. Animal identifiers are namespaced historical:/young: to prevent
slide-name collisions. Transplanted animals are not included.

Each group's data are exported as individual plaque values (animal columns),
animal means (genotype columns), and tidy tables. Six pairwise two-sided Welch
comparisons use animal means as independent replicates. Benjamini-Hochberg FDR
covers all features and comparisons within each marker dataset, and separately
all 24 OC halo/ratio tests (four outcomes x six comparisons).
The young cohort has three APP-WT and two APP-FIRE animals available; only
animals with measured values enter a test.

OC halo means OC-positive voxels outside MX04; volume ratio and surface ratio
are OC divided by MX04. Animal means average individual plaque ratios.
Halo statistics verify saved central/halo masks and measurements before export.
See `morphology_results/ttests/OC_halo_and_ratios/run_*/README.md` for file details.
PCA and decision-tree transfer can also be run with `bash run.sh profiles`.
PCA uses only historical controls to fit the scaler/components, then projects
both young groups; one plot shows individual plaques and another animal means.
The saved all-historical and holdout-trained decision trees are applied without
retraining. Historical holdout precision/recall/F1 remain historical validation
results; they are not presented as young-cohort accuracy.

## Storage and validation

LIF conversion uses lossless LZW compression; saved binary masks use lossless
zlib compression. The optional training subset uses hard links to preprocessed
images to avoid storing duplicate pixel data. Working ilastik projects are
copies; headless inference uses the original classifiers read-only.
Processing stops before the next conversion/preprocessing/segmentation job if
less than 2 GiB remains. No original data or prior output is deleted.
The classifiers have not been retrained on young plaques. Review the saved QC
images against fluorescence before interpreting cohort differences.
