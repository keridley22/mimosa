# Modular Plaque Pipeline — fast preprocessing + morphology + decision tree

## Pipeline

1. Converted multichannel OME-TIFF input
2. Fast rolling-ball preprocessing (background estimated at 0.25x XY resolution)
3. Manual ilastik Pixel Classification
4. H5 probabilities -> binary 3D masks
5. 3D morphology, central plaque extraction, OC halo and combined feature table
6. Decision-tree grid search, cross-validation and held-out 80/20 evaluation

## Configuration

Edit only `plaque_pipeline/config.py` for normal use.

Important paths:
- `raw_input_dir`
- `output_root`
- `genotype_csv`

Preprocessed images are saved to:

```
<output_root>/preprocessed_ome_tiffs/MX04/
<output_root>/preprocessed_ome_tiffs/OC/
```

The ML-ready feature table is saved to:

```
<output_root>/morphology_results/combined/decision_tree_features.csv
```

Decision-tree outputs are saved to:

```
<output_root>/decision_tree_results/
```

## Commands

```bash
python main.py preprocess
python main.py ilastik_instructions
python main.py h5_to_masks
python main.py morphology
python main.py decision_tree
```

After manual ilastik, run everything remaining with:

```bash
python main.py post_ilastik
```

If masks already exist and you only want morphology + ML:

```bash
python main.py analysis
```

## Combined central-plaque metrics

The pipeline writes the requested metrics:

- solidity_oc
- complexity_oc
- major_axis_length_m04
- solidity_m04
- elongation_oc
- minor_axis_length_m04
- surface_area_oc
- area_voxels_m04
- sphericity_m04
- volume_um3_m04
- volume_ratio_oc_m04
- surface_area_ratio_oc_m04
- surface_area_m04
- sphericity_oc
- minor_axis_length_oc
- area_voxels_oc
- major_axis_length_oc
- extent_oc
- extent_m04
- elongation_m04
- complexity_m04
- volume_um3_oc
- oc_halo_volume_um3

It also records `oc_halo_voxels` and `oc_halo_fraction_of_oc` as useful QC/secondary metrics.

Definitions:
- elongation = major axis / minor axis
- complexity = volume / surface area
- volume_ratio_oc_m04 = OC central volume / MX04 central volume
- surface_area_ratio_oc_m04 = OC central surface area / MX04 central surface area
- OC halo = central OC mask AND NOT central MX04 mask

## Decision-tree workflow

The workflow follows the supplied `supervisedMLmodels.ipynb` logic while preventing test-set leakage:

1. Restrict to configured training genotypes (default APP-WT and APP-FIRE).
2. Make a stratified 80/20 train/test split.
3. Run GridSearchCV only on the 80% training set.
4. Rank metrics using the tuned decision tree's feature importances; non-zero metrics are reported as selected by the tree.
5. Cross-validate the selected hyperparameter set on the training data.
6. Evaluate once on the untouched 20% test set.
7. Refit the chosen tree on all labelled training data and save it for transfer/prediction experiments.

Outputs include:
- `grid_search_results.csv`
- `feature_importance_ranking.csv`
- `best_model_cross_validation_scores.csv`
- `held_out_predictions.csv`
- `classification_report.csv`
- `confusion_matrix.csv`
- `metrics.json`
- `decision_tree.png`
- `tree_rules.txt`
- `best_dt_combined.joblib`
