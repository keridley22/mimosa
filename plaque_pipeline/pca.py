"""Descriptive PCA of plaque profiles and mean animal profiles."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .config import CFG
from .decision_tree import check_marker_features, file_hash, prepare_table

GROUPS = ('APP-WT', 'APP-FIRE', 'APP-FIRE+transplant')
COLOURS = {'APP-WT': '#0072B2', 'APP-FIRE': '#D55E00', 'APP-FIRE+transplant': '#009E73'}
LABELS = {'APP-WT': 'APP-WT', 'APP-FIRE': 'APP-FIRE', 'APP-FIRE+transplant': 'Transplanted'}


def select_features(frame, cfg):
    """Keep physical volume rather than counting the same volume twice in PCA."""
    selected, rows = [], []
    aliases = {'area_voxels_m04': 'volume_um3_m04', 'area_voxels_oc': 'volume_um3_oc'}
    factor = cfg.voxel_xy_um**2 * cfg.voxel_z_um
    for feature in cfg.model_features:
        duplicate = feature in aliases and aliases[feature] in cfg.model_features
        if duplicate and not np.allclose(frame[feature]*factor, frame[aliases[feature]], rtol=1e-6):
            raise ValueError(f'{feature} is not consistent with physical volume')
        rows.append(dict(feature=feature, included_in_pca=not duplicate,
                         reason=f'duplicate_of_{aliases[feature]}' if duplicate else 'retained_tree_feature'))
        if not duplicate:
            selected.append(feature)
    return selected, pd.DataFrame(rows)


def aggregate_animals(frame, features):
    if frame.groupby('animal_id').Genotype.nunique().gt(1).any():
        raise ValueError('An animal has conflicting genotypes')
    keys = ['animal_id', 'original_animal_id', 'cohort', 'Genotype']
    grouped = frame.groupby(keys, sort=True, dropna=False)
    result = grouped[features].mean().reset_index()
    counts = grouped.size().rename('n_plaques').reset_index()
    return result.merge(counts, on=keys, validate='one_to_one')


def fit_profiles(frame, features, fit_scope):
    if fit_scope not in ('all', 'historical'):
        raise ValueError('PCA fit scope must be all or historical')
    fit_mask = pd.Series(True, index=frame.index) if fit_scope == 'all' else frame.cohort.eq('historical')
    values = frame[features].to_numpy(dtype=float)
    if not np.isfinite(values).all() or int(fit_mask.sum()) < 3:
        raise ValueError('PCA requires complete finite profiles and at least three fitting observations')
    fit_values = values[fit_mask.to_numpy()]
    scaler = StandardScaler().fit(fit_values)
    constant = scaler.var_ == 0
    active = [f for f, is_constant in zip(features, constant) if not is_constant]
    if len(active) < 2:
        raise ValueError('Fewer than two nonconstant features remain')
    scaler = StandardScaler().fit(frame.loc[fit_mask, active])
    standardized = scaler.transform(frame[active])
    model = PCA(svd_solver='full', whiten=False).fit(standardized[fit_mask.to_numpy()])
    if np.count_nonzero(model.explained_variance_ > 1e-12) < 2:
        raise ValueError('Fewer than two informative principal components remain')
    coordinates = model.transform(standardized)
    identity = [c for c in ['image_name', 'animal_id', 'original_animal_id', 'cohort', 'Genotype', 'n_plaques'] if c in frame]
    scores = frame[identity].copy()
    scores['used_to_fit_pca'] = fit_mask.to_numpy()
    components = [f'PC{i+1}' for i in range(coordinates.shape[1])]
    for i, component in enumerate(components):
        scores[component] = coordinates[:, i]
    variance = pd.DataFrame(dict(component=components, variance=model.explained_variance_,
                                 fraction_explained=model.explained_variance_ratio_,
                                 cumulative_fraction=np.cumsum(model.explained_variance_ratio_)))
    weights = pd.DataFrame(model.components_.T, index=pd.Index(active, name='feature'), columns=components)
    scaling = pd.DataFrame(dict(feature=active, mean=scaler.mean_, population_sd=scaler.scale_))
    return dict(scores=scores, variance=variance, weights=weights, scaling=scaling,
                scaler=scaler, model=model, active_features=active,
                constant_features=[f for f, c in zip(features, constant) if c])


def plot_pca(result, level, fit_scope, output):
    scores, variance = result['scores'], result['variance']
    fig, ax = plt.subplots(figsize=(8.5, 6.5), layout='constrained')
    for group in GROUPS:
        data = scores[scores.Genotype.eq(group)]
        ax.scatter(data.PC1, data.PC2, c=COLOURS[group], s=52 if level == 'plaque' else 90,
                   alpha=.7 if level == 'plaque' else .95, edgecolors='white', linewidths=.6,
                   label=f'{LABELS[group]} (n={len(data)})', zorder=3)
        if level == 'animal' and group == 'APP-FIRE+transplant':
            for _, row in data.iterrows():
                ax.annotate(row.original_animal_id, (row.PC1, row.PC2), xytext=(6, 7),
                            textcoords='offset points', fontsize=9, color=COLOURS[group], zorder=4)
    ax.axhline(0, color='#bbbbbb', linewidth=.7, zorder=0)
    ax.axvline(0, color='#bbbbbb', linewidth=.7, zorder=0)
    variance_label = 'historical variance' if fit_scope == 'historical' else 'variance'
    ax.set_xlabel(f"PC1 ({variance.fraction_explained.iloc[0]:.1%} {variance_label})", fontsize=11)
    ax.set_ylabel(f"PC2 ({variance.fraction_explained.iloc[1]:.1%} {variance_label})", fontsize=11)
    ax.set_title('PCA of plaque feature profiles' if level == 'plaque' else 'PCA of mean animal feature profiles',
                 fontsize=15, pad=34)
    subtitle = ('One point per plaque' if level == 'plaque' else 'One point per animal; features averaged across plaques')
    subtitle += f"\n{len(result['active_features'])} standardized features; " + ('all three groups fitted jointly' if fit_scope == 'all' else 'historical fit; transplanted profiles projected')
    ax.text(.5, 1.015, subtitle, transform=ax.transAxes, ha='center', va='bottom', fontsize=9, color='#555555')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, loc='best', fontsize=10)
    ax.margins(x=.15, y=.12)
    for extension in ('png', 'pdf'):
        fig.savefig(output / f'{level}_pca.{extension}', dpi=250, bbox_inches='tight')
    plt.close(fig)


def run_pca(cfg=CFG, fit_scope='historical', output_dir=None):
    historical_path = cfg.previous_output_root / 'morphology_results/combined/decision_tree_features.csv'
    current_path = cfg.feature_table_path
    hashes = dict(historical=file_hash(historical_path), current=file_hash(current_path))
    historical, excluded_historical = prepare_table(historical_path, 'historical', cfg)
    current, excluded_current = prepare_table(current_path, 'transplanted', cfg)
    for frame, path in [(historical, historical_path), (current, current_path)]:
        check_marker_features(frame, path.parent, cfg)
    transplanted = current[current.Genotype.eq('APP-FIRE+transplant')]
    unpooled_controls = current[current.Genotype.ne('APP-FIRE+transplant')].copy()
    if set(historical.Genotype) != set(cfg.train_labels) or transplanted.empty:
        raise ValueError('PCA requires historical APP-WT, historical APP-FIRE and transplanted profiles')
    plaques = pd.concat([historical, transplanted], ignore_index=True)
    features, selection = select_features(plaques, cfg)
    animals = aggregate_animals(plaques, features)
    results = {level: fit_profiles(frame, features, fit_scope)
               for level, frame in [('plaque', plaques), ('animal', animals)]}
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = Path(output_dir) if output_dir else cfg.output_root / 'pca_results' / f'run_{stamp}'
    output.mkdir(parents=True, exist_ok=False)
    selection.to_csv(output / 'feature_selection.csv', index=False)
    plaques.to_csv(output / 'plaque_feature_profiles.csv', index=False)
    animals.to_csv(output / 'animal_mean_feature_profiles.csv', index=False)
    pd.concat([excluded_historical, excluded_current], ignore_index=True).to_csv(output / 'excluded_plaques.csv', index=False)
    unpooled_controls.to_csv(output / 'new_cohort_controls_not_pooled.csv', index=False)
    for level, result in results.items():
        for name in ('scores', 'variance', 'scaling'):
            result[name].to_csv(output / f'{level}_{name}.csv', index=False)
        result['weights'].to_csv(output / f'{level}_component_weights.csv')
        joblib.dump(dict(scaler=result['scaler'], pca=result['model'], features=result['active_features'],
                         fit_scope=fit_scope, aggregation='none' if level == 'plaque' else 'raw_feature_mean_per_animal'),
                    output / f'{level}_pca_model.joblib')
        plot_pca(result, level, fit_scope, output)
    if hashes != dict(historical=file_hash(historical_path), current=file_hash(current_path)):
        raise RuntimeError('An input changed during PCA; rerun using stable input tables')
    manifest = dict(created_utc=stamp, fit_scope=fit_scope, features=features, sklearn_version=sklearn.__version__,
                    historical_source=str(historical_path), current_source=str(current_path), source_sha256=hashes,
                    standardization='feature z-scores; population standard deviation; no log transform or whitening',
                    animal_aggregation='Arithmetic mean of each feature across included plaques, before fitting scaler/PCA',
                    plot_counts={level: result['scores'].Genotype.value_counts().to_dict() for level, result in results.items()},
                    constant_features_dropped={level: result['constant_features'] for level, result in results.items()},
                    PC1_PC2_variance={level: result['variance'].fraction_explained.iloc[:2].tolist() for level, result in results.items()},
                    new_cohort_controls_pooled=False)
    (output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False))
    report = f'''# PCA of historical controls and transplanted plaques

The plots use {len(plaques)} plaques from {len(animals)} animals: historical APP-WT,
historical APP-FIRE, and corrected APP-FIRE+transplant measurements.
The seven newly acquired APP-FIRE control plaques remain outside the comparison,
consistent with the decision-tree analysis. Exclusions and all input profiles are saved.

## Preprocessing and fitting

The decision tree's feature set was used, omitting the two voxel-count columns
because each is exactly proportional to its corresponding physical-volume column.
This leaves {len(features)} distinct input columns. Other correlated features are retained.
The decision tree itself is unchanged. `feature_selection.csv` records every inclusion.

Features are centered and scaled to unit population standard deviation before PCA.
No log transform, clipping, outlier removal, batch correction or whitening is applied.
Constant features, if any, are omitted separately at each level and recorded in the manifest.
Fit scope: **{fit_scope}**. {'All three groups enter the scaler and PCA fits together. This is exploratory, unsupervised visualization.' if fit_scope == 'all' else 'Only historical profiles enter the scaler and PCA fits; transplanted profiles are transformed using those fitted parameters.'}
Group labels are used only for selecting/displaying cohorts, not as PCA target variables.

For the animal plot, each feature is first averaged across an animal's included plaques.
The animal means then receive their own scaling and PCA. Each animal contributes one
observation to that fit. Ratios are averaged as measured per plaque, rather than
recomputed from averaged numerator/denominator values.
The two plots therefore have different fitted axes; their coordinates must not be
compared numerically across plots. Plaque-level fitting weights each plaque equally,
so animals with more plaques contribute more observations there.

## Plots and variance

| Plot | PC1 variance | PC2 variance | Combined |
|---|---:|---:|---:|
'''
    for level, result in results.items():
        v = result['variance'].fraction_explained.iloc[:2]
        report += f'| {level} | {v.iloc[0]:.2%} | {v.iloc[1]:.2%} | {v.sum():.2%} |\n'
    report += '''
PNG and vector PDF versions of both plots are included. Scores identify every plaque
or animal. Transplanted animal IDs are labelled on the animal plot; all IDs are in its scores CSV.
`component_weights.csv` files contain unit-norm eigenvector coefficients, not feature-PC correlations.
Scaling parameters, all explained-variance fractions and fitted models are saved for reproducibility.
PCA separation does not by itself establish a treatment effect; acquisition cohort
and residual segmentation differences can contribute to the feature variation.

Rerun: `bash run.sh pca` (historical fit with transplanted projection, as requested).
The explicit equivalent is `bash run.sh pca --pca-fit historical`.
An exploratory joint fit is available with `--pca-fit all`. Each run creates a new dated result folder.

Methods: [scikit-learn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
and [feature standardization](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).
'''
    (output / 'README.md').write_text(report)
    print(json.dumps(manifest['plot_counts'], indent=2))
    print(json.dumps(manifest['PC1_PC2_variance'], indent=2))
    print(f'PCA results: {output}')
    return output
