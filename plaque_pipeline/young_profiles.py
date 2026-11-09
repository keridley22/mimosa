"""Historical-reference PCA and frozen decision-tree projections for young plaques."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from .decision_tree import (prepare_table, check_marker_features, file_hash, predict_table, animal_summary)
from .pca import select_features, aggregate_animals, fit_profiles

COLOURS = {'APP-WT': '#0072B2', 'APP-FIRE': '#D55E00',
           'APP-WT young': '#009E73', 'APP-FIRE young': '#CC79A7'}


def plot_profiles(result, level, groups, output):
    scores, variance = result['scores'], result['variance']
    fig, ax = plt.subplots(figsize=(9, 7), layout='constrained')
    for group in groups:
        data = scores.loc[scores.Genotype.eq(group)]
        ax.scatter(data.PC1, data.PC2, color=COLOURS[group],
                   marker='D' if group.endswith('young') else 'o',
                   s=55 if level == 'plaque' else 90, alpha=.75,
                   edgecolors='white', linewidths=.6, label=f'{group} (n={len(data)})')
        if level == 'animal' and group.endswith('young'):
            for row in data.itertuples():
                ax.annotate(row.original_animal_id, (row.PC1, row.PC2), xytext=(5, 6),
                            textcoords='offset points', fontsize=9, color=COLOURS[group])
    ax.axhline(0, color='#cccccc', linewidth=.7, zorder=0)
    ax.axvline(0, color='#cccccc', linewidth=.7, zorder=0)
    ax.set_xlabel(f'PC1 ({variance.fraction_explained.iloc[0]:.1%} historical variance)')
    ax.set_ylabel(f'PC2 ({variance.fraction_explained.iloc[1]:.1%} historical variance)')
    title = 'Plaque feature profiles' if level == 'plaque' else 'Mean animal feature profiles'
    ax.set_title(f'{title}\nHistorical PCA fit; young profiles projected', fontsize=15, pad=15)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, fontsize=10)
    ax.margins(.15)
    for extension in ('png', 'pdf'):
        fig.savefig(output / f'{level}_pca.{extension}', dpi=220, bbox_inches='tight')
    plt.close(fig)


def project_saved_tree(bundle_path, frame, features, model_name):
    before = file_hash(bundle_path)
    bundle = joblib.load(bundle_path)
    if bundle['features'] != list(features) or bundle['training_cohort'] != 'historical_only':
        raise ValueError('Saved tree features or training cohort differ from the requested reference')
    if list(bundle['model'].classes_) != ['APP-FIRE', 'APP-WT']:
        raise ValueError('Saved tree classes are not historical APP-WT and APP-FIRE')
    result = predict_table(bundle['model'], frame, list(features), model_name)
    if file_hash(bundle_path) != before:
        raise RuntimeError('Saved reference model changed during prediction')
    return result


def run_young_profiles(cfg, output_dir=None, tree_reference=None):
    reference = Path(tree_reference) if tree_reference else (
        cfg.previous_output_root.parent / 'transplanted_plaque_pipeline_outputs_corrected_20260922'
        / 'decision_tree_results/run_20260924T114218821935Z')
    historical_path = cfg.previous_output_root / 'morphology_results/combined/decision_tree_features.csv'
    current_path = cfg.feature_table_path
    sources = [historical_path, current_path, reference/'final_tree.joblib', reference/'evaluation_tree.joblib',
               reference/'historical_model_inputs.csv', reference/'run_manifest.json']
    hashes = {str(p): file_hash(p) for p in sources}
    historical, old_omitted = prepare_table(historical_path, 'historical', cfg)
    young, new_omitted = prepare_table(current_path, 'young', cfg)
    if set(young.Genotype) != set(cfg.current_comparison_genotypes):
        raise ValueError('Both young groups need complete profiles for projection')
    for frame, path in ((historical, historical_path), (young, current_path)):
        check_marker_features(frame, path.parent, cfg)
    # Require the same historical rows and measurements used to train the frozen tree.
    fitted = pd.read_csv(reference/'historical_model_inputs.csv').set_index('image_name')
    current_reference = historical.set_index('image_name')
    checked = ['animal_id', 'Genotype', *cfg.model_features]
    pd.testing.assert_frame_equal(fitted[checked].sort_index(), current_reference[checked].sort_index(),
                                  check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-10)
    combined = pd.concat([historical, young], ignore_index=True)
    features, feature_selection = select_features(combined, cfg)
    animals = aggregate_animals(combined, features)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = Path(output_dir) if output_dir else cfg.output_root/'profile_analysis'/f'run_{stamp}'
    output.mkdir(parents=True, exist_ok=False)
    pca_out, tree_out = output/'pca', output/'decision_tree'
    pca_out.mkdir(); tree_out.mkdir()
    combined.to_csv(pca_out/'plaque_feature_profiles.csv', index=False)
    animals.to_csv(pca_out/'animal_mean_feature_profiles.csv', index=False)
    feature_selection.to_csv(pca_out/'feature_selection.csv', index=False)
    pd.concat([old_omitted, new_omitted], ignore_index=True).to_csv(output/'excluded_profiles.csv', index=False)
    counts = {}
    for level, frame in (('plaque', combined), ('animal', animals)):
        result = fit_profiles(frame, features, 'historical')
        for name in ('scores', 'variance', 'scaling'):
            result[name].to_csv(pca_out/f'{level}_{name}.csv', index=False)
        result['weights'].to_csv(pca_out/f'{level}_component_weights.csv')
        joblib.dump(dict(scaler=result['scaler'], pca=result['model'], features=result['active_features'],
                         fit_scope='historical'), pca_out/f'{level}_pca_model.joblib')
        plot_profiles(result, level, cfg.analysis_genotypes, pca_out)
        counts[level] = frame.Genotype.value_counts().to_dict()
    summaries = []
    for filename, name in (('final_tree.joblib', 'all_historical_model'),
                           ('evaluation_tree.joblib', 'historical_holdout_training_model')):
        predictions = project_saved_tree(reference/filename, young, cfg.model_features, name)
        predictions.to_csv(tree_out/f'young_plaque_predictions_{name}.csv', index=False)
        summary = animal_summary(predictions)
        summary.to_csv(tree_out/f'young_animal_summary_{name}.csv', index=False)
        group = summary.groupby('Genotype').agg(
            n_animals=('animal_id', 'size'), n_plaques=('n_plaques', 'sum'),
            mean_animal_fraction_APP_WT_like=('fraction_APP-WT_like', 'mean'),
            SD_animal_fraction_APP_WT_like=('fraction_APP-WT_like', 'std')).reset_index()
        group.to_csv(tree_out/f'young_group_summary_{name}.csv', index=False)
        summaries.append((name, summary))
    for filename in ('evaluation_metrics.csv', 'holdout_classification_report.csv',
                     'holdout_classification_report.md', 'animal_split_membership.csv'):
        shutil.copy2(reference/filename, tree_out/f'historical_{filename}')
    primary = summaries[0][1].sort_values(['Genotype', 'animal_id'])
    fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
    values = primary['fraction_APP-WT_like'].to_numpy()*100
    x = list(range(len(primary)))
    ax.bar(x, values, color='#0072B2', label='APP-WT-like')
    ax.bar(x, 100-values, bottom=values, color='#D55E00', label='APP-FIRE-like')
    ax.set_xticks(x, [f'{r.original_animal_id}\n{r.Genotype}' for r in primary.itertuples()], fontsize=9)
    ax.set(ylim=(0, 105), ylabel='Plaques classified (%)', title='Young plaques projected through the saved historical tree')
    ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(1, 1))
    ax.spines[['top', 'right']].set_visible(False)
    for extension in ('png', 'pdf'):
        fig.savefig(tree_out/f'young_animal_predictions.{extension}', dpi=220, bbox_inches='tight')
    plt.close(fig)
    if hashes != {str(p): file_hash(p) for p in sources}:
        raise RuntimeError('A reference model or input table changed during projection')
    (output/'manifest.json').write_text(json.dumps(dict(
        created_utc=stamp, source_sha256=hashes, tree_reference=str(reference), counts=counts,
        pca_fit_scope='historical_only', tree_refit=False, young_used_for_training=False,
        pca_features=features, tree_features=list(cfg.model_features),
        code_sha256=file_hash(Path(__file__))), indent=2)+'\n')
    (output/'README.md').write_text(f'''# Young plaque profiles in the historical reference

PCA includes historical APP-WT/APP-FIRE and both young groups. Plaque counts:
{counts['plaque']}. Animal counts: {counts['animal']}.
Scaling and PCA are fitted only to historical profiles; young profiles are
projected into that space. Separate fits are used for individual plaques and
animal means. Animal features are averaged before scaling. Duplicate voxel-count
features are removed, leaving {len(features)} PCA features. No log transform or
whitening is used. PNG/PDF plots, coordinates, loadings and scaling are in `pca/`.

Decision-tree predictions in `decision_tree/` use the unchanged saved historical
trees from {reference}. The primary tree was trained on all historical controls;
the other tree is the unchanged holdout-trained model. No young data enter
training or parameter selection. Plaque predictions, animal fractions and group
means of animal fractions are saved separately for each tree. Leaf fractions
are the model's stored class fractions, not calibrated biological probabilities.

The historical 20% holdout accuracy/precision/recall/F1 and animal split are copied
for reference. Those are historical validation results, not young-cohort accuracy.
APP-WT-like/APP-FIRE-like describe resemblance to historical plaque morphology;
the young animals retain their recorded genotypes. Age, acquisition and
segmentation differences can affect this projection.

Complete cases are used for profile analysis; excluded profiles and reasons are
in `excluded_profiles.csv`. Morphology t-tests retain per-feature valid values
and are saved separately under `morphology_results/ttests/`.
Reproduce with `bash run.sh profiles`.
''')
    print(f'Saved young PCA and historical-tree projections: {output}', flush=True)
    return output
