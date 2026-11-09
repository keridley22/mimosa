"""Historical WT/FIRE validation and independent transplanted phenotype prediction."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

from .config import CFG


def tree_parameters(cfg):
    return dict(criterion=cfg.decision_tree_criterion, max_depth=cfg.decision_tree_max_depth,
                min_samples_leaf=cfg.decision_tree_min_samples_leaf,
                min_samples_split=cfg.decision_tree_min_samples_split, random_state=cfg.random_state)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_marker_features(frame, folder, cfg):
    """Check feature units and identities against the same-marker central tables."""
    feature_to_central = dict(area_voxels='area_voxels', volume_um3='volume_um3', extent='extent',
                             surface_area='surface_area_um2', complexity='complexity_volume_surface',
                             sphericity='sphericity', solidity='solidity', elongation='elongation',
                             major_axis_length='major_axis_length_um', minor_axis_length='minor_axis_length_um')
    for marker, suffix in [('MX04', 'm04'), ('OC', 'oc')]:
        central = pd.read_csv(folder / f'overall_{marker}_central.csv').set_index('image_name', verify_integrity=True)
        if not set(frame.image_name).issubset(central.index):
            raise ValueError(f'Missing {marker} central records for feature table')
        central = central.loc[frame.image_name]
        for feature, source_column in feature_to_central.items():
            if not np.allclose(frame[f'{feature}_{suffix}'].to_numpy(dtype=float),
                               central[source_column].to_numpy(dtype=float), equal_nan=True):
                raise ValueError(f'{feature}_{suffix} differs from {marker} central measurements')
        if not np.allclose(central.volume_um3, central.area_voxels * cfg.voxel_xy_um**2 * cfg.voxel_z_um,
                           equal_nan=True):
            raise ValueError(f'{marker} volume calibration differs from configuration')
    for feature, numerator, denominator in [
        ('volume_ratio_oc_m04', 'volume_um3_oc', 'volume_um3_m04'),
        ('surface_area_ratio_oc_m04', 'surface_area_oc', 'surface_area_m04'),
    ]:
        expected = frame[numerator] / frame[denominator].replace(0, np.nan)
        if not np.allclose(frame[feature], expected, equal_nan=True):
            raise ValueError(f'Inconsistent {feature}')


def prepare_table(path, cohort, cfg):
    frame = pd.read_csv(path)
    required = {'image_name', 'Genotype', *cfg.model_features}
    identity = 'animal_id' if 'animal_id' in frame else 'sample_id'
    required.add(identity)
    if not required.issubset(frame):
        raise ValueError(f'{path}: missing columns {sorted(required-set(frame))}')
    if frame[['image_name', 'Genotype', identity]].isna().any().any():
        raise ValueError(f'Missing image/genotype/animal identity in {path}')
    if frame.image_name.duplicated().any():
        raise ValueError(f'Duplicate plaque identities in {path}')
    frame['Genotype'] = frame.Genotype.str.strip().replace(cfg.genotype_map)
    frame['original_animal_id'] = frame[identity].astype(str)
    frame['animal_id'] = cohort + ':' + frame.original_animal_id
    frame['cohort'] = cohort
    if frame.groupby('animal_id').Genotype.nunique().gt(1).any():
        raise ValueError(f'An animal has conflicting genotypes in {path}')
    features = list(cfg.model_features)
    frame[features] = frame[features].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    exclusions = cfg.historical_excluded_images if cohort == 'historical' else cfg.excluded_images
    reasons = []
    for _, row in frame.iterrows():
        reason = []
        if row.image_name in exclusions:
            reason.append('configured_image_exclusion')
        if cohort == 'historical' and row.Genotype not in cfg.train_labels:
            reason.append('outside_training_genotypes')
        missing = [f for f in features if pd.isna(row[f])]
        if missing:
            reason.append('missing_or_nonfinite_features:' + ','.join(missing))
        reasons.append(';'.join(reason))
    frame['exclusion_reason'] = reasons
    excluded = frame[frame.exclusion_reason.ne('')].copy()
    included = frame[frame.exclusion_reason.eq('')].drop(columns='exclusion_reason').copy()
    included = included.sort_values(['animal_id', 'image_name']).reset_index(drop=True)
    return included, excluded


def animal_table(frame):
    if frame.groupby('animal_id').Genotype.nunique().gt(1).any():
        raise ValueError('Conflicting animal genotypes')
    return frame[['animal_id', 'Genotype']].drop_duplicates().sort_values('animal_id').reset_index(drop=True)


def animal_holdout(frame, cfg):
    animals = animal_table(frame)
    train, test = train_test_split(animals, test_size=cfg.test_size, stratify=animals.Genotype,
                                   random_state=cfg.random_state)
    train_idx = np.flatnonzero(frame.animal_id.isin(train.animal_id))
    test_idx = np.flatnonzero(frame.animal_id.isin(test.animal_id))
    return train_idx, test_idx


def animal_cv_splits(frame, cfg):
    animals = animal_table(frame)
    folds = min(cfg.decision_tree_cv_folds, int(animals.Genotype.value_counts().min()))
    if folds < 2:
        raise ValueError('At least two historical animals per class are required')
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=cfg.random_state)
    for train, test in cv.split(animals.animal_id, animals.Genotype):
        yield (np.flatnonzero(frame.animal_id.isin(animals.iloc[train].animal_id)),
               np.flatnonzero(frame.animal_id.isin(animals.iloc[test].animal_id)))


def predict_table(model, frame, features, model_name):
    identity = ['image_name', 'animal_id', 'original_animal_id', 'cohort', 'Genotype']
    result = frame[identity].copy()
    result['model'] = model_name
    result['predicted_label'] = model.predict(frame[features])
    probabilities = model.predict_proba(frame[features])
    for index, label in enumerate(model.classes_):
        result[f'leaf_fraction_{label}'] = probabilities[:, index]
    result['tree_leaf'] = model.apply(frame[features])
    result['prediction_scope'] = 'phenotype_prediction'
    return result


def evaluate(predictions, label, classes):
    truth, predicted = predictions.Genotype, predictions.predicted_label
    return dict(evaluation=label, n_plaques=len(predictions), n_animals=predictions.animal_id.nunique(),
                accuracy=accuracy_score(truth, predicted),
                balanced_accuracy=balanced_accuracy_score(truth, predicted),
                macro_precision=precision_score(truth, predicted, labels=classes, average='macro', zero_division=0),
                macro_recall=recall_score(truth, predicted, labels=classes, average='macro', zero_division=0),
                macro_f1=f1_score(truth, predicted, labels=classes, average='macro', zero_division=0),
                animal_mean_plaque_accuracy=predictions.assign(correct=truth.eq(predicted)).groupby('animal_id').correct.mean().mean())


def save_holdout_classification_report(predictions, classes, output):
    """Report the saved holdout labels without retraining or changing the split."""
    report = classification_report(predictions.Genotype, predictions.predicted_label,
                                   labels=classes, output_dict=True, zero_division=0)
    rows = []
    for label in [*classes, 'macro avg', 'weighted avg']:
        row = report[label]
        rows.append(dict(class_or_average=label, precision=row['precision'], recall=row['recall'],
                         f1_score=row['f1-score'], support=int(row['support'])))
    table = pd.DataFrame(rows)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / 'holdout_classification_report.csv', index=False)
    accuracy = float(accuracy_score(predictions.Genotype, predictions.predicted_label))
    text = f'''# 20% animal holdout: precision, recall and F1

Accuracy: **{accuracy:.2%}** ({int(predictions.Genotype.eq(predictions.predicted_label).sum())}/{len(predictions)} plaques).
This is the existing holdout test: {len(predictions)} plaques from {predictions.animal_id.nunique()} animals.
Metrics are plaque-level. Animals were kept together in the train/test split.

| Class or average | Precision | Recall | F1 | Plaques |
|---|---:|---:|---:|---:|
'''
    for row in rows:
        text += f"| {row['class_or_average']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1_score']:.4f} | {row['support']} |\n"
    text += '''
Each class is evaluated as that class versus the other class. Macro averages
weight the two classes equally; weighted averages use their true plaque counts.
Precision is the fraction of predictions for a class that are correct. Recall
is the fraction of that class's true plaques recovered. F1 is their harmonic mean.
The saved predictions and split were not changed to produce this report.

Method: [scikit-learn classification report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html).
'''
    (output / 'holdout_classification_report.md').write_text(text)
    return table


def animal_summary(predictions):
    rows = []
    for animal, frame in predictions.groupby('animal_id', sort=True):
        row = dict(animal_id=animal, original_animal_id=frame.original_animal_id.iloc[0],
                   Genotype=frame.Genotype.iloc[0], n_plaques=len(frame))
        for label in ('APP-WT', 'APP-FIRE'):
            row[f'n_{label}_like'] = int(frame.predicted_label.eq(label).sum())
            row[f'fraction_{label}_like'] = float(frame.predicted_label.eq(label).mean())
            row[f'mean_leaf_fraction_{label}'] = float(frame[f'leaf_fraction_{label}'].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def save_tree(model, features, output, name, title):
    (output / f'{name}_rules.txt').write_text(export_text(model, feature_names=features, decimals=5))
    pd.DataFrame(dict(feature=features, importance=model.feature_importances_)).sort_values(
        'importance', ascending=False).to_csv(output / f'{name}_feature_importance.csv', index=False)
    fig, ax = plt.subplots(figsize=(max(16, model.get_n_leaves()*2.2), max(9, (model.get_depth()+1)*1.7)))
    plot_tree(model, feature_names=features, class_names=list(model.classes_), filled=True,
              rounded=True, precision=3, fontsize=9, ax=ax)
    ax.set_title(title + '\nClass-count order: ' + ', '.join(model.classes_), fontsize=15, pad=18)
    fig.savefig(output / f'{name}.png', dpi=160, bbox_inches='tight')
    fig.savefig(output / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


def save_summary_plot(cv_predictions, transplant_summary, classes, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    matrix = confusion_matrix(cv_predictions.Genotype, cv_predictions.predicted_label, labels=classes)
    axes[0].imshow(matrix, cmap='Blues', vmin=0)
    for (row, col), value in np.ndenumerate(matrix):
        axes[0].text(col, row, str(value), ha='center', va='center', fontsize=18,
                     color='white' if value > matrix.max()*.55 else 'black')
    axes[0].set(xticks=[0, 1], yticks=[0, 1], xticklabels=classes, yticklabels=classes,
                xlabel='Predicted class', ylabel='Recorded genotype', title='Historical validation\nAnimals kept separate across CV folds')
    x = np.arange(len(transplant_summary))
    wt = transplant_summary['fraction_APP-WT_like'].to_numpy() * 100
    axes[1].bar(x, wt, color='#3274a1', label='APP-WT-like')
    axes[1].bar(x, 100-wt, bottom=wt, color='#dd8452', label='APP-FIRE-like')
    axes[1].set(xticks=x, xticklabels=transplant_summary.original_animal_id, ylim=(0, 112),
                ylabel='Plaques classified (%)', title='Transplanted plaque predictions\nTree refitted on all historical controls')
    for i, row in transplant_summary.reset_index(drop=True).iterrows():
        axes[1].text(i, 103, f'n={row.n_plaques}', ha='center', fontsize=10)
    axes[1].legend(loc='upper center', bbox_to_anchor=(.5, -.13), ncol=2)
    for extension in ('png', 'pdf'):
        fig.savefig(output / f'validation_and_transplanted_predictions.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def run_decision_tree(cfg=CFG, output_dir=None):
    historical_path = cfg.previous_output_root / 'morphology_results/combined/decision_tree_features.csv'
    current_path = cfg.feature_table_path
    source_hashes = dict(historical=file_hash(historical_path), current=file_hash(current_path))
    historical, historical_excluded = prepare_table(historical_path, 'historical', cfg)
    current, current_excluded = prepare_table(current_path, 'transplanted', cfg)
    if set(historical.Genotype) != set(cfg.train_labels):
        raise ValueError('Both historical APP-WT and APP-FIRE classes are required')
    for frame, path in [(historical, historical_path), (current, current_path)]:
        check_marker_features(frame, path.parent, cfg)
    transplanted = current[current.Genotype.eq('APP-FIRE+transplant')].copy()
    external_controls = current[current.Genotype.ne('APP-FIRE+transplant')].copy()
    if transplanted.empty:
        raise ValueError('No complete transplanted plaques available')
    features, classes = list(cfg.model_features), list(cfg.train_labels)
    if len(features) != len(set(features)):
        raise ValueError('Duplicate model features')
    train_idx, test_idx = animal_holdout(historical, cfg)
    split_pairs = list(animal_cv_splits(historical, cfg))
    for train, test in [(train_idx, test_idx), *split_pairs]:
        if set(historical.iloc[train].animal_id) & set(historical.iloc[test].animal_id):
            raise AssertionError('Animal leakage in model split')
        if set(historical.iloc[train].Genotype) != set(classes) or set(historical.iloc[test].Genotype) != set(classes):
            raise ValueError('A split is missing a genotype')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = Path(output_dir) if output_dir else cfg.decision_tree_dir / f'run_{stamp}'
    output.mkdir(parents=True, exist_ok=False)
    historical.to_csv(output / 'historical_model_inputs.csv', index=False)
    current.to_csv(output / 'new_cohort_model_inputs.csv', index=False)
    pd.concat([historical_excluded, current_excluded], ignore_index=True).to_csv(output / 'excluded_plaques.csv', index=False)

    split = historical[['image_name', 'animal_id', 'Genotype']].copy()
    split['holdout_split'] = 'train'
    split.loc[test_idx, 'holdout_split'] = 'test'
    evaluation_model = DecisionTreeClassifier(**tree_parameters(cfg)).fit(
        historical.iloc[train_idx][features], historical.iloc[train_idx].Genotype)
    held_out = predict_table(evaluation_model, historical.iloc[test_idx], features, 'animal_holdout_model')
    held_out['prediction_scope'] = 'unseen_historical_animals'
    held_out.to_csv(output / 'held_out_predictions.csv', index=False)
    save_holdout_classification_report(held_out, classes, output)
    metrics = [evaluate(held_out, 'animal_holdout_test', classes)]
    cv_predictions, cv_metrics = [], []
    for fold, (train, test) in enumerate(split_pairs, 1):
        model = DecisionTreeClassifier(**tree_parameters(cfg)).fit(historical.iloc[train][features], historical.iloc[train].Genotype)
        predicted = predict_table(model, historical.iloc[test], features, f'cv_fold_{fold}')
        predicted['fold'] = fold
        predicted['prediction_scope'] = 'unseen_historical_animals'
        split.loc[test, 'cv_test_fold'] = fold
        cv_predictions.append(predicted)
        cv_metrics.append(dict(fold=fold, **evaluate(predicted, f'cv_fold_{fold}', classes)))
    cv_predictions = pd.concat(cv_predictions).sort_values(['animal_id', 'image_name'])
    if cv_predictions.image_name.duplicated().any() or len(cv_predictions) != len(historical):
        raise AssertionError('Cross-validation must predict every historical plaque exactly once')
    cv_predictions.to_csv(output / 'grouped_cv_predictions.csv', index=False)
    pd.DataFrame(cv_metrics).to_csv(output / 'grouped_cv_fold_metrics.csv', index=False)
    split.to_csv(output / 'animal_split_membership.csv', index=False)
    metrics.append(evaluate(cv_predictions, 'animal_grouped_cv_out_of_fold', classes))

    final_model = DecisionTreeClassifier(**tree_parameters(cfg)).fit(historical[features], historical.Genotype)
    historical_predictions = predict_table(final_model, historical, features, 'all_historical_model')
    historical_predictions['prediction_scope'] = 'training_resubstitution_not_independent'
    historical_predictions.to_csv(output / 'historical_self_predictions.csv', index=False)
    metrics.append(evaluate(historical_predictions, 'training_resubstitution_not_independent', classes))
    pd.DataFrame(metrics).to_csv(output / 'evaluation_metrics.csv', index=False)
    for name, predictions in [('held_out', held_out), ('grouped_cv', cv_predictions)]:
        pd.DataFrame(confusion_matrix(predictions.Genotype, predictions.predicted_label, labels=classes),
                     index=pd.Index(classes, name='true_genotype'), columns=classes).to_csv(output / f'{name}_confusion_matrix.csv')

    prediction = predict_table(final_model, transplanted, features, 'all_historical_model')
    prediction.to_csv(output / 'transplanted_plaque_predictions.csv', index=False)
    summary = animal_summary(prediction)
    summary.to_csv(output / 'transplanted_animal_summary.csv', index=False)
    group_summary = dict(n_animals=len(summary), n_plaques=len(prediction),
                         animal_mean_fraction_APP_WT_like=float(summary['fraction_APP-WT_like'].mean()),
                         pooled_plaque_fraction_APP_WT_like=float(prediction.predicted_label.eq('APP-WT').mean()))
    pd.DataFrame([group_summary]).to_csv(output / 'transplanted_group_summary.csv', index=False)
    predict_table(evaluation_model, transplanted, features, 'animal_holdout_model').to_csv(
        output / 'transplanted_predictions_evaluation_model.csv', index=False)
    if not external_controls.empty:
        predict_table(final_model, external_controls, features, 'all_historical_model').to_csv(
            output / 'new_cohort_controls_predictions.csv', index=False)

    for name, model in [('evaluation_tree', evaluation_model), ('final_tree', final_model)]:
        joblib.dump(dict(model=model, features=features, classes=list(model.classes_),
                         tree_params=tree_parameters(cfg), training_cohort='historical_only'), output / f'{name}.joblib')
        save_tree(model, features, output, name,
                  'Historical APP-WT versus APP-FIRE: ' + ('animal holdout training subset' if name == 'evaluation_tree' else 'all historical controls'))
    save_summary_plot(cv_predictions, summary, classes, output)
    if source_hashes != dict(historical=file_hash(historical_path), current=file_hash(current_path)):
        raise RuntimeError('An input feature table changed during analysis; rerun using stable inputs.')
    manifest = dict(created_utc=stamp, historical_source=str(historical_path), current_source=str(current_path),
                    historical_sha256=source_hashes['historical'], current_sha256=source_hashes['current'],
                    features=features, tree_parameters=tree_parameters(cfg), sklearn_version=sklearn.__version__,
                    requested_test_fraction_animals=cfg.test_size, training_animals=sorted(historical.iloc[train_idx].animal_id.unique()),
                    held_out_animals=sorted(historical.iloc[test_idx].animal_id.unique()), cv_folds=len(split_pairs),
                    historical_animal_identity='animal_id if available; otherwise recorded sample_id',
                    hyperparameter_search=False, transplant_data_used_for_training=False,
                    marker_and_volume_checks='passed', group_summary=group_summary, evaluations=metrics)
    (output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False))
    report = f'''# Historical decision tree and transplanted predictions

Historical training cohort: {len(historical)} plaques from {historical.animal_id.nunique()} animals.
Transplanted cohort: {len(transplanted)} plaques from {transplanted.animal_id.nunique()} animals.
Excluded rows: {len(historical_excluded)} historical and {len(current_excluded)} new-cohort rows; reasons are saved.

| Evaluation | Plaques | Animals | Accuracy | Balanced accuracy |
|---|---:|---:|---:|---:|
'''
    for row in metrics:
        report += f"| {row['evaluation']} | {row['n_plaques']} | {row['n_animals']} | {row['accuracy']:.1%} | {row['balanced_accuracy']:.1%} |\n"
    report += f'''
The final tree classified {prediction.predicted_label.eq('APP-WT').sum()} of {len(prediction)} transplanted plaques as APP-WT-like.
The mean APP-WT-like fraction across animals was {group_summary['animal_mean_fraction_APP_WT_like']:.1%}.

## Method and interpretation

The existing {len(features)} features and fixed tree settings were retained: {tree_parameters(cfg)}.
No feature selection, parameter search, rescaling, imputation, or transplant-based tuning was performed.
The 20% holdout split is stratified over animals, then expanded to plaques. Actual counts are recorded above.
The separate {len(split_pairs)}-fold validation also splits unique animals, stratified by genotype, then expands to plaques.
Each historical plaque receives one out-of-fold prediction. These estimates use overlapping historical data and are not independent experiments.
Historical `sample_id` is used as the animal unit when PF animal IDs are absent, as in the supplied metadata.
Training is plaque-level with equal plaque weights. Animal-average plaque accuracy is also reported separately.

After validation, a tree with the same fixed settings is refitted to all historical controls and applied to transplanted plaques.
Historical self-predictions are training resubstitution and are not an independent performance estimate.
The frozen holdout-trained tree's transplanted predictions are also saved for comparison with the exact evaluated tree.
New-cohort control animals are predicted separately and never pooled into historical training.

APP-WT-like and APP-FIRE-like describe the model's morphological classifications, not changed genotypes.
Transplanted plaques have no verified two-class phenotype labels, so transplant accuracy is not calculated.
`leaf_fraction_*` values are training-class fractions in the reached leaf, not calibrated confidence probabilities.
The historical and transplanted acquisition cohorts differ; batch effects or residual segmentation differences can affect transfer.
Do not interpret a WT-like fraction alone as evidence that transplantation restored normal biology.

## Outputs and rerun

Start with `evaluation_metrics.csv`, `holdout_classification_report.csv`, `transplanted_animal_summary.csv`, and `validation_and_transplanted_predictions.png`.
Individual predictions, split membership, source checksums, exclusion records, saved models and tree diagrams are included.
Rerun from the installed pipeline directory with `bash run.sh decision_tree`. Each run gets a new dated folder.

Methods reference: [scikit-learn grouped cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data)
and [decision-tree class fractions](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html).
'''
    (output / 'README.md').write_text(report)
    print(pd.DataFrame(metrics).to_string(index=False))
    print(summary.to_string(index=False))
    print(f'Decision-tree results: {output}')
    return output
