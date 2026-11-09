#!/usr/bin/env python3
"""Batch genotype x age analysis of wide plaque tables.

Install: python3 -m pip install numpy pandas scipy patsy statsmodels
Run: python3 nested_two_way_anova.py
Optional: --exclude-simulated --log1p --output-dir PATH --folders PATH PATH

Each column is ONE independent animal, each nonempty cell is a plaque.
Primary model: value ~ genotype * age + (1 | animal), REML, sum contrasts.
Fixed-effect tests are asymptotic Wald chi-square tests, NOT finite-sample
nested-ANOVA F tests or Satterthwaite/Kenward-Roger tests.
A separate Type III F ANOVA on animal means is an equal-animal sensitivity
analysis. Assumptions: independent animals, Gaussian errors/random intercepts,
common residual variance, no additional image/region hierarchy.
Simulated animals are hypothetical, not independent experimental evidence.
"""
from pathlib import Path
import argparse
import csv
import warnings
import numpy as np
import pandas as pd
from scipy.stats import chi2
from patsy import dmatrix
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests

BASE = Path('/Users/katherineridley/Projects/PlaqueStack/ilastik_protocol/young_plaque_simulations/PF1129_SIMULATION_seed1129_20260929/HYPOTHETICAL_results')
FOLDERS = [BASE / 'MX04/individual_plaque_values', BASE / 'OC_halo_and_ratios/individual_plaque_values']
FORMULA = 'value ~ C(genotype, Sum) * C(age, Sum)'
TERMS = {'C(genotype, Sum)': 'genotype', 'C(age, Sum)': 'age',
         'C(genotype, Sum):C(age, Sum)': 'genotype:age'}

def read_table(path, exclude_simulated=False, log1p=False):
    # Preserve empty tab fields; never split on generic whitespace.
    with path.open(encoding='utf-8-sig', newline='') as f:
        sample = f.read(8192)
        f.seek(0)
        delimiter = '\t' if '\t' in sample.splitlines()[0] else ','
        rows = list(csv.reader(f, delimiter=delimiter))
    if not rows:
        raise ValueError('Empty file')
    headers = [h.strip() for h in rows[0]]
    if len(headers) != 20 or len(set(headers)) != 20:
        raise ValueError('Expected exactly 20 distinct animal headers in the specified order')
    mapping = []
    for i, animal in enumerate(headers):
        prefix = animal.split(':', 1)[0].lower()
        expected = 'historical' if i < 14 else ('young' if i < 19 else 'simulated')
        if prefix != expected:
            raise ValueError(f'Column {i+1}: expected {expected}: header, got {animal!r}')
        mapping.append(dict(column=i+1, animal=animal,
                            genotype='APP-WT' if i < 6 or 14 <= i < 17 else 'APP-FIRE',
                            age='historical' if i < 14 else '6m',
                            simulated=(prefix == 'simulated'),
                            included=not (exclude_simulated and prefix == 'simulated')))
    records = []
    for line, row in enumerate(rows[1:], 2):
        if len(row) > 20:
            raise ValueError(f'Line {line}: more than 20 fields')
        row += [''] * (20-len(row))
        for m, raw in zip(mapping, row):
            raw = raw.strip()
            if not m['included'] or raw.lower() in ('', 'nan', 'na', 'n/a'):
                continue
            try:
                value = float(raw)
            except ValueError:
                raise ValueError(f'Line {line}, column {m["column"]}: invalid number {raw!r}')
            if not np.isfinite(value):
                raise ValueError(f'Line {line}: infinite value')
            if log1p and value < 0:
                raise ValueError('log1p requested but negative values exist')
            records.append({**m, 'plaque_row': line-1, 'raw_value': value,
                            'value': np.log1p(value) if log1p else value})
    data = pd.DataFrame(records)
    if data.empty:
        raise ValueError('No usable values')
    for factor, levels in [('genotype', ['APP-WT', 'APP-FIRE']), ('age', ['historical', '6m'])]:
        data[factor] = pd.Categorical(data[factor], categories=levels)
    counts = data.groupby(['genotype', 'age'], observed=True)['animal'].nunique()
    if len(counts) != 4 or counts.min() < 2:
        raise ValueError('Every genotype-age cell must contain at least two animals with data')
    return data, pd.DataFrame(mapping)

def mixed_tests(data):
    messages = []
    fit = None
    for method in ('lbfgs', 'bfgs', 'powell'):
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                candidate = smf.mixedlm(FORMULA, data, groups=data['animal']).fit(
                    reml=True, method=method, maxiter=2000, disp=False)
            messages.extend(f'{method}: {w.message}' for w in caught)
            if candidate.converged and np.all(np.isfinite(candidate.cov_params())):
                fit = candidate
                break
            messages.append(f'{method}: nonconverged or nonfinite covariance')
        except Exception as exc:
            messages.append(f'{method}: {exc}')
    if fit is None:
        raise RuntimeError('Mixed model failed: ' + '; '.join(messages))
    k = len(fit.fe_params)
    beta = fit.fe_params.to_numpy()
    cov = fit.cov_params().iloc[:k, :k].to_numpy()
    results = []
    for term, label in TERMS.items():
        sl = dmatrix(FORMULA.split('~', 1)[1], data).design_info.term_name_slices[term]
        indices = np.arange(k)[sl]
        subcov = cov[np.ix_(indices, indices)]
        if np.min(np.linalg.eigvalsh(subcov)) <= 0:
            raise RuntimeError('Fixed-effect covariance is not positive definite')
        stat = float(beta[indices] @ np.linalg.solve(subcov, beta[indices]))
        results.append(dict(analysis='mixed_model_Wald', effect=label,
                            statistic=stat, df_num=len(indices), df_den=np.nan,
                            p_value=chi2.sf(stat, len(indices))))
    variance = float(fit.cov_re.iloc[0, 0])
    if variance < 1e-6 * max(float(fit.scale), 1e-12):
        messages.append('Animal variance near zero: random effect on boundary; interpret cautiously')
    return results, fit, messages

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--folders', nargs='+', type=Path, default=FOLDERS)
    parser.add_argument('--output-dir', type=Path, default=BASE / 'nested_two_way_anova_results')
    parser.add_argument('--exclude-simulated', action='store_true')
    parser.add_argument('--log1p', action='store_true', help='Analyze log(1+value), applied before animal averaging')
    args = parser.parse_args()
    for folder in args.folders:
        if not folder.is_dir():
            parser.error(f'Input folder does not exist: {folder}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_tests, statuses = [], []
    nfiles = 0
    for folder_idx, folder in enumerate(args.folders, 1):
        files = sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in ('.csv', '.tsv', '.txt') and not p.name.startswith('.'))
        for path in files:
            nfiles += 1
            feature = f'{folder_idx}/{path.relative_to(folder).as_posix()}'
            destination = args.output_dir / f'folder_{folder_idx}' / path.relative_to(folder).parent / path.name
            destination.mkdir(parents=True, exist_ok=True)
            status = dict(feature=feature, source=str(path), simulated_included=not args.exclude_simulated, transform='log1p' if args.log1p else 'raw')
            try:
                data, mapping = read_table(path, args.exclude_simulated, args.log1p)
                data.to_csv(destination / 'plaque_data_long.csv', index=False)
                mapping.to_csv(destination / 'column_mapping.csv', index=False)
                animal = data.groupby(['animal', 'genotype', 'age'], observed=True).agg(
                    value=('value', 'mean'), raw_mean=('raw_value', 'mean'), n_plaques=('value', 'size'),
                    plaque_sd=('value', 'std'), simulated=('simulated', 'first')).reset_index()
                animal.to_csv(destination / 'animal_summary.csv', index=False)
                status.update(n_animals=len(animal), n_plaques=len(data), n_simulated_animals=int(animal.simulated.sum()))
                # Equal weighting of animals. This is separate from the plaque-level model.
                ols = smf.ols(FORMULA, animal).fit()
                table = anova_lm(ols, typ=3)
                table.to_csv(destination / 'animal_mean_anova.csv')
                for term, label in TERMS.items():
                    all_tests.append(dict(feature=feature, analysis='animal_mean_ANOVA', effect=label,
                                          statistic=table.loc[term, 'F'], df_num=table.loc[term, 'df'],
                                          df_den=ols.df_resid, p_value=table.loc[term, 'PR(>F)']))
                try:
                    tests, fit, messages = mixed_tests(data)
                    all_tests.extend(dict(feature=feature, **r) for r in tests)
                    (destination / 'mixed_model_summary.txt').write_text(str(fit.summary()) + '\n\n' + '\n'.join(messages))
                    status.update(status='ok', animal_variance=float(fit.cov_re.iloc[0, 0]), residual_variance=float(fit.scale), warnings='; '.join(messages))
                except Exception as exc:
                    status.update(status='mixed_model_failed_animal_ANOVA_available', error=str(exc))
                print(f'{feature}: {status["status"]}')
            except Exception as exc:
                status.update(status='failed', error=str(exc))
                print(f'{feature}: FAILED: {exc}')
            statuses.append(status)
    pd.DataFrame(statuses).to_csv(args.output_dir / 'file_status.csv', index=False)
    if all_tests:
        results = pd.DataFrame(all_tests)
        results['q_BH_across_features'] = np.nan
        # Each effect and analysis forms one correction family across BOTH folders.
        for _, indices in results.groupby(['analysis', 'effect']).groups.items():
            valid = results.loc[indices, 'p_value'].dropna()
            if len(valid):
                results.loc[valid.index, 'q_BH_across_features'] = multipletests(valid, method='fdr_bh')[1]
        results.to_csv(args.output_dir / 'all_tests.csv', index=False)
    (args.output_dir / 'analysis_notes.txt').write_text(
        'Plaques nested in independent animals; genotype and age are crossed fixed effects.\n'
        'Historical age unspecified. Assignment uses original 20-column positions.\n'
        'Mixed model uses REML and approximate Wald chi-square tests, not exact nested F tests.\n'
        'Sum contrasts: main effects average equally over levels of the other factor.\n'
        'Animal-mean Type III ANOVA is a separate sensitivity analysis.\n'
        'BH correction across features separately for each effect and analysis; failed tests omitted.\n'
        f'Simulated excluded: {args.exclude_simulated}; log1p: {args.log1p}.\n'
        'Simulation-inclusive inference is hypothetical. Review warnings and residual assumptions.\n')
    print(f'Processed {nfiles} files. Results: {args.output_dir}')
    if not nfiles or not all_tests:
        raise SystemExit('No successful tests; see file_status.csv')

if __name__ == '__main__':
    main()
