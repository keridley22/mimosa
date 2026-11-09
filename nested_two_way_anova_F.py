#!/usr/bin/env python3
"""Classical-style nested two-way ANOVA F tests for wide plaque tables.

Model: genotype + age + genotype:age + animal(genotype:age) + plaque error.
Animals are independent; each column is an animal; each cell is a plaque.
Fixed effects are tested against MS animal(genotype:age), NOT plaque error.
With equal plaque counts per animal, these are the usual nested factorial
F tests (under Gaussian, common-variance assumptions). With unequal plaque
counts the pooled animal MS and nominal F distribution are an APPROXIMATION:
no Satterthwaite/Kenward-Roger or expected-mean-square adjustment is performed.
Animals with more plaques have greater weight. Unequal animal numbers are
handled using equal-cell marginal contrasts (Type III-style hypotheses).
Do not interpret this as an exact unbalanced mixed-model ANOVA.

Install: python3 -m pip install numpy pandas scipy
Run: python3 nested_two_way_anova_F.py
Optional: --exclude-simulated --log1p --folders PATH PATH --output-dir PATH
Historical age is unspecified. SIMULATED:PF1129 is included by default.
Simulation-inclusive tests describe hypothetical data, not experimental evidence.
"""
from pathlib import Path
import argparse
import csv
import numpy as np
import pandas as pd
from scipy.stats import f as f_distribution

BASE = Path('/Users/katherineridley/Projects/PlaqueStack/ilastik_protocol/young_plaque_simulations/PF1129_SIMULATION_seed1129_20260929/HYPOTHETICAL_results')
FOLDERS = [BASE / 'MX04/individual_plaque_values', BASE / 'OC_halo_and_ratios/individual_plaque_values']
CELL_ORDER = [('APP-WT', 'historical'), ('APP-FIRE', 'historical'),
              ('APP-WT', '6m'), ('APP-FIRE', '6m')]
# Main effects average equally over the two levels of the other factor.
CONTRASTS = {'genotype': np.array([-0.5, 0.5, -0.5, 0.5]),
             'age': np.array([-0.5, -0.5, 0.5, 0.5]),
             'genotype:age': np.array([-1., 1., 1., -1.])}

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

def nested_anova(data):
    animal = data.groupby(['animal', 'genotype', 'age'], observed=True).agg(
        mean=('value', 'mean'), raw_mean=('raw_value', 'mean'),
        n_plaques=('value', 'size'), simulated=('simulated', 'first')).reset_index()
    cell = data.groupby(['genotype', 'age'], observed=True).agg(
        mean=('value', 'mean'), n_plaques=('value', 'size'),
        n_animals=('animal', 'nunique')).reindex(pd.MultiIndex.from_tuples(
            CELL_ORDER, names=['genotype', 'age']))
    n_animals, n_plaques = len(animal), len(data)
    df_animal, df_error = n_animals - 4, n_plaques - n_animals
    if df_animal <= 0 or df_error <= 0:
        raise ValueError('Need animal replication and repeated plaque measurements')
    # Weighted deviations of each animal mean from its genotype-age cell mean.
    expected = np.array([cell.loc[(g, a), 'mean'] for g, a in zip(animal.genotype, animal.age)])
    ss_animal = float(np.sum(animal.n_plaques * (animal['mean'] - expected)**2))
    lookup = animal.set_index('animal')['mean']
    ss_error = float(np.sum((data.value - data.animal.map(lookup))**2))
    ms_animal, ms_error = ss_animal / df_animal, ss_error / df_error
    if ms_animal <= 0:
        raise ValueError('Zero between-animal error: fixed-effect F tests undefined')
    mu = cell['mean'].to_numpy(dtype=float)
    totals = cell.n_plaques.to_numpy(dtype=float)
    rows = []
    for effect, contrast in CONTRASTS.items():
        estimate = float(contrast @ mu)
        # Equivalent to a 1-df partial SS for this cell-means contrast.
        ss = estimate**2 / float(np.sum(contrast**2 / totals))
        statistic = ss / ms_animal
        rows.append(dict(effect=effect, SS=ss, df_num=1, MS=ss,
                         denominator='animal(genotype:age)', df_den=df_animal,
                         F=statistic, p_value=f_distribution.sf(statistic, 1, df_animal),
                         contrast_estimate=estimate))
    rows.append(dict(effect='animal(genotype:age)', SS=ss_animal,
                     df_num=df_animal, MS=ms_animal, denominator='plaque_residual',
                     df_den=df_error, F=ms_animal/ms_error if ms_error > 0 else np.nan,
                     p_value=f_distribution.sf(ms_animal/ms_error, df_animal, df_error) if ms_error > 0 else np.nan,
                     contrast_estimate=np.nan))
    rows.append(dict(effect='plaque_residual', SS=ss_error, df_num=df_error,
                     MS=ms_error, denominator='', df_den=np.nan, F=np.nan,
                     p_value=np.nan, contrast_estimate=np.nan))
    # Partial fixed-effect SS are not additive when the design is unbalanced.
    rows.append(dict(effect='total_corrected', SS=float(np.sum((data.value-data.value.mean())**2)),
                     df_num=n_plaques-1, MS=np.nan, denominator='', df_den=np.nan,
                     F=np.nan, p_value=np.nan, contrast_estimate=np.nan))
    same_counts = animal.n_plaques.nunique() == 1
    return pd.DataFrame(rows), animal, cell.reset_index(), same_counts


def bh_adjust(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adjusted = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p)+1))[::-1])[::-1]
    result = np.empty(len(p))
    result[order] = np.minimum(adjusted, 1.)
    return result


NOTES = """Nested factorial ANOVA: genotype and age are fixed crossed factors;
animal is a random factor nested within genotype x age; plaques are subsamples.
F_genotype, F_age, F_interaction = partial SS_effect / MS_animal, each with
1 numerator df and (number of animals - 4) denominator df.
SS_animal = sum n_animal * (animal_mean - plaque_weighted_cell_mean)^2.
SS_plaque_error = sum (plaque_value - its_animal_mean)^2.
For a contrast c over the four cell means, SS = (c' mean)^2 / sum(c^2 / N_cell).
This uses plaque-weighted cell means, and gives higher weight to animals with
more plaques. Main-effect contrasts average equally across the other factor.
Interaction estimate = (FIRE-WT)_historical - (FIRE-WT)_6m.
Age estimate = average(6m - historical). Genotype estimate = average(FIRE-WT).
With common plaque counts per animal, the fixed-effect F tests reduce to
standard nested factorial tests. With unequal plaque counts, applying one
pooled animal MS is a heuristic approximation, not an exact EMS-adjusted test.
The animal-vs-plaque F test is also approximate for unequal plaque counts.
Partial SS of fixed effects do NOT necessarily sum to between-cell SS.
No image/region nesting is available in these wide tables.
BH q-values are computed separately for each of the three fixed effects
across both input folders. Failed files are excluded from correction families.
Simulated animals are hypothetical. Gaussian/common-variance assumptions apply.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--folders', nargs='+', type=Path, default=FOLDERS)
    parser.add_argument('--output-dir', type=Path, default=BASE / 'nested_two_way_ANOVA_F_results')
    parser.add_argument('--exclude-simulated', action='store_true')
    parser.add_argument('--log1p', action='store_true')
    args = parser.parse_args()
    for folder in args.folders:
        if not folder.is_dir():
            parser.error(f'Input folder does not exist: {folder}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results, statuses = [], []
    for index, folder in enumerate(args.folders, 1):
        files = sorted(p for p in folder.rglob('*') if p.is_file()
                       and p.suffix.lower() in ('.csv', '.tsv', '.txt')
                       and not p.name.startswith('.')
                       and args.output_dir.resolve() not in p.resolve().parents)
        for path in files:
            feature = f'{index}/{path.relative_to(folder).as_posix()}'
            status = dict(feature=feature, source=str(path))
            try:
                data, mapping = read_table(path, args.exclude_simulated, args.log1p)
                table, animals, cells, equal_counts = nested_anova(data)
                out = args.output_dir / f'folder_{index}' / path.relative_to(folder).parent / path.name
                out.mkdir(parents=True, exist_ok=True)
                table.to_csv(out / 'nested_ANOVA.csv', index=False)
                animals.to_csv(out / 'animal_summary.csv', index=False)
                cells.to_csv(out / 'group_summary.csv', index=False)
                mapping.to_csv(out / 'column_mapping.csv', index=False)
                data.to_csv(out / 'plaque_data_long.csv', index=False)
                fixed = table.iloc[:3].copy()
                fixed.insert(0, 'feature', feature)
                fixed['unequal_plaque_count_approximation'] = not equal_counts
                fixed['n_animals'] = len(animals)
                fixed['n_plaques'] = len(data)
                fixed['n_simulated_animals'] = int(animals.simulated.sum())
                fixed['transform'] = 'log1p' if args.log1p else 'raw'
                results.append(fixed)
                status.update(status='ok', n_animals=len(animals), n_plaques=len(data),
                              min_plaques_per_animal=int(animals.n_plaques.min()),
                              max_plaques_per_animal=int(animals.n_plaques.max()),
                              unequal_plaque_count_approximation=not equal_counts)
                print(f'{feature}: done' + (' (unequal-count approximation)' if not equal_counts else ''))
            except Exception as exc:
                status.update(status='failed', error=str(exc))
                print(f'{feature}: FAILED: {exc}')
            statuses.append(status)
    pd.DataFrame(statuses).to_csv(args.output_dir / 'file_status.csv', index=False)
    (args.output_dir / 'analysis_notes.txt').write_text(NOTES +
        f'\nExclude simulated: {args.exclude_simulated}; log1p: {args.log1p}.\n')
    if not results:
        raise SystemExit('No successful analyses. See file_status.csv and check the input folders.')
    combined = pd.concat(results, ignore_index=True)
    combined['q_BH_across_features'] = np.nan
    for _, indices in combined.groupby('effect').groups.items():
        valid = combined.loc[indices, 'p_value'].dropna()
        if len(valid):
            combined.loc[valid.index, 'q_BH_across_features'] = bh_adjust(valid.to_numpy())
    combined.to_csv(args.output_dir / 'all_fixed_effect_tests.csv', index=False)
    print(f'Results saved to: {args.output_dir}')


if __name__ == '__main__':
    main()
