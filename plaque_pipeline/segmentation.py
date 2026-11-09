"""Apply original same-marker classifiers with explicit input/output provenance."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import h5py
import pandas as pd
import tifffile

from .io_utils import strip_tiff_suffix
from .probabilities import probability_to_zyx


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def build_command(cfg, channel, input_path, output_path):
    return [str(cfg.ilastik_executable), '--headless', '--readonly',
            f'--project={cfg.source_project(channel)}', '--export_source=Probabilities',
            '--output_format=compressed hdf5', '--output_axis_order=zyxc', '--export_dtype=float32',
            f'--output_filename_format={output_path}', str(input_path)]


def validate_probability_file(path, input_path):
    with h5py.File(path, 'r') as h:
        data = h['exported_data']
        tags = data.attrs.get('axistags')
        axes = ''.join(a['key'] for a in json.loads(tags)['axes']) if tags else 'zyxc'
        foreground = probability_to_zyx(data[()], 0, axes)
    with tifffile.TiffFile(input_path) as t:
        if foreground.shape != t.series[0].shape:
            raise ValueError(f'Probability/input shape mismatch: {path}')


def can_resume(output_path, provenance):
    """Never accept an untracked or differently generated probability export."""
    if not output_path.exists():
        return False
    record_path = output_path.with_suffix('.provenance.json')
    if not record_path.exists():
        raise ValueError(f'Untracked existing probabilities: {output_path}. Use a fresh output root.')
    record = json.loads(record_path.read_text())
    if any(record.get(k) != v for k, v in provenance.items()):
        raise ValueError(f'Input/model provenance changed for {output_path}. Use a fresh output root.')
    if record.get('output_sha256') != sha256(output_path):
        raise ValueError(f'Probability checksum changed: {output_path}')
    return True


def segment_original_projects(cfg):
    if not cfg.ilastik_executable.is_file():
        raise FileNotFoundError(cfg.ilastik_executable)
    log_path = cfg.output_root / 'preprocessing_log.csv'
    log = pd.read_csv(log_path)
    if log.empty or not (log.mx04_channel_index.eq(cfg.mx04_channel_index).all()
                         and log.oc_channel_index.eq(cfg.oc_channel_index).all()):
        raise ValueError('Preprocessing channel routing does not match this configuration. Rerun preprocessing.')
    audit = cfg.output_root / 'input_audit' / 'segmentation'
    audit.mkdir(parents=True, exist_ok=True)
    jobs = []
    for channel in (cfg.mx04_tag, cfg.oc_tag):
        project = cfg.source_project(channel)
        with h5py.File(project, 'r') as h:
            labels = [x.decode() for x in h['PixelClassification/LabelNames'][()]]
            if labels != ['Plaque', 'Background'] or 'PixelClassification/ClassifierForests' not in h:
                raise ValueError(f'Unexpected classifier/classes: {project}')
        inputs = sorted((cfg.preprocessed_dir / channel).glob('*.ome.tif*'))
        logged_inputs = {Path(p).resolve() for p in log[f'{channel.lower()}_path']}
        if not inputs or len(logged_inputs) != len(log) or {p.resolve() for p in inputs} != logged_inputs:
            raise ValueError(f'{channel} inputs differ from preprocessing log. Rerun preprocessing.')
        model_hash = sha256(project)
        for input_path in inputs:
            output_path = cfg.probability_dir / channel / f'{strip_tiff_suffix(input_path)}_Probabilities.h5'
            provenance = dict(marker=channel, raw_channel_index=getattr(cfg, f'{channel.lower()}_channel_index'),
                              source_project=str(project), project_sha256=model_hash,
                              input_path=str(input_path), input_sha256=sha256(input_path))
            jobs.append((channel, input_path, output_path, provenance))
    environment = dict(os.environ, LAZYFLOW_THREADS=str(cfg.ilastik_threads),
                       LAZYFLOW_TOTAL_RAM_MB=str(cfg.ilastik_ram_mb))
    for index, (channel, input_path, output_path, provenance) in enumerate(jobs, 1):
        from .conversion import check_free_space
        check_free_space(cfg.output_root)
        print(f'[{index}/{len(jobs)}] {channel}: {input_path.name}', flush=True)
        if can_resume(output_path, provenance):
            print('  Verified completed export; skipping.', flush=True)
            continue
        temporary = output_path.parent / '_incomplete' / output_path.name
        temporary.parent.mkdir(parents=True, exist_ok=True)
        command = build_command(cfg, channel, input_path, temporary)
        run_log = audit / f'{strip_tiff_suffix(input_path)}.log'
        with run_log.open('w') as stream:
            result = subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'ilastik failed ({result.returncode}); see {run_log}')
        validate_probability_file(temporary, input_path)
        output_hash = sha256(temporary)
        # A completed file is published only after successful prediction/validation.
        record_path = output_path.with_suffix('.provenance.json')
        record_temp = record_path.with_suffix('.json.tmp')
        record_temp.write_text(json.dumps(dict(provenance, output_sha256=output_hash, command=command), indent=2))
        record_temp.replace(record_path)
        temporary.replace(output_path)
        print(f'  Saved validated probabilities: {output_path.name}', flush=True)
    print('Both marker batches are complete. Next: bash run.sh post_ilastik', flush=True)
