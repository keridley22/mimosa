"""Resolve slide/section keys to biological animals without silently guessing."""
from __future__ import annotations

import re
import pandas as pd


def sample_key(value: str) -> str | None:
    keys = set(re.findall(r"(?:^|_)([A-Z]_\d+)(?=_|$)", str(value)))
    keys.update(f"{a}_{b}" for a, b in re.findall(r"slide([A-Z])(\d+)(?=_|$)", str(value)))
    if len(keys) > 1:
        raise ValueError(f"Ambiguous sample identifiers in {value!r}: {sorted(keys)}")
    return next(iter(keys), None)


def load_metadata(cfg) -> dict[str, dict[str, str]]:
    df = pd.read_csv(cfg.genotype_csv, dtype=str).fillna("")
    required = {"filename", "animal_id", "Genotype"}
    if not required.issubset(df.columns):
        raise ValueError(f"Metadata requires {sorted(required)}")
    lookup = {}
    animal_groups = {}
    for row in df.to_dict("records"):
        key = sample_key(row["filename"])
        animal = row["animal_id"].strip()
        group = cfg.genotype_map.get(row["Genotype"].strip(), row["Genotype"].strip())
        group = cfg.metadata_genotype_map.get(group, group)
        if not key or not animal or not group:
            raise ValueError(f"Incomplete metadata row: {row}")
        record = {"sample_id": key, "animal_id": animal, "Genotype": group}
        if key in lookup and lookup[key] != record:
            raise ValueError(f"Conflicting metadata for {key}")
        if animal in animal_groups and animal_groups[animal] != group:
            raise ValueError(f"Animal {animal} belongs to multiple genotypes")
        lookup[key] = record
        animal_groups[animal] = group
    return lookup


def resolve_metadata(image_name: str, lookup) -> dict[str, str]:
    key = sample_key(image_name)
    if key not in lookup:
        raise ValueError(f"No metadata match for {image_name!r} (key={key!r})")
    return lookup[key].copy()
