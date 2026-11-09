#!/usr/bin/env python3
from __future__ import annotations

import argparse

from plaque_pipeline.config import CFG
from plaque_pipeline.decision_tree import train_decision_tree
from plaque_pipeline.io_utils import ensure_dir
from plaque_pipeline.morphology import process_morphology
from plaque_pipeline.preprocess import preprocess_ometiff_folder
from plaque_pipeline.probabilities import h5_to_all_masks


def print_manual_ilastik_instructions() -> None:
    ensure_dir(CFG.probability_dir / CFG.mx04_tag)
    ensure_dir(CFG.probability_dir / CFG.oc_tag)
    print(f"""
Manual ilastik checkpoint
-------------------------
MX04 training: {CFG.training_subset_dir / CFG.mx04_tag}
MX04 batch:    {CFG.mx04_preprocessed_dir}
MX04 H5 out:   {CFG.probability_dir / CFG.mx04_tag}

OC training:   {CFG.training_subset_dir / CFG.oc_tag}
OC batch:      {CFG.oc_preprocessed_dir}
OC H5 out:     {CFG.probability_dir / CFG.oc_tag}

Then run:
  python main.py h5_to_masks
  python main.py morphology
  python main.py decision_tree

Or after ilastik:
  python main.py post_ilastik
""")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plaque OME-TIFF -> ilastik -> morphology -> decision tree")
    parser.add_argument(
        "stage",
        choices=[
            "preprocess", "ilastik_instructions", "h5_to_masks", "morphology",
            "decision_tree", "post_ilastik", "analysis"
        ],
    )
    args = parser.parse_args()
    ensure_dir(CFG.output_root)

    if args.stage == "preprocess":
        preprocess_ometiff_folder(CFG)
        print_manual_ilastik_instructions()
    elif args.stage == "ilastik_instructions":
        print_manual_ilastik_instructions()
    elif args.stage == "h5_to_masks":
        h5_to_all_masks(CFG)
    elif args.stage == "morphology":
        process_morphology(CFG)
    elif args.stage == "decision_tree":
        train_decision_tree(CFG)
    elif args.stage == "post_ilastik":
        h5_to_all_masks(CFG)
        process_morphology(CFG)
        train_decision_tree(CFG)
    elif args.stage == "analysis":
        process_morphology(CFG)
        train_decision_tree(CFG)


if __name__ == "__main__":
    main()
