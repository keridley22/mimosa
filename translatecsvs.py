"""Standalone compatibility entry point; uses the transplanted configuration."""
from plaque_pipeline.config import CFG
from plaque_pipeline.stats import run_default_ttests

if __name__ == "__main__":
    run_default_ttests(CFG)
