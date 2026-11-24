#!/usr/bin/env python
"""Turn results/timings.csv into charts and a Markdown table for the README/blog post."""
import argparse
from pathlib import Path

from scbench.plots import make_report

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="results")
    make_report(Path(p.parse_args().results))
