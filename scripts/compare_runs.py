#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(p: str):
    return json.loads(Path(p).read_text())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    args = p.parse_args()
    print(f"{'run':32} {'score':>8} {'levels':>9} {'actions':>9} {'sec':>8}")
    for path in args.runs:
        d = load(path)
        s = d.get("scorecard") or {}
        print(f"{Path(path).name[:32]:32} {float(s.get('score') or 0):8.4f} {str(s.get('total_levels_completed'))+'/'+str(s.get('total_levels')):>9} {str(s.get('total_actions')):>9} {d.get('elapsed_seconds',0):8.1f}")


if __name__ == "__main__":
    main()
