#!/usr/bin/env python
from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("env_dir")
    args = p.parse_args()
    rows = []
    for path in glob.glob(str(Path(args.env_dir) / "*" / "*" / "metadata.json")):
        d = json.load(open(path, encoding="utf-8"))
        baselines = d.get("baseline_actions", [])
        rows.append(
            {
                "game": d["game_id"],
                "tags": d.get("tags", []),
                "levels": len(baselines),
                "human_actions": sum(baselines),
                "mean": round(statistics.mean(baselines), 1) if baselines else 0,
            }
        )
    print(json.dumps({"games": len(rows), "levels": sum(r["levels"] for r in rows), "rows": sorted(rows, key=lambda x: x["game"])}, indent=2))


if __name__ == "__main__":
    main()
