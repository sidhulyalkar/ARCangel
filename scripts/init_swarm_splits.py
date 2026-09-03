from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from arc3lab.arena.splits import SplitRegistry


def main() -> int:
    ap = argparse.ArgumentParser(description="Discover ARC environments and initialize swarm splits")
    ap.add_argument("--root", default="artifacts/arena/v013")
    ap.add_argument("--salt", default=os.getenv("ARCANGEL_SPLIT_SALT", ""))
    ap.add_argument("--dev-fraction", type=float, default=0.60)
    ap.add_argument("--validation-fraction", type=float, default=0.20)
    args = ap.parse_args()
    if not args.salt:
        raise ValueError("set ARCANGEL_SPLIT_SALT or pass --salt; do not commit the private salt")

    from arc_agi import Arcade

    arc = Arcade()
    game_ids = sorted(str(item.game_id) for item in arc.get_environments())
    if len(game_ids) < 3:
        raise RuntimeError("too few environments to create DEV/VALIDATION/BLIND splits")

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "game_ids.txt").write_text("\n".join(game_ids) + "\n")
    registry = SplitRegistry.build(
        game_ids,
        salt=args.salt,
        dev_fraction=args.dev_fraction,
        validation_fraction=args.validation_fraction,
    )
    registry.write(root / "splits.public.json", root / "splits.private.json")
    print(
        json.dumps(
            {
                "game_count": len(game_ids),
                "dev": len(registry.dev),
                "validation": len(registry.validation),
                "blind": len(registry.blind),
                "public_registry": str(root / "splits.public.json"),
                "private_registry": str(root / "splits.private.json"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
