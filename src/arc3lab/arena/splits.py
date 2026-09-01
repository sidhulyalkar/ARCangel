from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True, slots=True)
class SplitRegistry:
    dev: tuple[str, ...]
    validation: tuple[str, ...]
    blind: tuple[str, ...]
    salt: str

    @classmethod
    def build(
        cls,
        game_ids: Iterable[str],
        *,
        salt: str,
        dev_fraction: float = 0.60,
        validation_fraction: float = 0.20,
    ) -> "SplitRegistry":
        if not 0 < dev_fraction < 1:
            raise ValueError("dev_fraction must be between 0 and 1")
        if not 0 < validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        if dev_fraction + validation_fraction >= 1:
            raise ValueError("dev + validation fractions must leave a blind split")
        unique = sorted({str(game_id) for game_id in game_ids})
        buckets: dict[str, list[str]] = {"dev": [], "validation": [], "blind": []}
        for game_id in unique:
            digest = hashlib.sha256(f"{salt}:{game_id}".encode()).digest()
            unit = int.from_bytes(digest[:8], "big") / float(2**64)
            if unit < dev_fraction:
                buckets["dev"].append(game_id)
            elif unit < dev_fraction + validation_fraction:
                buckets["validation"].append(game_id)
            else:
                buckets["blind"].append(game_id)
        return cls(
            dev=tuple(buckets["dev"]),
            validation=tuple(buckets["validation"]),
            blind=tuple(buckets["blind"]),
            salt=salt,
        )

    def ids(self, split: str) -> tuple[str, ...]:
        if split not in {"dev", "validation", "blind"}:
            raise ValueError(f"unknown split {split}")
        return getattr(self, split)

    def public_dict(self) -> dict[str, object]:
        """Research-facing view. Blind game identities are intentionally absent."""
        return {
            "dev": list(self.dev),
            "validation": list(self.validation),
            "blind_count": len(self.blind),
            "salt_hash": hashlib.sha256(self.salt.encode()).hexdigest(),
        }

    def private_dict(self) -> dict[str, object]:
        return {
            "dev": list(self.dev),
            "validation": list(self.validation),
            "blind": list(self.blind),
            "salt": self.salt,
        }

    def write(self, public_path: str | Path, private_path: str | Path) -> None:
        Path(public_path).write_text(json.dumps(self.public_dict(), indent=2) + "\n")
        Path(private_path).write_text(json.dumps(self.private_dict(), indent=2) + "\n")
