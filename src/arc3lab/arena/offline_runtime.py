from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


_KAGGLE_ENVIRONMENT_CANDIDATES = (
    Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files"),
    Path("/kaggle/input/arc-prize-2026-arc-agi-3/environment_files"),
)


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _contains_games(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        return next(path.glob("*/*/metadata.json"), None) is not None or next(
            path.glob("*/metadata.json"), None
        ) is not None
    except OSError:
        return False


def _validate_authoritative_path(path: str | Path, *, source: str) -> Path:
    resolved = _resolved(path)
    if _contains_games(resolved):
        return resolved
    raise FileNotFoundError(
        f"{source} points to an invalid ARC-AGI-3 environment directory: {resolved}. "
        "Expected a local environment_files tree containing metadata.json."
    )


def _autodiscovery_candidates() -> Iterable[Path]:
    seen: set[str] = set()

    def emit(path: Path) -> Path | None:
        resolved = _resolved(path)
        key = str(resolved)
        if key in seen:
            return None
        seen.add(key)
        return resolved

    for candidate in _KAGGLE_ENVIRONMENT_CANDIDATES:
        row = emit(candidate)
        if row is not None:
            yield row

    local = emit(Path("environment_files"))
    if local is not None:
        yield local

    kaggle_root = Path("/kaggle/input")
    if kaggle_root.is_dir():
        try:
            discovered = sorted(kaggle_root.glob("**/environment_files"))
        except OSError:
            discovered = []
        for candidate in discovered:
            row = emit(candidate)
            if row is not None:
                yield row


def discover_environment_dir(explicit: str | Path | None = None) -> Path:
    """Resolve a local ARC-AGI-3 environment tree without contacting the network.

    Explicit configuration is authoritative. If the caller passes a path, or an
    ``ENVIRONMENTS_DIR`` value already exists, an invalid path is an error rather
    than permission to silently evaluate against a different game tree.
    """

    if explicit:
        return _validate_authoritative_path(explicit, source="explicit environments_dir")

    env_value = os.getenv("ENVIRONMENTS_DIR", "").strip()
    if env_value:
        return _validate_authoritative_path(env_value, source="ENVIRONMENTS_DIR")

    attempted: list[str] = []
    for candidate in _autodiscovery_candidates():
        attempted.append(str(candidate))
        if _contains_games(candidate):
            return candidate
    raise FileNotFoundError(
        "Could not locate a local ARC-AGI-3 environment_files directory containing metadata.json. "
        f"Attempted: {attempted}"
    )


def configure_offline_environment(
    explicit: str | Path | None = None,
    *,
    recordings_dir: str | Path | None = None,
) -> Path:
    """Fail closed into the ARC toolkit's local-only mode for research evaluation."""

    environment_dir = discover_environment_dir(explicit)
    os.environ["OPERATION_MODE"] = "OFFLINE"
    os.environ["ENVIRONMENTS_DIR"] = str(environment_dir)
    if recordings_dir is not None:
        os.environ["RECORDINGS_DIR"] = str(Path(recordings_dir).resolve(strict=False))
    return environment_dir


def open_offline_arcade(
    explicit: str | Path | None = None,
    *,
    recordings_dir: str | Path | None = None,
):
    """Construct Arcade with explicit OFFLINE authority, independent of ambient env state."""

    environment_dir = configure_offline_environment(
        explicit,
        recordings_dir=recordings_dir,
    )
    from arc_agi import Arcade, OperationMode

    kwargs = {
        "operation_mode": OperationMode.OFFLINE,
        "environments_dir": str(environment_dir),
    }
    if recordings_dir is not None:
        kwargs["recordings_dir"] = str(Path(recordings_dir).resolve(strict=False))
    return Arcade(**kwargs)
