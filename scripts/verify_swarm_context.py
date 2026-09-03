from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

from arc3lab.arena.research_context import assert_research_payload_safe


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify one exported ARCangel swarm context capsule")
    ap.add_argument("--capsule", required=True)
    ap.add_argument("--receipt", required=True)
    args = ap.parse_args()

    capsule = Path(args.capsule)
    receipt_path = Path(args.receipt)
    if not capsule.exists():
        raise FileNotFoundError(capsule)
    if not receipt_path.exists():
        raise FileNotFoundError(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    actual_sha = hashlib.sha256(capsule.read_bytes()).hexdigest()
    if actual_sha != str(receipt.get("capsule_sha256", "")):
        raise RuntimeError("swarm context capsule SHA-256 does not match receipt")
    if receipt.get("format") != "arcangel-swarm-context-v1":
        raise RuntimeError("unknown swarm context capsule format")
    if receipt.get("research_evidence_scope") != ["dev", "validation"]:
        raise RuntimeError("swarm context receipt does not prove DEV/VALIDATION-only evidence")

    with tarfile.open(capsule, "r:gz") as archive:
        names = archive.getnames()
        if any("/blind/" in f"/{name.lower()}/" or "blind-results" in name.lower() for name in names):
            raise RuntimeError("capsule contains a private BLIND artifact path")
        if any(name.startswith("repo/artifacts/arena/") for name in names):
            raise RuntimeError("capsule contains private arena artifact paths")
        handle = archive.extractfile("arena/scorecard.json")
        if handle is None:
            raise RuntimeError("capsule has no research scorecard")
        scorecard = json.loads(handle.read())
        assert_research_payload_safe(scorecard)
        if scorecard.get("evidence_scope") != ["dev", "validation"]:
            raise RuntimeError("capsule scorecard is not DEV/VALIDATION-only")
        if int(scorecard.get("result_count", -1)) != int(receipt.get("development_result_count", -2)):
            raise RuntimeError("capsule development result count does not match receipt")

        expected_hashes = dict(receipt.get("included_file_sha256") or {})
        for rel, expected in expected_hashes.items():
            member = archive.extractfile(f"repo/{rel}")
            if member is None:
                raise RuntimeError(f"receipt references missing capsule file: {rel}")
            if hashlib.sha256(member.read()).hexdigest() != expected:
                raise RuntimeError(f"capsule file hash mismatch: {rel}")

    print(
        json.dumps(
            {
                "status": "SWARM_CONTEXT_VERIFIED",
                "capsule": str(capsule),
                "capsule_sha256": actual_sha,
                "git_head": receipt.get("git_head"),
                "development_result_count": receipt.get("development_result_count"),
                "included_files": len(receipt.get("included_file_sha256") or {}),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
