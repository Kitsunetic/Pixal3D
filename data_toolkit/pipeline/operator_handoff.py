from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re

from .orchestrator import (
    _atomic_write_bytes_nofollow,
    _read_regular_bytes_nofollow,
    _required_checkpoint_dict,
)


HANDOFF_REMEDIATION_SCHEMA_VERSION = 1
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_CHUNK = re.compile(r"chunk[0-9]{3}")


def _identifier(value: str, description: str) -> None:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"invalid {description}: {value!r}")


def _timestamp(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("operator handoff timestamp must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat()


def _checkpoint_payload(checkpoint) -> bytes:
    value = asdict(checkpoint)
    _required_checkpoint_dict(value, Path("pipeline.json"))
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def preserve_operator_handoff_attempts(
    batch_root: Path,
    *,
    evidence_root: Path,
    unit_id: str,
    node_id: str,
    lease_token: str,
    reason: str,
    now: datetime,
) -> tuple[tuple[str, str, int, int], ...]:
    """Refund commands interrupted by a token-fenced operator handoff."""

    for value, description in (
        (unit_id, "work unit id"),
        (node_id, "node id"),
        (lease_token, "lease token"),
    ):
        _identifier(value, description)
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("operator handoff reason must be non-empty")
    remediated_at = _timestamp(now)
    batch_root = Path(batch_root)
    evidence_root = Path(evidence_root)
    checkpoints_root = batch_root / "chunk_checkpoints"

    candidates = []
    if checkpoints_root.is_dir():
        for chunk_root in sorted(checkpoints_root.iterdir()):
            if (
                _CHUNK.fullmatch(chunk_root.name) is None
                or not chunk_root.is_dir()
                or chunk_root.is_symlink()
            ):
                continue
            path = chunk_root / "pipeline.json"
            payload = _read_regular_bytes_nofollow(path, missing_ok=True)
            if payload is None:
                continue
            try:
                value = json.loads(payload)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"invalid operator handoff checkpoint: {path}: {error}"
                ) from error
            checkpoint = _required_checkpoint_dict(value, path)
            expected_shard = (
                f"{batch_root.parent.name}-{chunk_root.name}"
            )
            if (
                checkpoint.shard_id != expected_shard
                or checkpoint.gate != "production"
            ):
                raise ValueError(
                    f"operator handoff checkpoint identity mismatch: {path}"
                )
            if checkpoint.active_attempt is None:
                continue
            command = str(checkpoint.active_attempt["command"])
            before = int(checkpoint.active_attempt["attempt"])
            if before <= 0:
                raise ValueError(
                    f"operator handoff cannot refund attempt zero: {path}"
                )
            checkpoint.attempts[command] = before - 1
            checkpoint.active_attempt = None
            corrected = _checkpoint_payload(checkpoint)
            candidates.append(
                (
                    chunk_root.name,
                    command,
                    before,
                    before - 1,
                    path,
                    payload,
                    corrected,
                )
            )

    evidence_root.mkdir(parents=True, exist_ok=False)
    refunded = []
    for (
        chunk_id,
        command,
        before,
        after,
        path,
        original,
        corrected,
    ) in candidates:
        _atomic_write_bytes_nofollow(
            evidence_root / f"{chunk_id}.pipeline.before.json",
            original,
        )
        _atomic_write_bytes_nofollow(path, corrected)
        refunded.append(
            {
                "chunk_id": chunk_id,
                "command": command,
                "attempts_before": before,
                "attempts_after": after,
            }
        )

    remediation = {
        "schema_version": HANDOFF_REMEDIATION_SCHEMA_VERSION,
        "unit_id": unit_id,
        "node_id": node_id,
        "lease_token": lease_token,
        "reason": reason,
        "remediated_at": remediated_at,
        "batch_root": str(batch_root),
        "refunded_attempts": refunded,
        "checkpoint_sha256": {
            item[0]: {
                "before": sha256(item[5]).hexdigest(),
                "after": sha256(item[6]).hexdigest(),
            }
            for item in candidates
        },
    }
    _atomic_write_bytes_nofollow(
        evidence_root / "remediation.json",
        json.dumps(
            remediation,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8"),
    )
    return tuple(
        (
            item["chunk_id"],
            item["command"],
            item["attempts_before"],
            item["attempts_after"],
        )
        for item in refunded
    )
