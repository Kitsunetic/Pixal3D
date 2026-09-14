from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from .orchestrator import (
    _atomic_write_bytes_nofollow,
    _read_regular_bytes_nofollow,
    _required_checkpoint_dict,
)

HANDOFF_REMEDIATION_SCHEMA_VERSION = 2
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_CHUNK = re.compile(r"chunk[0-9]{3}")


@dataclass(frozen=True, slots=True)
class IsolatedHandoff:
    checkpoint_path: Path
    evidence_root: Path
    shard_id: str
    reason: str
    now: datetime


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


def preserve_isolated_handoff_attempt(
    handoff: IsolatedHandoff,
) -> tuple[str, int, int] | None:
    """Refund an active attempt interrupted by an isolated worker handoff."""

    _identifier(handoff.shard_id, "shard id")
    if not isinstance(handoff.reason, str) or not handoff.reason.strip():
        raise ValueError("isolated handoff reason must be non-empty")
    remediated_at = _timestamp(handoff.now)
    checkpoint_path = Path(handoff.checkpoint_path)
    evidence_root = Path(handoff.evidence_root)
    payload = _read_regular_bytes_nofollow(checkpoint_path)
    if payload is None:
        raise FileNotFoundError(checkpoint_path)
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"invalid isolated handoff checkpoint: {checkpoint_path}: {error}"
        ) from error
    checkpoint = _required_checkpoint_dict(value, checkpoint_path)
    if checkpoint.shard_id != handoff.shard_id or checkpoint.gate != "production":
        raise ValueError(
            f"isolated handoff checkpoint identity mismatch: {checkpoint_path}"
        )

    refunded: tuple[str, int, int] | None = None
    corrected = payload
    active_attempt = checkpoint.active_attempt
    if active_attempt is not None:
        command = active_attempt["command"]
        before = active_attempt["attempt"]
        assert isinstance(command, str)
        assert isinstance(before, int) and not isinstance(before, bool)
        if before <= 0:
            raise ValueError(
                f"isolated handoff cannot refund attempt zero: {checkpoint_path}"
            )
        checkpoint.attempts[command] = before - 1
        checkpoint.active_attempt = None
        corrected = _checkpoint_payload(checkpoint)
        refunded = (command, before, before - 1)

    evidence_root.mkdir(parents=True, exist_ok=False)
    _atomic_write_bytes_nofollow(
        evidence_root / "pipeline.before.json",
        payload,
    )
    if refunded is not None:
        _atomic_write_bytes_nofollow(checkpoint_path, corrected)
    remediation = {
        "schema_version": HANDOFF_REMEDIATION_SCHEMA_VERSION,
        "shard_id": handoff.shard_id,
        "reason": handoff.reason,
        "remediated_at": remediated_at,
        "checkpoint_path": str(checkpoint_path),
        "refunded_attempt": (
            {
                "command": refunded[0],
                "attempts_before": refunded[1],
                "attempts_after": refunded[2],
            }
            if refunded is not None
            else None
        ),
        "checkpoint_sha256": {
            "before": sha256(payload).hexdigest(),
            "after": sha256(corrected).hexdigest(),
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
    return refunded


def preserve_operator_handoff_attempts(
    checkpoint_root: Path,
    *,
    shard_id: str,
    evidence_root: Path,
    unit_id: str,
    node_id: str,
    lease_token: str,
    reason: str,
    now: datetime,
) -> tuple[tuple[str, str, int, int], ...]:
    """Refund commands interrupted by a token-fenced operator handoff."""

    for value, description in (
        (shard_id, "shard id"),
        (unit_id, "work unit id"),
        (node_id, "node id"),
        (lease_token, "lease token"),
    ):
        _identifier(value, description)
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("operator handoff reason must be non-empty")
    remediated_at = _timestamp(now)
    checkpoint_root = Path(checkpoint_root)
    evidence_root = Path(evidence_root)

    candidates = []
    if checkpoint_root.is_dir():
        for chunk_root in sorted(checkpoint_root.iterdir()):
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
            expected_shard = f"{shard_id}-{chunk_root.name}"
            if (
                checkpoint.shard_id != expected_shard
                or checkpoint.gate != "production"
            ):
                raise ValueError(
                    f"operator handoff checkpoint identity mismatch: {path}"
                )
            if checkpoint.active_attempt is None:
                continue
            command = checkpoint.active_attempt["command"]
            before = checkpoint.active_attempt["attempt"]
            assert isinstance(command, str)
            assert isinstance(before, int) and not isinstance(before, bool)
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
        "checkpoint_root": str(checkpoint_root),
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
