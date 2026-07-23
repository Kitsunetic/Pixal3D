from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import json
from pathlib import PurePosixPath
import re

from .config import PipelineConfig


PIPELINE_VERSION = "pixal3d-mv-v2"
REVIEW_BASELINE_COMMIT = "1d36d55d7a15e32d85ce892a7a2b91f9809d149a"
APPROVED_HISTORICAL_COMMITS = frozenset(
    {
        "0f4b290f3d23419f9389e669734cb7b8a50ec817",
        "fc26830338348e17929d7a734a4015ed4dca7bdd",
    }
)
_HEX_40 = re.compile(r"[0-9a-f]{40}")
_HEX_64 = re.compile(r"[0-9a-f]{64}")
_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "config_hash",
        "pipeline_version",
        "approved_at",
        "reviewed_against_commit",
        "compatible_tool_commits",
        "output_affecting_changed_paths",
    }
)
_COMMIT_KEYS = frozenset(
    {
        "tool_commit",
        "changed_paths",
        "changed_paths_sha256",
    }
)


class OutputCompatibilityValidationError(ValueError):
    pass


@dataclass(frozen=True)
class OutputCompatibility:
    compatible_tool_commits: frozenset[str]
    evidence_sha256: str | None


def _reject_duplicate_json_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise OutputCompatibilityValidationError(
                f"duplicate JSON key: {key}"
            )
        value[key] = item
    return value


def _require_exact_keys(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise OutputCompatibilityValidationError(
            f"invalid {label} keys"
        )


def _validate_changed_paths(value, label):
    if not isinstance(value, list) or not value:
        raise OutputCompatibilityValidationError(
            f"{label} must be a non-empty list"
        )
    if value != sorted(set(value)):
        raise OutputCompatibilityValidationError(
            f"{label} must be sorted and unique"
        )
    for path in value:
        if not isinstance(path, str) or not path:
            raise OutputCompatibilityValidationError(
                f"invalid {label} entry"
            )
        pure = PurePosixPath(path)
        if (
            pure.is_absolute()
            or pure.as_posix() != path
            or path == "."
            or ".." in pure.parts
        ):
            raise OutputCompatibilityValidationError(
                f"unsafe {label} entry: {path!r}"
            )


def _changed_paths_sha256(paths):
    return sha256(
        "".join(f"{path}\n" for path in paths).encode("utf-8")
    ).hexdigest()


def _validate_timestamp(value):
    if not isinstance(value, str):
        raise OutputCompatibilityValidationError(
            "approved_at must be a string"
        )
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise OutputCompatibilityValidationError(
            "approved_at must be an RFC 3339 timestamp"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise OutputCompatibilityValidationError(
            "approved_at must include a timezone"
        )


def _validate_payload(payload, config):
    _require_exact_keys(
        payload, _TOP_LEVEL_KEYS, "output compatibility attestation"
    )
    if payload["schema_version"] != 1 or isinstance(
        payload["schema_version"], bool
    ):
        raise OutputCompatibilityValidationError(
            "unsupported output compatibility schema"
        )
    if payload["artifact_type"] != "output_compatibility_attestation":
        raise OutputCompatibilityValidationError(
            "invalid output compatibility artifact type"
        )
    if (
        config.pipeline_version != PIPELINE_VERSION
        or payload["pipeline_version"] != PIPELINE_VERSION
    ):
        raise OutputCompatibilityValidationError(
            "output compatibility pipeline version mismatch"
        )
    if (
        not isinstance(payload["config_hash"], str)
        or not _HEX_64.fullmatch(payload["config_hash"])
        or payload["config_hash"] != config.config_hash()
    ):
        raise OutputCompatibilityValidationError(
            "output compatibility config hash mismatch"
        )
    _validate_timestamp(payload["approved_at"])
    if payload["reviewed_against_commit"] != REVIEW_BASELINE_COMMIT:
        raise OutputCompatibilityValidationError(
            "output compatibility review baseline mismatch"
        )
    output_paths = payload["output_affecting_changed_paths"]
    if not isinstance(output_paths, list) or output_paths:
        raise OutputCompatibilityValidationError(
            "output-affecting changed paths must be an empty list"
        )
    entries = payload["compatible_tool_commits"]
    if not isinstance(entries, list):
        raise OutputCompatibilityValidationError(
            "compatible tool commits must be a list"
        )
    commits = []
    for index, entry in enumerate(entries):
        _require_exact_keys(entry, _COMMIT_KEYS, f"commit entry {index}")
        commit = entry["tool_commit"]
        if not isinstance(commit, str) or not _HEX_40.fullmatch(commit):
            raise OutputCompatibilityValidationError(
                f"invalid tool commit in entry {index}"
            )
        paths = entry["changed_paths"]
        _validate_changed_paths(paths, f"changed paths for {commit}")
        digest = entry["changed_paths_sha256"]
        if (
            not isinstance(digest, str)
            or not _HEX_64.fullmatch(digest)
            or digest != _changed_paths_sha256(paths)
        ):
            raise OutputCompatibilityValidationError(
                f"changed-path digest mismatch for {commit}"
            )
        commits.append(commit)
    if len(commits) != len(set(commits)):
        raise OutputCompatibilityValidationError(
            "duplicate compatible tool commit"
        )
    if frozenset(commits) != APPROVED_HISTORICAL_COMMITS:
        raise OutputCompatibilityValidationError(
            "compatible tool commit set mismatch"
        )
    return frozenset(commits)


def load_output_compatibility(
    config: PipelineConfig,
) -> OutputCompatibility:
    # Import lazily so the compatibility module can reuse the pipeline's
    # hardened no-follow reader without adding an orchestrator import cycle.
    from .orchestrator import _read_regular_bytes_nofollow

    path = (
        config.paths.data2_root / "control/output_compatibility.json"
    )
    try:
        payload_bytes = _read_regular_bytes_nofollow(path, missing_ok=True)
    except OSError as error:
        raise OutputCompatibilityValidationError(
            f"unsafe output compatibility attestation: {path}: {error}"
        ) from error
    if payload_bytes is None:
        return OutputCompatibility(frozenset(), None)
    try:
        payload = json.loads(
            payload_bytes,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OutputCompatibilityValidationError(
            "invalid output compatibility JSON"
        ) from error
    commits = _validate_payload(payload, config)
    return OutputCompatibility(
        compatible_tool_commits=commits,
        evidence_sha256=sha256(payload_bytes).hexdigest(),
    )
