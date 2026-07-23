from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
import json

import pytest

from data_toolkit.pipeline.config import PathConfig
from data_toolkit.pipeline.output_compatibility import (
    APPROVED_HISTORICAL_COMMITS,
    REVIEW_BASELINE_COMMIT,
    OutputCompatibilityValidationError,
    load_output_compatibility,
)


def _changed_paths_digest(paths):
    return sha256(
        "".join(f"{path}\n" for path in paths).encode("utf-8")
    ).hexdigest()


@pytest.fixture
def compatibility_config(config, tmp_path):
    return replace(
        config,
        paths=PathConfig(
            data2_root=tmp_path / "data2",
            data3_root=tmp_path / "data3",
            local_root=tmp_path / "local",
        ),
    )


def _valid_payload(config):
    path_sets = (
        (
            "data_toolkit/pipeline/gpu_policy.py",
            "data_toolkit/pipeline/supervisor.py",
        ),
        (
            "data_toolkit/pipeline/evidence.py",
            "data_toolkit/pipeline/work_queue.py",
        ),
    )
    return {
        "schema_version": 1,
        "artifact_type": "output_compatibility_attestation",
        "config_hash": config.config_hash(),
        "pipeline_version": "pixal3d-mv-v2",
        "approved_at": datetime.now(timezone.utc).isoformat(),
        "reviewed_against_commit": REVIEW_BASELINE_COMMIT,
        "compatible_tool_commits": [
            {
                "tool_commit": commit,
                "changed_paths": list(paths),
                "changed_paths_sha256": _changed_paths_digest(paths),
            }
            for commit, paths in zip(
                sorted(APPROVED_HISTORICAL_COMMITS), path_sets, strict=True
            )
        ],
        "output_affecting_changed_paths": [],
    }


def _write_payload(config, payload):
    path = (
        config.paths.data2_root / "control/output_compatibility.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True))
    return path


def test_missing_output_compatibility_approves_no_historical_commits(
    compatibility_config,
):
    result = load_output_compatibility(compatibility_config)

    assert result.compatible_tool_commits == frozenset()
    assert result.evidence_sha256 is None


def test_exact_output_compatibility_attestation_is_accepted_without_mutation(
    compatibility_config,
):
    path = _write_payload(
        compatibility_config, _valid_payload(compatibility_config)
    )
    before = path.read_bytes()

    result = load_output_compatibility(compatibility_config)

    assert (
        result.compatible_tool_commits
        == APPROVED_HISTORICAL_COMMITS
    )
    assert result.evidence_sha256 == sha256(before).hexdigest()
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.update({"unknown": True}),
        lambda payload: payload["compatible_tool_commits"].append(
            dict(payload["compatible_tool_commits"][0])
        ),
        lambda payload: payload["compatible_tool_commits"][0].update(
            {"tool_commit": "short"}
        ),
        lambda payload: payload.update({"config_hash": "f" * 64}),
        lambda payload: payload.update({"pipeline_version": "other"}),
        lambda payload: payload.update(
            {"reviewed_against_commit": "e" * 40}
        ),
        lambda payload: payload["compatible_tool_commits"][0].update(
            {"changed_paths_sha256": "0" * 64}
        ),
        lambda payload: payload["compatible_tool_commits"][0].update(
            {"changed_paths": list(reversed(
                payload["compatible_tool_commits"][0]["changed_paths"]
            ))}
        ),
        lambda payload: payload.update(
            {"output_affecting_changed_paths": ["render.py"]}
        ),
        lambda payload: payload.update(
            {"approved_at": "2026-07-23T12:00:00"}
        ),
    ],
)
def test_invalid_output_compatibility_attestation_fails_closed(
    compatibility_config, mutate
):
    payload = _valid_payload(compatibility_config)
    mutate(payload)
    _write_payload(compatibility_config, payload)

    with pytest.raises(OutputCompatibilityValidationError):
        load_output_compatibility(compatibility_config)


def test_symlinked_output_compatibility_attestation_fails_closed(
    compatibility_config,
):
    control = compatibility_config.paths.data2_root / "control"
    control.mkdir(parents=True)
    target = control / "actual.json"
    target.write_text(
        json.dumps(_valid_payload(compatibility_config), sort_keys=True)
    )
    (control / "output_compatibility.json").symlink_to(target)

    with pytest.raises(OutputCompatibilityValidationError, match="unsafe"):
        load_output_compatibility(compatibility_config)
