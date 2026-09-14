import json
from datetime import datetime, timezone

from data_toolkit.pipeline.operator_handoff import (
    IsolatedHandoff,
    preserve_isolated_handoff_attempt,
    preserve_operator_handoff_attempts,
)

NOW = datetime(2026, 7, 28, 2, 30, tzinfo=timezone.utc)


def _checkpoint(*, active=True):
    return {
        "schema_version": 3,
        "shard_id": "HSSD-00000-chunk001",
        "completed_commands": ["stage_raw", "dump_mesh"],
        "attempts": {"stage_raw": 1, "dump_mesh": 1, "dump_pbr": 3},
        "active_attempt": (
            {"command": "dump_pbr", "attempt": 3}
            if active
            else None
        ),
        "quality_outcomes": {},
        "gate": "production",
    }


def test_operator_handoff_refunds_only_the_active_command_attempt(tmp_path):
    checkpoint_root = (
        tmp_path
        / "control/checkpoints/HSSD/HSSD-00000/chunks/batch009"
    )
    checkpoint = (
        checkpoint_root / "chunk001/pipeline.json"
    )
    checkpoint.parent.mkdir(parents=True)
    original = json.dumps(_checkpoint(), sort_keys=True).encode()
    checkpoint.write_bytes(original)
    evidence_root = tmp_path / "control/remediations/handoff-token"

    result = preserve_operator_handoff_attempts(
        checkpoint_root,
        shard_id="HSSD-00000",
        evidence_root=evidence_root,
        unit_id="HSSD--HSSD-00000--batch009",
        node_id="node16",
        lease_token="handoff-token",
        reason="operator concurrency reload",
        now=NOW,
    )

    repaired = json.loads(checkpoint.read_text())
    assert result == (("chunk001", "dump_pbr", 3, 2),)
    assert repaired["attempts"]["dump_pbr"] == 2
    assert repaired["attempts"]["dump_mesh"] == 1
    assert repaired["active_attempt"] is None
    assert (
        evidence_root / "chunk001.pipeline.before.json"
    ).read_bytes() == original
    evidence = json.loads(
        (evidence_root / "remediation.json").read_text()
    )
    assert evidence["schema_version"] == 2
    assert evidence["checkpoint_root"] == str(checkpoint_root)
    assert evidence["unit_id"] == "HSSD--HSSD-00000--batch009"
    assert evidence["refunded_attempts"] == [
        {
            "attempts_after": 2,
            "attempts_before": 3,
            "chunk_id": "chunk001",
            "command": "dump_pbr",
        }
    ]


def test_operator_handoff_records_noop_when_no_command_is_active(tmp_path):
    checkpoint_root = (
        tmp_path
        / "control/checkpoints/HSSD/HSSD-00000/chunks/batch009"
    )
    checkpoint = (
        checkpoint_root / "chunk001/pipeline.json"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text(json.dumps(_checkpoint(active=False)))
    evidence_root = tmp_path / "control/remediations/handoff-token"

    result = preserve_operator_handoff_attempts(
        checkpoint_root,
        shard_id="HSSD-00000",
        evidence_root=evidence_root,
        unit_id="HSSD--HSSD-00000--batch009",
        node_id="node16",
        lease_token="handoff-token",
        reason="operator concurrency reload",
        now=NOW,
    )

    assert result == ()
    assert json.loads(checkpoint.read_text())["attempts"]["dump_pbr"] == 3
    evidence = json.loads(
        (evidence_root / "remediation.json").read_text()
    )
    assert evidence["schema_version"] == 2
    assert evidence["checkpoint_root"] == str(checkpoint_root)
    assert evidence["refunded_attempts"] == []


def test_isolated_handoff_refunds_active_attempt_with_evidence(tmp_path):
    checkpoint = tmp_path / "control/checkpoints/source/source-00002/batch000.json"
    checkpoint.parent.mkdir(parents=True)
    value = _checkpoint()
    value["shard_id"] = "source-00002"
    original = json.dumps(value, sort_keys=True).encode()
    checkpoint.write_bytes(original)
    evidence_root = tmp_path / "control/remediations/isolated-stop"

    result = preserve_isolated_handoff_attempt(
        IsolatedHandoff(
            checkpoint_path=checkpoint,
            evidence_root=evidence_root,
            shard_id="source-00002",
            reason="foreign GPU process appeared",
            now=NOW,
        )
    )

    repaired = json.loads(checkpoint.read_text())
    assert result == ("dump_pbr", 3, 2)
    assert repaired["attempts"]["dump_pbr"] == 2
    assert repaired["active_attempt"] is None
    assert (evidence_root / "pipeline.before.json").read_bytes() == original
    evidence = json.loads((evidence_root / "remediation.json").read_text())
    assert evidence["shard_id"] == "source-00002"
    assert evidence["refunded_attempt"] == {
        "attempts_after": 2,
        "attempts_before": 3,
        "command": "dump_pbr",
    }
