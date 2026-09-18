#!/usr/bin/env python3
"""06_finalize: v2 batch 결과를 검증하고 별도 prepared-v2 경로로 publish한다."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from data_toolkit.preprocess._common.runtime import (
    add_batch_location_arguments,
    add_config_argument,
    add_rank_arguments,
    apply_batch_defaults,
    atomic_write_json,
    atomic_write_jsonl,
    config_get,
    load_config,
    PREPARED_V2_COMPLETION_FILE,
    completed_batch_reason,
    read_jsonl,
    resolve_work_root,
    require_batch_ownership,
    stage_root,
    successful,
    write_stage_info,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    add_batch_location_arguments(parser)
    add_rank_arguments(parser)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--prepared-root", type=Path, default=None)
    parser.add_argument("--publish", action="store_true", help="검증된 05_encode 결과를 prepared-v2에 publish")
    arguments = parser.parse_args()
    config, config_path = load_config(arguments.config)
    apply_batch_defaults(arguments, config)
    batch_index = require_batch_ownership(arguments, config)
    prepared_root = arguments.prepared_root or Path(config_get(config, "paths", "prepared_root"))
    work_root = resolve_work_root(arguments, config)
    output = stage_root(work_root, "06", "finalize")
    skip_reason = completed_batch_reason(
        config, arguments.source, arguments.shard, arguments.batch, prepared_root,
    )
    if skip_reason is not None:
        report = {
            "stage": "06_finalize", "source": arguments.source, "shard": arguments.shard,
            "batch": arguments.batch, "assets": 0, "published": False,
            "batch_index": batch_index, "world_size": arguments.world_size, "rank": arguments.rank,
            "skipped_completed": True, "skip_reason": skip_reason,
            "skipped_existing_prepared": skip_reason == "legacy_prepared",
            "skipped_published_prepared_v2": skip_reason == "prepared_v2",
        }
        atomic_write_jsonl(output / "manifest.jsonl", [])
        atomic_write_json(output / "report.json", report)
        write_stage_info(output, config=str(config_path), **report)
        print(output / "report.json")
        return 0
    manifest = arguments.manifest or stage_root(work_root, "05", "encode") / "manifest.jsonl"
    records = successful(read_jsonl(manifest))
    if not records:
        parser.error("05_encode의 성공 asset이 없습니다")
    encode_root = stage_root(work_root, "05", "encode")
    required = (encode_root / "shape", encode_root / "pbr", encode_root / "ss")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        parser.error(f"latent 출력이 누락되었습니다: {missing}")

    report = {
        "stage": "06_finalize", "source": arguments.source, "shard": arguments.shard,
        "batch": arguments.batch, "assets": len(records), "encode_root": str(encode_root),
        "published": False, "batch_index": batch_index,
        "world_size": arguments.world_size, "rank": arguments.rank,
    }
    if arguments.publish:
        target = prepared_root / arguments.source / arguments.shard / arguments.batch
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            parser.error(f"기존 publish 결과가 있습니다: {target}")
        with tempfile.TemporaryDirectory(dir=target.parent, prefix=f".{arguments.batch}.") as temporary_dir:
            temporary = Path(temporary_dir) / arguments.batch
            shutil.copytree(encode_root, temporary)
            atomic_write_json(
                temporary / PREPARED_V2_COMPLETION_FILE,
                {
                    "schema": "pixal3d-preprocess-v2-completion-v1",
                    "source": arguments.source,
                    "shard": arguments.shard,
                    "batch": arguments.batch,
                    "assets": len(records),
                },
            )
            os.replace(temporary, target)
        report.update(published=True, prepared_path=str(target))
    atomic_write_jsonl(output / "manifest.jsonl", records)
    atomic_write_json(output / "report.json", report)
    write_stage_info(output, config=str(config_path), **report)
    print(output / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
