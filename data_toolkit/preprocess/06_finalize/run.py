#!/usr/bin/env python3
"""06_finalize: 로컬 tar를 학습 loader로 검증한 뒤 prepared에 합친다."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
from pathlib import Path
import subprocess
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
    completed_batch_reason,
    read_jsonl,
    resolve_work_root,
    require_batch_ownership,
    stage_root,
    successful,
    write_stage_info,
)
from data_toolkit.preprocess._common.encode_partition import (
    exclude_configured_assets,
    parse_view_indices,
    skipped_oversized_asset_ids,
)
from data_toolkit.preprocess._common.finalize_packs import (
    BatchIdentity,
    BatchPublication,
    build_local_packs,
    family_sources,
    missing_members,
)
from data_toolkit.preprocess._common.finalize_publish import publish_staged
from data_toolkit.preprocess._common.finalize_training import validate_training_load


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    add_batch_location_arguments(parser)
    add_rank_arguments(parser)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--prepared-root", type=Path, default=None)
    parser.add_argument("--local-temp-root", type=Path, default=None)
    parser.add_argument("--publish", action="store_true", help="학습 loader 검증 뒤 8개 family tar를 prepared에 합침")
    arguments = parser.parse_args()
    config, config_path = load_config(arguments.config)
    apply_batch_defaults(arguments, config)
    batch_index = require_batch_ownership(arguments, config)
    prepared_root = arguments.prepared_root or Path(config_get(config, "paths", "prepared_root"))
    local_temp_root = arguments.local_temp_root or Path(config_get(config, "paths", "local_temp_root"))
    control_root = Path(config_get(config, "paths", "control_root"))
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
        }
        atomic_write_jsonl(output / "manifest.jsonl", [])
        atomic_write_json(output / "report.json", report)
        write_stage_info(output, config=str(config_path), **report)
        print(output / "report.json")
        return 0
    # 05 may run as disjoint high- and low-voxel partitions.  Its per-partition
    # manifests are operational logs, not the completion truth for a batch.
    manifest = arguments.manifest or stage_root(work_root, "04", "voxelize") / "manifest.jsonl"
    records = exclude_configured_assets(successful(read_jsonl(manifest)), config)
    skipped_oversized = skipped_oversized_asset_ids(stage_root(work_root, "05", "encode"))
    records = [record for record in records if record["asset_id"] not in skipped_oversized]
    if not records:
        parser.error("04_voxelize의 성공 asset이 없습니다")
    encode = config_get(config, "stages", "encode")
    if (
        tuple(int(value) for value in str(encode["resolutions"]).split(",")) != (256, 512, 1024)
        or int(encode["ss_resolution"]) != 64
        or parse_view_indices(str(encode["view_indices"])) != (0, 1)
    ):
        parser.error("06 pack layout은 shape/PBR 256,512,1024, SS 64, view 0-1을 요구합니다")
    publication = BatchPublication(
        identity=BatchIdentity(arguments.source, arguments.shard, arguments.batch),
        work_root=work_root,
        prepared_root=prepared_root,
        control_root=control_root,
        assets=tuple(sorted(record["asset_id"] for record in records)),
        batch_assets=tuple(sorted(
            (control_root / "shards" / arguments.source / arguments.shard / f"{arguments.batch}.txt")
            .read_text(encoding="utf-8").splitlines()
        )),
        config_hash=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        tool_commit="",
    )
    absent = missing_members(family_sources(publication))
    if absent:
        examples = ", ".join(str(path) for path in absent[:3])
        parser.error(
            f"03_render/05_encode pack 입력이 불완전합니다: {len(absent)}개 파일 누락 "
            f"(예: {examples})"
        )

    report = {
        "stage": "06_finalize", "source": arguments.source, "shard": arguments.shard,
        "batch": arguments.batch, "assets": len(records),
        "skipped_oversized": len(skipped_oversized),
        "published": False, "batch_index": batch_index,
        "world_size": arguments.world_size, "rank": arguments.rank,
    }
    if arguments.publish:
        repository = Path(__file__).resolve().parents[3]
        tool_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repository,
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", "data_toolkit/preprocess"],
            cwd=repository, check=True, capture_output=True, text=True,
        ).stdout.strip()
        if dirty:
            tool_commit += "+dirty"
        publication = replace(publication, tool_commit=tool_commit)
        local_temp_root = local_temp_root.resolve()
        local_temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{arguments.shard}-{arguments.batch}.", dir=local_temp_root,
        ) as temporary:
            staging = Path(temporary)
            entries = build_local_packs(publication, staging / "packs")
            training = validate_training_load(publication, entries, staging / "training")
            entries = publish_staged(publication, entries)
        report.update(
            published=True, loader_validated=True,
            loader_sample_asset=training.sample_asset,
            loader_configs=list(training.configs_checked),
            prepared_path=str(prepared_root),
            packs={family: entry["pack"] for family, entry in entries.items()},
        )
    atomic_write_jsonl(output / "manifest.jsonl", records)
    atomic_write_json(output / "report.json", report)
    write_stage_info(output, config=str(config_path), **report)
    print(output / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
