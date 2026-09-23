#!/usr/bin/env python3
"""01_manifest: control batch을 압축 해제된 direct GLB 경로로 고정한다."""

from __future__ import annotations

import argparse
import csv
import fcntl
import os
from pathlib import Path
import re
import sqlite3
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from data_toolkit.preprocess._common.runtime import (
    add_batch_location_arguments,
    add_config_argument,
    add_rank_arguments,
    apply_batch_defaults,
    atomic_write_jsonl,
    config_get,
    load_config,
    completed_batch_reason,
    require_batch_ownership,
    resolve_work_root,
    sha256_file,
    stage_root,
    write_stage_info,
)


OBJECT_ID_PATTERN = re.compile(r"/3d-models/([^/?#]+)")


def object_id_from_identifier(value: str) -> str:
    match = OBJECT_ID_PATTERN.search(value)
    if match is None:
        raise ValueError(f"Sketchfab object id를 찾을 수 없습니다: {value!r}")
    return match.group(1)


def default_index_path(work_root: Path) -> Path:
    return work_root.parents[1] / "_index" / "asset_index.sqlite"


def direct_paths(raw_root: Path, glb_roots: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for glb_root in glb_roots:
        root = raw_root / glb_root
        if not root.is_dir():
            raise FileNotFoundError(f"direct GLB root를 찾을 수 없습니다: {root}")
        for path in root.rglob("*.glb"):
            if path.stem in result:
                raise RuntimeError(f"중복 object id가 있습니다: {path.stem}")
            result[path.stem] = path
    return result


def build_index(
    index_path: Path,
    metadata_csv: Path,
    raw: dict,
) -> int:
    raw_root = Path(raw["root"])
    glb_roots = [str(item) for item in raw["glb_roots"]]
    mode = raw["mode"]
    if mode != "direct_glb":
        raise ValueError(f"unsupported raw mode: {mode}; direct_glb만 지원합니다")
    direct = direct_paths(raw_root, glb_roots)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=index_path.parent, prefix=f".{index_path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
        connection = sqlite3.connect(temporary)
        try:
            connection.execute(
                "CREATE TABLE asset_index (asset_id TEXT PRIMARY KEY, object_id TEXT NOT NULL, raw_kind TEXT NOT NULL, raw_path TEXT NOT NULL)"
            )
            rows: list[tuple[str, str, str, str]] = []
            with metadata_csv.open(encoding="utf-8", newline="") as stream:
                for record in csv.DictReader(stream):
                    object_id = object_id_from_identifier(record["file_identifier"])
                    direct_path = direct.get(object_id)
                    if direct_path is not None:
                        rows.append((record["sha256"], object_id, "direct", str(direct_path.resolve())))
            connection.executemany(
                "INSERT INTO asset_index(asset_id, object_id, raw_kind, raw_path) VALUES (?, ?, ?, ?)",
                rows,
            )
            connection.commit()
        finally:
            connection.close()
        os.replace(temporary, index_path)
        temporary = None
        return len(rows)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def index_uses_direct_glbs(index_path: Path) -> bool:
    connection = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
    try:
        return connection.execute(
            "SELECT COUNT(*) FROM asset_index WHERE raw_kind != 'direct'"
        ).fetchone()[0] == 0
    finally:
        connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    add_batch_location_arguments(parser)
    add_rank_arguments(parser)
    parser.add_argument("--instances", type=Path, default=None)
    parser.add_argument("--metadata-csv", type=Path, default=None)
    parser.add_argument("--index-path", type=Path, default=None)
    parser.add_argument("--build-index", action="store_true")
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="기존 index를 현재 direct GLB 설정으로 다시 생성합니다.",
    )
    parser.add_argument("--hash-input", action="store_true")
    arguments = parser.parse_args()
    config, config_path = load_config(arguments.config)
    apply_batch_defaults(arguments, config)
    batch_index = require_batch_ownership(arguments, config)
    work_root = resolve_work_root(arguments, config)
    control_root = Path(config_get(config, "paths", "control_root"))
    instances = arguments.instances or control_root / "shards" / arguments.source / arguments.shard / f"{arguments.batch}.txt"
    metadata_csv = arguments.metadata_csv or control_root / "metadata" / arguments.source / "metadata.csv"
    index_path = (arguments.index_path or default_index_path(work_root)).resolve()
    raw = dict(config_get(config, "raw"))
    if arguments.build_index or arguments.rebuild_index:
        lock_path = index_path.with_name(f".{index_path.name}.lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("w") as lock_stream:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
            if index_path.is_file() and not arguments.rebuild_index:
                print(f"using existing index: {index_path}")
            else:
                indexed = build_index(index_path, metadata_csv, raw)
                print(f"indexed {indexed} assets: {index_path}")
    if not index_path.is_file():
        parser.error(f"asset index가 없습니다. 한 번 --build-index로 실행하세요: {index_path}")
    if not index_uses_direct_glbs(index_path):
        parser.error(
            "기존 index에 archive_7z record가 남아 있습니다. "
            "압축 해제 완료 후 --rebuild-index로 다시 생성하세요."
        )

    output = stage_root(work_root, "01", "manifest")
    skip_reason = completed_batch_reason(config, arguments.source, arguments.shard, arguments.batch)
    if skip_reason is not None:
        atomic_write_jsonl(output / "manifest.jsonl", [])
        write_stage_info(
            output, stage="01_manifest", config=str(config_path), source=arguments.source,
            shard=arguments.shard, batch=arguments.batch, count=0, succeeded=0,
            batch_index=batch_index, world_size=arguments.world_size, rank=arguments.rank,
            skipped_completed=True, skip_reason=skip_reason,
            skipped_existing_prepared=skip_reason == "legacy_prepared",
        )
        print(output / "manifest.jsonl")
        return 0

    asset_ids = [line.strip() for line in instances.read_text().splitlines() if line.strip()]
    connection = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
    try:
        resolved = {
            asset_id: row for asset_id in asset_ids if (row := connection.execute(
                "SELECT object_id, raw_kind, raw_path FROM asset_index WHERE asset_id = ?", (asset_id,)
            ).fetchone()) is not None
        }
    finally:
        connection.close()
    records = []
    for asset_id in asset_ids:
        row = resolved.get(asset_id)
        if row is None:
            records.append({"asset_id": asset_id, "status": "error", "error": "configured raw source에 asset이 없습니다"})
            continue
        object_id, raw_kind, raw_path = row
        record = {
            "asset_id": asset_id, "object_id": object_id, "raw_kind": raw_kind,
            "raw_path": raw_path,
            "status": "ok",
        }
        if arguments.hash_input and raw_kind == "direct":
            record["raw_sha256"] = sha256_file(Path(raw_path))
        records.append(record)
    atomic_write_jsonl(output / "manifest.jsonl", records)
    write_stage_info(
        output, stage="01_manifest", config=str(config_path), source=arguments.source,
        shard=arguments.shard, batch=arguments.batch, instances=str(instances),
        metadata_csv=str(metadata_csv), index_path=str(index_path), count=len(records),
        batch_index=batch_index, world_size=arguments.world_size, rank=arguments.rank,
        succeeded=sum(record["status"] == "ok" for record in records),
    )
    print(output / "manifest.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
