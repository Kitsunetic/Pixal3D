#!/usr/bin/env python3
"""01_manifest: control batch을 direct GLB 또는 7z archive member로 고정한다."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
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
    apply_batch_defaults,
    atomic_write_jsonl,
    config_get,
    load_config,
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


def direct_paths(raw_root: Path, glb_root: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in (raw_root / glb_root).rglob("*.glb"):
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
    glb_root = str(raw["glb_root"])
    mode = raw["mode"]
    direct = direct_paths(raw_root, glb_root)
    object_paths: dict[str, str] = {}
    if mode == "archive_7z":
        with gzip.open(Path(raw["object_path_index"]), "rt", encoding="utf-8") as stream:
            object_paths = json.load(stream)
    elif mode != "direct_glb":
        raise ValueError(f"unsupported raw mode: {mode}")

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
                "CREATE TABLE asset_index (asset_id TEXT PRIMARY KEY, object_id TEXT NOT NULL, raw_kind TEXT NOT NULL, raw_path TEXT, archive_path TEXT, archive_member TEXT)"
            )
            rows: list[tuple[str, str, str, str | None, str | None, str | None]] = []
            with metadata_csv.open(encoding="utf-8", newline="") as stream:
                for record in csv.DictReader(stream):
                    object_id = object_id_from_identifier(record["file_identifier"])
                    direct_path = direct.get(object_id)
                    if direct_path is not None:
                        rows.append((record["sha256"], object_id, "direct", str(direct_path.resolve()), None, None))
                        continue
                    if mode != "archive_7z":
                        continue
                    relative = object_paths.get(object_id)
                    if relative is None:
                        continue
                    member_path = Path(relative)
                    archive_path = raw_root / glb_root / f"{member_path.parent.name}.7z"
                    if archive_path.is_file():
                        rows.append((
                            record["sha256"], object_id, "archive_7z", None,
                            str(archive_path.resolve()), member_path.relative_to(glb_root).as_posix(),
                        ))
            connection.executemany(
                "INSERT INTO asset_index(asset_id, object_id, raw_kind, raw_path, archive_path, archive_member) VALUES (?, ?, ?, ?, ?, ?)",
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    add_batch_location_arguments(parser)
    parser.add_argument("--instances", type=Path, default=None)
    parser.add_argument("--metadata-csv", type=Path, default=None)
    parser.add_argument("--index-path", type=Path, default=None)
    parser.add_argument("--build-index", action="store_true")
    parser.add_argument("--hash-input", action="store_true")
    arguments = parser.parse_args()
    config, config_path = load_config(arguments.config)
    apply_batch_defaults(arguments, config)
    work_root = resolve_work_root(arguments, config)
    control_root = Path(config_get(config, "paths", "control_root"))
    instances = arguments.instances or control_root / "shards" / arguments.source / arguments.shard / f"{arguments.batch}.txt"
    metadata_csv = arguments.metadata_csv or control_root / "metadata" / arguments.source / "metadata.csv"
    index_path = (arguments.index_path or default_index_path(work_root)).resolve()
    raw = dict(config_get(config, "raw"))
    if arguments.build_index:
        indexed = build_index(index_path, metadata_csv, raw)
        print(f"indexed {indexed} assets: {index_path}")
    if not index_path.is_file():
        parser.error(f"asset index가 없습니다. 한 번 --build-index로 실행하세요: {index_path}")

    asset_ids = [line.strip() for line in instances.read_text().splitlines() if line.strip()]
    connection = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
    try:
        resolved = {
            asset_id: row for asset_id in asset_ids if (row := connection.execute(
                "SELECT object_id, raw_kind, raw_path, archive_path, archive_member FROM asset_index WHERE asset_id = ?", (asset_id,)
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
        object_id, raw_kind, raw_path, archive_path, archive_member = row
        record = {
            "asset_id": asset_id, "object_id": object_id, "raw_kind": raw_kind,
            "raw_path": raw_path, "archive_path": archive_path, "archive_member": archive_member,
            "status": "ok",
        }
        if arguments.hash_input and raw_kind == "direct":
            record["raw_sha256"] = sha256_file(Path(raw_path))
        records.append(record)
    output = stage_root(work_root, "01", "manifest")
    atomic_write_jsonl(output / "manifest.jsonl", records)
    write_stage_info(
        output, stage="01_manifest", config=str(config_path), source=arguments.source,
        shard=arguments.shard, batch=arguments.batch, instances=str(instances),
        metadata_csv=str(metadata_csv), index_path=str(index_path), count=len(records),
        succeeded=sum(record["status"] == "ok" for record in records),
    )
    print(output / "manifest.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
