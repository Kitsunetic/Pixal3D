"""배치 단위 전처리 단계의 공통 I/O와 경로 규약."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import tempfile
from contextlib import contextmanager
from typing import Any, Iterator, Iterable, Mapping

import yaml


DEFAULT_SOURCE = "ObjaverseXL_sketchfab"
DEFAULT_SHARD = "ObjaverseXL_sketchfab-00000"
DEFAULT_BATCH = "batch000"
DEFAULT_WORK_ROOT = Path("data/preprocess_v2")
DEFAULT_PREPARED_ROOT = Path("/home/rvi/ns2/youngwoo/pixal3d/prepared-v2")


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_config_path() -> Path:
    return repository_root() / "data_toolkit/preprocess/config/default.yaml"


def load_config(path: Path | None) -> tuple[dict[str, Any], Path]:
    config_path = (path or default_config_path()).resolve()
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"preprocess config must be a mapping: {config_path}")
    return config, config_path


def config_get(config: Mapping[str, Any], *keys: str) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise KeyError(".".join(keys))
        value = value[key]
    return value


def require_single_visible_cuda_device() -> str:
    """GPU 단계는 launcher가 노출한 정확히 하나의 CUDA device만 사용한다."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    devices = [device.strip() for device in visible.split(",") if device.strip()]
    if len(devices) != 1 or devices[0] == "-1":
        raise RuntimeError(
            "GPU 단계는 CUDA_VISIBLE_DEVICES로 정확히 하나의 device를 지정해야 합니다 "
            "(예: CUDA_VISIBLE_DEVICES=5)."
        )
    return visible


def default_batch_root(
    source: str = DEFAULT_SOURCE,
    shard: str = DEFAULT_SHARD,
    batch: str = DEFAULT_BATCH,
) -> Path:
    return DEFAULT_WORK_ROOT / source / shard / batch


def add_config_argument(parser: Any) -> None:
    parser.add_argument("--config", type=Path, default=default_config_path())


def add_batch_location_arguments(parser: Any) -> None:
    parser.add_argument("--source", default=None)
    parser.add_argument("--shard", default=None)
    parser.add_argument("--batch", default=None)
    parser.add_argument(
        "--work-root",
        type=Path,
        default=None,
        help="배치의 로컬 작업 루트. 기본값: ./data/preprocess_v2/<source>/<shard>/<batch>",
    )


def apply_batch_defaults(arguments: Any, config: Mapping[str, Any]) -> None:
    arguments.source = arguments.source or config_get(config, "dataset", "source")
    arguments.shard = arguments.shard or config_get(config, "batch", "shard")
    arguments.batch = arguments.batch or config_get(config, "batch", "name")


def resolve_work_root(arguments: Any, config: Mapping[str, Any]) -> Path:
    if arguments.work_root is not None:
        return Path(arguments.work_root).resolve()
    return (
        Path(config_get(config, "paths", "work_root"))
        / arguments.source / arguments.shard / arguments.batch
    ).resolve()


def stage_root(work_root: Path, number: str, name: str) -> Path:
    return work_root / f"{number}_{name}"


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(),
    )


def atomic_write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    payload = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    ).encode()
    atomic_write_bytes(path, payload)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def atomic_pickle_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_stage_info(stage_directory: Path, **values: Any) -> None:
    atomic_write_json(stage_directory / "stage.json", values)


def write_legacy_metadata(path: Path, records: Iterable[Mapping[str, Any]], flag: str) -> None:
    """Legacy voxel scripts require a compact ``metadata.csv`` surface."""
    rows = [{"sha256": str(record["asset_id"]), flag: True} for record in records]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", dir=path.parent, prefix=f".{path.name}.", newline="", delete=False
        ) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=("sha256", flag))
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def successful(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(record) for record in records if record.get("status") == "ok"]


@contextmanager
def materialize_glb(record: Mapping[str, Any], scratch_root: Path, archive_binary: str) -> Iterator[Path]:
    """Yield a direct GLB, extracting one archive member into temporary local storage if needed."""
    kind = record.get("raw_kind", "direct")
    if kind == "direct":
        yield Path(str(record["raw_path"]))
        return
    if kind != "archive_7z":
        raise ValueError(f"unsupported raw kind: {kind!r}")
    archive_path = Path(str(record["archive_path"]))
    member = str(record["archive_member"])
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch_root, prefix=f"{record['asset_id'][:12]}.") as temporary:
        subprocess.run(
            [archive_binary, "x", "-y", f"-o{temporary}", str(archive_path), member],
            stdout=subprocess.DEVNULL,
            check=True,
        )
        extracted = Path(temporary) / member
        if not extracted.is_file():
            raise RuntimeError(f"archive extraction did not create {member}")
        yield extracted
