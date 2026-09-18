"""배치 단위 전처리 단계의 공통 I/O와 경로 규약."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import tempfile
from typing import Any, Iterable, Mapping

import yaml


DEFAULT_SOURCE = "ObjaverseXL_sketchfab"
DEFAULT_SHARD = "ObjaverseXL_sketchfab-00000"
DEFAULT_BATCH = "batch000"
DEFAULT_WORK_ROOT = Path("/home/rvi/ns3/youngwoo/pixal3d/preprocess_v2")
DEFAULT_PREPARED_ROOT = Path("/home/rvi/ns3/youngwoo/pixal3d/prepared-v2")
PREPARED_V2_COMPLETION_FILE = "completion.json"
LEGACY_PREPARED_FAMILIES = frozenset(
    {
        "common",
        "SS-64",
        "shape-256",
        "shape-512",
        "shape-1024",
        "PBR-256",
        "PBR-512",
        "PBR-1024",
    }
)


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


def add_rank_arguments(parser: Any) -> None:
    """Require a deterministic global batch shard for every stage invocation."""
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--rank", type=int, required=True)


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


def batch_index(config: Mapping[str, Any], source: str, shard: str, batch: str) -> int:
    """Return the stable index of a control batch without persisting scheduler state."""
    control_root = Path(config_get(config, "paths", "control_root"))
    shard_root = control_root / "shards" / source
    paths = sorted(shard_root.glob("*/batch*.txt"), key=lambda path: path.relative_to(shard_root).as_posix())
    target = shard_root / shard / f"{batch}.txt"
    try:
        return paths.index(target)
    except ValueError as error:
        raise ValueError(f"control batch를 찾을 수 없습니다: {target}") from error


def require_batch_ownership(arguments: Any, config: Mapping[str, Any]) -> int:
    """Validate ``batch_index % world_size == rank`` and return the batch index."""
    if arguments.world_size <= 0:
        raise ValueError("--world-size는 양수여야 합니다")
    if not 0 <= arguments.rank < arguments.world_size:
        raise ValueError("--rank는 0 이상 --world-size 미만이어야 합니다")
    index = batch_index(config, arguments.source, arguments.shard, arguments.batch)
    owner = index % arguments.world_size
    if owner != arguments.rank:
        raise ValueError(
            f"{arguments.source}/{arguments.shard}/{arguments.batch}의 batch_index={index}는 "
            f"world_size={arguments.world_size}에서 rank={owner}의 소유입니다"
        )
    return index


def legacy_prepared_batch_complete(
    config: Mapping[str, Any], source: str, shard: str, batch: str
) -> bool:
    """Return whether the legacy production prepared index proves a full batch pack."""
    root = Path(config_get(config, "paths", "existing_prepared_root"))
    index_path = root / "index" / source / f"{shard}.json"
    if not index_path.is_file():
        return False
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid legacy prepared index: {index_path}: {error}") from error
    if (
        index.get("gate") != "production"
        or index.get("source") != source
        or index.get("shard_id") != shard
    ):
        return False
    entries = index.get("batches", {}).get(batch)
    return isinstance(entries, Mapping) and set(entries) == LEGACY_PREPARED_FAMILIES


def prepared_v2_batch_complete(
    config: Mapping[str, Any], source: str, shard: str, batch: str,
    prepared_root: Path | None = None,
) -> bool:
    """Return whether an atomically published v2 batch has its completion marker."""
    root = prepared_root or Path(config_get(config, "paths", "prepared_root"))
    target = root / source / shard / batch
    marker = target / PREPARED_V2_COMPLETION_FILE
    if not marker.is_file():
        return False
    try:
        completion = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid prepared-v2 completion marker: {marker}: {error}") from error
    return (
        completion.get("schema") == "pixal3d-preprocess-v2-completion-v1"
        and completion.get("source") == source
        and completion.get("shard") == shard
        and completion.get("batch") == batch
        and isinstance(completion.get("assets"), int)
        and all((target / name).is_dir() for name in ("shape", "pbr", "ss"))
    )


def completed_batch_reason(
    config: Mapping[str, Any], source: str, shard: str, batch: str,
    prepared_root: Path | None = None,
) -> str | None:
    if legacy_prepared_batch_complete(config, source, shard, batch):
        return "legacy_prepared"
    if prepared_v2_batch_complete(config, source, shard, batch, prepared_root):
        return "prepared_v2"
    return None


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


def direct_glb_path(record: Mapping[str, Any]) -> Path:
    """Return the already-extracted direct GLB referenced by a manifest record."""
    kind = record.get("raw_kind", "direct")
    if kind != "direct":
        raise ValueError(f"unsupported raw kind: {kind!r}; direct GLB만 지원합니다")
    path = Path(str(record["raw_path"]))
    if not path.is_file():
        raise FileNotFoundError(f"direct GLB를 찾을 수 없습니다: {path}")
    return path
