"""Publish locally validated batch packs into the shared prepared index."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import TypedDict

from data_toolkit.pipeline.atomic_io import atomic_write_json
from data_toolkit.pipeline.packing import PACK_FAMILIES, file_sha
from data_toolkit.preprocess._common.finalize_packs import (
    BatchIdentity,
    BatchPublication,
    PublicationError,
    StagedPack,
    pack_relative,
)


class IndexEntry(TypedDict):
    pack: str
    pack_sha256: str
    manifest: str
    manifest_sha256: str


class ShardIndex(TypedDict):
    gate: str
    source: str
    shard_id: str
    batches: dict[str, dict[str, IndexEntry]]


@contextmanager
def _shard_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _read_index(path: Path, identity: BatchIdentity) -> ShardIndex:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {
            "gate": "production", "source": identity.source,
            "shard_id": identity.shard, "batches": {},
        }
    except json.JSONDecodeError as error:
        raise PublicationError(path, f"invalid shard index JSON: {error}") from error
    if (
        not isinstance(raw, dict)
        or raw.get("gate") != "production"
        or raw.get("source") != identity.source
        or raw.get("shard_id") != identity.shard
        or not isinstance(raw.get("batches"), dict)
    ):
        raise PublicationError(path, "invalid shard index identity")
    batches: dict[str, dict[str, IndexEntry]] = {}
    for batch, entries in raw["batches"].items():
        if not isinstance(batch, str) or not isinstance(entries, dict) or set(entries) != set(PACK_FAMILIES):
            raise PublicationError(path, "incomplete shard index batch")
        parsed: dict[str, IndexEntry] = {}
        for family, entry in entries.items():
            if not isinstance(entry, dict) or set(entry) != set(IndexEntry.__annotations__):
                raise PublicationError(path, "invalid shard index entry")
            expected_pack = pack_relative(BatchIdentity(identity.source, identity.shard, batch), family)
            pack_name = entry.get("pack")
            manifest_name = entry.get("manifest")
            pack_sha256 = entry.get("pack_sha256")
            manifest_sha256 = entry.get("manifest_sha256")
            if (
                not isinstance(pack_name, str)
                or pack_name != expected_pack.as_posix()
                or not isinstance(manifest_name, str)
                or manifest_name != expected_pack.with_suffix(".tar.manifest.json").as_posix()
                or not isinstance(pack_sha256, str)
                or not isinstance(manifest_sha256, str)
            ):
                raise PublicationError(path, "non-canonical shard index entry")
            parsed[family] = {
                "pack": pack_name,
                "pack_sha256": pack_sha256,
                "manifest": manifest_name,
                "manifest_sha256": manifest_sha256,
            }
        batches[batch] = parsed
    return {
        "gate": "production", "source": identity.source,
        "shard_id": identity.shard, "batches": batches,
    }


def _copy_verified(source: Path, destination: Path, expected_sha256: str) -> None:
    with source.open("rb") as source_stream, destination.open("xb") as target_stream:
        shutil.copyfileobj(source_stream, target_stream, length=8 * 1024 * 1024)
        target_stream.flush()
        os.fsync(target_stream.fileno())
    if file_sha(destination) != expected_sha256:
        raise PublicationError(destination, "NS2 staged copy checksum mismatch")


def publish_staged(
    publication: BatchPublication,
    staged: Mapping[str, StagedPack],
) -> dict[str, IndexEntry]:
    """Copy verified local packs to NS2 and append one index entry last."""
    identity = publication.identity
    for component in (identity.source, identity.shard, identity.batch):
        if component in ("", ".", "..") or Path(component).name != component or "\\" in component:
            raise PublicationError(publication.prepared_root, f"unsafe path component: {component}")
    if set(staged) != set(PACK_FAMILIES):
        raise PublicationError(publication.prepared_root, "incomplete staged pack families")
    publication.prepared_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{identity.batch}.", dir=publication.prepared_root,
    ) as temporary:
        ns2_staging = Path(temporary)
        for family in PACK_FAMILIES:
            pack = staged[family]
            _copy_verified(pack.archive, ns2_staging / f"{family}.tar", pack.pack_sha256)
            _copy_verified(
                pack.manifest,
                ns2_staging / f"{family}.tar.manifest.json",
                file_sha(pack.manifest),
            )

        index_path = publication.prepared_root / "index" / identity.source / f"{identity.shard}.json"
        lock = publication.prepared_root / ".locks" / identity.source / f"{identity.shard}.lock"
        with _shard_lock(lock):
            index = _read_index(index_path, identity)
            if identity.batch in index["batches"]:
                raise PublicationError(index_path, "batch is already published")
            for family in PACK_FAMILIES:
                destination = publication.prepared_root / pack_relative(identity, family)
                if destination.exists() or destination.with_suffix(".tar.manifest.json").exists():
                    raise PublicationError(destination, "unindexed pack already exists")
            entries: dict[str, IndexEntry] = {}
            for family in PACK_FAMILIES:
                relative = pack_relative(identity, family)
                destination = publication.prepared_root / relative
                destination_manifest = destination.with_suffix(".tar.manifest.json")
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(ns2_staging / f"{family}.tar", destination)
                os.replace(ns2_staging / f"{family}.tar.manifest.json", destination_manifest)
                entries[family] = {
                    "pack": relative.as_posix(),
                    "pack_sha256": staged[family].pack_sha256,
                    "manifest": relative.with_suffix(".tar.manifest.json").as_posix(),
                    "manifest_sha256": file_sha(destination_manifest),
                }
            index["batches"][identity.batch] = entries
            index["batches"] = {
                batch: index["batches"][batch] for batch in sorted(index["batches"])
            }
            atomic_write_json(index_path, index)
            return entries
