"""Build verified legacy-layout batch packs in a local staging directory."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

from data_toolkit.pipeline.atomic_io import atomic_write_json
from data_toolkit.pipeline.packing import (
    PACK_FAMILIES,
    build_pack,
    verify_pack,
)


FAMILY_DIRECTORIES: Final[Mapping[str, Path]] = {
    "common": Path("common"),
    "SS-64": Path("ss/64"),
    "shape-256": Path("shape/256"),
    "shape-512": Path("shape/512"),
    "shape-1024": Path("shape/1024"),
    "PBR-256": Path("pbr/256"),
    "PBR-512": Path("pbr/512"),
    "PBR-1024": Path("pbr/1024"),
}
_LATENT_LAYOUT: Final[Mapping[str, tuple[str, Path]]] = {
    "SS-64": ("ss", Path("ss_latents/ss_enc_conv3d_16l8_fp16_64_view")),
    "shape-256": ("shape", Path("shape_latents/shape_enc_next_dc_f16c32_fp16_256_view")),
    "shape-512": ("shape", Path("shape_latents/shape_enc_next_dc_f16c32_fp16_512_view")),
    "shape-1024": ("shape", Path("shape_latents/shape_enc_next_dc_f16c32_fp16_1024_view")),
    "PBR-256": ("pbr", Path("pbr_latents/tex_enc_next_dc_f16c32_fp16_256_view_fix")),
    "PBR-512": ("pbr", Path("pbr_latents/tex_enc_next_dc_f16c32_fp16_512_view_fix")),
    "PBR-1024": ("pbr", Path("pbr_latents/tex_enc_next_dc_f16c32_fp16_1024_view_fix")),
}


@dataclass(frozen=True, slots=True)
class BatchIdentity:
    source: str
    shard: str
    batch: str


@dataclass(frozen=True, slots=True)
class BatchPublication:
    identity: BatchIdentity
    work_root: Path
    prepared_root: Path
    control_root: Path
    assets: tuple[str, ...]
    batch_assets: tuple[str, ...]
    config_hash: str
    tool_commit: str


@dataclass(frozen=True, slots=True)
class FamilySource:
    root: Path
    members: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class PublicationError(Exception):
    path: Path
    reason: str

    def __str__(self) -> str:
        return f"{self.path}: {self.reason}"


@dataclass(frozen=True, slots=True)
class StagedPack:
    archive: Path
    manifest: Path
    pack_sha256: str


def pack_relative(identity: BatchIdentity, family: str) -> Path:
    """Return the production-compatible relative tar path in prepared."""
    return FAMILY_DIRECTORIES[family] / identity.source / identity.shard / f"{identity.batch}.tar"


def family_sources(publication: BatchPublication) -> dict[str, FamilySource]:
    """Enumerate only the render and latent members required by the pack contract."""
    render_members = tuple(
        Path("renders_cond") / asset / name
        for asset in publication.assets
        for name in (*(f"{view:03d}.png" for view in range(8)), "transforms.json")
    )
    sources = {
        "common": FamilySource(publication.work_root / "03_render", render_members)
    }
    for family, (stage_name, directory) in _LATENT_LAYOUT.items():
        members = tuple(
            directory / asset / filename
            for asset in publication.assets
            for view in (0, 1)
            for filename in (f"view{view:02d}.npz", f"view{view:02d}_scale.json")
        )
        sources[family] = FamilySource(
            publication.work_root / "05_encode" / stage_name, members
        )
    return sources


def missing_members(sources: Mapping[str, FamilySource]) -> tuple[Path, ...]:
    """List absent input files before any tar is staged."""
    return tuple(
        source.root / member
        for source in sources.values()
        for member in source.members
        if not (source.root / member).is_file()
    )


def build_local_packs(publication: BatchPublication, staging_root: Path) -> dict[str, StagedPack]:
    """Create and verify all eight packs before any prepared write."""
    sources = family_sources(publication)
    absent = missing_members(sources)
    if absent:
        raise PublicationError(absent[0], f"{len(absent)} required pack members are missing")
    staged: dict[str, StagedPack] = {}
    for family in PACK_FAMILIES:
        archive = staging_root / f"{family}.tar"
        source = sources[family]
        manifest = build_pack(
            source.root, list(source.members), archive, publication.identity.shard,
            batch_id=publication.identity.batch, family=family,
            config_hash=publication.config_hash, tool_commit=publication.tool_commit,
            asset_sha256s=publication.batch_assets,
            included_asset_sha256s=publication.assets,
        )
        manifest_path = archive.with_suffix(".tar.manifest.json")
        verify_pack(archive, manifest_path)
        atomic_write_json(
            manifest_path,
            asdict(replace(manifest, validated_at=datetime.now(timezone.utc).isoformat())),
        )
        staged[family] = StagedPack(archive, manifest_path, manifest.pack_sha256)
    return staged
