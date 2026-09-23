"""Exercise the real training datasets against locally built family tar files."""

from __future__ import annotations

from collections.abc import Mapping
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import tarfile
from typing import Final, assert_never
from unittest.mock import patch

import numpy as np
from torch.utils.data import DataLoader

from data_toolkit.preprocess._common.finalize_packs import (
    BatchPublication,
    PublicationError,
    StagedPack,
)


TRAINING_CONFIGS: Final[tuple[str, ...]] = (
    "ss_flow_img_dit_1_3B_32_bf16_proj_finetune_ft64.json",
    "slat_flow_img2shape_dit_1_3B_256_bf16_proj_finetune.json",
    "slat_flow_img2shape_dit_1_3B_256_bf16_proj_finetune_ft512.json",
    "slat_flow_img2shape_dit_1_3B_512_bf16_proj_finetune_ft1024.json",
    "slat_flow_imgshape2tex_dit_1_3B_256_bf16_proj_finetune.json",
    "slat_flow_imgshape2tex_dit_1_3B_512_bf16_proj_finetune.json",
    "slat_flow_imgshape2tex_dit_1_3B_512_bf16_proj_finetune_ft1024.json",
)


@dataclass(frozen=True, slots=True)
class TrainingCheck:
    sample_asset: str
    configs_checked: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TrainingAnchor:
    asset: str
    aesthetic_score: float


def _eligible_asset(publication: BatchPublication) -> TrainingAnchor:
    metadata = (
        publication.control_root / "metadata" / publication.identity.source / "metadata.csv"
    )
    included_assets = frozenset(publication.assets)
    try:
        with metadata.open(encoding="utf-8", newline="") as stream:
            scores = {
                row["sha256"]: float(row["aesthetic_score"])
                for row in csv.DictReader(stream)
                if row["sha256"] in included_assets and row["aesthetic_score"]
            }
    except FileNotFoundError as error:
        raise PublicationError(metadata, "training aesthetic metadata is missing") from error
    for asset in publication.assets:
        if scores.get(asset, 0.0) < 4.5:
            continue
        eligible = True
        for resolution, limit in ((256, 8192), (512, 8192), (1024, 32768)):
            shape = publication.work_root / f"05_encode/shape/shape_latents/shape_enc_next_dc_f16c32_fp16_{resolution}_view"
            pbr = publication.work_root / f"05_encode/pbr/pbr_latents/tex_enc_next_dc_f16c32_fp16_{resolution}_view_fix"
            paths = (
                *(shape / asset / f"view{view:02d}.npz" for view in (0, 1)),
                *(pbr / asset / f"view{view:02d}.npz" for view in (0, 1)),
            )
            if any(_token_count(path) > limit for path in paths):
                eligible = False
                break
        if eligible:
            return TrainingAnchor(asset, scores[asset])
    raise PublicationError(metadata, "no asset passes the training aesthetic/token limits")


def _token_count(path: Path) -> int:
    with np.load(path) as payload:
        return len(payload["coords"])


def _extract_sample(
    staged: Mapping[str, StagedPack], sample_asset: str, destination: Path,
) -> None:
    for pack in staged.values():
        with tarfile.open(pack.archive, "r:") as archive:
            selected = [
                member for member in archive
                if sample_asset in Path(member.name).parts
            ]
            for member in selected:
                if not member.isfile():
                    raise PublicationError(pack.archive, "training sample contains non-file tar member")
                target = destination / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                stream = archive.extractfile(member)
                if stream is None:
                    raise PublicationError(pack.archive, f"unreadable tar member: {member.name}")
                with stream, target.open("wb") as output:
                    shutil.copyfileobj(stream, output)


def _latent_tokens(path: Path) -> int:
    with np.load(path) as payload:
        if not np.isfinite(payload["feats"]).all():
            raise PublicationError(path, "non-finite latent features")
        return len(payload["coords"])


def _write_sample_metadata(root: Path, anchor: TrainingAnchor) -> None:
    sample_asset = anchor.asset
    render_root = root / "renders_cond"
    flags = {
        "sha256": sample_asset,
        "aesthetic_score": anchor.aesthetic_score,
        "cond_rendered": True,
        **{f"{family}_latent_view{view:02d}_encoded": True
           for family in ("ss", "shape", "pbr") for view in (0, 1)},
    }
    with (render_root / "metadata.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=flags)
        writer.writeheader()
        writer.writerow(flags)
    for resolution in (256, 512, 1024):
        shape = root / f"shape_latents/shape_enc_next_dc_f16c32_fp16_{resolution}_view"
        pbr = root / f"pbr_latents/tex_enc_next_dc_f16c32_fp16_{resolution}_view_fix"
        shape_tokens = []
        pbr_tokens = []
        for view in (0, 1):
            shape_path = shape / sample_asset / f"view{view:02d}.npz"
            pbr_path = pbr / sample_asset / f"view{view:02d}.npz"
            shape_tokens.append(_latent_tokens(shape_path))
            pbr_tokens.append(_latent_tokens(pbr_path))
            with np.load(shape_path) as shape_data, np.load(pbr_path) as pbr_data:
                if not np.array_equal(shape_data["coords"], pbr_data["coords"]):
                    raise PublicationError(pbr_path, "shape/PBR coordinates differ")
        for directory, column, counts in (
            (shape, "shape_latent_tokens", shape_tokens),
            (pbr, "pbr_latent_tokens", pbr_tokens),
        ):
            with (directory / "metadata.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=("sha256", column))
                writer.writeheader()
                writer.writerow({"sha256": sample_asset, column: max(counts)})


def validate_training_load(
    publication: BatchPublication, staged: Mapping[str, StagedPack], scratch_root: Path,
) -> TrainingCheck:
    """Unpack a qualified sample and read it through seven real training configs."""
    anchor = _eligible_asset(publication)
    _extract_sample(staged, anchor.asset, scratch_root)
    _write_sample_metadata(scratch_root, anchor)
    from pixal3d import datasets

    configuration_root = Path(__file__).resolve().parents[3] / "configs/gen"
    checked = []
    for name in TRAINING_CONFIGS:
        configuration = json.loads((configuration_root / name).read_text(encoding="utf-8"))
        args = configuration["dataset"]["args"]
        resolution = int(args.get("resolution", 256))
        roots = {
            publication.identity.source: {
                "render_cond": str(scratch_root / "renders_cond"),
                "ss_latent": str(scratch_root / "ss_latents/ss_enc_conv3d_16l8_fp16_64_view"),
                "shape_latent": str(scratch_root / f"shape_latents/shape_enc_next_dc_f16c32_fp16_{resolution}_view"),
                "pbr_latent": str(scratch_root / f"pbr_latents/tex_enc_next_dc_f16c32_fp16_{resolution}_view_fix"),
            }
        }
        dataset = getattr(datasets, configuration["dataset"]["name"])(json.dumps(roots), **args)
        if len(dataset) != 1:
            raise PublicationError(scratch_root, f"{name}: training sample filtered out")
        for view in (0, 1):
            # StandardDatasetBase.__getitem__ retries recursively on failure.
            # Read each view directly first so a bad sample fails instead of looping.
            with patch.object(np.random, "randint", return_value=view):
                dataset.get_instance(roots[publication.identity.source], anchor.asset)
        collate = getattr(dataset, "collate_fn", None)
        sample = next(iter(DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=collate)))
        match sample:
            case [batch]:
                pass
            case dict() as batch:
                pass
            case unreachable:
                assert_never(unreachable)
        if batch["_sha256"] != [anchor.asset]:
            raise PublicationError(scratch_root, f"{name}: loader returned a different asset")
        if tuple(batch["cond"].shape) != (1, 3, int(args["image_size"]), int(args["image_size"])):
            raise PublicationError(scratch_root, f"{name}: invalid conditioning image shape")
        checked.append(name)
    return TrainingCheck(anchor.asset, tuple(checked))
