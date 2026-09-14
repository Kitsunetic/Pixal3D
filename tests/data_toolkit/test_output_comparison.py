from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from data_toolkit.pipeline.output_comparison import (
    OutputComparisonError,
    compare_prepared_outputs,
)
from data_toolkit.pipeline.packing import build_pack

ASSET = "a" * 64


def _write_prepared_output(
    prepared: Path, *, alpha: int = 255, latent: int = 1, red: int = 0
) -> None:
    source = prepared / "source"
    render = source / "renders_cond" / ASSET
    render.mkdir(parents=True)
    Image.fromarray(np.array([[[red, 0, 0, alpha]]], dtype=np.uint8), "RGBA").save(
        render / "000.png"
    )
    (render / "transforms.json").write_text('{"frames":[{"radius":1.0}]}')
    latent_path = source / "shape_latents" / ASSET / "view00.npz"
    latent_path.parent.mkdir(parents=True)
    np.savez_compressed(latent_path, feats=np.array([latent], dtype=np.float32))
    members = [path.relative_to(source) for path in source.rglob("*") if path.is_file()]
    build_pack(
        source,
        members,
        prepared / "common" / "Synthetic" / "Synthetic-00000" / "batch000.tar",
        "Synthetic-00000",
        batch_id="batch000",
        family="common",
        config_hash="b" * 64,
        tool_commit="c" * 40,
        asset_sha256s=(ASSET,),
        completed_count=1,
        quarantined_count=0,
        gate="smoke",
    )


def test_compare_prepared_outputs_accepts_stochastic_rgb(tmp_path: Path):
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_prepared_output(reference, red=0)
    _write_prepared_output(candidate, red=255)

    report = compare_prepared_outputs(reference, candidate)

    assert report.pack_count == 1
    assert report.member_count == 3
    assert report.alpha_images == 1
    assert report.alpha_exact_images == 1
    assert report.latent_arrays == 1
    assert report.latent_exact_arrays == 1


def test_compare_prepared_outputs_rejects_alpha_difference(tmp_path: Path):
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_prepared_output(reference)
    _write_prepared_output(candidate, alpha=0)

    with pytest.raises(OutputComparisonError, match="alpha mismatch"):
        compare_prepared_outputs(reference, candidate)


def test_compare_prepared_outputs_rejects_latent_difference(tmp_path: Path):
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_prepared_output(reference)
    _write_prepared_output(candidate, latent=2)

    with pytest.raises(OutputComparisonError, match="NPZ array mismatch"):
        compare_prepared_outputs(reference, candidate)
