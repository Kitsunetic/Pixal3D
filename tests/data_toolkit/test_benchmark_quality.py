import json
from pathlib import Path

import numpy as np
from PIL import Image

from data_toolkit.benchmark_quality import compare_benchmark_outputs


def _write_fixture(root: Path, feature: float, rgb: int = 128) -> None:
    image = root / "renders_cond/asset/000.png"
    latent = root / "shape_latents/model/asset/view00.npz"
    transform = root / "renders_cond/asset/transforms.json"
    image.parent.mkdir(parents=True)
    latent.parent.mkdir(parents=True)
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    rgba[2:6, 2:6, :3] = rgb
    rgba[2:6, 2:6, 3] = 255
    Image.fromarray(rgba, mode="RGBA").save(image)
    np.savez_compressed(
        latent,
        coords=np.array([[1, 2, 3]], dtype=np.uint8),
        feats=np.array([[feature, -0.25]], dtype=np.float32),
    )
    transform.write_text(
        json.dumps(
            {
                "render_seed": 444,
                "frames": [
                    {
                        "camera_angle_x": 0.5,
                        "radius": 2.0,
                        "transform_matrix": np.eye(4).tolist(),
                    }
                ],
            }
        )
    )


def test_diagnostic_policy_accepts_bounded_latent_noise_and_reports_rgb(tmp_path):
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_fixture(reference, 0.5, rgb=128)
    _write_fixture(candidate, 0.5005, rgb=64)

    report = compare_benchmark_outputs(reference, candidate)

    assert report["passed"] is True
    assert report["min_rgb_psnr"] < 50
    assert report["min_alpha_iou"] == 1.0
    assert report["max_latent_relative_l2"] < 0.002


def test_seeded_rgb_policy_enforces_per_image_psnr(tmp_path):
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_fixture(reference, 0.5, rgb=128)
    _write_fixture(candidate, 0.5, rgb=64)

    report = compare_benchmark_outputs(
        reference, candidate, rgb_policy="required", min_rgb_psnr=50
    )

    assert report["passed"] is False
    assert any("RGB PSNR" in failure for failure in report["failures"])


def test_coordinate_and_latent_thresholds_are_hard_failures(tmp_path):
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_fixture(reference, 0.5)
    _write_fixture(candidate, 0.6)
    latent = candidate / "shape_latents/model/asset/view00.npz"
    np.savez_compressed(
        latent,
        coords=np.array([[9, 2, 3]], dtype=np.uint8),
        feats=np.array([[0.6, -0.25]], dtype=np.float32),
    )

    report = compare_benchmark_outputs(reference, candidate)

    assert report["passed"] is False
    assert any("coords" in failure for failure in report["failures"])
    assert any("relative L2" in failure for failure in report["failures"])


def test_empty_or_missing_roots_fail_closed(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()

    report = compare_benchmark_outputs(empty, tmp_path / "missing")

    assert report["passed"] is False
    assert any("no comparable artifacts" in item for item in report["failures"])
    assert any("not a directory" in item for item in report["failures"])
