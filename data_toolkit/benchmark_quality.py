# pyright: reportMissingImports=false
"""Reproducible quality gate for Pixal3D preprocessing outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

MIN_ALPHA_IOU = 1.0
DEFAULT_MIN_SEEDED_RGB_PSNR = 50.0
DEFAULT_MAX_LATENT_RELATIVE_L2 = 0.002
CAMERA_ATOL = 1e-6
ARTIFACT_SUFFIXES = {".json", ".npz", ".png", ".vxz"}
EXACT_ARRAY_KEYS = {"coords", "coord", "voxel_indices", "intersected"}
Array: TypeAlias = NDArray[np.generic]


def image_psnr(reference: Array, candidate: Array) -> float:
    if reference.shape != candidate.shape:
        raise ValueError(
            f"image shape mismatch: {reference.shape} != {candidate.shape}"
        )
    error = reference.astype(np.float64) - candidate.astype(np.float64)
    mse = float(np.mean(error * error))
    return float("inf") if mse == 0.0 else 20.0 * math.log10(255.0 / math.sqrt(mse))


def _alpha_iou(reference: Array, candidate: Array) -> float:
    reference_mask = np.greater(reference, 0)
    candidate_mask = np.greater(candidate, 0)
    union = np.logical_or(reference_mask, candidate_mask).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(reference_mask, candidate_mask).sum() / union)


def _artifact_paths(root: Path) -> set[Path]:
    return {
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix in ARTIFACT_SUFFIXES
        and "raw" not in path.relative_to(root).parts
    }


def _relative_l2(reference: Array, candidate: Array) -> float:
    expected = reference.astype(np.float64, copy=False)
    actual = candidate.astype(np.float64, copy=False)
    denominator = max(float(np.linalg.norm(expected.ravel())), 1e-12)
    return float(np.linalg.norm((expected - actual).ravel()) / denominator)


def _compare_mapping(
    relative: Path,
    reference: dict[str, Array],
    candidate: dict[str, Array],
    failures: list[str],
    max_latent_relative_l2: float,
) -> list[float]:
    relative_l2: list[float] = []
    if reference.keys() != candidate.keys():
        failures.append(f"{relative}: array keys differ")
        return relative_l2
    for key, expected in reference.items():
        actual = candidate[key]
        label = f"{relative}:{key}"
        if expected.shape != actual.shape or expected.dtype != actual.dtype:
            failures.append(f"{label}: shape or dtype differs")
            continue
        if np.issubdtype(actual.dtype, np.number) and not np.all(np.isfinite(actual)):
            failures.append(f"{label}: non-finite candidate values")
            continue
        if key in EXACT_ARRAY_KEYS or relative.suffix == ".vxz":
            if not np.array_equal(expected, actual):
                failures.append(f"{label}: exact values differ")
            continue
        if not np.issubdtype(actual.dtype, np.number):
            if not np.array_equal(expected, actual):
                failures.append(f"{label}: values differ")
            continue
        value = _relative_l2(expected, actual)
        relative_l2.append(value)
        if value > max_latent_relative_l2:
            failures.append(
                f"{label}: relative L2 {value:.8f} > {max_latent_relative_l2:.8f}"
            )
    return relative_l2


def _read_vxz(path: Path) -> dict[str, Array]:
    import o_voxel

    coords, attributes = cast(
        tuple[torch.Tensor, dict[str, torch.Tensor]],
        cast(object, o_voxel.io.read_vxz(str(path), num_threads=1)),
    )
    arrays: dict[str, Array] = {"coords": coords.cpu().numpy()}
    arrays.update({key: value.cpu().numpy() for key, value in attributes.items()})
    return arrays


def _json_close(reference: Any, candidate: Any) -> bool:
    if isinstance(reference, dict) and isinstance(candidate, dict):
        return reference.keys() == candidate.keys() and all(
            _json_close(reference[key], candidate[key]) for key in reference
        )
    if isinstance(reference, list) and isinstance(candidate, list):
        return len(reference) == len(candidate) and all(
            _json_close(expected, actual)
            for expected, actual in zip(reference, candidate, strict=True)
        )
    if isinstance(reference, (int, float)) and isinstance(candidate, (int, float)):
        return math.isclose(reference, candidate, rel_tol=0.0, abs_tol=CAMERA_ATOL)
    return reference == candidate


def compare_benchmark_outputs(
    reference_root: Path,
    candidate_root: Path,
    *,
    rgb_policy: str = "diagnostic",
    min_rgb_psnr: float = DEFAULT_MIN_SEEDED_RGB_PSNR,
    max_latent_relative_l2: float = DEFAULT_MAX_LATENT_RELATIVE_L2,
) -> dict[str, Any]:
    failures: list[str] = []
    if not reference_root.is_dir():
        failures.append(f"reference root is not a directory: {reference_root}")
    if not candidate_root.is_dir():
        failures.append(f"candidate root is not a directory: {candidate_root}")
    reference_paths = _artifact_paths(reference_root)
    candidate_paths = _artifact_paths(candidate_root)
    if not reference_paths:
        failures.append("reference contains no comparable artifacts")
    if not candidate_paths:
        failures.append("candidate contains no comparable artifacts")
    if reference_paths != candidate_paths:
        missing = sorted(str(path) for path in reference_paths - candidate_paths)
        extra = sorted(str(path) for path in candidate_paths - reference_paths)
        failures.append(f"artifact set differs; missing={missing}, extra={extra}")

    psnrs: list[float] = []
    alpha_ious: list[float] = []
    latent_relative_l2: list[float] = []
    compared_npz = 0
    compared_vxz = 0
    for relative in sorted(reference_paths & candidate_paths):
        expected_path = reference_root / relative
        actual_path = candidate_root / relative
        if relative.suffix == ".png":
            expected = np.asarray(Image.open(expected_path).convert("RGBA"))
            actual = np.asarray(Image.open(actual_path).convert("RGBA"))
            if expected.shape != actual.shape:
                failures.append(f"{relative}: image shape differs")
                continue
            psnrs.append(image_psnr(expected[:, :, :3], actual[:, :, :3]))
            alpha_ious.append(_alpha_iou(expected[:, :, 3], actual[:, :, 3]))
            if not np.array_equal(expected[:, :, 3], actual[:, :, 3]):
                failures.append(f"{relative}: alpha values differ")
        elif relative.suffix == ".npz":
            with (
                np.load(expected_path) as expected_file,
                np.load(actual_path) as actual_file,
            ):
                latent_relative_l2.extend(
                    _compare_mapping(
                        relative,
                        dict(expected_file),
                        dict(actual_file),
                        failures,
                        max_latent_relative_l2,
                    )
                )
            compared_npz += 1
        elif relative.suffix == ".vxz":
            _compare_mapping(
                relative,
                _read_vxz(expected_path),
                _read_vxz(actual_path),
                failures,
                0.0,
            )
            compared_vxz += 1
        else:
            expected_json = json.loads(expected_path.read_text())
            actual_json = json.loads(actual_path.read_text())
            if not _json_close(expected_json, actual_json):
                failures.append(f"{relative}: JSON values differ")

    min_psnr = min(psnrs, default=float("inf"))
    min_iou = min(alpha_ious, default=1.0)
    if rgb_policy == "required" and min_psnr < min_rgb_psnr:
        failures.append(f"minimum RGB PSNR {min_psnr:.4f} < {min_rgb_psnr}")
    if min_iou < MIN_ALPHA_IOU:
        failures.append(f"minimum alpha IoU {min_iou:.6f} < {MIN_ALPHA_IOU}")
    return {
        "passed": not failures,
        "failures": failures,
        "rgb_policy": rgb_policy,
        "min_rgb_psnr": min_psnr,
        "min_alpha_iou": min_iou,
        "max_latent_relative_l2": max(latent_relative_l2, default=0.0),
        "compared_png": len(psnrs),
        "compared_npz": compared_npz,
        "compared_vxz": compared_vxz,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument(
        "--rgb-policy", choices=("diagnostic", "required"), default="diagnostic"
    )
    parser.add_argument("--min-rgb-psnr", type=float, default=50.0)
    parser.add_argument("--max-latent-relative-l2", type=float, default=0.002)
    args = parser.parse_args()
    report = compare_benchmark_outputs(
        args.reference,
        args.candidate,
        rgb_policy=args.rgb_policy,
        min_rgb_psnr=args.min_rgb_psnr,
        max_latent_relative_l2=args.max_latent_relative_l2,
    )
    print(json.dumps(report, allow_nan=True, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
