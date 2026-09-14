"""Compare two prepared Pixal3D publications without extracting their packs."""

from __future__ import annotations

import argparse
import json
import tarfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from .packing import verify_pack
from .validation import ValidationError


class OutputComparisonError(ValueError):
    """Raised when two prepared publications are not output-compatible."""


@dataclass(frozen=True, slots=True)
class OutputComparisonReport:
    pack_count: int
    member_count: int
    alpha_images: int
    alpha_exact_images: int
    latent_arrays: int
    latent_exact_arrays: int

    def with_member(
        self, *, alpha: bool = False, latent_arrays: int = 0
    ) -> OutputComparisonReport:
        return OutputComparisonReport(
            self.pack_count,
            self.member_count + 1,
            self.alpha_images + int(alpha),
            self.alpha_exact_images + int(alpha),
            self.latent_arrays + latent_arrays,
            self.latent_exact_arrays + latent_arrays,
        )


def _prepared_packs(root: Path) -> dict[str, Path]:
    packs = {path.relative_to(root).as_posix(): path for path in root.rglob("*.tar")}
    if not packs:
        raise OutputComparisonError(f"no prepared packs found: {root}")
    return packs


def _member_map(bundle: tarfile.TarFile, label: str) -> dict[str, tarfile.TarInfo]:
    members: dict[str, tarfile.TarInfo] = {}
    for member in bundle:
        if not member.isfile() or member.name in members:
            raise OutputComparisonError(f"invalid {label} tar member: {member.name}")
        members[member.name] = member
    return members


def _member_bytes(bundle: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    stream = bundle.extractfile(member)
    if stream is None:
        raise OutputComparisonError(f"unreadable tar member: {member.name}")
    with stream:
        return stream.read()


def _same_content(
    reference: tarfile.TarFile,
    reference_member: tarfile.TarInfo,
    candidate: tarfile.TarFile,
    candidate_member: tarfile.TarInfo,
) -> bool:
    if reference_member.size != candidate_member.size:
        return False
    reference_stream = reference.extractfile(reference_member)
    candidate_stream = candidate.extractfile(candidate_member)
    if reference_stream is None or candidate_stream is None:
        raise OutputComparisonError(f"unreadable tar member: {reference_member.name}")
    with reference_stream, candidate_stream:
        while True:
            expected = reference_stream.read(8 * 1024 * 1024)
            actual = candidate_stream.read(8 * 1024 * 1024)
            if expected != actual:
                return False
            if not expected:
                return True


def _compare_npz(reference: bytes, candidate: bytes, label: str) -> int:
    try:
        with (
            np.load(BytesIO(reference), allow_pickle=False) as expected,
            np.load(BytesIO(candidate), allow_pickle=False) as actual,
        ):
            if set(expected.files) != set(actual.files):
                raise OutputComparisonError(f"NPZ key mismatch: {label}")
            arrays = 0
            for key in expected.files:
                arrays += 1
                if not np.array_equal(expected[key], actual[key]):
                    raise OutputComparisonError(f"NPZ array mismatch: {label}:{key}")
    except OutputComparisonError:
        raise
    except (OSError, ValueError) as error:
        raise OutputComparisonError(f"invalid NPZ member: {label}") from error
    return arrays


def _compare_alpha(reference: bytes, candidate: bytes, label: str) -> None:
    try:
        with (
            Image.open(BytesIO(reference)) as expected_image,
            Image.open(BytesIO(candidate)) as actual_image,
        ):
            expected = np.asarray(expected_image.convert("RGBA"))
            actual = np.asarray(actual_image.convert("RGBA"))
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise OutputComparisonError(f"invalid render image: {label}") from error
    if expected.shape != actual.shape or not np.array_equal(
        expected[..., 3], actual[..., 3]
    ):
        raise OutputComparisonError(f"alpha mismatch: {label}")


def _compare_pack(
    reference_path: Path, candidate_path: Path, report: OutputComparisonReport
) -> OutputComparisonReport:
    try:
        with (
            tarfile.open(reference_path, mode="r:") as reference,
            tarfile.open(candidate_path, mode="r:") as candidate,
        ):
            reference_members = _member_map(reference, "reference")
            candidate_members = _member_map(candidate, "candidate")
            if set(reference_members) != set(candidate_members):
                raise OutputComparisonError(
                    f"tar member set mismatch: {reference_path.name}"
                )
            for name in sorted(reference_members):
                reference_member = reference_members[name]
                candidate_member = candidate_members[name]
                label = f"{reference_path.name}:{name}"
                if name.endswith(".npz"):
                    arrays = _compare_npz(
                        _member_bytes(reference, reference_member),
                        _member_bytes(candidate, candidate_member),
                        label,
                    )
                    report = report.with_member(latent_arrays=arrays)
                elif name.endswith(".png"):
                    _compare_alpha(
                        _member_bytes(reference, reference_member),
                        _member_bytes(candidate, candidate_member),
                        label,
                    )
                    report = report.with_member(alpha=True)
                else:
                    if not _same_content(
                        reference, reference_member, candidate, candidate_member
                    ):
                        raise OutputComparisonError(f"content mismatch: {label}")
                    report = report.with_member()
    except (OSError, tarfile.TarError) as error:
        raise OutputComparisonError(
            f"invalid prepared pack: {reference_path}"
        ) from error
    return report


def compare_prepared_outputs(
    reference_root: Path, candidate_root: Path
) -> OutputComparisonReport:
    """Require deterministic prepared outputs to match across two pipeline runs."""
    reference_root = Path(reference_root)
    candidate_root = Path(candidate_root)
    reference_packs = _prepared_packs(reference_root)
    candidate_packs = _prepared_packs(candidate_root)
    if set(reference_packs) != set(candidate_packs):
        raise OutputComparisonError("prepared pack set mismatch")
    for path in reference_packs.values():
        try:
            verify_pack(path, path.with_suffix(path.suffix + ".manifest.json"))
        except ValidationError as error:
            raise OutputComparisonError(f"invalid reference pack: {path}") from error
    for path in candidate_packs.values():
        try:
            verify_pack(path, path.with_suffix(path.suffix + ".manifest.json"))
        except ValidationError as error:
            raise OutputComparisonError(f"invalid candidate pack: {path}") from error
    report = OutputComparisonReport(len(reference_packs), 0, 0, 0, 0, 0)
    for relative in sorted(reference_packs):
        report = _compare_pack(
            reference_packs[relative], candidate_packs[relative], report
        )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args(argv)
    try:
        report = compare_prepared_outputs(args.reference, args.candidate)
    except OutputComparisonError as error:
        print(str(error))
        return 1
    print(json.dumps(asdict(report), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
