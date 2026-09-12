from concurrent.futures import ThreadPoolExecutor
import importlib
from hashlib import sha256
import inspect
from io import BytesIO
import json
from pathlib import Path
import stat
import subprocess
import sys
import tarfile
from threading import Barrier
from types import SimpleNamespace
import zipfile

import pandas as pd
import pytest


def test_adapter_download_contract():
    for name in ("ABO", "HSSD", "3D-FUTURE", "Toys4k", "ObjaverseXL"):
        module = importlib.import_module(f"data_toolkit.datasets.{name}")
        parameters = inspect.signature(module.download).parameters
        assert "metadata" in parameters
        assert "output_dir" in parameters


def test_public_metadata_paths(monkeypatch):
    seen = []
    monkeypatch.setattr(
        pd, "read_csv", lambda path: seen.append(path) or pd.DataFrame()
    )
    for name in ("HSSD", "3D-FUTURE", "Toys4k"):
        importlib.import_module(f"data_toolkit.datasets.{name}").get_metadata()
    assert seen == [
        "hf://datasets/JeffreyXiang/TRELLIS-500K/HSSD.csv",
        "hf://datasets/JeffreyXiang/TRELLIS-500K/3D-FUTURE.csv",
        "hf://datasets/JeffreyXiang/TRELLIS-500K/Toys4k.csv",
    ]


def test_hssd_snapshot_is_bounded_and_only_verified_files_are_returned(
    monkeypatch, tmp_path
):
    module = importlib.import_module("data_toolkit.datasets.HSSD")
    good = b"verified"
    calls = []

    monkeypatch.setattr(module.huggingface_hub, "whoami", lambda: {"name": "test"})

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        root = Path(kwargs["local_dir"])
        (root / "models").mkdir(parents=True)
        (root / "models/good.glb").write_bytes(good)
        (root / "models/bad.glb").write_bytes(b"wrong")

    monkeypatch.setattr(
        module.huggingface_hub, "snapshot_download", snapshot_download
    )
    metadata = pd.DataFrame(
        [
            {
                "file_identifier": "models/good.glb",
                "sha256": sha256(good).hexdigest(),
            },
            {"file_identifier": "models/bad.glb", "sha256": "0" * 64},
        ]
    )

    result = module.download(metadata, str(tmp_path), max_workers=64)

    assert calls == [
        {
            "repo_id": "hssd/hssd-models",
            "repo_type": "dataset",
            "allow_patterns": ["models/good.glb", "models/bad.glb"],
            "local_dir": str(tmp_path / "raw"),
            "max_workers": 8,
        }
    ]
    assert result.to_dict("records") == [
        {
            "sha256": sha256(good).hexdigest(),
            "local_path": "raw/models/good.glb",
        }
    ]


def _write_zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, contents in members.items():
            archive.writestr(name, contents)


def test_3d_future_extracts_only_selected_verified_directory(tmp_path):
    module = importlib.import_module("data_toolkit.datasets.3D-FUTURE")
    selected_image = b"selected image"
    mesh = b"mtllib model.mtl\nmesh"
    material = b"map_Kd texture.png\n"
    texture = b"texture"
    archive_path = tmp_path / "3D-FUTURE-model.zip"
    _write_zip(
        archive_path,
        {
            "3D-FUTURE-model/selected/image.jpg": selected_image,
            "3D-FUTURE-model/selected/model.mtl": material,
            "3D-FUTURE-model/selected/raw_model.obj": mesh,
            "3D-FUTURE-model/selected/texture.png": texture,
            "3D-FUTURE-model/ignored/image.jpg": b"ignored",
            "3D-FUTURE-model/ignored/raw_model.obj": b"ignored mesh",
        },
    )
    metadata = pd.DataFrame(
        [
            {
                "file_identifier": "3D-FUTURE-model/selected",
                "sha256": sha256(selected_image).hexdigest(),
            }
        ]
    )

    result = module.download(metadata, str(tmp_path), max_workers=64)

    assert result.to_dict("records") == [
        {
            "sha256": sha256(selected_image).hexdigest(),
            "local_path": "raw/3D-FUTURE-model/selected/raw_model.obj",
            "content_sha256": sha256(mesh).hexdigest(),
            "companion_files": json.dumps(
                {
                    "raw/3D-FUTURE-model/selected/image.jpg": sha256(
                        selected_image
                    ).hexdigest(),
                    "raw/3D-FUTURE-model/selected/model.mtl": sha256(
                        material
                    ).hexdigest(),
                    "raw/3D-FUTURE-model/selected/texture.png": sha256(
                        texture
                    ).hexdigest(),
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
    ]
    assert not (tmp_path / "raw/3D-FUTURE-model/ignored").exists()


@pytest.mark.parametrize("name", ["../escape", "/absolute"])
def test_3d_future_rejects_unsafe_zip_members_before_extraction(tmp_path, name):
    module = importlib.import_module("data_toolkit.datasets.3D-FUTURE")
    _write_zip(
        tmp_path / "3D-FUTURE-model.zip",
        {
            "3D-FUTURE-model/selected/image.jpg": b"selected",
            name: b"unsafe",
        },
    )

    with pytest.raises(ValueError, match="Unsafe ZIP member"):
        module.download(pd.DataFrame(columns=["file_identifier", "sha256"]), tmp_path)

    assert not (tmp_path / "raw/3D-FUTURE-model").exists()


def test_toys4k_rejects_symlink_member_before_extraction(tmp_path):
    module = importlib.import_module("data_toolkit.datasets.Toys4k")
    archive_path = tmp_path / "toys4k_blend_files.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("toys4k_blend_files/selected.blend", b"blend")
        symlink = zipfile.ZipInfo("toys4k_blend_files/link.blend")
        symlink.create_system = 3
        symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(symlink, "selected.blend")

    with pytest.raises(ValueError, match="Unsafe ZIP member"):
        module.download(pd.DataFrame(columns=["file_identifier", "sha256"]), tmp_path)

    assert not (tmp_path / "raw/toys4k_blend_files").exists()


def _write_tar(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w") as archive:
        for name, contents in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(contents)
            archive.addfile(info, BytesIO(contents))


def test_abo_extracts_only_selected_verified_member(tmp_path):
    module = importlib.import_module("data_toolkit.datasets.ABO")
    raw = tmp_path / "raw"
    raw.mkdir()
    selected = b"selected glb"
    _write_tar(
        raw / "abo-3dmodels.tar",
        {
            "3dmodels/original/selected.glb": selected,
            "3dmodels/original/ignored.glb": b"ignored",
        },
    )
    metadata = pd.DataFrame(
        [
            {
                "file_identifier": "selected.glb",
                "sha256": sha256(selected).hexdigest(),
            }
        ]
    )

    result = module.download(metadata, str(tmp_path), max_workers=64)

    assert result.to_dict("records") == [
        {
            "sha256": sha256(selected).hexdigest(),
            "local_path": "raw/3dmodels/original/selected.glb",
        }
    ]
    assert not (raw / "3dmodels/original/ignored.glb").exists()


def test_abo_reuses_verified_local_file_without_opening_archive(
    monkeypatch, tmp_path
):
    module = importlib.import_module("data_toolkit.datasets.ABO")
    contents = b"already downloaded glb"
    relative = "raw/3dmodels/original/existing.glb"
    local = tmp_path / relative
    local.parent.mkdir(parents=True)
    local.write_bytes(contents)
    metadata = pd.DataFrame(
        [
            {
                "file_identifier": "existing.glb",
                "local_path": relative,
                "sha256": sha256(contents).hexdigest(),
            }
        ]
    )
    monkeypatch.setattr(
        module.tarfile,
        "open",
        lambda *_args, **_kwargs: pytest.fail("archive should not be opened"),
    )
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("archive should not be downloaded"),
    )

    result = module.download(metadata, str(tmp_path), max_workers=8)

    assert result.to_dict("records") == [
        {"sha256": sha256(contents).hexdigest(), "local_path": relative}
    ]


@pytest.mark.parametrize("link_type", [tarfile.SYMTYPE, tarfile.LNKTYPE])
def test_abo_rejects_tar_links_before_extraction(tmp_path, link_type):
    module = importlib.import_module("data_toolkit.datasets.ABO")
    raw = tmp_path / "raw"
    raw.mkdir()
    with tarfile.open(raw / "abo-3dmodels.tar", "w") as archive:
        info = tarfile.TarInfo("3dmodels/original/link.glb")
        info.type = link_type
        info.linkname = "../../outside"
        archive.addfile(info)

    with pytest.raises(ValueError, match="Unsafe TAR member"):
        module.download(pd.DataFrame(columns=["file_identifier", "sha256"]), tmp_path)


def test_objaverse_download_processes_are_bounded(monkeypatch, tmp_path):
    module = importlib.import_module("data_toolkit.datasets.ObjaverseXL")
    contents = b"object"
    metadata = pd.DataFrame(
        [
            {
                "file_identifier": "object.glb",
                "sha256": sha256(contents).hexdigest(),
            }
        ]
    )
    monkeypatch.setattr(module.oxl, "get_annotations", lambda: metadata.copy())
    seen = []

    def download_objects(annotations, **kwargs):
        seen.append(kwargs)
        path = tmp_path / "raw/object.glb"
        path.write_bytes(contents)
        return {"object.glb": str(path)}

    monkeypatch.setattr(module.oxl, "download_objects", download_objects)

    module.download(metadata, str(tmp_path), max_workers=64)

    assert seen[0]["processes"] == 8


def test_objaverse_download_reuses_verified_local_glb(monkeypatch, tmp_path):
    module = importlib.import_module("data_toolkit.datasets.ObjaverseXL")
    contents = b"existing objaverse glb"
    relative = "raw/hf-objaverse-v1/glbs/000-000/object.glb"
    local = tmp_path / relative
    local.parent.mkdir(parents=True)
    local.write_bytes(contents)
    digest = sha256(contents).hexdigest()
    metadata = pd.DataFrame(
        [
            {
                "sha256": digest,
                "file_identifier": "object.glb",
                "local_path": relative,
                "content_sha256": digest,
            }
        ]
    )
    monkeypatch.setattr(
        module.oxl,
        "get_annotations",
        lambda: pytest.fail("verified local assets need no remote annotations"),
    )
    monkeypatch.setattr(
        module.oxl,
        "download_objects",
        lambda *_args, **_kwargs: pytest.fail(
            "verified local assets must not be downloaded again"
        ),
    )

    result = module.download(metadata, str(tmp_path), max_workers=8)

    assert result.to_dict("records") == [
        {"sha256": digest, "local_path": relative}
    ]


def test_objaverse_download_uses_asset_sha_when_content_digest_is_missing(
    monkeypatch, tmp_path
):
    module = importlib.import_module("data_toolkit.datasets.ObjaverseXL")
    contents = b"existing objaverse glb"
    digest = sha256(contents).hexdigest()
    relative = "raw/objects/object.glb"
    local = tmp_path / relative
    local.parent.mkdir(parents=True)
    local.write_bytes(contents)
    metadata = pd.DataFrame(
        [
            {
                "sha256": digest,
                "file_identifier": "object.glb",
                "local_path": relative,
                "content_sha256": float("nan"),
            }
        ]
    )
    monkeypatch.setattr(
        module.oxl,
        "get_annotations",
        lambda: pytest.fail("a verified local asset needs no remote lookup"),
    )

    result = module.download(metadata, str(tmp_path), max_workers=1)

    assert result.to_dict("records") == [
        {"sha256": digest, "local_path": relative}
    ]


@pytest.mark.parametrize(
    "unsafe_path", ["../outside.glb", "/tmp/outside.glb", "raw\\outside.glb"]
)
def test_objaverse_instance_rejects_unsafe_metadata_paths(tmp_path, unsafe_path):
    module = importlib.import_module("data_toolkit.datasets.ObjaverseXL")

    result = module._process_instance(
        (
            {"sha256": "a" * 64, "local_path": unsafe_path},
            str(tmp_path),
            lambda *_args: pytest.fail("unsafe paths must not reach the worker"),
        )
    )

    assert result is None


def test_objaverse_instance_rejects_unsafe_zip_members(tmp_path):
    module = importlib.import_module("data_toolkit.datasets.ObjaverseXL")
    archive = tmp_path / "raw/github/repos/example/repository.zip"
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("models/object.glb", b"valid")
        bundle.writestr("../escaped.glb", b"unsafe")

    result = module._process_instance(
        (
            {
                "sha256": "a" * 64,
                "local_path": (
                    "raw/github/repos/example/repository.zip/models/object.glb"
                ),
            },
            str(tmp_path),
            lambda *_args: pytest.fail("unsafe archives must not be consumed"),
        )
    )

    assert result is None


def test_objaverse_instance_rejects_zip_symlinks(tmp_path):
    module = importlib.import_module("data_toolkit.datasets.ObjaverseXL")
    archive = tmp_path / "raw/github/repos/example/repository.zip"
    archive.parent.mkdir(parents=True)
    link = zipfile.ZipInfo("models/link.glb")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(link, "../../outside.glb")

    result = module._process_instance(
        (
            {
                "sha256": "a" * 64,
                "local_path": (
                    "raw/github/repos/example/repository.zip/models/link.glb"
                ),
            },
            str(tmp_path),
            lambda *_args: pytest.fail("zip links must not be consumed"),
        )
    )

    assert result is None


def test_objaverse_instance_rejects_duplicate_zip_members(tmp_path):
    module = importlib.import_module("data_toolkit.datasets.ObjaverseXL")
    archive = tmp_path / "raw/github/repos/example/repository.zip"
    archive.parent.mkdir(parents=True)
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("models/object.glb", b"first")
            bundle.writestr("models/object.glb", b"second")

    result = module._process_instance(
        (
            {
                "sha256": "a" * 64,
                "local_path": (
                    "raw/github/repos/example/repository.zip/models/object.glb"
                ),
            },
            str(tmp_path),
            lambda *_args: pytest.fail("ambiguous archives must not be consumed"),
        )
    )

    assert result is None


def test_objaverse_instance_streams_only_selected_regular_zip_member(tmp_path):
    module = importlib.import_module("data_toolkit.datasets.ObjaverseXL")
    archive = tmp_path / "raw/github/repos/example/repository.zip"
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("models/object.glb", b"selected")
        bundle.writestr("models/other.glb", b"other")

    result = module._process_instance(
        (
            {
                "sha256": "a" * 64,
                "local_path": (
                    "raw/github/repos/example/repository.zip/models/object.glb"
                ),
            },
            str(tmp_path),
            lambda path, digest: (Path(path).read_bytes(), digest),
        )
    )

    assert result == (b"selected", "a" * 64)


def test_objaverse_instance_pins_direct_file_before_callback(tmp_path):
    module = importlib.import_module("data_toolkit.datasets.ObjaverseXL")
    local = tmp_path / "raw/objects/object.glb"
    replacement = tmp_path / "replacement.glb"
    local.parent.mkdir(parents=True)
    local.write_bytes(b"verified")
    replacement.write_bytes(b"replacement")

    def replace_then_read(path, _digest):
        local.unlink()
        local.symlink_to(replacement)
        return Path(path).read_bytes()

    result = module._process_instance(
        (
            {
                "sha256": "a" * 64,
                "local_path": "raw/objects/object.glb",
            },
            str(tmp_path),
            replace_then_read,
        )
    )

    assert result == b"verified"


@pytest.mark.parametrize("mode", ["outside", "symlink", "digest"])
def test_objaverse_download_rejects_unverified_dependency_paths(
    monkeypatch, tmp_path, mode
):
    module = importlib.import_module("data_toolkit.datasets.ObjaverseXL")
    expected = sha256(b"expected").hexdigest()
    metadata = pd.DataFrame(
        [{"sha256": expected, "file_identifier": "object.glb"}]
    )
    monkeypatch.setattr(module.oxl, "get_annotations", lambda: metadata.copy())
    raw = tmp_path / "raw"
    raw.mkdir()
    if mode == "outside":
        returned = tmp_path.parent / "outside.glb"
        returned.write_bytes(b"expected")
    elif mode == "symlink":
        target = raw / "target.glb"
        target.write_bytes(b"expected")
        returned = raw / "object.glb"
        returned.symlink_to(target)
    else:
        returned = raw / "object.glb"
        returned.write_bytes(b"wrong")
    monkeypatch.setattr(
        module.oxl,
        "download_objects",
        lambda *_args, **_kwargs: {"object.glb": str(returned)},
    )

    with pytest.raises(ValueError, match="downloaded Objaverse asset"):
        module.download(metadata, str(tmp_path), max_workers=1)


def test_download_wrapper_maps_alias_and_merges_records(monkeypatch, tmp_path):
    module = importlib.import_module("data_toolkit.download")
    root = tmp_path / "source"
    root.mkdir()
    pd.DataFrame(
        [
            {"sha256": "a" * 64, "file_identifier": "a.glb"},
            {"sha256": "b" * 64, "file_identifier": "b.glb"},
        ]
    ).to_csv(root / "metadata.csv", index=False)
    calls = []

    def add_args(parser):
        parser.add_argument("--source", default="sketchfab")

    def download(metadata, output_dir, **kwargs):
        calls.append((metadata.copy(), output_dir, kwargs))
        return metadata[["sha256"]].assign(
            local_path=lambda frame: "raw/" + frame["sha256"] + ".glb"
        )

    adapter = SimpleNamespace(add_args=add_args, download=download)
    imported = []

    def import_adapter(name):
        imported.append(name)
        return adapter

    monkeypatch.setattr(module, "_import_adapter", import_adapter)
    for rank in (0, 1):
        module.main(
            [
                "ObjaverseXL_github",
                "--root",
                str(root),
                "--rank",
                str(rank),
                "--world_size",
                "2",
                "--max_workers",
                "5",
                "--record_prefix",
                "batch000_",
            ]
        )

    assert imported == ["ObjaverseXL", "ObjaverseXL"]
    assert [call[2]["source"] for call in calls] == ["github", "github"]
    assert [call[2]["max_workers"] for call in calls] == [5, 5]
    merged = pd.read_csv(root / "raw/metadata.csv")
    assert merged["sha256"].tolist() == ["a" * 64, "b" * 64]
    assert sorted(path.name for path in (root / "raw/new_records").iterdir()) == [
        "part_batch000_0.csv",
        "part_batch000_1.csv",
    ]
    assert not (root / "raw/metadata.csv.tmp").exists()


@pytest.mark.parametrize(
    "invocation",
    [
        ["data_toolkit/download.py"],
        ["-m", "data_toolkit.download"],
    ],
)
def test_download_adapter_imports_work_in_script_and_module_modes(invocation):
    repository = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [sys.executable, *invocation, "ABO", "--help"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_merge_preserves_existing_rows_and_updates_non_null_fields(tmp_path):
    module = importlib.import_module("data_toolkit.download")
    raw = tmp_path / "raw"
    new_records = raw / "new_records"
    new_records.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "sha256": "a" * 64,
                "local_path": "raw/existing.glb",
                "provenance": "keep",
            },
            {
                "sha256": "b" * 64,
                "local_path": "raw/old.glb",
                "provenance": "retain",
            },
        ]
    ).to_csv(raw / "metadata.csv", index=False)
    pd.DataFrame(
        [
            {"sha256": "b" * 64, "local_path": "raw/new.glb"},
            {"sha256": "c" * 64, "local_path": "raw/added.glb"},
        ]
    ).to_csv(new_records / "part_0.csv", index=False)

    module._merge_download_records(tmp_path)

    merged = pd.read_csv(raw / "metadata.csv").set_index("sha256")
    assert merged.loc["a" * 64].to_dict() == {
        "local_path": "raw/existing.glb",
        "provenance": "keep",
    }
    assert merged.loc["b" * 64].to_dict() == {
        "local_path": "raw/new.glb",
        "provenance": "retain",
    }
    assert merged.loc["c" * 64, "local_path"] == "raw/added.glb"


def test_parallel_rank_publication_and_merge_has_no_lost_rows(tmp_path):
    module = importlib.import_module("data_toolkit.download")
    raw = tmp_path / "raw"
    (raw / "new_records").mkdir(parents=True)
    pd.DataFrame(
        [{"sha256": "f" * 64, "local_path": "raw/existing.glb"}]
    ).to_csv(raw / "metadata.csv", index=False)
    worker_count = 8
    published = Barrier(worker_count)

    def publish_and_merge(rank):
        frame = pd.DataFrame(
            [
                {
                    "sha256": f"{rank:064x}",
                    "local_path": f"raw/{rank}.glb",
                }
            ]
        )
        module._publish_download_records(tmp_path, rank, frame)
        published.wait()
        module._merge_download_records(tmp_path)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        list(executor.map(publish_and_merge, range(worker_count)))

    merged = pd.read_csv(raw / "metadata.csv")
    assert merged["sha256"].tolist() == [
        *(f"{rank:064x}" for rank in range(worker_count)),
        "f" * 64,
    ]
    assert not [
        path for path in raw.rglob("*") if path.is_file() and ".tmp" in path.name
    ]
