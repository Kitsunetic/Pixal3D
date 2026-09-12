import os
import subprocess
import sys

from data_toolkit.pipeline.inherited_fd import pass_fds_for_path


def test_private_asset_alias_survives_exec(tmp_path):
    source = tmp_path / "source.glb"
    alias = tmp_path / "asset.glb"
    source.write_bytes(b"pinned")
    descriptor = os.open(source, os.O_RDONLY)
    try:
        alias.symlink_to(f"/proc/self/fd/{descriptor}")
        source.unlink()
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; import sys; "
                    "sys.stdout.buffer.write(Path(sys.argv[1]).read_bytes())"
                ),
                str(alias),
            ],
            check=True,
            capture_output=True,
            pass_fds=pass_fds_for_path(alias),
        )
    finally:
        os.close(descriptor)

    assert result.stdout == b"pinned"


def test_normal_asset_path_does_not_inherit_descriptors(tmp_path):
    assert pass_fds_for_path(tmp_path / "asset.glb") == ()
