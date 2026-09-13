from pathlib import Path


def test_production_overlay_exports_baked_runtime_identity() -> None:
    repository = Path(__file__).resolve().parents[2]
    dockerfile = (
        repository / "docker/production-fast-overlay.Dockerfile"
    ).read_text()

    assert "ENV PYTHONPATH=/root/dev/Pixal3D-fast" in dockerfile
    assert "ENV PIXAL3D_TOOL_COMMIT=${PIXAL3D_RUNTIME_COMMIT}" in dockerfile
