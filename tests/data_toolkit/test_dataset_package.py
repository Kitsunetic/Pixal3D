from importlib.machinery import PathFinder
from pathlib import Path


def test_local_dataset_adapters_are_not_shadowed_by_installed_package(
    tmp_path: Path,
) -> None:
    # Given an unrelated regular package with the same top-level name.
    external_package = tmp_path / "datasets"
    external_package.mkdir()
    (external_package / "__init__.py").write_text("ORIGIN = 'external'\n")
    toolkit_root = Path(__file__).resolve().parents[2] / "data_toolkit"

    # When Python resolves the package using the script's search path.
    specification = PathFinder.find_spec(
        "datasets", [str(toolkit_root), str(tmp_path)]
    )

    # Then the local adapters must take precedence over site packages.
    assert specification is not None
    assert specification.origin == str(
        toolkit_root / "datasets" / "__init__.py"
    )
