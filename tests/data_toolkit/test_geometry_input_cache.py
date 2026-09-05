from data_toolkit.pipeline import geometry_input_cache


def test_load_or_prepare_reuses_matching_source(tmp_path, monkeypatch):
    monkeypatch.setattr(geometry_input_cache, "_CACHE_ROOT", tmp_path / "cache")
    source = tmp_path / "mesh.pickle"
    source.write_bytes(b"source-v1")
    calls = 0

    def prepare():
        nonlocal calls
        calls += 1
        return {"calls": calls}

    first = geometry_input_cache.load_or_prepare(
        "mesh", "asset", source, prepare
    )
    second = geometry_input_cache.load_or_prepare(
        "mesh", "asset", source, prepare
    )

    assert first == {"calls": 1}
    assert second == first
    assert calls == 1


def test_load_or_prepare_invalidates_changed_source(tmp_path, monkeypatch):
    monkeypatch.setattr(geometry_input_cache, "_CACHE_ROOT", tmp_path / "cache")
    source = tmp_path / "mesh.pickle"
    source.write_bytes(b"source-v1")
    calls = 0

    def prepare():
        nonlocal calls
        calls += 1
        return calls

    assert geometry_input_cache.load_or_prepare("mesh", "asset", source, prepare) == 1
    source.write_bytes(b"source-v2-with-a-different-size")
    assert geometry_input_cache.load_or_prepare("mesh", "asset", source, prepare) == 2
