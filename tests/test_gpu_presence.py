from tools.gpu_presence import parse_device_ids, payload_elements


def test_payload_elements_uses_float32_bytes():
    assert payload_elements(16) == 16 * 1024 * 1024 // 4


def test_parse_device_ids_rejects_duplicate_or_negative_ids():
    assert parse_device_ids("0,2,7") == [0, 2, 7]
    for value in ("0,0", "-1", ""):
        try:
            parse_device_ids(value)
        except ValueError:
            pass
        else:
            raise AssertionError(value)
