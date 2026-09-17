from contextlib import contextmanager

import pytest
import torch

import data_toolkit.encode_ss_latent as legacy_ss
import data_toolkit.pipeline.encoder_preprocessing as preprocessing
from data_toolkit.encode_ss_latent_view import _dense_ss_batch
from data_toolkit.encode_ss_latent import _dense_ss_volume


def _old_dense_batch(
    coordinate_sets: list[torch.Tensor], resolution: int
) -> torch.Tensor:
    """Mirror the former per-sample CPU densification contract."""
    samples = []
    for coordinates in coordinate_sets:
        coordinates = coordinates.long()
        sample = torch.zeros(
            1,
            resolution,
            resolution,
            resolution,
            dtype=torch.long,
        )
        sample[:, coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]] = 1
        samples.append(sample)
    return torch.stack(samples, dim=0).float()


def test_dense_ss_preprocessing_exposes_allocation_transfer_and_scatter(monkeypatch):
    # Given: two coordinate sets and a profiler replacement that records sections.
    stages = []

    @contextmanager
    def record_stage(name, device=None):
        stages.append((name, torch.device(device) if device is not None else None))
        yield

    monkeypatch.setattr(preprocessing, "profile_stage", record_stage)
    coordinate_sets = [
        torch.tensor([[0, 1, 2]], dtype=torch.long),
        torch.tensor([[3, 2, 1]], dtype=torch.long),
    ]

    # When: SS occupancy is constructed on its target device.
    _dense_ss_batch(coordinate_sets, resolution=4, device=torch.device("cpu"))

    # Then: allocation, each transfer, and each scatter have separate timings.
    assert [name for name, _ in stages] == [
        "ss.volume.allocate",
        "ss.coordinates.transfer",
        "ss.volume.scatter",
        "ss.coordinates.transfer",
        "ss.volume.scatter",
    ]


def test_dense_ss_batch_matches_legacy_occupancy_for_multiple_and_empty_sets():
    # Given: sparse coordinate sets accepted by the latent validator.
    coordinate_sets = [
        torch.tensor([[0, 1, 2], [3, 2, 1], [3, 2, 1]], dtype=torch.uint8),
        torch.empty((0, 3), dtype=torch.uint8),
        torch.tensor([[1, 1, 1]], dtype=torch.uint8),
    ]

    # When: the batch is densified on the encoder device.
    actual = _dense_ss_batch(coordinate_sets, resolution=4, device=torch.device("cpu"))

    # Then: occupancy, shape, dtype, and device retain the legacy contract.
    assert actual.shape == (3, 1, 4, 4, 4)
    assert actual.dtype == torch.float32
    assert actual.device.type == "cpu"
    assert torch.equal(actual, _old_dense_batch(coordinate_sets, resolution=4))


def test_legacy_dense_ss_volume_matches_former_loader_and_encoder_input():
    # Given: sparse coordinates loaded by the legacy encoder's worker thread.
    coordinates = torch.tensor(
        [[0, 1, 2], [3, 2, 1], [3, 2, 1]], dtype=torch.long
    )

    # When: the encoder input is densified on its CPU device.
    actual = _dense_ss_volume(
        coordinates, resolution=4, device=torch.device("cpu")
    )

    # Then: it retains the former one-item batch occupancy contract.
    expected = _old_dense_batch([coordinates], resolution=4)
    assert actual.shape == (1, 1, 4, 4, 4)
    assert actual.dtype == torch.float32
    assert actual.device.type == "cpu"
    assert torch.equal(actual, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_dense_ss_batch_places_cpu_coordinates_and_occupancy_on_cuda():
    # Given: CPU-loaded sparse coordinates and a CUDA encoder device.
    coordinate_sets = [
        torch.tensor([[0, 0, 0], [2, 1, 3]], dtype=torch.uint8),
        torch.empty((0, 3), dtype=torch.uint8),
    ]

    # When: densification targets CUDA.
    actual = _dense_ss_batch(coordinate_sets, resolution=4, device=torch.device("cuda"))

    # Then: only the final dense float occupancy is allocated on CUDA.
    assert actual.device.type == "cuda"
    assert actual.dtype == torch.float32
    torch.testing.assert_close(
        actual.cpu(), _old_dense_batch(coordinate_sets, resolution=4)
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_legacy_dense_ss_volume_allocates_occupancy_only_on_cuda(monkeypatch):
    # Given: CPU-loaded coordinates and a CUDA legacy encoder.
    coordinates = torch.tensor([[0, 0, 0], [2, 1, 3]], dtype=torch.long)
    expected = _old_dense_batch([coordinates], resolution=4)
    allocation_devices = []
    real_zeros = torch.zeros

    def record_zeros(*args, **kwargs):
        allocation_devices.append(kwargs["device"])
        return real_zeros(*args, **kwargs)

    monkeypatch.setattr(legacy_ss.torch, "zeros", record_zeros)

    # When: its one-item batch is densified for that encoder device.
    actual = _dense_ss_volume(
        coordinates, resolution=4, device=torch.device("cuda")
    )

    # Then: the dense float occupancy was not allocated on CPU.
    assert allocation_devices == [torch.device("cuda")]
    assert actual.device.type == "cuda"
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual.cpu(), expected)
