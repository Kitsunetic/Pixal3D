import torch

from data_toolkit import encode_latent_bundle
import pixal3d.models as models


def test_bundle_encoder_uses_forward_only_sparse_convolution_path(tmp_path, monkeypatch):
    instances = tmp_path / "instances.txt"
    instances.write_text("a" * 64 + "\n")
    model = torch.nn.Linear(1, 1)
    observed = []

    class ObserveNeedsGrad(torch.autograd.Function):
        @staticmethod
        def forward(ctx, weight):
            observed.append(ctx.needs_input_grad[0])
            return weight

        @staticmethod
        def backward(ctx, gradient):
            return gradient

    monkeypatch.setattr(models, "from_pretrained", lambda _path: model)

    def run_leaf(_script, _arguments):
        encoder = models.from_pretrained("shape")
        with torch.no_grad():
            ObserveNeedsGrad.apply(encoder.weight)

    monkeypatch.setattr(encode_latent_bundle, "_run_leaf", run_leaf)
    encode_latent_bundle.main([
        "--root", str(tmp_path),
        "--instances", str(instances),
        "--dual_grid_root", str(tmp_path),
        "--pbr_voxel_root", str(tmp_path),
        "--shape_latent_root", str(tmp_path),
        "--pbr_latent_root", str(tmp_path),
        "--ss_latent_root", str(tmp_path),
        "--resolutions", "1024", "--ss_resolution", "64",
        "--loader_workers", "1", "--saver_workers", "1",
        "--latent_dtype", "float16",
        "--micro_batch_sizes", "64:1,1024:1",
        "--gpu_memory_target_percent", "90",
    ])

    assert observed and not any(observed)
