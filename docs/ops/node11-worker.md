# Node11 worker handoff

The Node11 worker runs in the isolated `youngwoo_pixal3d` container. It shares the
production queue and data roots with the other workers, but keeps scratch data in
`/root/node11/data/pixal3d`.

Status checks from the Pixal3D checkout:

```bash
PYTHONPATH=. /home/youngwoo/miniconda3/envs/pixal3d/bin/python -m \
  data_toolkit.pipeline.cli workers \
  --config data_toolkit/configs/multiview_preprocess.yaml \
  --action status \
  --worker-registry /root/data2/pixal3d/control/runtime/workers.json
```

To drain only Node11, let its current lease finish, and prevent new claims:

```bash
PYTHONPATH=. /home/youngwoo/miniconda3/envs/pixal3d/bin/python -m \
  data_toolkit.pipeline.cli workers \
  --config data_toolkit/configs/multiview_preprocess.yaml \
  --action drain --node-id node11 \
  --worker-registry /root/data2/pixal3d/control/runtime/workers.json
```

Reactivate it after the node is ready:

```bash
PYTHONPATH=. /home/youngwoo/miniconda3/envs/pixal3d/bin/python -m \
  data_toolkit.pipeline.cli workers \
  --config data_toolkit/configs/multiview_preprocess.yaml \
  --action activate --node-id node11 \
  --worker-registry /root/data2/pixal3d/control/runtime/workers.json
```

The supervisor is a detached process in `youngwoo_pixal3d` and must retain these
environment variables when restarted:

```text
PYTHONPATH=/root/dev/Pixal3D-node11
PIXAL3D_LOCAL_FREE_PERCENT=5
PIXAL3D_LOCAL_FREE_GIB=120
PIXAL3D_TOOL_COMMIT=6c524737b66cbf99c7e1af3264d28973f487c7b8
```

Do not run `queue --action init`; it would replace the shared queue. Draining or
removing Node11 does not alter Node16/Node17 registrations.

## Grafana container visibility

When a batch is in a CPU-only stage, `nvidia-smi` may show no active GPU
process even though the `youngwoo_pixal3d` container is processing data. The
GPU presence watchdog keeps a small CUDA allocation inside that same container
so DCGM/cAdvisor can associate the container name with the GPU metrics. It does
not participate in the Pixal3D queue or preprocessing worker.

The allocation is 16 MiB of float32 payload per GPU 0–7. CUDA context overhead
is additional and is driver-dependent; the watchdog exits if any allocation
cannot be made, leaving the production worker untouched.

The service template is
`ops/systemd/pixal3d-node11-gpu-presence.service`. On Node11, install and start
it as the `youngwoo` user (the existing container is not restarted):

```bash
mkdir -p "$HOME/.config/systemd/user"
cp /home/youngwoo/dev/Pixal3D-node11/ops/systemd/pixal3d-node11-gpu-presence.service \
  "$HOME/.config/systemd/user/"
systemctl --user daemon-reload
systemctl --user enable --now pixal3d-node11-gpu-presence.service
```

Verify the presence process and the unchanged preprocessing processes:

```bash
docker exec youngwoo_pixal3d pgrep -af '[g]pu_presence.py'
docker exec youngwoo_pixal3d nvidia-smi \
  --query-compute-apps=pid,process_name,used_memory --format=csv
pgrep -af 'data_toolkit.pipeline.cli (supervisor|worker)'
```

To remove only the visibility helper:

```bash
systemctl --user disable --now pixal3d-node11-gpu-presence.service
rm -f "$HOME/.config/systemd/user/pixal3d-node11-gpu-presence.service"
```
