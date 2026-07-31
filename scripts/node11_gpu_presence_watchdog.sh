#!/usr/bin/env bash
set -euo pipefail

container_name="${PIXAL3D_PRESENCE_CONTAINER:-youngwoo_pixal3d}"
gpu_ids="${PIXAL3D_PRESENCE_GPUS:-0,1,2,3,4,5,6,7}"
payload_mib="${PIXAL3D_PRESENCE_PAYLOAD_MIB:-16}"
interval_seconds="${PIXAL3D_PRESENCE_INTERVAL_SECONDS:-30}"
docker_bin="${DOCKER_BIN:-/usr/bin/docker}"
python_bin="${PIXAL3D_PRESENCE_PYTHON:-/opt/conda/envs/pixal3d/bin/python}"
utility_path="${PIXAL3D_PRESENCE_UTILITY:-/root/dev/Pixal3D-node11/tools/gpu_presence.py}"

container_running() {
    [[ "$("$docker_bin" inspect -f '{{.State.Running}}' "$container_name" 2>/dev/null || true)" == "true" ]]
}

presence_running() {
    "$docker_bin" exec "$container_name" pgrep -f '[g]pu_presence.py' >/dev/null 2>&1
}

start_presence() {
    "$docker_bin" exec -d "$container_name" env PYTHONUNBUFFERED=1 \
        "$python_bin" "$utility_path" \
        --devices "$gpu_ids" \
        --payload-mib "$payload_mib" \
        --interval-seconds "$interval_seconds" >/dev/null
}

while :; do
    if container_running && ! presence_running; then
        start_presence || true
    fi
    sleep "$interval_seconds"
done
