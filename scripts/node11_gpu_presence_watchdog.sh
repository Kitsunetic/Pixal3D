#!/usr/bin/env bash
set -euo pipefail

container_name="${PIXAL3D_PRESENCE_CONTAINER:-youngwoo_pixal3d}"
gpu_ids="${PIXAL3D_PRESENCE_GPUS:-0,1,2,3,4,5,6,7}"
payload_mib="${PIXAL3D_PRESENCE_PAYLOAD_MIB:-16}"
interval_seconds="${PIXAL3D_PRESENCE_INTERVAL_SECONDS:-30}"
docker_bin="${DOCKER_BIN:-/usr/bin/docker}"
python_bin="${PIXAL3D_PRESENCE_PYTHON:-/opt/conda/envs/pixal3d/bin/python}"
utility_path="${PIXAL3D_PRESENCE_UTILITY:-/root/dev/Pixal3D-node11/tools/gpu_presence.py}"
local_mode="${PIXAL3D_PRESENCE_LOCAL:-0}"
log_path="${PIXAL3D_PRESENCE_LOG:-/tmp/pixal3d-gpu-presence.log}"

container_running() {
    if [[ "$local_mode" == "1" ]]; then
        return 0
    fi
    [[ "$("$docker_bin" inspect -f '{{.State.Running}}' "$container_name" 2>/dev/null || true)" == "true" ]]
}

presence_running() {
    if [[ "$local_mode" == "1" ]]; then
        local pid cmd
        while read -r pid; do
            [[ -z "$pid" || "$pid" == "$$" || "$pid" == "$PPID" ]] && continue
            cmd="$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)"
            if [[ "$cmd" == "$python_bin "* && "$cmd" == *"gpu_presence.py"* ]]; then
                return 0
            fi
        done < <(pgrep -f '[g]pu_presence.py' || true)
        return 1
    fi
    "$docker_bin" exec "$container_name" pgrep -f '[g]pu_presence.py' >/dev/null 2>&1
}

start_presence() {
    if [[ "$local_mode" == "1" ]]; then
        nohup "$python_bin" "$utility_path" \
            --devices "$gpu_ids" \
            --payload-mib "$payload_mib" \
            --interval-seconds "$interval_seconds" >>"$log_path" 2>&1 &
        return
    fi
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
