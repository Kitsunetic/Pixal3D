#!/usr/bin/env python3
"""고정 modulo shard를 순차 처리하는 로컬 rank launcher.

이 launcher는 scheduler가 아니다. 같은 제어 batch 목록에서
``batch_index % world_size == rank``인 batch만 고르고, 각 batch에 대해
stage ``run.py``를 하나씩 호출한다. 중간 산출물은 해당 node의 local
``data/preprocess_v2``에만 남는다.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from data_toolkit.preprocess._common.runtime import (
    add_config_argument,
    completed_batch_reason,
    config_get,
    load_config,
)


STAGES = ("01", "02", "03", "04", "05", "06")
STAGE_NAMES = {
    "01": "manifest",
    "02": "dump",
    "03": "render",
    "04": "voxelize",
    "05": "encode",
    "06": "finalize",
}


def control_batches(config: dict, source: str) -> list[tuple[int, str, str]]:
    control_root = Path(config_get(config, "paths", "control_root"))
    shard_root = control_root / "shards" / source
    paths = sorted(shard_root.glob("*/batch*.txt"), key=lambda path: path.relative_to(shard_root).as_posix())
    if not paths:
        raise RuntimeError(f"control batch가 없습니다: {shard_root}")
    return [(index, path.parent.name, path.stem) for index, path in enumerate(paths)]


def parse_stages(value: str) -> tuple[str, ...]:
    stages = tuple(item.strip() for item in value.split(",") if item.strip())
    if not stages or any(stage not in STAGES for stage in stages):
        raise ValueError(f"--stages는 {','.join(STAGES)}의 comma 목록이어야 합니다")
    positions = [STAGES.index(stage) for stage in stages]
    if positions != sorted(set(positions)):
        raise ValueError("--stages는 중복 없이 오름차순이어야 합니다")
    return stages


def stage_program(root: Path, stage: str) -> Path:
    return root / "data_toolkit" / "preprocess" / f"{stage}_{STAGE_NAMES[stage]}" / "run.py"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument("--source", default=None)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--stages", default=",".join(STAGES))
    parser.add_argument("--publish", action="store_true", help="06 단계에서 prepared-v2로 publish")
    arguments = parser.parse_args()
    if arguments.world_size <= 0:
        parser.error("--world-size는 양수여야 합니다")
    if not 0 <= arguments.rank < arguments.world_size:
        parser.error("--rank는 0 이상 --world-size 미만이어야 합니다")
    try:
        stages = parse_stages(arguments.stages)
    except ValueError as error:
        parser.error(str(error))

    config, config_path = load_config(arguments.config)
    source = arguments.source or config_get(config, "dataset", "source")
    owned = [
        (index, shard, batch)
        for index, shard, batch in control_batches(config, source)
        if index % arguments.world_size == arguments.rank
    ]
    root = Path(__file__).resolve().parents[3]
    processed = skipped = soft_failed = 0
    for batch_index, shard, batch in owned:
        reason = completed_batch_reason(config, source, shard, batch)
        if reason is not None:
            skipped += 1
            print(f"skip {batch_index} {shard}/{batch}: {reason}", flush=True)
            continue
        print(f"start {batch_index} {shard}/{batch}", flush=True)
        for stage in stages:
            command = [
                sys.executable, str(stage_program(root, stage)),
                "--config", str(config_path), "--source", source,
                "--shard", shard, "--batch", batch,
                "--world-size", str(arguments.world_size), "--rank", str(arguments.rank),
            ]
            if stage == "01" and processed == 0:
                command.append("--build-index")
            if stage == "06" and arguments.publish:
                command.append("--publish")
            result = subprocess.run(command, cwd=root)
            if result.returncode == 2:
                soft_failed += 1
                print(f"asset-level errors in {shard}/{batch} stage {stage}; continuing", file=sys.stderr, flush=True)
                continue
            if result.returncode != 0:
                print(f"stopped at {shard}/{batch} stage {stage} (exit {result.returncode})", file=sys.stderr)
                return result.returncode
        processed += 1
    print(
        f"rank={arguments.rank}/{arguments.world_size}: processed={processed} "
        f"skipped={skipped} soft_failed_stages={soft_failed}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
