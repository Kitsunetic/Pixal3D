#!/usr/bin/env python3
"""Keep a bounded CUDA allocation alive for container-level GPU visibility.

This utility is deliberately independent from Pixal3D's preprocessing worker. It
only creates one small tensor per selected GPU and exits cleanly on termination.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from threading import Event
from typing import Iterable


MIN_PAYLOAD_MIB = 1
MAX_PAYLOAD_MIB = 128
DEFAULT_PAYLOAD_MIB = 16
DEFAULT_INTERVAL_SECONDS = 30.0


def payload_elements(payload_mib: int) -> int:
    """Return the number of float32 elements in ``payload_mib`` MiB."""

    if not isinstance(payload_mib, int):
        raise TypeError("payload_mib must be an integer")
    if not MIN_PAYLOAD_MIB <= payload_mib <= MAX_PAYLOAD_MIB:
        raise ValueError(
            f"payload_mib must be between {MIN_PAYLOAD_MIB} and "
            f"{MAX_PAYLOAD_MIB} MiB"
        )
    return payload_mib * 1024 * 1024 // 4


def parse_device_ids(value: str) -> list[int]:
    """Parse a comma-separated list of unique non-negative CUDA ordinals."""

    if not value or not value.strip():
        raise ValueError("at least one CUDA device id is required")
    result: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item or not item.isdigit():
            raise ValueError(f"invalid CUDA device id: {item!r}")
        device_id = int(item)
        if device_id in result:
            raise ValueError(f"duplicate CUDA device id: {device_id}")
        result.append(device_id)
    return result


def _status_line(device_ids: Iterable[int], payload_mib: int) -> str:
    return json.dumps(
        {
            "event": "gpu_presence",
            "devices": list(device_ids),
            "payload_mib": payload_mib,
            "pid": os.getpid(),
        },
        sort_keys=True,
    )


def run_presence(
    *,
    device_ids: list[int],
    payload_mib: int,
    interval_seconds: float,
) -> int:
    """Allocate and retain one bounded tensor per device until signalled."""

    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    elements = payload_elements(payload_mib)

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - only reached in deployment
        print(f"gpu presence requires PyTorch: {exc}", file=sys.stderr)
        return 2

    if not torch.cuda.is_available():
        print("CUDA is not available", file=sys.stderr)
        return 2
    device_count = torch.cuda.device_count()
    invalid = [device_id for device_id in device_ids if device_id >= device_count]
    if invalid:
        print(
            f"CUDA device ids {invalid} are not visible; "
            f"device_count={device_count}",
            file=sys.stderr,
        )
        return 2

    stop = Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    allocations = []
    try:
        for device_id in device_ids:
            device = torch.device(f"cuda:{device_id}")
            # zeros_ forces physical allocation instead of only reserving an
            # uncommitted virtual range in the CUDA allocator.
            tensor = torch.empty(elements, dtype=torch.float32, device=device)
            tensor.zero_()
            torch.cuda.synchronize(device)
            allocations.append(tensor)
        print(_status_line(device_ids, payload_mib), flush=True)
        while not stop.wait(interval_seconds):
            print(_status_line(device_ids, payload_mib), flush=True)
    except Exception as exc:  # noqa: BLE001 - deployment utility must clean up
        print(f"gpu presence allocation failed: {exc}", file=sys.stderr)
        return 2
    finally:
        allocations.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--devices",
        default=None,
        help="comma-separated CUDA ordinals (default: all visible devices)",
    )
    parser.add_argument(
        "--payload-mib",
        type=int,
        default=DEFAULT_PAYLOAD_MIB,
        choices=range(MIN_PAYLOAD_MIB, MAX_PAYLOAD_MIB + 1),
        metavar="MIB",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=DEFAULT_INTERVAL_SECONDS,
        metavar="SECONDS",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.devices is None:
            import torch

            if not torch.cuda.is_available():
                print("CUDA is not available", file=sys.stderr)
                return 2
            device_ids = list(range(torch.cuda.device_count()))
        else:
            device_ids = parse_device_ids(args.devices)
        return run_presence(
            device_ids=device_ids,
            payload_mib=args.payload_mib,
            interval_seconds=args.interval_seconds,
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
