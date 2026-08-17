#!/usr/bin/env python
"""Inspect all physical GPUs and print safe training recommendations."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dma_wdf.utils.gpu_inspector import (  # noqa: E402
    physical_gpu_rejection_reasons,
    query_physical_gpus,
    recommend_physical_gpus,
    visible_logical_mapping,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only physical GPU inventory and recommendation."
    )
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--min-free-mib", type=float, default=8192)
    parser.add_argument("--max-used-mib", type=float, default=2048)
    parser.add_argument("--max-util-percent", type=float, default=20)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    statuses = query_physical_gpus()
    recommended = recommend_physical_gpus(
        statuses,
        count=args.count,
        minimum_free_memory_mib=args.min_free_mib,
        maximum_used_memory_mib=args.max_used_mib,
        maximum_gpu_utilization_percent=args.max_util_percent,
    )
    payload = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "thresholds": {
            "minimum_free_memory_mib": args.min_free_mib,
            "maximum_used_memory_mib": args.max_used_mib,
            "maximum_gpu_utilization_percent": args.max_util_percent,
        },
        "physical_gpus": [
            {
                **status.state_dict(),
                "rejection_reasons": physical_gpu_rejection_reasons(
                    status,
                    minimum_free_memory_mib=args.min_free_mib,
                    maximum_used_memory_mib=args.max_used_mib,
                    maximum_gpu_utilization_percent=args.max_util_percent,
                ),
            }
            for status in statuses
        ],
        "visible_logical_mapping": visible_logical_mapping(statuses),
        "recommended_physical_indices": [
            status.physical_index for status in recommended
        ],
        "enough_recommended_gpus": len(recommended) == args.count,
    }
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    print("物理GPU检查（只读，不会占用显存）")
    print(
        f"CUDA_VISIBLE_DEVICES="
        f"{payload['cuda_visible_devices'] or '<未设置>'}"
    )
    print()
    for row in payload["physical_gpus"]:
        reasons = row["rejection_reasons"]
        state = "可用" if not reasons else "忙碌: " + ", ".join(reasons)
        print(
            f"物理GPU {row['physical_index']}: {row['name']}, "
            f"used={row['used_memory_mib']:.0f} MiB, "
            f"free={row['free_memory_mib']:.0f} MiB, "
            f"util={row['utilization_percent']:.0f}% -> {state}"
        )
    print()
    if not payload["enough_recommended_gpus"]:
        print(
            f"没有找到 {args.count} 张满足阈值的物理GPU；"
            "不要启动正式训练。"
        )
        raise SystemExit(2)
    selected = ",".join(
        str(value) for value in payload["recommended_physical_indices"]
    )
    print(f"建议物理GPU: {selected}")
    print("运行前执行：")
    print(f"  export CUDA_VISIBLE_DEVICES={selected}")
    print("设置后这些卡会重新编号为 PyTorch 逻辑 cuda:0, cuda:1, ...")
    print("训练脚本继续使用 --device auto，不要填写物理编号。")


if __name__ == "__main__":
    main()
