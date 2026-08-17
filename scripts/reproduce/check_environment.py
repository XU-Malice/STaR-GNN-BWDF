#!/usr/bin/env python
"""Report reproducibility-critical software and hardware versions."""

from __future__ import annotations

import importlib
import json
import platform
import sys


EXPECTED_MAJOR_MINOR = (3, 11)
PACKAGES = ("torch", "numpy", "pandas", "yaml", "pyarrow", "scipy")


def main() -> None:
    if sys.version_info[:2] != EXPECTED_MAJOR_MINOR:
        raise RuntimeError(
            f"Python 3.11 is required; found {platform.python_version()}"
        )
    versions: dict[str, str] = {}
    for name in PACKAGES:
        module = importlib.import_module(name)
        versions[name] = str(getattr(module, "__version__", "unknown"))
    torch = importlib.import_module("torch")
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_count": int(torch.cuda.device_count()),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("Environment import check: PASS")


if __name__ == "__main__":
    main()
