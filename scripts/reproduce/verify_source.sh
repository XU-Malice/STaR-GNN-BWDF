#!/usr/bin/env bash
# 验证公开源码、配置、测试和文档在下载后没有被修改。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

test -f SOURCE_CHECKSUMS.sha256 || {
  echo "ERROR：缺少 SOURCE_CHECKSUMS.sha256" >&2
  exit 1
}

sha256sum -c SOURCE_CHECKSUMS.sha256
echo "公开源码文件校验：PASS"
