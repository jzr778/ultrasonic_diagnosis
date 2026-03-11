#!/usr/bin/env bash
# 独立运行脚本：与 offline_avm_generate 放在同一目录即可，不依赖 perception 仓库路径。
# 优先使用包内 lib/ 和 runfiles 中的库，避免对方机器缺 OpenCV 4.2 等依赖。
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${SCRIPT_DIR}/offline_avm_generate"

if [ ! -x "$BIN" ]; then
  echo "Error: 未找到可执行文件 $BIN"
  exit 1
fi

# 包内 lib/ 目录（打包时放入的 OpenCV 4.2 等）
LIB_DIR="${SCRIPT_DIR}/lib"
if [ -d "$LIB_DIR" ]; then
  export LD_LIBRARY_PATH="${LIB_DIR}:${LD_LIBRARY_PATH}"
  # runfiles 里缺 libopencv_core.so.3.2，用包内 4.2 做 3.2 符号链接供加载
  WRAPPER_DIR="${WRAPPER_DIR:-/tmp/opencv32_to_42_run}"
  mkdir -p "$WRAPPER_DIR"
  for name in core imgproc imgcodecs highgui videoio video; do
    src="${LIB_DIR}/libopencv_${name}.so.4.2"
    [ -e "$src" ] && ln -sf "$src" "${WRAPPER_DIR}/libopencv_${name}.so.3.2" 2>/dev/null || true
  done
  export LD_LIBRARY_PATH="${WRAPPER_DIR}:${LD_LIBRARY_PATH}"
fi

# 若无 lib/，尝试用系统 OpenCV 4.2 做 3.2 符号链接
if [ ! -d "$LIB_DIR" ] || [ ! -e "${LIB_DIR}/libopencv_core.so.4.2" ]; then
  SYS_OPENCV="${SYS_OPENCV:-/usr/lib/x86_64-linux-gnu}"
  WRAPPER_DIR="${WRAPPER_DIR:-/tmp/opencv32_to_42_run}"
  mkdir -p "$WRAPPER_DIR"
  for name in core imgproc imgcodecs highgui videoio video; do
    src="${SYS_OPENCV}/libopencv_${name}.so.4.2"
    [ -e "$src" ] && ln -sf "$src" "${WRAPPER_DIR}/libopencv_${name}.so.3.2" 2>/dev/null || true
  done
  export LD_LIBRARY_PATH="${WRAPPER_DIR}:${LD_LIBRARY_PATH}"
fi

exec "$BIN" "$@"