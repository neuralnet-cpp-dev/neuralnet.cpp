#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
#  lgpl_verify.sh — LGPL 可替换库二进制验证（Phase 3-C；docs/internal/34 §7）
#
#  证明"改 libnn.so → 不重编消费方 → 换 .so → 消费方行为随新 .so 变化"：
#   1) 符号级：libnn.so 导出 nnn_* C ABI；consumer 二进制不含引擎实现符号
#      （引擎全在 .so，consumer 只 include C ABI + IR 头）；
#   2) 运行时替换：编一个 matmul 输出 ×2 的 patched libnn.so，用
#      LD_LIBRARY_PATH 指向它跑**同一个** nn_consumer_test 二进制（不重编）
#      → matmul 数值 FAIL（值翻倍，行为随 .so 变）；
#      → 恢复原 .so → 全 PASS（行为复原）。
#  前置：nn_consumer_test 与 libnn.so 已构建（build/test、build/lib）。
#  用法：bash bench/lgpl_verify.sh
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail
cd "$(dirname "$0")/.."

ROOT="$PWD"
BUILD="$ROOT/build"
CONSUMER="$BUILD/test/nn_consumer_test"
LIBNN="$BUILD/lib/libnn.so"
PATDIR="$(mktemp -d)"
trap 'rm -rf "$PATDIR"' EXIT

failures=0

echo "═══ [1/3] 符号级：引擎在 .so，consumer 无引擎实现 ═══"
libnn_syms="$(nm -D --defined-only "$LIBNN" 2>/dev/null || true)"
if [[ "$libnn_syms" == *nnn_matmul* ]]; then
    echo "  [PASS] libnn.so 导出 nnn_matmul（C ABI）"
else
    echo "  [FAIL] libnn.so 未导出 nnn_matmul"
    failures=$((failures+1))
fi
if ! [ -f "$CONSUMER" ]; then
    echo "  [FAIL] 找不到 $CONSUMER（先构建 nn_consumer_test）"
    exit 1
fi
consumer_syms="$(nm "$CONSUMER" 2>/dev/null || true)"
n_engine_syms="$(printf '%s\n' "$consumer_syms" | grep -cE '_ZN2nn.*(CpuEngine|GpuEngine)' || true)"
if [ "${n_engine_syms:-0}" -gt 0 ]; then
    echo "  [FAIL] consumer 二进制含引擎实现符号（引擎被内联，LGPL 不可替换）"
    failures=$((failures+1))
else
    echo "  [PASS] consumer 二进制无引擎实现符号（引擎全在 libnn.so）"
fi

echo "═══ [2/3] 编 patched libnn.so（matmul 输出 ×2）═══"
# 在 nnn_matmul 结果上加可观测偏移（不改主仓源码，临时补丁）
python3 - "$ROOT/src/nn_api.cpp" "$PATDIR/nn_api.cpp" <<'PY'
import sys
src = open(sys.argv[1]).read()
anchor = "    auto r = e->e->matmul(A->t, B->t, transA != 0, transB != 0, Precision::F32);"
hack = anchor + "\n    { auto rs = e->e->scale_inplace(*r, 2.0f); if (!rs) { set_err(rs.error().message); return NNN_ERR; } }"
assert anchor in src, "nn_api.cpp 结构变化，补丁锚点失效"
open(sys.argv[2], "w").write(src.replace(anchor, hack))
PY
# 定位 Vulkan include：优先读构建缓存，其次在常见 SDK 路径查找
VK_INC="$(cmake -LA -N "$BUILD" 2>/dev/null | sed -n 's/^Vulkan_INCLUDE_DIR:.*=//p' | head -1 || true)"
if [ -z "$VK_INC" ] || [ ! -d "$VK_INC" ]; then
    VK_INC="$(find /home/ethan/vulkan /usr/local /usr -maxdepth 4 -type d -name shaderc 2>/dev/null | head -1 || true)"
    [ -n "$VK_INC" ] && VK_INC="$(dirname "$VK_INC")"
fi
if [ -z "$VK_INC" ] || [ ! -f "$VK_INC/shaderc/shaderc.h" ]; then
    echo "  [FAIL] 找不到 Vulkan include（shaderc/shaderc.h）；设 NN_VULKAN_INCLUDE"; exit 1
fi
VK_LIB="$(dirname "$VK_INC")/lib"
c++ -DNN_HAS_VULKAN -I"$BUILD/generated" -I"$ROOT/include" -isystem "$VK_INC" \
    -std=c++26 -fno-exceptions -fPIC -shared -fexperimental-library -Wno-pass-failed -O2 \
    "$PATDIR/nn_api.cpp" -o "$PATDIR/libnn.so" \
    -L"$VK_LIB" -lvulkan -lstdc++exp
echo "  [PASS] patched libnn.so 编译完成 → $PATDIR/libnn.so"

echo "═══ [3/3] 运行时替换：同一 consumer 二进制，行为随 .so 变 ═══"
# patched：matmul ×2 → matmul 数值应 FAIL（行为变化）；eval_expr 不受影响仍 PASS
patched_out="$(LD_LIBRARY_PATH="$PATDIR" NN_VULKAN_DEVICE=850M "$CONSUMER" 2>&1 || true)"
if [[ "$patched_out" == *"[FAIL] matmul 数值"* ]]; then
    echo "  [PASS] patched .so 下 matmul 数值 FAIL（输出 ×2，行为随 .so 变）"
else
    echo "  [FAIL] patched .so 未改变 consumer 行为（替换未生效？）"
    failures=$((failures+1))
fi
if [[ "$patched_out" == *"[PASS] eval_expr 数值"* ]]; then
    echo "  [PASS] eval_expr 不受 matmul 补丁影响（仍 PASS，补丁精准）"
else
    echo "  [FAIL] eval_expr 也被破坏（补丁范围过大）"
    failures=$((failures+1))
fi
# 恢复原 .so → 全 PASS（同一二进制）
orig_out="$(NN_VULKAN_DEVICE=850M "$CONSUMER" 2>&1 || true)"
if [[ "$orig_out" == *"nn_consumer_test: 0 failure(s)"* ]]; then
    echo "  [PASS] 原 .so 下全 PASS（同一二进制，行为复原）"
else
    echo "  [FAIL] 原 .so 未复原"
    failures=$((failures+1))
fi

echo
if [ "$failures" -eq 0 ]; then
    echo "✅ LGPL 可替换验证通过：改 libnn.so → 不重编消费方 → 换 .so → 行为随 .so 变化"
else
    echo "❌ LGPL 可替换验证失败：$failures 处"
fi
exit "$failures"
