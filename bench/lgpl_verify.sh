#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
#  lgpl_verify.sh — LGPL 可替换库二进制验证（引擎 .so 化版；docs/internal/34 §10）
#
#  证明"改 libnn-*.so → 不重编消费方 → 换 .so → 消费方行为随新 .so 变化"：
#   1) 符号级：libnn-cpu.so 导出 nn::create_cpu_engine；消费方二进制不含
#      引擎实现符号（引擎全在 .so，消费方只 include 抽象层头 + 链 .so）；
#   2) 运行时替换：在临时目录复制 submodule 源码，给 CpuEngine::matmul_impl
#      注入"输出 ×2"补丁，编出 patched libnn-cpu.so，用 LD_LIBRARY_PATH
#      指向它跑**同一个** mnist_train 二进制（不重编）→ 训练 loss 显著不同
#      （行为随 .so 变）；→ 恢复原 .so → loss 复原（行为复原）。
#  前置：主仓库已构建（build/mnist_train、build/engine/libnn-cpu.so）。
#  用法：bash bench/lgpl_verify.sh
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail
cd "$(dirname "$0")/.."

ROOT="$PWD"
BUILD="$ROOT/build"
CONSUMER="$BUILD/mnist_train"
LIBCPU="$BUILD/engine/libnn-cpu.so"
SUBMOD="$ROOT/engine"
PATDIR="$(mktemp -d)"
trap 'rm -rf "$PATDIR"' EXIT

failures=0

echo "═══ [1/3] 符号级：引擎在 .so，消费方无引擎实现 ═══"
lib_syms="$(nm -D --defined-only -C "$LIBCPU" 2>/dev/null || true)"
if [[ "$lib_syms" == *"nn::create_cpu_engine()"* ]]; then
    echo "  [PASS] libnn-cpu.so 导出 nn::create_cpu_engine（C++ 直出工厂）"
else
    echo "  [FAIL] libnn-cpu.so 未导出 create_cpu_engine"
    failures=$((failures+1))
fi
if ! [ -f "$CONSUMER" ]; then
    echo "  [FAIL] 找不到 $CONSUMER（先构建 mnist_train）"
    exit 1
fi
consumer_syms="$(nm "$CONSUMER" 2>/dev/null || true)"
n_engine_syms="$(printf '%s\n' "$consumer_syms" | grep -cE '_ZN2nn.*(CpuEngine|GpuEngine)' || true)"
if [ "${n_engine_syms:-0}" -gt 0 ]; then
    echo "  [FAIL] 消费方二进制含引擎实现符号（引擎被内联，LGPL 不可替换）"
    failures=$((failures+1))
else
    echo "  [PASS] 消费方二进制无引擎实现符号（引擎全在 libnn-cpu.so）"
fi

echo "═══ [2/3] 临时目录编 patched libnn-cpu.so（matmul 输出 ×2）═══"
# 复制 submodule 源码（排除 .git/build），注入补丁后独立构建——不改动主仓/submodule。
cp -r "$SUBMOD" "$PATDIR/submod"
rm -rf "$PATDIR/submod/.git" "$PATDIR/submod/build"
python3 - "$PATDIR/submod/engine/include/neuralnet.cpp/compute_cpu_engine.hpp" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
anchor = """                if (P == Precision::F32) return r;"""
hack = """                if (P == Precision::F32) { auto sc = scale_inplace(*r, 2.0f); if (!sc) return std::unexpected(sc.error()); return r; }"""
assert anchor in s, "compute_cpu_engine.hpp 结构变化，补丁锚点失效"
open(p, "w").write(s.replace(anchor, hack))
PY
cmake -S "$PATDIR/submod" -B "$PATDIR/submod/build" -G Ninja \
      -DNN_ENABLE_GPU=OFF -DNN_BUILD_TESTS=OFF >/dev/null 2>&1
ninja -C "$PATDIR/submod/build" libnn-cpu.so >/dev/null 2>&1
echo "  [PASS] patched libnn-cpu.so 编译完成 → $PATDIR/submod/build/libnn-cpu.so"

echo "═══ [3/3] 运行时替换：同一 layer_bench 二进制，行为随 .so 变 ═══"
# 观测载体 = layer_bench --op matmul --hash（**显式调引擎 matmul_impl 虚方法**，
# 确定性输出位级 FNV hash；DSL 模板路径在消费方 TU 求值、不经 .so，故不适用）：
#   原 .so → hash_orig；patched .so → hash_pat（应不同）；恢复原 .so → hash_orig（复原）。
run_bench() {  # $1 = LD_LIBRARY_PATH（空 = 原 .so）
    local lp="${1:-}"
    if [ -n "$lp" ]; then
        LD_LIBRARY_PATH="$lp" "$BUILD/layer_bench" --op matmul --hash 2>&1 | grep -oE 'hash=[0-9a-f]+' || true
    else
        "$BUILD/layer_bench" --op matmul --hash 2>&1 | grep -oE 'hash=[0-9a-f]+' || true
    fi
}
orig_hash="$(run_bench "")"
patched_hash="$(run_bench "$PATDIR/submod/build")"
restore_hash="$(run_bench "")"
echo "  原 .so      matmul hash: $orig_hash"
echo "  patched .so matmul hash: $patched_hash"
echo "  恢复原 .so  matmul hash: $restore_hash"
if [ -n "$patched_hash" ] && [ -n "$orig_hash" ] && [ "$patched_hash" != "$orig_hash" ]; then
    echo "  [PASS] patched .so 下 matmul 输出 hash 不同（×2 生效，行为随 .so 变）"
else
    echo "  [FAIL] patched .so 未改变消费方行为（替换未生效？）"
    failures=$((failures+1))
fi
if [ -n "$restore_hash" ] && [ "$restore_hash" = "$orig_hash" ]; then
    echo "  [PASS] 恢复原 .so 后 hash 复原（同一二进制，行为复原）"
else
    echo "  [FAIL] 恢复原 .so 后 hash 未复原"
    failures=$((failures+1))
fi

echo
if [ "$failures" -eq 0 ]; then
    echo "✅ LGPL 可替换验证通过：改 libnn-cpu.so → 不重编消费方 → 换 .so → 行为随 .so 变化"
else
    echo "❌ LGPL 可替换验证失败：$failures 处"
fi
exit "$failures"
