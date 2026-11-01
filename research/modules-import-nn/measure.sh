#!/usr/bin/env bash
# ── measure.sh — import nn vs #include <nn.hpp> 编译速度对比 ──────────────────
# 口径：clang++ 24 / Linux / -O3 / 纯 CPU（无 NN_HAS_VULKAN）。每场景 3 轮取 min。
# 真实姿势（逐条实测验证，非纸面推断）：
#   · BMI 用 `--precompile -o X.pcm -x c++-module`（`-c` 会退化成 ELF 对象、`-x` 必须在尾）
#   · consumer 用 `-fmodule-file=nn=<pcm>`（`-fprebuilt-module-path` 对本用例不生效）
#   · 薄接口模块把 STL 留在全局模块 → 消费者需自带 <cstddef> 等（记为边界）
# 场景：
#   A include 冷（无 ccache）      —— 每 TU 重 parse 全库
#   B module 冷（BMI 建 + 首 consumer）—— 一次性成本
#   C include + ccache 命中        —— 现行主线热路径
#   D module 热（BMI 已建，改 consumer 只重编 consumer）
# 前后端拆分：-fsyntax-only≈前端；全量 -c≈前端+后端；差≈后端。
# 运行：cd research/modules-import-nn && ./measure.sh
# ───────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(dirname "$0")"
ROOT=../..
CXX=${CXX:-clang++}
FLAGS="-std=c++26 -O3"
INC="-I ${ROOT}/include"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT

min3() { # min3 <label> <cmd...>
    local label=$1; shift; local best=99999 rc=0
    for _ in 1 2 3; do
        local t0 t1 dt; t0=$(date +%s.%N)
        "$@" >/dev/null 2>&1 || rc=1
        t1=$(date +%s.%N)
        dt=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f",b-a}')
        best=$(awk -v a="$best" -v b="$dt" 'BEGIN{print (b<a)?b:a}')
    done
    [ "$rc" -ne 0 ] && { echo "$label: FAIL"; return 1; }
    echo "$label: ${best}s"
}

split() { # split <label> <cmd...>
    local label=$1; shift; local t0 t1 fe all be
    t0=$(date +%s.%N); "$@" -fsyntax-only >/dev/null 2>&1||true; t1=$(date +%s.%N)
    fe=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f",b-a}')
    t0=$(date +%s.%N); "$@" -c -o "$T/fb.o" >/dev/null 2>&1||true; t1=$(date +%s.%N)
    all=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f",b-a}')
    be=$(awk -v a="$all" -v b="$fe" 'BEGIN{printf "%.2f",(a-b>0)?a-b:0}')
    echo "    $label: 前端≈${fe}s  后端≈${be}s  全量≈${all}s"
}

echo "== 环境 =="; $CXX --version|head -1; echo "flags: $FLAGS $INC"; echo

echo "== A. include 冷（无 ccache）=="
min3 "  A-total" $CXX $FLAGS $INC -c consumer_include.cpp -o "$T/a.o"
split "A" $CXX $FLAGS $INC consumer_include.cpp
echo

echo "== B. module 冷（BMI 建造 + 首个 consumer 编译）=="
min3 "  B-bmi(一次性)" $CXX $FLAGS $INC --precompile -o "$T/nn.pcm" -x c++-module nn_module_thin.cpp
min3 "  B-consumer(首编)" $CXX $FLAGS -fmodule-file=nn="$T/nn.pcm" -c consumer_import_min.cpp -o "$T/b.o"
echo "  （B-bmi 是一次性成本；改 consumer 不必重付，见 D）"
echo

echo "== C. include + ccache 命中 =="
ccache -C >/dev/null 2>&1||true
ccache $CXX $FLAGS $INC -c consumer_include.cpp -o "$T/c0.o" >/dev/null 2>&1  # 冷填
min3 "  C-hit" ccache $CXX $FLAGS $INC -c consumer_include.cpp -o "$T/c.o"
echo

echo "== D. module 热（BMI 已建；改 consumer 只重编 consumer）=="
$CXX $FLAGS $INC --precompile -o "$T/nn.pcm" -x c++-module nn_module_thin.cpp >/dev/null 2>&1  # 预置 BMI
min3 "  D-consumer(改一行后)" $CXX $FLAGS -fmodule-file=nn="$T/nn.pcm" -c consumer_import_min.cpp -o "$T/d.o"
split "D" $CXX $FLAGS -fmodule-file=nn="$T/nn.pcm" consumer_import_min.cpp
echo

echo "== 参考：BMI 体积（AST 驻留代价）=="
ls -la "$T/nn.pcm" 2>/dev/null | awk '{printf "  nn.pcm: %.1f MB\n", $5/1048576}'
echo; echo "done."
