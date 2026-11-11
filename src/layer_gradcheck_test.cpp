// ── layer_gradcheck_test — 基础层数值梯度检查（合并） ───────────────────────
// 合并：swiglu_gradcheck + rmsnorm_gradcheck + softmax_gradcheck + conv2d_gradcheck
// 覆盖：SwiGLU / RMSNorm / Softmax 的数值梯度验证 + Conv2D（独立参考实现比对）
// ───────────────────────────────────────────────────────────────────────────

#include <cstdio>
#include <iostream>

// ── conv2d_gradcheck（独立参考实现比对）────────────────────────────────────
#define main test_conv2d_gradcheck
#define max_abs_diff max_abs_diff_conv2d
// ── prelude（聚合体头提升）───────────────────────────────────────
// 下方首个嵌套 .cpp 起即出现 import；MSVC 模块消费方对「import 之后再
// 文本 include 同一头」不合并实体（C2953/C2572 族，上游 microsoft/STL
// #6436 同类）。故把 flatten（本文件+全部嵌套 .cpp）的头在此先行包含；
// include guard 幂等，clang/GCC 路径无害。改动此块须同步嵌套文件的头集合。
#include <nn_error_macros.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <random>
#include <string>
#include <array>
#include <vector>
#include "test_common.hpp"

#include "conv2d_gradcheck.cpp"
#undef max_abs_diff
#undef main

// ── swiglu_gradcheck ───────────────────────────────────────────────────────
#define main test_swiglu_gradcheck
#include "swiglu_gradcheck.cpp"
#undef approx
#undef dot
#undef main

// ── rmsnorm_gradcheck ──────────────────────────────────────────────────────
#define main test_rmsnorm_gradcheck
#include "rmsnorm_gradcheck.cpp"
#undef approx
#undef dot
#undef main

// ── softmax_gradcheck ──────────────────────────────────────────────────────
#define main test_softmax_gradcheck
#include "softmax_gradcheck.cpp"
#undef approx
#undef dot
#undef main

int main(int argc, char* argv[])
{
    int failures = 0;

    std::puts("=== swiglu_gradcheck ===");
    failures += test_swiglu_gradcheck(argc, argv);

    std::puts("=== rmsnorm_gradcheck ===");
    failures += test_rmsnorm_gradcheck(argc, argv);

    std::puts("=== softmax_gradcheck ===");
    failures += test_softmax_gradcheck(argc, argv);

    std::puts("=== conv2d_gradcheck ===");
    failures += test_conv2d_gradcheck(argc, argv);

    std::printf("\nlayer_gradcheck_test: %d failure(s)\n", failures);
    return failures != 0 ? 1 : 0;
}