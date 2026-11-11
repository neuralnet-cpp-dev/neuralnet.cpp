// ── expr_gpu_test — 表达式/IR GPU 合并测试 ─────────────────────────────────
// 合并：fused_gpu_test + tensor_expr_test
// 覆盖：AOT 融合 shader 数值验证 / 表达式求值
// 注意：需要 Vulkan SDK；纯 CPU 构建返回 77（ctest SKIP）
// ───────────────────────────────────────────────────────────────────────────

#include <cstdio>
#include <iostream>

#ifndef NN_GPU_ENGINE
int main()
{
    std::cout << "[SKIP] 此程序需要 Vulkan SDK 支持。\n";
    return 77;
}
#else

// ── fused_gpu_test ─────────────────────────────────────────────────────────
#define main test_fused_gpu
#define max_abs_diff max_abs_diff_fused
// ── prelude（聚合体头提升）───────────────────────────────────────
// 下方首个嵌套 .cpp 起即出现 import；MSVC 模块消费方对「import 之后再
// 文本 include 同一头」不合并实体（C2953/C2572 族，上游 microsoft/STL
// #6436 同类）。故把 flatten（本文件+全部嵌套 .cpp）的头在此先行包含；
// include guard 幂等，clang/GCC 路径无害。改动此块须同步嵌套文件的头集合。
#include <nn_error_macros.hpp>
#include <cmath>
#include <iomanip>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "fused_gpu_test.cpp"
#undef max_abs_diff
#undef main

// ── tensor_expr_test ───────────────────────────────────────────────────────
#define main test_tensor_expr
#include "tensor_expr_test.cpp"
#undef main

int main()
{
    int failures = 0;

    std::puts("=== fused_gpu (AOT shader numerical) ===");
    int r = test_fused_gpu();
    if (r != 77) failures += r;

    std::puts("=== tensor_expr (expression evaluation) ===");
    r = test_tensor_expr();
    if (r != 77) failures += r;

    std::printf("\nexpr_gpu_test: %d failure(s)\n", failures);
    return failures != 0 ? 1 : 0;
}

#endif // NN_GPU_ENGINE
