// ── expr_cpu_test — 表达式/IR 纯 CPU 合并测试 ──────────────────────────────
// 合并：expr_dsl_test + expr_reduce_test + expr_matmul_test + expr_opt_test
// 覆盖：DSL 编译期融合 / 归约语义 / matmul IR 融合 / 优化 pass
// ───────────────────────────────────────────────────────────────────────────

#include <cstdio>
#include <iostream>

// ── expr_dsl_test ──────────────────────────────────────────────────────────
#define main test_expr_dsl
#define g_fail g_fail_dsl
// ── prelude（聚合体头提升）───────────────────────────────────────
// 下方首个嵌套 .cpp 起即出现 import；MSVC 模块消费方对「import 之后再
// 文本 include 同一头」不合并实体（C2953/C2572 族，上游 microsoft/STL
// #6436 同类）。故把 flatten（本文件+全部嵌套 .cpp）的头在此先行包含；
// include guard 幂等，clang/GCC 路径无害。改动此块须同步嵌套文件的头集合。
#include <cmath>
#include <vector>
#include "test_common.hpp"
#include <nn_error_macros.hpp>
#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <type_traits>

#include "expr_dsl_test.cpp"
#undef g_fail
#undef main

// ── expr_reduce_test ───────────────────────────────────────────────────────
#define main test_expr_reduce
#define g_fail g_fail_reduce
#define test_validation test_validation_reduce
#include "expr_reduce_test.cpp"
#undef test_validation
#undef g_fail
#undef main

// ── expr_matmul_test ───────────────────────────────────────────────────────
#define main test_expr_matmul
#define g_fail g_fail_matmul
#define test_validation test_validation_matmul
#include "expr_matmul_test.cpp"
#undef test_validation
#undef g_fail
#undef main

// ── expr_opt_test ──────────────────────────────────────────────────────────
#define main test_expr_opt
#define g_fail g_fail_opt
#define test_determinism test_determinism_opt
#include "expr_opt_test.cpp"
#undef test_determinism
#undef g_fail
#undef main

// ── expr_fold_test（fold 分块状态归约；独立函数，无宏重命名）───────────────
#include "expr_fold_test.cpp"

// ── expr_fused_key_test（"结构 = 表达式类型"：AOT 自登记的 key 契约）────────
#define main test_fused_key
#define g_fail g_fail_fused_key
#include "expr_fused_key_test.cpp"
#undef g_fail
#undef main

int main()
{
    int failures = 0;

    std::puts("=== expr_dsl (compile-time fusion + AOT) ===");
    failures += test_expr_dsl();

    std::puts("=== expr_reduce (reduction semantics) ===");
    failures += test_expr_reduce();

    std::puts("=== expr_matmul (matmul in IR fusion) ===");
    failures += test_expr_matmul();

    std::puts("=== expr_opt (canonicalize + CSE + regalloc) ===");
    failures += test_expr_opt();

    std::puts("=== expr_fold (block-state reduction) ===");
    failures += test_expr_fold();

    std::puts("=== expr_fused_key (structure == expression type) ===");
    failures += test_fused_key();

    std::printf("\nexpr_cpu_test: %d failure(s)\n", failures);
    return failures != 0 ? 1 : 0;
}