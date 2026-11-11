// ── doc_attn_test — 文档感知注意力合并测试 ──────────────────────────────────
// 合并：doc_attn_e2e_test
// 覆盖：端到端跨文档屏蔽不变性
// （掩码语义由 fold body 表达：spec 级覆盖见 expr_fold_test 三掩码，
//   doc 端到端覆盖见下方 e2e）
// ───────────────────────────────────────────────────────────────────────────

#include <cstdio>
#include <iostream>

#define main test_doc_attn_e2e
// ── prelude（聚合体头提升）───────────────────────────────────────
// 下方首个嵌套 .cpp 起即出现 import；MSVC 模块消费方对「import 之后再
// 文本 include 同一头」不合并实体（C2953/C2572 族，上游 microsoft/STL
// #6436 同类）。故把 flatten（本文件+全部嵌套 .cpp）的头在此先行包含；
// include guard 幂等，clang/GCC 路径无害。改动此块须同步嵌套文件的头集合。
#include <nn_error_macros.hpp>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include "doc_attn_e2e_test.cpp"
#undef main

int main(int argc, char* argv[])
{
    std::puts("=== doc_attn_e2e (cross-doc invariance) ===");
    const int failures = test_doc_attn_e2e(argc, argv);
    std::printf("\ndoc_attn_test: %d failure(s)\n", failures);
    return failures != 0 ? 1 : 0;
}
