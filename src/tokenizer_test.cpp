// ── tokenizer_test — 分词器正确性测试 ───────────────────────────────────────
// 聚合 tokenizer_consistency_test：BPE pre_tokenize 对拍 + 标记往返
// ───────────────────────────────────────────────────────────────────────────

#include <cstdio>
#include <iostream>

#define main test_tokenizer
// ── prelude（聚合体头提升）───────────────────────────────────────
// 下方首个嵌套 .cpp 起即出现 import；MSVC 模块消费方对「import 之后再
// 文本 include 同一头」不合并实体（C2953/C2572 族，上游 microsoft/STL
// #6436 同类）。故把 flatten（本文件+全部嵌套 .cpp）的头在此先行包含；
// include guard 幂等，clang/GCC 路径无害。改动此块须同步嵌套文件的头集合。
#include <algorithm>
#include <array>
#include <cstdint>
#include <random>
#include <regex>
#include <string>
#include <string_view>
#include <vector>
#include "test_common.hpp"

#include "tokenizer_consistency_test.cpp"
#undef main

int main()
{
    int failures = test_tokenizer();
    std::printf("\ntokenizer_test: %d failure(s)\n", failures);
    return failures != 0 ? 1 : 0;
}