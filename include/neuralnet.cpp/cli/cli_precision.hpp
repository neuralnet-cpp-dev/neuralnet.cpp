// ── cli_precision.hpp — --precision-* 参数解析的单一实现（#R8）──────────
// mnist_train 与 text_train_config 原先各有一份逐字重复的 parse_precision，
// 此处收敛为一份。行为（可接受词、报错文本、退出码）与两份旧实现完全一致。
//
// allow_t1_58：三值（T1_58）只适用于 param 槽（"该层权重是三值"）；
// compute/stable/optimizer 出现它一律在参数解析处拒绝，而不是拖到 init 期才报
// 能力校验错误（docs/internal/21-quantized-weights.md §4.1）。
#pragma once

#include <cstdlib>
#include <iostream>
#include <string>

#include "neuralnet.cpp/precision.hpp"

namespace nn::cli
{

inline nn::Precision parse_precision(const std::string& name, const char* flag,
                                     bool allow_t1_58 = false)
{
    if (name == "f16" || name == "half") return nn::Precision::F16;
    if (name == "f32" || name == "float") return nn::Precision::F32;
    if (name == "t1_58")
    {
        if (allow_t1_58) return nn::Precision::T1_58;
        std::cerr << "无效 --" << flag << ": t1_58（三值权重只适用于 --precision-param）"
                  << "，可选: f16, f32\n";
        std::exit(1);
    }
    std::cerr << "无效 --" << flag << ": " << name
              << "，可选: " << (allow_t1_58 ? "f16, f32, t1_58" : "f16, f32") << "\n";
    std::exit(1);
}

} // namespace nn::cli
