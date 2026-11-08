// ── cli_jit_warmup.hpp — JIT 预热（P3/D5，docs/internal/35）──────────────────
//
// 构建期 AOT 收集链删除后，融合 shader 全靠运行期 JIT（libnn-kernelgen.so）
// 现编并持久化进 .nnk 缓存。本助手在 GPU 上跑一遍**代表性 forward**
// （nn::model_warmup），把该模型用到的表达式全部预编进缓存：
//   · 消除训练/推理首步的 JIT 冷启动延迟（后续 = 缓存命中）；
//   · 配合 `nn_warmup` 构建期目标（常用架构预热产物随发布分发），
//     部署环境首跑可完全不依赖 kernelgen。
//
// 约定：
//   · CPU 引擎直接跳过（模板路径无需预热）；
//   · `--no-warmup`（各 app 的布尔开关）跳过；
//   · 失败由调用方 `NN_EXIT(..., 1, "预热失败: ")` 上抛（不静默降级）。
// ─────────────────────────────────────────────────────────────────────────────
#pragma once

#include <chrono>
#include <iostream>
#include <utility>

#include "neuralnet.cpp/compute_engine.hpp"
#include "neuralnet.cpp/compute_tensor.hpp"   // Device
#include "neuralnet.cpp/model_container.hpp"  // nn::model_warmup / Model

namespace nn::cli
{

// 跑一遍代表性 forward（输入形状由调用方按架构构造：值不重要、形状重要——
// JIT 键只含结构；GPU 才执行）。
[[nodiscard]] inline Result<void> jit_warmup(Model& model, ComputeEngine& engine,
                                             const Tensor& input, bool enabled)
{
    if (!enabled) return {};
    if (engine.device() == Device::CPU) return {};
    const auto t0 = std::chrono::steady_clock::now();
    NN_TRY(r, model_warmup(model, input));
    (void)r;
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
    std::cout << "[warmup] JIT 预热完成（" << ms
              << " ms，表达式已现编进 .nnk 缓存；--no-warmup 可跳过）\n";
    return {};
}

}  // namespace nn::cli
