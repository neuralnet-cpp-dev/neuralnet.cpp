// ── nn.opt — C++ 模块：优化器 + 学习率调度 ──────────
// 薄接口模块：GMF 文本包含域头（实体挂全局模块，与 #include 消费方可混链）；
// 导出 = `export namespace nn { … }` 逐符号（D7 命名空间级导出已证伪）。
// ⚠ 嵌套名（nn::cli::X / nn::dsl::X / nn::detail::X）必须**镜像原命名空间**
// 导出——`using nn::cli::X;` 写在 nn 块里会被 clang 扁平引入为 nn::X，原限定
// 路径不可见（实测）。宏不可导出（消费方 include <nn_error_macros.hpp>）。
module;
#include <neuralnet.cpp/compute_optimizer.hpp>
#include <neuralnet.cpp/lr_scheduler.hpp>
export module nn.opt;
export import nn.core;
export namespace nn {
    using nn::Adam;
    using nn::AdamW;
    using nn::AdamWInt8;
    using nn::AdamWQuant;
    using nn::LrScheduleConfig;
    using nn::Optimizer;
    using nn::OptimizerMeta;
    using nn::StepLrScheduleConfig;
    using nn::compute_epoch_lr;
    using nn::compute_step_lr;
    using nn::create_optimizer;
    using nn::find_optimizer;
    using nn::optimizer_names;
    using nn::optimizer_table;
    namespace cli {
        using nn::LrScheduleConfig;
        using nn::compute_epoch_lr;
    }
}
