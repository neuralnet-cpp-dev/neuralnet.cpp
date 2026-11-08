// ── nn.cli — C++ 模块：CLI 公共件 ──────────
// 薄接口模块：GMF 文本包含域头（实体挂全局模块，与 #include 消费方可混链）；
// 导出 = `export namespace nn { … }` 逐符号（D7 命名空间级导出已证伪）。
// ⚠ 嵌套名（nn::cli::X / nn::dsl::X / nn::detail::X）必须**镜像原命名空间**
// 导出——`using nn::cli::X;` 写在 nn 块里会被 clang 扁平引入为 nn::X，原限定
// 路径不可见（实测）。宏不可导出（消费方 include <nn_error_macros.hpp>）。
module;
#include <neuralnet.cpp/cli/cli_help.hpp>
#include <neuralnet.cpp/cli/cli_engine_factory.hpp>
#include <neuralnet.cpp/cli/cli_gpu_option.hpp>
#include <neuralnet.cpp/cli/cli_lr_scheduler.hpp>
#include <neuralnet.cpp/cli/cli_mnist_io.hpp>
#include <neuralnet.cpp/cli/cli_precision_warning.hpp>
#include <neuralnet.cpp/cli/cli_train_common.hpp>
#include <neuralnet.cpp/cli/text_train_checkpoint.hpp>
#include <neuralnet.cpp/cli/text_train_config.hpp>
#include <neuralnet.cpp/cli/text_train_loop.hpp>
#include <neuralnet.cpp/cli/text_train_recovery.hpp>
#include <neuralnet.cpp/cli/cli_jit_warmup.hpp>
export module nn.cli;
export import nn.core;
export namespace nn {
    namespace cli {
        using nn::cli::EngineConfig;
        using nn::cli::Help;
        using nn::cli::ResumeStateAdoption;
        using nn::cli::TextTrainCheckpointLog;
        using nn::cli::TextTrainMemProbe;
        using nn::cli::TrainCommonArgs;
        using nn::cli::TrainConfig;
        using nn::cli::adopt_resume_state;
        using nn::cli::create_engine;
        using nn::cli::dangerous_precision_warning;
        using nn::cli::evaluate_mnist;
        using nn::cli::is_forbidden_resume_flag;
        using nn::cli::jit_warmup;
        using nn::cli::load_mnist_csv;
        using nn::cli::load_optimizer_state;
        using nn::cli::make_text_train_mem_probe;
        using nn::cli::parse_gpu_option;
        using nn::cli::parse_text_train_args;
        using nn::cli::parse_train_common_args;
        using nn::cli::precision_offload_combo_error;
        using nn::cli::resume_text_train;
        using nn::cli::run_text_train_loop;
        using nn::cli::save_ckpt_with_state;
        using nn::cli::save_text_train_model;
        using nn::cli::text_train_ckpt_exit_code;
        using nn::cli::text_train_ckpt_path;
        using nn::cli::verify_dataset_identity;
    }
}
