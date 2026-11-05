// ── nn.cli — C++ 模块：CLI 公共件（帮助/引擎工厂/GPU 选择/训练入口装配/预热助手） ──────────
// 薄接口模块（同 nn.core，D7 实验结论）：GMF 文本包含域头（实体挂全局
// 模块，与 #include 消费方同一实体），`export using namespace nn;` 命名空间级
// re-export——不再逐符号列清单；CMake 侧 NN_MOD_*_HEADERS 是重建触发闭包。
// 约束：宏不可导出（消费方 include <nn_error_macros.hpp> 宏岛头）；STL
// 留在全局片段（消费方自带）。
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
export using namespace nn;
