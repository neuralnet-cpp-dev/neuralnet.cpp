// ── nn.train — C++ 模块：训练工厂与数据流（TextTrainer/RunDriver/样本/RL 管线） ──────────
// 薄接口模块（同 nn.core，D7 实验结论）：GMF 文本包含域头（实体挂全局
// 模块，与 #include 消费方同一实体），`export using namespace nn;` 命名空间级
// re-export——不再逐符号列清单；CMake 侧 NN_MOD_*_HEADERS 是重建触发闭包。
// 约束：宏不可导出（消费方 include <nn_error_macros.hpp> 宏岛头）；STL
// 留在全局片段（消费方自带）。
module;
#include <neuralnet.cpp/text_trainer.hpp>
#include <neuralnet.cpp/text_train_config.hpp>
#include <neuralnet.cpp/batch_source.hpp>
#include <neuralnet.cpp/run_driver.hpp>
#include <neuralnet.cpp/dataset_samples.hpp>
#include <neuralnet.cpp/rl_experience.hpp>
#include <neuralnet.cpp/rl_policy.hpp>
#include <neuralnet.cpp/rl_pipeline.hpp>
export module nn.train;
export import nn.core;
export using namespace nn;
