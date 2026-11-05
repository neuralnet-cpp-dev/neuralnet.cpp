// ── nn.model — C++ 模块：模型容器 / 规格 / 序列化 / 训练会话 ──────────
// 薄接口模块（同 nn.core，D7 实验结论）：GMF 文本包含域头（实体挂全局
// 模块，与 #include 消费方同一实体），`export using namespace nn;` 命名空间级
// re-export——不再逐符号列清单；CMake 侧 NN_MOD_*_HEADERS 是重建触发闭包。
// 约束：宏不可导出（消费方 include <nn_error_macros.hpp> 宏岛头）；STL
// 留在全局片段（消费方自带）。
module;
#include <neuralnet.cpp/model_container.hpp>
#include <neuralnet.cpp/model_serialization.hpp>
#include <neuralnet.cpp/model_spec.hpp>
#include <neuralnet.cpp/train_state.hpp>
#include <neuralnet.cpp/training_session.hpp>
#include <neuralnet.cpp/training_group_session.hpp>
export module nn.model;
export import nn.core;
export using namespace nn;
