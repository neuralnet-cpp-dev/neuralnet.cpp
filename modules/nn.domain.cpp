// ── nn.domain — C++ 模块：模型工厂（MNIST/GPT/RAPT-RLA/CNN） ──────────
// 薄接口模块（同 nn.core，D7 实验结论）：GMF 文本包含域头（实体挂全局
// 模块，与 #include 消费方同一实体），`export using namespace nn;` 命名空间级
// re-export——不再逐符号列清单；CMake 侧 NN_MOD_*_HEADERS 是重建触发闭包。
// 约束：宏不可导出（消费方 include <nn_error_macros.hpp> 宏岛头）；STL
// 留在全局片段（消费方自带）。
module;
#include <neuralnet.cpp/domain_mnist.hpp>
#include <neuralnet.cpp/domain_gpt.hpp>
#include <neuralnet.cpp/domain_rla.hpp>
#include <neuralnet.cpp/domain_cnn.hpp>
export module nn.domain;
export import nn.core;
export using namespace nn;
