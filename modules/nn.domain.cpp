// ── nn.domain — C++ 模块：模型工厂（MNIST/GPT/RAPT-RLA/CNN） ──────────
// 薄接口模块：GMF 文本包含域头（实体挂全局模块，与 #include 消费方可混链）；
// 导出 = `export namespace nn { … }` 逐符号（D7 命名空间级导出已证伪）。
// ⚠ 嵌套名（nn::cli::X / nn::dsl::X / nn::detail::X）必须**镜像原命名空间**
// 导出——`using nn::cli::X;` 写在 nn 块里会被 clang 扁平引入为 nn::X，原限定
// 路径不可见（实测）。宏不可导出（消费方 include <nn_error_macros.hpp>）。
module;
#include <neuralnet.cpp/domain_mnist.hpp>
#include <neuralnet.cpp/domain_gpt.hpp>
#include <neuralnet.cpp/domain_rla.hpp>
#include <neuralnet.cpp/domain_cnn.hpp>
export module nn.domain;
export import nn.core;
export namespace nn {
    using nn::CnnConfig;
    using nn::CnnConvSpec;
    using nn::GptConfig;
    using nn::MNIST_CNN_CONVS;
    using nn::MNIST_CNN_FC;
    using nn::MNIST_IMG_SIZE;
    using nn::MNIST_INPUT_DIM;
    using nn::MNIST_LAYER_DIMS;
    using nn::MNIST_NUM_CLASSES;
    using nn::MNIST_PATCH_SIZE;
    using nn::MNIST_TF_D_FF;
    using nn::MNIST_TF_D_MODEL;
    using nn::MNIST_TF_NUM_HEADS;
    using nn::MNIST_TF_NUM_LAYERS;
    using nn::RAPTConfig;
    using nn::build_cnn_model;
    using nn::build_cnn_model_from_spec;
    using nn::build_gpt_model;
    using nn::build_gpt_model_from_spec;
    using nn::build_mnist_mlp_model;
    using nn::build_mnist_model_from_spec;
    using nn::build_mnist_transformer_model;
    using nn::build_rapt_model;
    using nn::build_rapt_model_from_spec;
    using nn::cnn_config_from_spec;
    using nn::load_image_from_csv_line;
    using nn::make_cnn_spec;
    using nn::make_gpt_spec;
    using nn::make_mnist_transformer_spec;
    using nn::make_rapt_spec;
}
