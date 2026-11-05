// ── nn.layer — C++ 模块：层家族 + 位置编码 ──────────
// 薄接口模块：GMF 文本包含域头（实体挂全局模块，与 #include 消费方可混链）；
// 导出 = `export namespace nn { … }` 逐符号（D7 命名空间级导出已证伪）。
// ⚠ 嵌套名（nn::cli::X / nn::dsl::X / nn::detail::X）必须**镜像原命名空间**
// 导出——`using nn::cli::X;` 写在 nn 块里会被 clang 扁平引入为 nn::X，原限定
// 路径不可见（实测）。宏不可导出（消费方 include <nn_error_macros.hpp>）。
module;
#include <neuralnet.cpp/compute_layer.hpp>
export module nn.layer;
export import nn.core;
export namespace nn {
    using nn::ActivationOffloader;
    using nn::ActivationType;
    using nn::BatchNorm;
    using nn::BitLinear;
    using nn::CausalSelfAttention;
    using nn::CnnSpec;
    using nn::Conv2D;
    using nn::CrossAttention;
    using nn::Embedding;
    using nn::FeedForward;
    using nn::GPTBlock;
    using nn::GPTModel;
    using nn::GeLU;
    using nn::GptSpec;
    using nn::Layer;
    using nn::LayerNorm;
    using nn::Linear;
    using nn::MaxPool2D;
    using nn::MlpSpec;
    using nn::ModelRegistry;
    using nn::ModelRole;
    using nn::ModelSpec;
    using nn::ModelSpecBase;
    using nn::ModelType;
    using nn::MultiHeadAttention;
    using nn::NormPlace;
    using nn::NormType;
    using nn::OutputBundleLayer;
    using nn::PosEncodingType;
    using nn::PrecisionSet;
    using nn::PrecisionSupport;
    using nn::RAPTModel;
    using nn::RMSNorm;
    using nn::RaptSpec;
    using nn::ReLU;
    using nn::ReLU2GLU;
    using nn::ReLULinearAttention;
    using nn::RotaryEmbedding;
    using nn::RowBandPacking;
    using nn::SamplingConfig;
    using nn::Softmax;
    using nn::SwiGLU;
    using nn::TransformerEncoderLayer;
    using nn::TransformerSpec;
    using nn::WeightLoadIntent;
    using nn::WeightQuant;
    using nn::make_norm_layer;
    using nn::model_role_name;
    using nn::norm_place_name;
    using nn::rounding_precision_set;
    using nn::sample_next_token_;
    using nn::spec_matches;
    using nn::spec_summary;
    namespace dsl {
        using nn::dsl::rsqrt;
    }
    namespace expr {
        using nn::expr::AttnMaskKind;
        using nn::expr::make_fold_attn_o;
    }
}
