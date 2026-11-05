// ── nn.model — C++ 模块：模型容器 / 规格 / 序列化 / 训练会话 ──────────
// 薄接口模块：GMF 文本包含域头（实体挂全局模块，与 #include 消费方可混链）；
// 导出 = `export namespace nn { … }` 逐符号（D7 命名空间级导出已证伪）。
// ⚠ 嵌套名（nn::cli::X / nn::dsl::X / nn::detail::X）必须**镜像原命名空间**
// 导出——`using nn::cli::X;` 写在 nn 块里会被 clang 扁平引入为 nn::X，原限定
// 路径不可见（实测）。宏不可导出（消费方 include <nn_error_macros.hpp>）。
module;
#include <neuralnet.cpp/model_container.hpp>
#include <neuralnet.cpp/model_serialization.hpp>
#include <neuralnet.cpp/model_spec.hpp>
#include <neuralnet.cpp/train_state.hpp>
#include <neuralnet.cpp/training_session.hpp>
#include <neuralnet.cpp/training_group_session.hpp>
export module nn.model;
export import nn.core;
export namespace nn {
    using nn::CheckpointState;
    using nn::DatasetIdentity;
    using nn::MODEL_TAIL_BYTES;
    using nn::MODEL_TAIL_MAGIC;
    using nn::Model;
    using nn::NamedTrainingStepOutput;
    using nn::ResumePolicy;
    using nn::TextTrainState;
    using nn::TrainCounters;
    using nn::TrainState;
    using nn::TrainingConfig;
    using nn::TrainingGroupSession;
    using nn::TrainingSession;
    using nn::TrainingStepOptions;
    using nn::TrainingStepOutput;
    using nn::TrainingStepReport;
    using nn::dataset_mismatch_reason;
    using nn::load_any_model;
    using nn::load_model;
    using nn::load_text_file;
    using nn::load_train_state;
    using nn::model_warmup;
    using nn::peek_any_model_spec;
    using nn::peek_model_role;
    using nn::peek_model_spec;
    using nn::peek_train_state;
    using nn::read_pod;
    using nn::read_pod_span;
    using nn::save_model;
    using nn::sha256_file_hex;
    using nn::sha256_file_prefix;
    using nn::sha256_hex;
    using nn::train_state_from_kv;
    using nn::train_state_to_kv;
    using nn::verify_model_integrity;
    using nn::write_pod;
    using nn::write_pod_span;
    namespace detail {
        using nn::detail::read_matrix_v5;
        using nn::detail::spec_from_kv;
        using nn::detail::spec_to_kv;
    }
}
