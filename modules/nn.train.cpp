// ── nn.train — C++ 模块：训练工厂与数据流 ──────────
// 薄接口模块：GMF 文本包含域头（实体挂全局模块，与 #include 消费方可混链）；
// 导出 = `export namespace nn { … }` 逐符号（D7 命名空间级导出已证伪）。
// ⚠ 嵌套名（nn::cli::X / nn::dsl::X / nn::detail::X）必须**镜像原命名空间**
// 导出——`using nn::cli::X;` 写在 nn 块里会被 clang 扁平引入为 nn::X，原限定
// 路径不可见（实测）。宏不可导出（消费方 include <nn_error_macros.hpp>）。
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
export namespace nn {
    using nn::AdvantageConfig;
    using nn::BPETokenizer;
    using nn::BatchAt;
    using nn::BatchSampler;
    using nn::BatchSource;
    using nn::CategoricalDistribution;
    using nn::CharBPETokenizer;
    using nn::CheckpointProvider;
    using nn::ClippedPolicyObjective;
    using nn::ClippedValueObjective;
    using nn::CollectingMetricSink;
    using nn::Dataset;
    using nn::DatasetKind;
    using nn::DatasetSubset;
    using nn::DeterministicRng;
    using nn::EntropyBonusObjective;
    using nn::ExperienceSchema;
    using nn::GaeAdvantage;
    using nn::GroupAdvantage;
    using nn::GroupSpan;
    using nn::GrpoObjective;
    using nn::InMemoryCheckpointProvider;
    using nn::IndexedBatchSource;
    using nn::MaskedNormalizer;
    using nn::MetricRecord;
    using nn::MetricSink;
    using nn::MonteCarloReturn;
    using nn::NNDATASET_FILE_VERSION;
    using nn::NndatasetBuild;
    using nn::PolicyBatchView;
    using nn::PolicyDistribution;
    using nn::PpoConfig;
    using nn::PpoObjective;
    using nn::RLPipelineConfig;
    using nn::RLTrainingPipeline;
    using nn::ReinforceObjective;
    using nn::RolloutSource;
    using nn::RunDriver;
    using nn::RunDriverConfig;
    using nn::RunHooks;
    using nn::RunReport;
    using nn::SampledKlObjective;
    using nn::TensorRecord;
    using nn::TextCheckpointRequest;
    using nn::TextSample;
    using nn::TextSampleBatch;
    using nn::TextSampleSet;
    using nn::TextTrainConfig;
    using nn::TextTrainReport;
    using nn::TextTrainer;
    using nn::TextTrainerHooks;
    using nn::TextWindowConfig;
    using nn::Tokenizer;
    using nn::TrainerHooks;
    using nn::UniformBatchSampler;
    using nn::UpdateOutcome;
    using nn::UpdateRequest;
    using nn::ValueRegressionObjective;
    using nn::encode_docs_parallel;
    using nn::load_tokenizer_from_bytes;
    using nn::load_tokenizer_from_file;
    using nn::load_tokenizer_from_kvrec;
    using nn::load_tokenizer_from_model_blob;
    using nn::load_tokenizer_from_string;
    using nn::make_text_train_config;
    using nn::make_text_train_state;
    using nn::mark_assistant_span;
    using nn::read_text_docs;
    using nn::write_file_label;
    using nn::write_nndataset;
    using nn::write_nnvocab;
}
