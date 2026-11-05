// ── nn.dataset — C++ 模块：统一数据集（.nndataset/.nnvocab） ──────────
// 薄接口模块：GMF 文本包含域头（实体挂全局模块，与 #include 消费方可混链）；
// 导出 = `export namespace nn { … }` 逐符号（D7 命名空间级导出已证伪）。
// ⚠ 嵌套名（nn::cli::X / nn::dsl::X / nn::detail::X）必须**镜像原命名空间**
// 导出——`using nn::cli::X;` 写在 nn 块里会被 clang 扁平引入为 nn::X，原限定
// 路径不可见（实测）。宏不可导出（消费方 include <nn_error_macros.hpp>）。
module;
#include <neuralnet.cpp/dataset.hpp>
#include <neuralnet.cpp/dataset_format.hpp>
export module nn.dataset;
export import nn.core;
export import nn.kvrecord;
export namespace nn {
    using nn::Dataset;
    using nn::DatasetInfo;
    using nn::DatasetKind;
    using nn::DatasetSubset;
    using nn::FILE_LABEL_SIZE;
    using nn::FileLabel;
    using nn::NNDATASET_FILE_VERSION;
    using nn::NNDATASET_MAGIC;
    using nn::NNVOCAB_FILE_VERSION;
    using nn::NNVOCAB_MAGIC;
    using nn::NndatasetBuild;
    using nn::TextCorpus;
    using nn::TextDoc;
    using nn::check_file_label;
    using nn::decode_vocab_blob;
    using nn::doc_block_size;
    using nn::encode_doc_block;
    using nn::encode_vocab_blob;
    using nn::mark_assistant_span;
    using nn::parse_nnvocab;
    using nn::read_file_label;
    using nn::read_text_docs;
    using nn::write_file_label;
    using nn::write_nndataset;
    using nn::write_nnvocab;
}
