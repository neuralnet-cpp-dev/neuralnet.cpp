// ── nn.dataset — C++ 模块：统一数据集（.nndataset / .nnvocab / doc 块）────
// 薄接口模块（见 nn.core.cpp 头注释）：依赖域（nn.core / nn.kvrecord）经
// `export import` re-export，消费方只 `import nn.dataset;` 即可连带拿到
// core 与 kvrecord 的公开符号。
// dataset_samples.hpp（训练样本切窗）依赖 L1 algebra_matrix，本轮不模块化。
module;
#include <neuralnet.cpp/dataset.hpp>
#include <neuralnet.cpp/dataset_format.hpp>
export module nn.dataset;
export import nn.core;
export import nn.kvrecord;

export namespace nn {
    // dataset_format.hpp
    using nn::NNVOCAB_MAGIC;
    using nn::NNDATASET_MAGIC;
    using nn::NNVOCAB_FILE_VERSION;
    using nn::NNDATASET_FILE_VERSION;
    using nn::FILE_LABEL_SIZE;
    using nn::write_file_label;
    using nn::FileLabel;
    using nn::read_file_label;
    using nn::check_file_label;
    using nn::encode_vocab_blob;
    using nn::decode_vocab_blob;
    using nn::write_nnvocab;
    using nn::parse_nnvocab;
    // dataset.hpp
    using nn::DatasetKind;
    using nn::DatasetSubset;
    using nn::DatasetInfo;
    using nn::TextCorpus;
    using nn::TextDoc;
    using nn::read_text_docs;
    using nn::mark_assistant_span;
    using nn::doc_block_size;
    using nn::encode_doc_block;
    using nn::NndatasetBuild;
    using nn::write_nndataset;
    using nn::Dataset;
}
