// ── nn.tokenizer — C++ 模块：分词器（BPE / CharBPE / 词表持久化） ──────────
// 薄接口模块：GMF 文本包含域头（实体挂全局模块，与 #include 消费方可混链）；
// 导出 = `export namespace nn { … }` 逐符号（D7 命名空间级导出已证伪）。
// ⚠ 嵌套名（nn::cli::X / nn::dsl::X / nn::detail::X）必须**镜像原命名空间**
// 导出——`using nn::cli::X;` 写在 nn 块里会被 clang 扁平引入为 nn::X，原限定
// 路径不可见（实测）。宏不可导出（消费方 include <nn_error_macros.hpp>）。
module;
#include <neuralnet.cpp/domain_tokenizer.hpp>
export module nn.tokenizer;
export import nn.core;
export import nn.kvrecord;
export import nn.dataset;
export namespace nn {
    using nn::BPETokenizer;
    using nn::CharBPETokenizer;
    using nn::Tokenizer;
    using nn::encode_docs_parallel;
    using nn::load_tokenizer_from_bytes;
    using nn::load_tokenizer_from_file;
    using nn::load_tokenizer_from_kvrec;
    using nn::load_tokenizer_from_model_blob;
    using nn::load_tokenizer_from_string;
    using nn::make_tokenizer;
    using nn::peek_tokenizer_type;
}
