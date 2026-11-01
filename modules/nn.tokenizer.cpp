// ── nn.tokenizer — C++ 模块：分词器（BPE / CharBPE / 词表持久化）─────────
// 薄接口模块（见 nn.core.cpp 头注释）：依赖域（nn.core / nn.kvrecord /
// nn.dataset）经 `export import` re-export，消费方只 `import nn.tokenizer;`
// 即可连带拿到全部四个域的公开符号（tokenizer 闭包 = 12 头，spike 实测）。
module;
#include <neuralnet.cpp/domain_tokenizer.hpp>
export module nn.tokenizer;
export import nn.core;
export import nn.kvrecord;
export import nn.dataset;

export namespace nn {
    // domain_tokenizer_base.hpp
    using nn::Tokenizer;
    using nn::encode_docs_parallel;
    // domain_tokenizer_bpe.hpp
    using nn::BPETokenizer;
    // domain_tokenizer_charbpe.hpp
    using nn::CharBPETokenizer;
    using nn::peek_tokenizer_type;
    using nn::make_tokenizer;
    using nn::load_tokenizer_from_string;
    using nn::load_tokenizer_from_kvrec;
    using nn::load_tokenizer_from_bytes;
    using nn::load_tokenizer_from_model_blob;
    using nn::load_tokenizer_from_file;
}
