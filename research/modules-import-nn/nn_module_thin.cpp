// ── nn_module_thin.cpp — 「薄接口」模块：文本 include nn.hpp 于全局片段，再 re-export 其符号 ──
// STL / nn.hpp 的**声明**都留在全局模块（经 global module fragment 文本包含），
// 模块 nn 只做一件事：把 nn 命名空间的符号 re-export。避免把 418 个系统头纳入 purview。
// 局限：re-export 需逐符号列名（此处用 inline namespace + using 声明的可行子集做测量）。
module;
#include <neuralnet.cpp/nn.hpp>
export module nn;

// 把常用符号经 using 声明导出（测量用非穷尽清单；宏不导出）。
export namespace nn {
    using nn::Matrix;
    using nn::MatrixT;
    using nn::Scalar;
    using nn::Precision;
    using nn::f16;
    using nn::Result;
    using nn::Error;
    using nn::CpuEngine;
    using nn::Tensor;
}
