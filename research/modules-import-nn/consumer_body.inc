// ── consumer_body.inc — 两个 consumer 的**完全相同**的工作负载 ──────────────
// include 版与 import 版执行逐字相同的代码，保证测量可比。
// 只用非宏 API（宏不随 named module 导出）；不 include 任何 STL 头。
// 强制实例化若干模板（MatrixT/f16/Result 路径），使 TU 不是空壳。

double run_workload()
{
    using namespace nn;
    MatrixT<Precision::F32> a(8, 8, 0.5f);
    MatrixT<Precision::F32> b(8, 8);
    for (std::size_t i = 0; i < 8; ++i)
        for (std::size_t j = 0; j < 8; ++j)
            b.set_value(i, j, static_cast<Scalar>(i + j) * 0.25f);
    Scalar acc = 0;
    for (std::size_t i = 0; i < 8; ++i)
        for (std::size_t j = 0; j < 8; ++j)
            acc += a.at(i, j) * b.at(i, j);
    f16 h{0.375f};
    acc += static_cast<float>(h);
    return static_cast<double>(acc);
}
