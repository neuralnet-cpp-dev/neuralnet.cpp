// ═══════════════════════════════════════════════════════════════════════════
//  nn_api.cpp — libnn.so 的 C ABI 实现（Phase 3-A 最小闭环）
//
//  门面层：只做"句柄 ↔ 内部对象"转换 + 序列化解码，真正的计算全部走库内引擎
//  （nn::CpuEngine / GpuEngine + dsl / eval_expr）。libnn.so 编具体引擎 + 原语；
//  C++ 模板面（nn::dsl / Layer 组合）留在 header 由消费方实例化（docs/internal/34）。
//
//  句柄语义：nnn_tensor 持有自包含的 nn::Tensor（CPU = Matrix 拷贝；GPU =
//  shared_ptr<GpuBuffer> 自带 backend keepalive）→ 释放引擎后 tensor 仍有效。
//  跨边界错误用 nnn_status + 线程本地 last_error（STL std::expected 不跨界）。
// ═══════════════════════════════════════════════════════════════════════════

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>

#include <neuralnet.cpp/nn.hpp>
#include <neuralnet.cpp/nn_api.h>
#include <neuralnet.cpp/kernel_cache.hpp>   // decode_expr_spec（eval_expr 边界载荷）

using namespace nn;

// 句柄本体：定义在全局作用域，与 nn_api.h 的 `struct nnn_engine_t` 前向声明
// 是**同一类型**（匿名命名空间里会另造一份 → 句柄指针类型不匹配）。
struct nnn_engine_t
{
    std::shared_ptr<ComputeEngine> e;
};

struct nnn_tensor_t
{
    Tensor t;
};

namespace
{

thread_local std::string g_last_error;

inline void set_err(const std::string& msg) { g_last_error = msg; }

// 错误统一路径：失败时设 last_error + 返回 NNN_ERR
template <typename T>
[[nodiscard]] inline bool try_ok(const Result<T>& r, T* out)
{
    if (r) { *out = *r; return true; }
    set_err(r.error().message);
    return false;
}

inline Precision to_precision(int prec)
{
    return (prec == static_cast<int>(NNN_PREC_F16)) ? Precision::F16 : Precision::F32;
}

} // namespace

extern "C"
{

const char* nnn_last_error(void) { return g_last_error.empty() ? nullptr : g_last_error.c_str(); }

nnn_engine nnn_engine_create_cpu(void)
{
    auto* e = new (std::nothrow) nnn_engine_t;
    if (!e) { set_err("engine 分配失败"); return nullptr; }
    e->e = CpuEngine::create();
    return e;
}

nnn_engine nnn_engine_create_gpu(const char* device)
{
#ifdef NN_HAS_VULKAN
    auto* e = new (std::nothrow) nnn_engine_t;
    if (!e) { set_err("engine 分配失败"); return nullptr; }
    auto backend = GpuBackend::shared();
    std::string sel = device ? device : "";
    auto r = backend->initialize(sel);
    if (!r)
    {
        set_err("GPU 初始化失败（NNN_NO_GPU）：" + r.error().message);
        delete e;
        return nullptr;
    }
    e->e = GpuEngine::create(*backend);
    return e;
#else
    (void)device;
    set_err("本构建无 Vulkan（NNN_NO_GPU）");
    return nullptr;
#endif
}

void nnn_engine_destroy(nnn_engine e)
{
    delete e;
}

nnn_tensor nnn_tensor_create(nnn_engine e, size_t rows, size_t cols, int prec)
{
    if (!e) { set_err("null engine"); return nullptr; }
    auto* t = new (std::nothrow) nnn_tensor_t;
    if (!t) { set_err("tensor 分配失败"); return nullptr; }
    auto r = e->e->try_create_tensor(rows, cols, to_precision(prec));
    if (!r) { set_err(r.error().message); delete t; return nullptr; }
    t->t = std::move(*r);
    return t;
}

nnn_tensor nnn_tensor_from_matrix(nnn_engine e, const float* data, size_t rows, size_t cols,
                                  int prec)
{
    if (!e) { set_err("null engine"); return nullptr; }
    Matrix m(rows, cols);
    if (data) std::memcpy(m.span().data(), data, rows * cols * sizeof(float));
    else std::memset(m.span().data(), 0, rows * cols * sizeof(float));
    auto* t = new (std::nothrow) nnn_tensor_t;
    if (!t) { set_err("tensor 分配失败"); return nullptr; }
    auto r = e->e->from_matrix(m, to_precision(prec));
    if (!r) { set_err(r.error().message); delete t; return nullptr; }
    t->t = std::move(*r);
    return t;
}

size_t nnn_tensor_rows(nnn_engine e, nnn_tensor t)
{
    (void)e;
    return t ? t->t.rows() : 0;
}
size_t nnn_tensor_cols(nnn_engine e, nnn_tensor t)
{
    (void)e;
    return t ? t->t.cols() : 0;
}
int nnn_tensor_precision(nnn_engine e, nnn_tensor t)
{
    (void)e;
    if (!t) return -1;
    return static_cast<int>(t->t.precision());
}

nnn_status nnn_tensor_to_matrix(nnn_engine e, nnn_tensor t, float* out, size_t cap, size_t* out_n)
{
    if (!e || !t) { set_err("null engine/tensor"); return NNN_ERR; }
    auto r = e->e->to_matrix(t->t);
    if (!r) { set_err(r.error().message); return NNN_ERR; }
    const Matrix& m = *r;
    const size_t n = m.rows() * m.cols();
    if (out_n) *out_n = n;
    if (out == nullptr) return NNN_OK;               // 只查形状
    if (cap < n) { set_err("to_matrix 缓冲容量不足"); return NNN_ERR; }
    std::memcpy(out, m.span().data(), n * sizeof(float));
    return NNN_OK;
}

void nnn_tensor_destroy(nnn_engine e, nnn_tensor t)
{
    (void)e;
    delete t;
}

nnn_status nnn_matmul(nnn_engine e, nnn_tensor A, nnn_tensor B, int transA, int transB,
                      nnn_tensor* out)
{
    if (!e || !A || !B || !out) { set_err("null 参数"); return NNN_ERR; }
    auto r = e->e->matmul(A->t, B->t, transA != 0, transB != 0, Precision::F32);
    if (!r) { set_err(r.error().message); return NNN_ERR; }
    auto* t = new (std::nothrow) nnn_tensor_t;
    if (!t) { set_err("tensor 分配失败"); return NNN_ERR; }
    t->t = std::move(*r);
    *out = t;
    return NNN_OK;
}

nnn_status nnn_eval_expr(nnn_engine e, const uint8_t* spec, size_t len, const nnn_tensor* inputs,
                         size_t nin, uint32_t prec_sig, size_t rows, size_t cols, nnn_tensor* out)
{
    if (!e || !spec || !out) { set_err("null 参数"); return NNN_ERR; }
    auto spec_r = nn::fused::decode_expr_spec(spec, len);
    if (!spec_r) { set_err(spec_r.error().message); return NNN_ERR; }
    std::vector<Tensor> ins;
    ins.reserve(nin);
    for (size_t i = 0; i < nin; ++i) ins.push_back(inputs[i]->t);
    auto r = e->e->eval_expr(*spec_r, ins, rows, cols,
                                  (prec_sig & 0x10000u) ? Precision::F16 : Precision::F32);
    if (!r) { set_err(r.error().message); return NNN_ERR; }
    auto* t = new (std::nothrow) nnn_tensor_t;
    if (!t) { set_err("tensor 分配失败"); return NNN_ERR; }
    t->t = std::move(*r);
    *out = t;
    return NNN_OK;
}

} // extern "C"
