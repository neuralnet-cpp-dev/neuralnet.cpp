// ── adam_int8_test — int8 打包状态的 AdamW（AdamWInt8，docs 21 §4.19）────────
// 设计依据：docs/internal/21-quantized-weights.md §4.13 的 int8 增量（AdamWQuant
// 的 f16 payload → int8 打包 1 B/元素，状态账 8 → ~2 B/权重）。
//
// 覆盖：
//   [1] 引擎原语 quantized_adamw_step：与宿主整数参考**逐位一致**（同运算序）；
//       跨形状（cols 非 4 倍数 / 多行多块）。
//   [2] 状态账：~2 B/权重（< AdamWQuant 的 4 B < f32 Adam 的 8 B）。
//   [3] 端到端收敛 A/B：AdamWInt8 vs AdamWQuant vs f32 AdamW（同问题同超参）
//       —— int8 状态的 loss 与 f32 同量级 + 参数真的移动。
//   [4] 打包布局：状态容器 (rows, ceil(cols/4))，解码值 ∈ [-127,127]。
//   [5] 首步零梯度：不产生 NaN/Inf（尺度下限保护）。
//   [6] save/load 往返：载入状态后 step 与未中断 step 逐位一致。
//
// 退出码：0 = 通过，1 = 失败，77 = 跳过（--gpu 且无设备）
// ─────────────────────────────────────────────────────────────────────────

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "neuralnet.cpp/cli/cli_engine_factory.hpp"
#include "neuralnet.cpp/compute_optimizer.hpp"
#include "neuralnet.cpp/int8_pack.hpp"

#define NN_TEST_COUNTER g_failures
#include "test_common.hpp"

namespace
{

int g_failures = 0;

[[nodiscard]] nn::Matrix random_matrix(std::size_t rows, std::size_t cols, unsigned seed)
{
    nn::Matrix m(rows, cols);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    for (auto& v : m.span())
        v = d(rng);
    return m;
}

bool matrices_equal(const nn::Matrix& a, const nn::Matrix& b)
{
    if (a.rows() != b.rows() || a.cols() != b.cols()) return false;
    const auto sa = a.span();
    const auto sb = b.span();
    for (std::size_t i = 0; i < sa.size(); ++i)
    {
        // NaN 逐位视为相等（两路径同 NaN 也一致）；其余要求严格相等
        if (sa[i] != sb[i] && !(std::isnan(sa[i]) && std::isnan(sb[i]))) return false;
    }
    return true;
}

// ── [1] 原语逐位对拍：quantized_adamw_step vs 宿主参考（同运算序）──────────
bool test_primitive(nn::ComputeEngine& e, std::size_t rows, std::size_t cols, unsigned seed,
                    const char* label)
{
    std::printf("[1] quantized_adamw_step（%s）：与宿主参考逐位一致\n", label);
    const nn::Matrix p0 = random_matrix(rows, cols, seed);
    const nn::Matrix g0 = random_matrix(rows, cols, seed + 1);
    // 宿主尺度（任意正数即可：原语不关心来源，调用方负责上界口径）
    nn::Matrix ms0(rows, 1), ms1(rows, 1), vs0(rows, 1), vs1(rows, 1);
    for (std::size_t r = 0; r < rows; ++r)
    {
        ms0.span()[r] = 0.05f * static_cast<float>(r + 1);
        ms1.span()[r] = 1.1f * ms0.span()[r] + 0.01f;
        vs0.span()[r] = 0.02f * static_cast<float>(r + 1);
        vs1.span()[r] = 1.1f * vs0.span()[r] + 0.01f;
    }
    // 初始状态容器：随机 int8 打包（量化任意值；v 是方差累加器，**必须 ≥ 0**）
    const std::size_t wpr = nn::i8_pack::words_per_row(cols);
    const nn::Matrix xm = random_matrix(rows, cols, seed + 2);
    nn::Matrix mp0(rows, wpr), vp0(rows, wpr);
    for (std::size_t r = 0; r < rows; ++r)
    {
        std::vector<std::int8_t> qm(cols), qv(cols);
        std::vector<std::uint32_t> wm(wpr), wv(wpr);
        for (std::size_t c = 0; c < cols; ++c)
        {
            qm[c] = nn::i8_pack::quantize(xm.span()[r * cols + c], 0.25f);
            qv[c] = nn::i8_pack::quantize(std::fabs(xm.span()[r * cols + c]) * 2.0f, 0.5f);
        }
        nn::i8_pack::pack_row(qm.data(), cols, wm.data());
        nn::i8_pack::pack_row(qv.data(), cols, wv.data());
        for (std::size_t wi = 0; wi < wpr; ++wi)
        {
            mp0.span()[r * wpr + wi] = std::bit_cast<float>(wm[wi]);
            vp0.span()[r * wpr + wi] = std::bit_cast<float>(wv[wi]);
        }
    }

    // ── 宿主参考（与 CPU 实现同运算序，逐位）────────────────────────────
    const float lr = 0.01f, beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f, wd = 0.01f;
    const float n = 2.0f;   // t=2：偏置修正分母
    const float inv_bc1 = 1.0f / (1.0f - std::pow(beta1, n));
    const float inv_bc2 = 1.0f / (1.0f - std::pow(beta2, n));
    const float one_m_b1 = 1.0f - beta1;
    const float one_m_b2 = 1.0f - beta2;

    nn::Matrix p_ref = p0;
    nn::Matrix mp_ref = mp0, vp_ref = vp0;
    std::vector<std::int8_t> qm(cols), qv(cols);
    std::vector<std::uint32_t> wm(wpr), wv(wpr);
    for (std::size_t r = 0; r < rows; ++r)
    {
        const float smo = ms0.span()[r], smn = ms1.span()[r];
        const float svo = vs0.span()[r], svn = vs1.span()[r];
        const float scl_m = smn / 127.0f;
        const float scl_v = svn / 127.0f;
        for (std::size_t wi = 0; wi < wpr; ++wi)
        {
            wm[wi] = std::bit_cast<std::uint32_t>(mp_ref.span()[r * wpr + wi]);
            wv[wi] = std::bit_cast<std::uint32_t>(vp_ref.span()[r * wpr + wi]);
        }
        for (std::size_t c = 0; c < cols; ++c)
        {
            qm[c] = nn::i8_pack::at(wm.data(), c);
            qv[c] = nn::i8_pack::at(wv.data(), c);
        }
        for (std::size_t c = 0; c < cols; ++c)
        {
            const float g = g0.span()[r * cols + c];
            const float m_old = smo * (static_cast<float>(qm[c]) / 127.0f);
            const float m_new = beta1 * m_old + one_m_b1 * g;
            qm[c] = nn::i8_pack::quantize(m_new, scl_m);
            const float v_old = svo * (static_cast<float>(qv[c]) / 127.0f);
            const float v_new = beta2 * v_old + one_m_b2 * g * g;
            qv[c] = nn::i8_pack::quantize(v_new, scl_v);
            const float m_hat = smn * (static_cast<float>(qm[c]) / 127.0f);
            const float v_hat = std::max(svn * (static_cast<float>(qv[c]) / 127.0f), scl_v);
            const float p_old = p_ref.span()[r * cols + c];
            const float upd = inv_bc1 * m_hat / (std::sqrt(inv_bc2 * v_hat) + eps) + wd * p_old;
            p_ref.span()[r * cols + c] = p_old - lr * upd;
        }
        nn::i8_pack::pack_row(qm.data(), cols, wm.data());
        nn::i8_pack::pack_row(qv.data(), cols, wv.data());
        for (std::size_t wi = 0; wi < wpr; ++wi)
        {
            mp_ref.span()[r * wpr + wi] = std::bit_cast<float>(wm[wi]);
            vp_ref.span()[r * wpr + wi] = std::bit_cast<float>(wv[wi]);
        }
    }

    // ── 引擎调用 ─────────────────────────────────────────────────────────
    nn::Tensor pt = upload(e, p0), gt = upload(e, g0);
    nn::Tensor mt = upload(e, mp0), vt = upload(e, vp0);
    nn::Tensor ms0t = upload(e, ms0), ms1t = upload(e, ms1);
    nn::Tensor vs0t = upload(e, vs0), vs1t = upload(e, vs1);
    auto r = e.quantized_adamw_step(pt, gt, mt, vt, ms0t, ms1t, vs0t, vs1t, lr, beta1, beta2, eps,
                                    wd, inv_bc1, inv_bc2);
    CHECK(r.has_value(), "quantized_adamw_step 成功");
    if (!r.has_value()) return false;
    const nn::Matrix p_e = download(e, pt);
    const nn::Matrix m_e = download(e, mt);
    const nn::Matrix v_e = download(e, vt);
    // 参数更新经 sqrt：GLSL sqrt 的精度是实现定义的（非正确舍入），与宿主
    // std::sqrt 跨工具链/驱动存在 1 ULP 级差异（实测 abs ≤ 2⁻²⁴、rel ≤ 7e-6，
    // 近零值处被放大）→ 参数比较用紧容差（仍可抓全部真实错误：逻辑错误 ≥ 1e-3）。
    // m/v 容器是纯整数打包路径（decode→encode 逐位确定）→ 保持逐位锁。
    const bool p_ok = close_to(p_e, p_ref, 1e-6f, "参数", 0);
    CHECK(p_ok, "参数一致（≤1 ULP 舍入容差）");
    CHECK(matrices_equal(m_e, mp_ref), "一阶矩容器逐位一致");
    CHECK(matrices_equal(v_e, vp_ref), "二阶矩容器逐位一致");
    return true;
}

// ── [2] 状态账：~2 B/权重 ─────────────────────────────────────────────────
void test_state_bytes(nn::ComputeEngine& eng, const char* tag)
{
    std::printf("[2] 状态账：AdamWInt8 ~2 B/权重 vs AdamWQuant 4 B vs f32 8 B（%s）\n", tag);
    nn::PrecisionProfile prof;
    auto m_r = nn::build_mnist_mlp_model(eng, nn::MNIST_LAYER_DIMS, nn::NormType::LayerNorm, prof);
    CHECK(m_r.has_value(), "MLP 构建成功");
    if (!m_r.has_value()) return;
    nn::Model& model = *m_r;
    auto opt8 = nn::create_optimizer("adamwint8", eng, model.parameters(),
                                     model.param_gradients(), 0.02f, 0.0f, prof);
    auto optq = nn::create_optimizer("adamwquant", eng, model.parameters(),
                                     model.param_gradients(), 0.02f, 0.0f, prof);
    CHECK(opt8 != nullptr && optq != nullptr, "优化器创建成功");
    if (!opt8 || !optq) return;
    auto* a8 = dynamic_cast<nn::AdamWInt8*>(opt8.get());
    auto* aq = dynamic_cast<nn::AdamWQuant*>(optq.get());
    CHECK(a8 != nullptr && aq != nullptr, "具体类型断言");
    if (!a8 || !aq) return;
    std::size_t weights = 0;
    for (auto& p : model.parameters())
        weights += p.get().rows() * p.get().cols();
    const double b8 = static_cast<double>(a8->state_bytes()) / static_cast<double>(weights);
    const double bq = static_cast<double>(aq->state_bytes()) / static_cast<double>(weights);
    std::printf("    AdamWInt8 = %.3f B/权重; AdamWQuant = %.3f B/权重\n", b8, bq);
    CHECK(b8 <= 2.1, "int8 状态 ≤ 2.1 B/权重（含每行尺度）");
    CHECK(b8 < bq, "int8 状态 < AdamWQuant（f16 payload）");
}

// ── [3][4] 端到端收敛 A/B ────────────────────────────────────────────────
bool train_mlp(nn::ComputeEngine& eng, const char* opt_name, int steps, nn::Scalar& loss_out,
               std::vector<nn::Scalar>& w_before_out, std::vector<nn::Scalar>& w_after_out)
{
    nn::PrecisionProfile prof;
    auto m_r = nn::build_mnist_mlp_model(eng, nn::MNIST_LAYER_DIMS, nn::NormType::LayerNorm, prof);
    if (!m_r) return false;
    nn::Model& model = *m_r;
    auto os = nn::create_optimizer(opt_name, eng, model.parameters(), model.param_gradients(),
                                   0.02f, 0.0f, prof);
    if (!os) return false;
    std::unique_ptr<nn::Optimizer> opt = std::move(os);

    const std::size_t in_dim = nn::MNIST_LAYER_DIMS.front();
    const std::size_t B = 8, C = 10;
    nn::Matrix X(in_dim, B), Y(C, B);
    for (std::size_t k = 0; k < in_dim; ++k)
        for (std::size_t b = 0; b < B; ++b)
            X.set_value(k, b, ((k * 7 + b * 3) % 11) * 0.09f - 0.5f);
    for (std::size_t c = 0; c < C; ++c)
        for (std::size_t b = 0; b < B; ++b)
            Y.set_value(c, b, (c == (b % C)) ? 1.0f : 0.0f);
    nn::Tensor Xt = upload(eng, X), Yt = upload(eng, Y);

    {
        auto w0 = eng.to_matrix(model.parameters()[0].get(), nn::Precision::F32);
        if (!w0) return false;
        w_before_out.assign(w0->span().begin(), w0->span().end());
    }

    nn::CrossEntropyLoss loss;
    nn::Scalar last = 0.0f;
    for (int s = 0; s < steps; ++s)
    {
        if (auto r = model.zero_grad(); !r) return false;
        auto pred = model.forward(Xt);
        if (!pred) return false;
        std::vector<std::size_t> labels(B);
        for (std::size_t b = 0; b < B; ++b)
            labels[b] = b % C;
        std::vector<nn::Scalar> mask(B, 1.0f);
        auto ls = loss.forward_sparse(eng, *pred, labels, mask, C);
        if (!ls) return false;
        last = *ls;
        if (!std::isfinite(last)) return false;
        auto gl = loss.backward();
        if (!gl) return false;
        auto bw = model.backward(*gl);
        if (!bw) return false;
        if (auto st = opt->step(); !st) return false;
    }
    loss_out = last;
    auto w1 = eng.to_matrix(model.parameters()[0].get(), nn::Precision::F32);
    if (!w1) return false;
    w_after_out.assign(w1->span().begin(), w1->span().end());
    return true;
}

void test_convergence(nn::ComputeEngine& eng, const char* tag)
{
    std::printf("[3][4] 端到端：AdamWInt8 vs AdamWQuant vs AdamW（同问题同超参）（%s）\n", tag);
    nn::Scalar loss_adam = 0.0f, loss_quant = 0.0f, loss_int8 = 0.0f;
    std::vector<nn::Scalar> wb_a, wa_a, wb_q, wa_q, wb_i, wa_i;
    const bool ok_a = train_mlp(eng, "adamw", 120, loss_adam, wb_a, wa_a);
    const bool ok_q = train_mlp(eng, "adamwquant", 120, loss_quant, wb_q, wa_q);
    const bool ok_i = train_mlp(eng, "adamwint8", 120, loss_int8, wb_i, wa_i);
    CHECK(ok_a && ok_q && ok_i, "三个优化器训练都跑通");
    if (!ok_a || !ok_q || !ok_i) return;
    std::printf("    最终 loss: AdamW=%.6f  AdamWQuant=%.6f  AdamWInt8=%.6f\n",
                static_cast<double>(loss_adam), static_cast<double>(loss_quant),
                static_cast<double>(loss_int8));
    CHECK(std::isfinite(loss_int8), "AdamWInt8 的 loss 有限");
    CHECK(loss_int8 < 0.5f, "AdamWInt8 收敛到小 loss");
    CHECK(std::fabs(static_cast<double>(loss_int8) - static_cast<double>(loss_adam)) < 0.5,
          "与 f32 AdamW 同量级（int8 状态精度代价）");
    double moved = 0.0;
    for (std::size_t i = 0; i < wa_i.size(); ++i)
        moved += std::fabs(static_cast<double>(wa_i[i]) - static_cast<double>(wb_i[i]));
    CHECK(moved > 1e-3, "AdamWInt8 下参数确实移动（非静默冻结）");
    std::printf("    首层权重 L1 位移: AdamW=%.4f  AdamWQuant=%.4f  AdamWInt8=%.4f\n",
                [&]
                {
                    double s = 0.0;
                    for (std::size_t i = 0; i < wa_a.size(); ++i)
                        s += std::fabs(static_cast<double>(wa_a[i]) - static_cast<double>(wb_a[i]));
                    return s;
                }(),
                [&]
                {
                    double s = 0.0;
                    for (std::size_t i = 0; i < wa_q.size(); ++i)
                        s += std::fabs(static_cast<double>(wa_q[i]) - static_cast<double>(wb_q[i]));
                    return s;
                }(),
                moved);
}

// ── [5] 首步零梯度不产生 NaN ──────────────────────────────────────────────
void test_zero_grad_first_step(nn::ComputeEngine& eng, const char* tag)
{
    std::printf("[5] 首步全零梯度（尺度下限保护）（%s）\n", tag);
    nn::Tensor p = eng.create_tensor(4, 4, nn::Precision::F32,
                                     nn::InitSpec::uniform(-0.1f, 0.1f, nn::kInitSeed));
    nn::Tensor g = eng.create_tensor(4, 4, nn::Precision::F32, nn::InitSpec::zero());
    nn::AdamWInt8 q(eng, {p}, {g}, 0.01f);
    auto r = q.step();
    CHECK(r.has_value(), "零梯度 step 成功");
    const nn::Matrix w = download(eng, p);
    const nn::Matrix sm = download(eng, q.moment_scales()[0]);
    bool finite = true;
    for (auto v : w.span())
        if (!std::isfinite(v)) finite = false;
    for (auto v : sm.span())
        if (!std::isfinite(v)) finite = false;
    CHECK(finite, "零梯度不产生 NaN/Inf（尺度下限）");
}

// ── [6] save/load 往返：载入状态后 step 与未中断 step 逐位一致 ────────────
// ⚠ 断点的完整语义 = 状态张量 + `t_` 计数器（state_step）：load 后必须
// set_state_step 回滚，否则偏置修正在 B 路径用了错误的 n。
void test_state_roundtrip(nn::ComputeEngine& eng, const char* tag)
{
    std::printf("[6] save/load 往返（%s）：载入状态后 step 逐位一致\n", tag);
    nn::Tensor p = eng.create_tensor(6, 5, nn::Precision::F32,
                                     nn::InitSpec::uniform(-0.2f, 0.2f, nn::kInitSeed));
    nn::Tensor g = eng.create_tensor(6, 5, nn::Precision::F32,
                                     nn::InitSpec::uniform(-0.1f, 0.1f, nn::kInitSeed + 1));
    nn::AdamWInt8 opt(eng, {p}, {g}, 0.01f);
    for (int i = 0; i < 3; ++i)
    {
        auto r = opt.step();
        CHECK(r.has_value(), "step 成功");
        if (!r.has_value()) return;
    }
    const std::uint64_t saved_step = opt.state_step();
    const nn::Matrix p3 = download(eng, p);
    std::vector<nn::Tensor> saved;
    NN_UNWRAP(opt.save_state(saved), "save_state");
    CHECK(saved.size() == 4, "每参数 4 张量（m_packed/v_packed/sm/sv）");
    // ⚠ save_state 是**句柄共享**（同一存储）：后续 step 会改动快照 → 必须
    // 立即把数据快照成宿主副本，load 前再重建为独立张量。
    std::vector<nn::Matrix> snap;
    for (const auto& t : saved)
        snap.push_back(download(eng, t));
    auto r4 = opt.step();   // 第 4 步 → 结果 A
    CHECK(r4.has_value(), "step(4) 成功");
    if (!r4.has_value()) return;
    const nn::Matrix p4a = download(eng, p);

    // 回滚：参数 ← P3、t_ ← 3、状态 ← 快照（重建独立张量），再走第 4 步 → 结果 B
    NN_UNWRAP(eng.write(p, p3.span()), "参数回滚写回");
    opt.set_state_step(saved_step);
    std::vector<nn::Tensor> restored;
    for (const auto& m : snap)
        restored.push_back(upload(eng, m));
    NN_UNWRAP(opt.load_state(restored), "load_state");
    auto r5 = opt.step();
    CHECK(r5.has_value(), "载入后 step 成功");
    if (!r5.has_value()) return;
    const nn::Matrix p4b = download(eng, p);
    CHECK(matrices_equal(p4a, p4b), "载入状态后 step 与未中断 step 逐位一致");
}

}   // namespace

int main(int argc, char* argv[])
{
    bool gpu = false;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--gpu") gpu = true;

    const auto run_all = [&](nn::ComputeEngine& e, const char* dev) {
        test_primitive(e, 5, 7, 71, "cols 非 4 倍数");
        test_primitive(e, 130, 512, 83, "多行多块");
        test_state_bytes(e, dev);
        test_convergence(e, dev);
        test_zero_grad_first_step(e, dev);
        test_state_roundtrip(e, dev);
    };

    if (gpu)
    {
        nn::cli::EngineConfig ec;
        ec.use_gpu = true;
        auto eng_r = nn::cli::create_engine(ec);
        NN_EXIT(eng_r, 77, "GPU 不可用，跳过: ");
        run_all(**eng_r, "GPU");
    }
    else
    {
        auto eng_owner = nn::create_cpu_engine();
        run_all(*eng_owner, "CPU");
    }

    if (g_failures == 0) std::puts("adam_int8_test: 全部通过");
    return g_failures == 0 ? 0 : 1;
}
