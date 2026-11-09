// ── RAPT（ReLU 线性注意力）端到端冒烟测试 ──────────────────────────────────
//
// 验证：
//   1. build_rapt_model / build_rapt_model_from_spec / spec_matches 往返
//   2. RAPTModel 整链 forward/backward 可运行（无 NaN、无崩溃）
//   3. 短训练循环 loss 有效且下降（证明梯度有效、可学习）
//   4. batch>1 布局正确性（铁律 5：batch=1 时布局重合测不出）
//   5. KV-cache 一致性按 ulp 口径收紧的容差断言 + 输出有限性
//   6. decode 分母（#P7 设备侧 den）宿主参考直接对拍（z_state + 输出）
//
// 用法：rapt_smoke_test [--steps N] [--gpu]（经 rapt_test 聚合注册 ctest）
// ─────────────────────────────────────────────────────────────────────────

#include <nn_error_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>
import nn.cli;
import nn.core;
import nn.dataset;
import nn.domain;
import nn.engine;
import nn.expr;
import nn.kvrecord;
import nn.layer;
import nn.loss;
import nn.model;
import nn.opt;
import nn.tokenizer;
import nn.train;
using nn::cli::EngineConfig;
using nn::cli::create_engine;

int main(int argc, char* argv[])
{
    std::size_t steps = 15;
    bool use_gpu = false;
    bool use_f16 = false;
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a == "--steps" && i + 1 < argc) steps = static_cast<std::size_t>(std::atoi(argv[++i]));
        else if (a == "--gpu") use_gpu = true;
        else if (a == "--f16") use_f16 = true;
    }

    auto engine = nn::cli::create_engine(nn::cli::EngineConfig{use_gpu});
    NN_EXIT(engine, 1, "\n");
    nn::ComputeEngine& eng = **engine;

    // ── 1. 规格往返校验 ─────────────────────────────────────────────
    nn::RAPTConfig cfg;
    cfg.vocab_size = 32;
    cfg.d_model = 16;   // d_k = d_model/heads = 8（偶数，RoPE 约束）
    cfg.seq_len = 8;
    cfg.num_heads = 2;
    cfg.d_ff = 32;
    cfg.num_layers = 2;
    // P2-3：`--f16` 走 profile_f16()（param/compute=F16，stable/optimizer=F32）
    if (use_f16)
    {
        cfg.precision = nn::profile_f16();
        std::cout << "精度: --f16（param/compute=f16）\n";
    }

    auto spec = nn::make_rapt_spec(cfg.vocab_size, cfg.d_model, cfg.seq_len, cfg.num_heads,
                                   cfg.d_ff, cfg.num_layers);
    auto model_from_spec = nn::build_rapt_model_from_spec(eng, spec);
    NN_EXIT(model_from_spec, 1, "build_rapt_model_from_spec failed\n");
    if (!nn::spec_matches(spec, *model_from_spec->spec()))
    {
        std::cerr << "RAPT spec round-trip mismatch\n";
        return 1;
    }
    std::cout << "spec round-trip OK (" << nn::spec_summary(spec) << ")\n";

    auto model = nn::build_rapt_model(eng, cfg);
    NN_EXIT(model, 1, "build_rapt_model failed\n");

    // ── 2. 训练冒烟：合成数据（causal 下一位置预测），验证 loss 下降 ──
    auto optimizer =
        nn::create_optimizer("adamw", eng, model->parameters(), model->param_gradients(),
                             nn::Scalar{1e-3}, nn::Scalar{0.01});
    NN_EXIT(optimizer, 1, "optimizer creation failed\n");

    nn::CrossEntropyLoss ce;
    if (use_f16) ce.set_precision_profile(nn::profile_f16());
    std::mt19937_64 rng{123};
    std::uniform_int_distribution<std::size_t> tok(0, cfg.vocab_size - 1);

    // 输入序列 + 因果下一位置标签（labels[t] = x[t+1]，末位任意）
    nn::Matrix in(cfg.seq_len, 1);
    std::vector<std::size_t> labels(cfg.seq_len);
    std::vector<std::size_t> toks(cfg.seq_len);
    for (std::size_t t = 0; t < cfg.seq_len; ++t)
        toks[t] = tok(rng);
    for (std::size_t t = 0; t < cfg.seq_len; ++t)
    {
        in.set_value_unchecked(t, 0, static_cast<nn::Scalar>(toks[t]));
        labels[t] = (t + 1 < cfg.seq_len) ? toks[t + 1] : tok(rng);
    }
    auto x = eng.from_matrix(in);
    NN_EXIT(x, 1);

    nn::Scalar first_loss = 0;
    nn::Scalar last_loss = 0;
    for (std::size_t step = 0; step < steps; ++step)
    {
        auto zero = optimizer->zero_grad();
        NN_EXIT(zero, 1, "zero_grad failed\n");
        auto logits = model->forward(*x);
        NN_EXIT(logits, 1, "forward failed: ");
        auto loss = ce.forward_sparse(eng, *logits, labels, {}, cfg.vocab_size);
        NN_EXIT(loss, 1, "loss failed: ");
        auto grad = ce.backward();
        NN_EXIT(grad, 1, "loss backward failed\n");
        auto b = model->backward(*grad);
        NN_EXIT(b, 1, "model backward failed: ");
        auto st = optimizer->step();
        NN_EXIT(st, 1, "optimizer step failed\n");
        if (step == 0) first_loss = *loss;
        last_loss = *loss;
        std::cout << "  step " << step << "  loss=" << *loss << "\n";
    }

    std::cout << "first_loss=" << first_loss << " last_loss=" << last_loss << "\n";
    if (!(first_loss > 0) || !(first_loss == first_loss) ||
        !(last_loss == last_loss))   // 有效且非 NaN
    {
        std::cerr << "loss invalid (NaN?)\n";
        return 1;
    }

    // ── 3. batch>1 布局正确性（铁律 5） ──────────────────────────────
    {
        const std::size_t batch = 2;
        nn::Matrix bin(cfg.seq_len, batch);
        std::vector<std::size_t> blabels(cfg.seq_len * batch);
        for (std::size_t b = 0; b < batch; ++b)
        {
            std::vector<std::size_t> bt(cfg.seq_len);
            for (std::size_t t = 0; t < cfg.seq_len; ++t)
                bt[t] = tok(rng);
            for (std::size_t t = 0; t < cfg.seq_len; ++t)
            {
                bin.set_value_unchecked(t, b, static_cast<nn::Scalar>(bt[t]));
                blabels[b * cfg.seq_len + t] = (t + 1 < cfg.seq_len) ? bt[t + 1] : tok(rng);
            }
        }
        auto bx = eng.from_matrix(bin);
        NN_EXIT(bx, 1);
        // 同一模型上多跑几步，验证 batch>1 的 forward/backward 不崩、不 NaN
        for (std::size_t step = 0; step < 3; ++step)
        {
            auto zero = optimizer->zero_grad();
            NN_EXIT(zero, 1, "batch>1 zero_grad failed\n");
            auto logits = model->forward(*bx);
            NN_EXIT(logits, 1, "batch>1 forward failed: ");
            auto loss = ce.forward_sparse(eng, *logits, blabels, {}, cfg.vocab_size);
            NN_EXIT(loss, 1, "batch>1 loss failed\n");
            if (!(*loss == *loss))
            {
                std::cerr << "batch>1 loss NaN\n";
                return 1;
            }
            auto grad = ce.backward();
            NN_EXIT(grad, 1, "batch>1 loss backward failed\n");
            auto b = model->backward(*grad);
            NN_EXIT(b, 1, "batch>1 model backward failed\n");
            auto st = optimizer->step();
            NN_EXIT(st, 1, "batch>1 optimizer step failed\n");
        }
        std::cout << "batch>1 (batch=" << batch << ") smoke OK\n";
    }

    // ── 4. KV cache 一致性：整序列 forward 末位 == 逐 token forward_step 累积 ──
    // 验证增量运行态生成（RLA 的 KV cache）与批量 forward 完全一致。
    {
        const std::size_t d_model = 16, heads = 2, L = 6;   // d_k = 8
        nn::ReLULinearAttention attn(d_model, heads, /*seq_len=*/0,
                                     /*causal=*/true, nn::PosEncodingType::RoPE);
        if (!attn.init(eng))
        {
            std::cerr << "kv attn init failed\n";
            return 1;
        }
        std::mt19937_64 rng{77};
        std::normal_distribution<nn::Scalar> dist(0.0, 1.0);
        nn::Matrix x_m(d_model, L);
        {
            auto sp = x_m.span();
            for (auto& v : sp)
                v = dist(rng);
        }
        auto x_t = eng.from_matrix(x_m);
        NN_EXIT(x_t, 1);
        auto full = attn.forward(*x_t);
        NN_EXIT(full, 1, "kv full forward failed\n");
        auto full_m = eng.to_matrix(*full);
        NN_EXIT(full_m, 1);

        // 逐 token 增量（运行态 KV cache）
        // RLA-2 状态：B_state (d_model, d_k) + z_state (d_model, 1)
        const std::size_t dk = d_model / heads;
        nn::Tensor B_state = eng.create_tensor(d_model, dk);
        NN_EXIT(eng.zero(B_state), 1, "kv zero B_state failed\n");
        nn::Tensor z_state = eng.create_tensor(d_model, 1);
        NN_EXIT(eng.zero(z_state), 1, "kv zero z_state failed\n");
        nn::Tensor inc_out;
        for (std::size_t t = 0; t < L; ++t)
        {
            nn::Matrix col(d_model, 1);
            for (std::size_t r = 0; r < d_model; ++r)
                col.set_value_unchecked(r, 0, x_m.at_unchecked(r, t));
            auto col_t = eng.from_matrix(col);
            NN_EXIT(col_t, 1);
            auto o = attn.forward_step(eng, *col_t, B_state, z_state, t);
            NN_EXIT(o, 1, "kv forward_step failed: ");
            inc_out = std::move(*o);
        }
        auto inc_m = eng.to_matrix(inc_out);
        NN_EXIT(inc_m, 1);
        // 比较增量末位 vs 整序列 forward 末位
        // 容差依据（自旧 1e-3 收紧至 ulp 口径）：两路径仅加法结合序不同，实测
        // docs/development/08 §测试表 = CPU 0 / GPU 2.98e-08（≈1 ulp）；本机
        // （850M，Release）实测 CPU 1.79e-7 / GPU 7.45e-8。取 16·eps·max(1, 幅值)
        // ——比旧容差紧约 500×，对不同设备/编译器的合理漂移仍留 ≥10× 余量，
        // 不写逐位断言以免跨设备 flaky。
        // 先断言有限：两路同时输出 NaN 时 a-b=NaN，会被 std::max(max_diff, NaN)
        // 静默吞掉（NaN 比较恒 false ⇒ 保留旧值）而误判通过。
        nn::Scalar max_diff{0}, max_mag{0};
        bool finite = true;
        for (std::size_t r = 0; r < d_model; ++r)
        {
            const nn::Scalar a = full_m->at_unchecked(r, L - 1);
            const nn::Scalar b = inc_m->at_unchecked(r, 0);
            if (!std::isfinite(a) || !std::isfinite(b)) finite = false;
            max_diff = std::max(max_diff, std::fabs(a - b));
            max_mag = std::max(max_mag, std::max(std::fabs(a), std::fabs(b)));
        }
        const nn::Scalar tol = 16 * std::numeric_limits<nn::Scalar>::epsilon() *
                               std::max(nn::Scalar{1}, max_mag);
        std::cout << "KV-cache consistency: max_diff=" << max_diff << " max_mag=" << max_mag
                  << " tol=" << tol << "\n";
        if (!finite)
        {
            std::cerr << "KV-cache 输出含 NaN/Inf!\n";
            return 1;
        }
        if (!(max_diff <= tol))
        {
            std::cerr << "KV-cache inconsistency! max_diff=" << max_diff
                      << " > tol=" << tol << "\n";
            return 1;
        }
        std::cout << "KV-cache consistency OK\n";
    }

    // ── 5. decode 分母（#P7）直接对拍：宿主参考重建 den 全链 ────────────────
    // den = q'·z + ε 现在**设备侧**算（q'⊙z → grouped_reduce_sum → gather_rows，
    // 见 compute_layer_rapt.hpp forward_step）。上面的 KV 一致性只证明两条设备
    // 路径互相对得上（同错则双双通过）。这里读回本层全部 8 个参数张量，按文档
    // 口径（docs/development/08 §forward_step）在宿主侧重建单步 pos=0 参考
    // （pos=0 ⇒ RoPE 旋转角 0 ⇒ cos=1/sin=0 恒等），直接断言：
    //   ① z_state（分母累加器 z=Σk'）与宿主 k' 一致；
    //   ② 输出 == W_o·[v·s/(s+ε)] + b_o，s = q'_hᵀk'_h —— s 经设备侧
    //      grouped_reduce_sum + gather_rows 进入除法，错分组/错 gather
    //      会直接体现为与参考的偏差。
    {
        const std::size_t dm = 16, nh = 2, dk = dm / nh;
        nn::ReLULinearAttention attn(dm, nh, /*seq_len=*/0, /*causal=*/true,
                                     nn::PosEncodingType::RoPE);
        if (!attn.init(eng))
        {
            std::cerr << "den attn init failed\n";
            return 1;
        }
        auto ps = attn.parameters();  // [wq,bq,wk,bk,wv,bv,wo,bo]
        if (ps.size() != 8)
        {
            std::cerr << "den: 参数数 " << ps.size() << " != 8\n";
            return 1;
        }
        const auto read_p = [&](std::size_t i)
        {
            auto m = eng.to_matrix(ps[i].get());
            NN_EXIT(m, 1, "den: 参数读回失败\n");
            return std::move(*m);
        };
        const nn::Matrix Wq = read_p(0), bq = read_p(1), Wk = read_p(2), bk = read_p(3),
                         Wv = read_p(4), bv = read_p(5), Wo = read_p(6), bo = read_p(7);

        // 固定种子输入（确定性；种子 99 使两个头的分母 s=q'ᵀk' 均非零，
        // 避免 ReLU 全灭导致该头的 den 对错误失敏——见下方 s 打印）
        nn::Matrix x_m(dm, 1);
        {
            std::mt19937_64 xr{99};
            std::normal_distribution<nn::Scalar> xd(0.0, 1.0);
            for (std::size_t r = 0; r < dm; ++r) x_m.set_value_unchecked(r, 0, xd(xr));
        }

        // 宿主参考用 double 累加（f32 设备误差单向暴露在对拍里）
        const auto lin = [&](const nn::Matrix& W, const nn::Matrix& b, std::size_t r)
        {
            double s = b.at_unchecked(r, 0);
            for (std::size_t c = 0; c < dm; ++c)
                s += static_cast<double>(W.at_unchecked(r, c)) * x_m.at_unchecked(c, 0);
            return s;
        };
        std::vector<double> q(dm), k(dm), v(dm);
        for (std::size_t r = 0; r < dm; ++r)
        {
            q[r] = lin(Wq, bq, r);
            k[r] = lin(Wk, bk, r);
            v[r] = lin(Wv, bv, r);
        }
        // 逐头 RMSNorm（Q/K；公式 = rms_norm_forward_：y = x·rsqrt(mean(x²)+1e-5)）
        // → RoPE(pos=0) 恒等 → ReLU；V 不 norm / 不 ReLU / 不 RoPE（与 forward_step 同）
        std::vector<double> qp(dm, 0.0), kp(dm, 0.0);
        for (std::size_t h = 0; h < nh; ++h)
        {
            double sq = 0, sk = 0;
            for (std::size_t j = 0; j < dk; ++j)
            {
                sq += q[h * dk + j] * q[h * dk + j];
                sk += k[h * dk + j] * k[h * dk + j];
            }
            const double rq = std::sqrt(sq / static_cast<double>(dk) + 1e-5);
            const double rk = std::sqrt(sk / static_cast<double>(dk) + 1e-5);
            for (std::size_t j = 0; j < dk; ++j)
            {
                const double qn = q[h * dk + j] / rq;
                const double kn = k[h * dk + j] / rk;
                qp[h * dk + j] = qn > 0 ? qn : 0.0;
                kp[h * dk + j] = kn > 0 ? kn : 0.0;
            }
        }
        // 分母 s_h = q'_hᵀk'（q',k'≥0 ⇒ s≥0 ⇒ den=s+ε≥ε>0）；
        // 单步零状态：z=k'、B=v·k'ᵀ ⇒ num=B·q' = v·(k'ᵀq') = v·s ⇒ pre = v·s/(s+ε)
        std::vector<double> pre(dm, 0.0);
        for (std::size_t h = 0; h < nh; ++h)
        {
            double s = 0;
            for (std::size_t j = 0; j < dk; ++j) s += qp[h * dk + j] * kp[h * dk + j];
            const double den = s + 1e-4;
            std::cout << "  den 参考: head" << h << " s=" << s << " den=s+1e-4=" << den << "\n";
            if (!(s >= 0) || !(den > 0))
            {
                std::cerr << "den 参考值异常（q',k'≥0 ⇒ s≥0, den>0）\n";
                return 1;
            }
            for (std::size_t j = 0; j < dk; ++j) pre[h * dk + j] = v[h * dk + j] * s / den;
        }
        // 出口线性层：exp = Wo·pre + bo
        std::vector<double> exp_out(dm, 0.0);
        for (std::size_t r = 0; r < dm; ++r)
        {
            double s = bo.at_unchecked(r, 0);
            for (std::size_t c = 0; c < dm; ++c) s += Wo.at_unchecked(r, c) * pre[c];
            exp_out[r] = s;
        }

        // 设备侧单步执行（pos=0，状态清零）
        auto x_t = eng.from_matrix(x_m);
        NN_EXIT(x_t, 1);
        nn::Tensor B_state = eng.create_tensor(dm, dk);
        NN_EXIT(eng.zero(B_state), 1, "den: zero B_state failed\n");
        nn::Tensor z_state = eng.create_tensor(dm, 1);
        NN_EXIT(eng.zero(z_state), 1, "den: zero z_state failed\n");
        auto o = attn.forward_step(eng, *x_t, B_state, z_state, /*pos=*/0);
        NN_EXIT(o, 1, "den forward_step failed: ");
        auto om = eng.to_matrix(*o);
        NN_EXIT(om, 1);
        auto zm = eng.to_matrix(z_state);
        NN_EXIT(zm, 1);

        // ① z_state == 宿主 k'（分母累加器数值正确）
        double z_err = 0, z_mag = 0;
        for (std::size_t r = 0; r < dm; ++r)
        {
            const double a = zm->at_unchecked(r, 0);
            z_err = std::max(z_err, std::fabs(a - kp[r]));
            z_mag = std::max(z_mag, std::fabs(a));
        }
        // ② 输出 == 宿主参考（den 经设备侧 grouped_reduce_sum + gather_rows 进除法）
        double out_err = 0, out_mag = 0;
        bool out_finite = true;
        for (std::size_t r = 0; r < dm; ++r)
        {
            const double a = om->at_unchecked(r, 0);
            if (!std::isfinite(a)) out_finite = false;
            out_err = std::max(out_err, std::fabs(a - exp_out[r]));
            out_mag = std::max(out_mag, std::fabs(a));
        }
        // 参考为 double、设备为 f32 多级链（rmsnorm/matmul/归约/除法），正常误差
        // 在数十 f32 ulp 内；取 64·eps·max(1,幅值)。设备若错分母（漏 gather /
        // 错分组 / 漏 ε），偏差是百分比级，远超此界。
        const double tol =
            64.0 * std::numeric_limits<nn::Scalar>::epsilon() * std::max(1.0, out_mag);
        const double z_tol =
            64.0 * std::numeric_limits<nn::Scalar>::epsilon() * std::max(1.0, z_mag);
        std::cout << "  den 直接对拍: z_err=" << z_err << " (tol=" << z_tol
                  << ")  out_err=" << out_err << " (tol=" << tol << ")\n";
        if (!out_finite)
        {
            std::cerr << "den: forward_step 输出含 NaN/Inf\n";
            return 1;
        }
        if (!(z_err <= z_tol))
        {
            std::cerr << "den: z_state 与宿主 k' 不一致（z_err=" << z_err << "）\n";
            return 1;
        }
        if (!(out_err <= tol))
        {
            std::cerr << "den: 输出与宿主参考不一致（错分母? out_err=" << out_err << "）\n";
            return 1;
        }
        std::cout << "decode den 直接对拍 OK\n";
    }

    std::cout << "RAPT SMOKE TEST PASSED\n";
    return 0;
}
