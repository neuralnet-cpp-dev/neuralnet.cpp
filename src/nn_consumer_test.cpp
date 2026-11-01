// ═══════════════════════════════════════════════════════════════════════════
//  nn_consumer_test — 外部消费者验证 libnn.so（Phase 3-A；docs/internal/34）
//
//  目的：证明"链 libnn.so 而非 header-only 引擎"可用 —— LGPL 可替换库二进制的
//  前提。本 TU **不 include compute_engine.hpp / nn.hpp**（不实例化引擎），只：
//    · `nn_api.h`（C ABI 句柄）→ 引擎/张量/原语全在 libnn.so；
//    · `kernel_cache.hpp` + `expr_spec.hpp`（**模板/IR 面，留 header**）→ 仅用于
//      encode_expr_spec 序列化 eval_expr 的载荷。
//  结果与手算参考对比（CPU 引擎确定性）。
// ═══════════════════════════════════════════════════════════════════════════

#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

#include <neuralnet.cpp/nn_api.h>
#include <neuralnet.cpp/kernel_cache.hpp>   // encode_expr_spec（边界载荷，header 面）
#include <neuralnet.cpp/expr_spec.hpp>      // ExprSpec / ExprInstr / ExprView 类型

static int failures = 0;
static void check(const char* name, bool ok)
{
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok) ++failures;
}

int main()
{
    // ── 1. CPU 引擎 matmul（C ABI 最小闭环）───────────────────────────
    nnn_engine e = nnn_engine_create_cpu();
    check("engine_create_cpu", e != nullptr);
    if (!e) return 1;

    // A = [[1,2,3],[4,5,6]], B = [[7,8],[9,10],[11,12]] → 2x2:
    //   [58, 64]   （1*7+2*9+3*11=58, 1*8+2*10+3*12=64）
    //   [139,154]  （4*7+5*9+6*11=139, 4*8+5*10+6*12=154）
    const float A[6] = {1, 2, 3, 4, 5, 6};
    const float B[6] = {7, 8, 9, 10, 11, 12};
    nnn_tensor At = nnn_tensor_from_matrix(e, A, 2, 3, NNN_PREC_F32);
    nnn_tensor Bt = nnn_tensor_from_matrix(e, B, 3, 2, NNN_PREC_F32);
    check("tensor_from_matrix", At && Bt);
    if (!At || !Bt) return 1;

    nnn_tensor Ct = nullptr;
    check("matmul(A,B)", nnn_matmul(e, At, Bt, 0, 0, &Ct) == NNN_OK && Ct != nullptr);
    check("matmul 形状 2x2",
          nnn_tensor_rows(e, Ct) == 2 && nnn_tensor_cols(e, Ct) == 2);

    float C[4];
    size_t n = 0;
    check("to_matrix", nnn_tensor_to_matrix(e, Ct, C, 4, &n) == NNN_OK && n == 4);
    const float Cexp[4] = {58, 64, 139, 154};
    bool mm_ok = true;
    for (int i = 0; i < 4; ++i) mm_ok = mm_ok && (std::fabs(C[i] - Cexp[i]) < 1e-4f);
    check("matmul 数值", mm_ok);

    nnn_tensor_destroy(e, Ct);
    nnn_tensor_destroy(e, Bt);
    nnn_tensor_destroy(e, At);

    // ── 1b. 数据操作/归约原语（Phase 3-B 补全）────────────────────────
    // transpose: A(2x3) → At(3x2): [[1,4],[2,5],[3,6]]
    {
        nnn_tensor T = nullptr;
        check("transpose", nnn_transpose(e, nnn_tensor_from_matrix(e, A, 2, 3, NNN_PREC_F32), &T) ==
                               NNN_OK && T);
        float Td[6];
        size_t tn = 0;
        check("transpose 形状 3x2", nnn_tensor_rows(e, T) == 3 && nnn_tensor_cols(e, T) == 2);
        nnn_tensor_to_matrix(e, T, Td, 6, &tn);
        const float Texp[6] = {1, 4, 2, 5, 3, 6};
        bool tok = tn == 6;
        for (int i = 0; i < 6 && tok; ++i) tok = (std::fabs(Td[i] - Texp[i]) < 1e-4f);
        check("transpose 数值", tok);
        nnn_tensor_destroy(e, T);
    }
    // col_reduce_sum: X(2x3) = [1,2,3; 4,5,6] → 每列和 (1,3): [5,7,9]
    {
        const float Xv[6] = {1, 2, 3, 4, 5, 6};
        nnn_tensor R = nullptr;
        check("col_reduce_sum", nnn_col_reduce_sum(e, nnn_tensor_from_matrix(e, Xv, 2, 3,
                                                                              NNN_PREC_F32),
                                                   &R) == NNN_OK && R);
        float Rd[3];
        size_t rn = 0;
        check("col_reduce_sum 形状 1x3", nnn_tensor_rows(e, R) == 1 && nnn_tensor_cols(e, R) == 3);
        nnn_tensor_to_matrix(e, R, Rd, 3, &rn);
        const float Rexp[3] = {5, 7, 9};
        bool rok = rn == 3;
        for (int i = 0; i < 3 && rok; ++i) rok = (std::fabs(Rd[i] - Rexp[i]) < 1e-4f);
        check("col_reduce_sum 数值", rok);
        nnn_tensor_destroy(e, R);
    }
    // scale_inplace + clone: S = clone(X); S *= 2 → [2,4,6,8,10,12]
    {
        const float Xv[6] = {1, 2, 3, 4, 5, 6};
        nnn_tensor S = nullptr;
        check("clone", nnn_clone(e, nnn_tensor_from_matrix(e, Xv, 2, 3, NNN_PREC_F32), &S) == NNN_OK &&
                           S);
        check("scale_inplace", nnn_scale_inplace(e, S, 2.0f) == NNN_OK);
        float Sd[6];
        size_t sn = 0;
        nnn_tensor_to_matrix(e, S, Sd, 6, &sn);
        bool sok = sn == 6;
        for (int i = 0; i < 6 && sok; ++i) sok = (std::fabs(Sd[i] - 2.0f * Xv[i]) < 1e-4f);
        check("clone+scale_inplace 数值 (X*2)", sok);
        nnn_tensor_destroy(e, S);
    }

    // ── 2. eval_expr（序列化 ExprSpec 边界载荷；x + 3）───────────────
    // 手工构造 out = input + const(3)：views=[Linear]，一条 Add。
    nn::ExprSpec spec;
    nn::ExprView lv;
    lv.kind = static_cast<std::uint8_t>(nn::ExprViewKind::Linear);
    lv.negate_first_half = 0;
    lv.param = 0;
    lv.param2 = 0;
    spec.views.push_back(lv);
    spec.instrs.push_back(nn::ExprInstr{
        static_cast<std::uint8_t>(nn::ExprOp::Add), /*dst=*/0,
        {static_cast<std::uint8_t>(nn::ExprOperandKind::Input), 0},
        {static_cast<std::uint8_t>(nn::ExprOperandKind::Const), 0},
        {0, 0}});
    spec.consts.push_back(3.0f);
    spec.num_regs = 1;

    const float X[6] = {1, 2, 3, 4, 5, 6};   // 2x3
    nnn_tensor Xt = nnn_tensor_from_matrix(e, X, 2, 3, NNN_PREC_F32);
    const std::vector<std::uint8_t> bytes = nn::fused::encode_expr_spec(spec);
    nnn_tensor Yt = nullptr;
    nnn_tensor ins[1] = {Xt};
    check("eval_expr(x+3)",
          nnn_eval_expr(e, bytes.data(), bytes.size(), ins, 1, /*prec_sig=*/0, /*rows=*/2,
                        /*cols=*/3, &Yt) == NNN_OK &&
              Yt != nullptr);
    float Y[6];
    n = 0;
    check("eval_expr to_matrix",
          nnn_tensor_to_matrix(e, Yt, Y, 6, &n) == NNN_OK && n == 6);
    bool ex_ok = true;
    for (int i = 0; i < 6; ++i) ex_ok = ex_ok && (std::fabs(Y[i] - (X[i] + 3.0f)) < 1e-4f);
    check("eval_expr 数值 (x+3)", ex_ok);

    nnn_tensor_destroy(e, Yt);
    nnn_tensor_destroy(e, Xt);

    // ── 3. GPU 引擎（可选；无卡 → NNN_NO_GPU 回退）──────────────────
    nnn_engine g = nnn_engine_create_gpu(nullptr);
    if (g)
    {
        nnn_tensor GAt = nnn_tensor_from_matrix(g, A, 2, 3, NNN_PREC_F32);
        nnn_tensor GBt = nnn_tensor_from_matrix(g, B, 3, 2, NNN_PREC_F32);
        nnn_tensor GCt = nullptr;
        bool gok = nnn_matmul(g, GAt, GBt, 0, 0, &GCt) == NNN_OK && GCt;
        float GC[4];
        size_t gn = 0;
        if (gok) gok = (nnn_tensor_to_matrix(g, GCt, GC, 4, &gn) == NNN_OK);
        bool gmm = gok && gn == 4;
        for (int i = 0; i < 4 && gmm; ++i) gmm = (std::fabs(GC[i] - Cexp[i]) < 1e-3f);
        check("GPU matmul（850M）", gmm);
        if (GCt) nnn_tensor_destroy(g, GCt);
        nnn_tensor_destroy(g, GBt);
        nnn_tensor_destroy(g, GAt);
        nnn_engine_destroy(g);
    }
    else
    {
        const char* err = nnn_last_error();
        std::printf("[INFO] 无 GPU（NNN_NO_GPU，回退 CPU）：%s\n", err ? err : "");
    }

    nnn_engine_destroy(e);
    std::printf("\nnn_consumer_test: %d failure(s)\n", failures);
    return failures != 0 ? 1 : 0;
}
