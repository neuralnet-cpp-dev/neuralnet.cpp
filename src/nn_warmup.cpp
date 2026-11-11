// ── nn_warmup — 构建期/发布期 JIT 预热目标（P3/D5，docs/internal/35）──────────
//
// 常用架构 × 代表性 shape 过一遍 **GPU forward** → 用到的表达式全部现编进
// kernels.nnk（.nnk 算子缓存），产物随发布物分发：
//   · 部署首跑无需现场依赖 libnn-kernelgen.so/glslc（配合各 app 的
//     model_warmup / --no-warmup 语义）；
//   · 同时充当"编译目标预热"（35 §P3 D5 入口②）。
//
// 覆盖（forward 即触发 eval_expr/fold/reduce/into 各入口的 JIT 编译）：
//   MNIST-MLP / MNIST-CNN(im2col+pool) / MNIST-Transformer(patch+attention fold)
//   / tiny GPT(attention fold + LN + GELU + head) / tiny RAPT(RoPE 线性注意力)。
//
// 用法：nn_warmup [--out <dir>]
//   缓存目录优先级（与 kernel_cache_path_from_env 同源）：
//     NN_KERNEL_CACHE_PATH > NN_KERNEL_CACHE_DIR > 本命令 --out（写入
//     NN_KERNEL_CACHE_DIR）> 可执行文件所在目录（缺省，便于把产物直接
//     落在 build/<preset>/ 里随发布拷走）。
//   无可用 Vulkan 设备 → 打印 SKIP 并返回 0（发布机无 GPU 不算失败）。
// ─────────────────────────────────────────────────────────────────────────────
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
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
using nn::fused::kernel_cache_path;

namespace
{
void set_cache_dir(const std::string& dir)
{
#ifdef _WIN32
    _putenv_s("NN_KERNEL_CACHE_DIR", dir.c_str());
#else
    setenv("NN_KERNEL_CACHE_DIR", dir.c_str(), 1);
#endif
}

std::string exe_dir(const char* argv0)
{
    const std::string s = argv0 != nullptr ? argv0 : "";
    const auto pos = s.find_last_of("/\\");
    if (pos == std::string::npos) return ".";
    if (pos == 0) return s.substr(0, 1);
    return s.substr(0, pos);
}

bool env_cache_configured()
{
    // 行为开关（RV-09：读取收敛到 core_env.hpp）。
    return !::nn::detail::env_string_raw("NN_KERNEL_CACHE_PATH").empty() ||
           !::nn::detail::env_string_raw("NN_KERNEL_CACHE_DIR").empty();
}
}  // namespace

int main(int argc, char** argv)
{
    // ── 缓存目录（在引擎/缓存首次触碰前落定）────────────────────────────────
    std::string out_dir;
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc)
        {
            out_dir = argv[++i];
        }
        else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0)
        {
            std::cout << "用法: nn_warmup [--out <dir>]\n"
                         "  GPU 常用架构前向预热 → kernels.nnk（--out 缺省 = 可执行文件目录；\n"
                         "  已设 NN_KERNEL_CACHE_PATH/DIR 环境变量时以环境变量为准）\n";
            return 0;
        }
        else
        {
            std::cerr << "未知参数: " << argv[i] << "\n";
            return 1;
        }
    }
    if (!env_cache_configured()) set_cache_dir(out_dir.empty() ? exe_dir(argv[0]) : out_dir);

#ifndef NN_GPU_ENGINE
    // GPU OFF（NN_ENABLE_GPU=OFF）：create_gpu_engine 符号不存在于 nn-cpu
    // （gpu_engine_so.cpp 只编进 libnn-gpu），无守卫引用 = 链接失败——与
    // cli_engine_factory 同一守卫口径（CMakeLists 1664 注释）。本工具只为
    // GPU kernel 缓存服务，无 GPU 即 SKIP 返回 0（与"无 Vulkan 设备"同语义）。
    std::cout << "[nn_warmup] SKIP: 本构建未启用 GPU 引擎（NN_ENABLE_GPU=OFF）\n";
    return 0;
#else
    // ── GPU 引擎（本目标只为 GPU kernel 缓存服务）────────────────────────────
    auto engine = nn::create_gpu_engine(nullptr);
    if (!engine)
    {
        const char* err = nn::last_error();
        std::cout << "[nn_warmup] SKIP: 无可用 Vulkan 设备（发布机无 GPU 不算失败）："
                  << (err != nullptr ? err : "(无错误信息)") << "\n";
        return 0;
    }
    if (engine->device() != nn::Device::GPU)
    {
        std::cout << "[nn_warmup] SKIP: create_gpu_engine 返回非 GPU 设备\n";
        return 0;
    }

    const std::string cache_path = nn::fused::kernel_cache_path();
    std::cout << "[nn_warmup] GPU: " << engine->describe().text() << "\n"
              << "[nn_warmup] 缓存: " << cache_path << "\n";

    int failures = 0;
    const auto warm = [&](const char* name, nn::Model& model, nn::Tensor&& x) -> void
    {
        const auto r = nn::model_warmup(model, x);
        if (!r)
        {
            std::cerr << "[nn_warmup][FAIL] " << name << ": " << r.error().message << "\n";
            ++failures;
        }
        else
        {
            std::cout << "[nn_warmup][ok] " << name << "\n";
        }
    };
    const auto zero = [&](std::size_t rows, std::size_t cols)
    { return engine->create_tensor(rows, cols, nn::Precision::F32, nn::InitSpec::zero()); };

    // 1. MNIST-MLP（Linear/ReLU/Norm 头）
    {
        auto m = nn::build_mnist_mlp_model(*engine);
        if (!m) { std::cerr << "[nn_warmup][FAIL] mlp build: " << m.error().message << "\n"; ++failures; }
        else warm("mnist-mlp (784x4)", *m, zero(784, 4));
    }
    // 2. MNIST-CNN（im2col / col2im / MaxPool / BatchNorm）
    {
        nn::CnnConfig c;   // mnist 默认：1×28×28，convs {6@5,16@5}，fc {120,10}
        c.convs.push_back({6, 5});
        c.convs.push_back({16, 5});
        c.fc_dims = {120, 10};
        auto m = nn::build_cnn_model(*engine, c);
        if (!m) { std::cerr << "[nn_warmup][FAIL] cnn build: " << m.error().message << "\n"; ++failures; }
        else warm("mnist-cnn (784x4)", *m, zero(784, 4));
    }
    // 3. MNIST-Transformer（patch embed + 注意力 fold + 位置编码）
    {
        auto m = nn::build_mnist_transformer_model(*engine);
        if (!m) { std::cerr << "[nn_warmup][FAIL] transformer build: " << m.error().message << "\n"; ++failures; }
        else warm("mnist-transformer (784x4)", *m, zero(784, 4));
    }
    // 4. tiny GPT（causal attention fold + GELU FFN + LM head；ids 形状 = (seq, batch)）
    {
        nn::GptConfig g;
        g.vocab_size = 256;
        g.d_model = 64;
        g.seq_len = 32;
        g.num_heads = 4;
        g.d_ff = 128;
        g.num_layers = 2;
        auto m = nn::build_gpt_model(*engine, g);
        if (!m) { std::cerr << "[nn_warmup][FAIL] gpt build: " << m.error().message << "\n"; ++failures; }
        else warm("gpt (32x2)", *m, zero(32, 2));
    }
    // 5. tiny RAPT（RoPE 线性注意力 scan 路径；同 (seq, batch) ids）
    {
        nn::RAPTConfig r;
        r.vocab_size = 256;
        r.d_model = 64;
        r.seq_len = 32;
        r.num_heads = 4;
        r.d_ff = 128;
        r.num_layers = 2;
        auto m = nn::build_rapt_model(*engine, r);
        if (!m) { std::cerr << "[nn_warmup][FAIL] rapt build: " << m.error().message << "\n"; ++failures; }
        else warm("rapt (32x2)", *m, zero(32, 2));
    }

    if (failures != 0)
    {
        std::cerr << "[nn_warmup] " << failures << " 项失败\n";
        return 1;
    }
    std::cout << "[nn_warmup] 完成，kernels.nnk = " << cache_path << "\n";
    return 0;
#endif // NN_GPU_ENGINE
}
