// ── gpu_batch_scope_test — GPU 单次推理 batch 作用域（issue #28）──────────
// 覆盖：
//   [1] CPU：InferenceScope 透传（begin/end 均 no-op），active()==true
//       （batch_active()==false ⇒ 自开；CPU 引擎自身 no-op）
//   [2] GPU：Model::forward 不再要求调用方手动 begin_batch/end_batch
//       （多 Conv 模型逐原语路径曾 VK_ERROR_OUT_OF_DEVICE_MEMORY）
//   [3] GPU：外层已持 batch 时 InferenceScope 被动透传（不重复 begin）
//   [4] GPU：外层 begin_batch + InferenceScope + forward 组合不撞
//       "Already in batch mode"
//   [5] #M7：model_warmup 首次 forward 失败后状态可回滚（生成器加载失败不被
//       "已尝试"latch 固化 → 二次调用仍报错而非静默成功/空指针崩溃；修正
//       NN_KERNELGEN_PATH 后二次 warmup 真正可用）
// 无 Vulkan 构建 / 无 GPU 设备 → 返回 77 = ctest SKIP。
// ─────────────────────────────────────────────────────────────────────────

#include <nn_error_macros.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
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

int g_fail = 0;
#define CHECK(cond, msg)                                                                  \
    do                                                                                    \
    {                                                                                     \
        if (!(cond))                                                                      \
        {                                                                                 \
            std::cout << "[FAIL] " << __FILE__ << ":" << __LINE__ << ": " << msg << "\n"; \
            ++g_fail;                                                                     \
        }                                                                                 \
    } while (0)

#ifdef NN_GPU_ENGINE
namespace
{

// 进程级临时 .nnk 缓存（隔离用户缓存，保证本用例的表达式 JIT 必现 miss）。
struct TempCacheFile
{
    std::string path;
    ~TempCacheFile()
    {
        std::remove(path.c_str());
        std::remove((path + ".tmp").c_str());
    }
};

// 多 Conv 小模型：issue #28 的触发形态（第二次 Conv 分配处 OOM）
nn::Result<nn::Model> make_conv_model(nn::ComputeEngine& eng)
{
    nn::Model m(eng);
    // 位置参数：(in_channels, out_channels, kernel, stride, padding, in_h, in_w)
    NN_TRY(c1, m.add<nn::Conv2D>(1, 8, 3, 1, 0, 8, 8));
    NN_TRY(c2, m.add<nn::Conv2D>(8, 16, 3, 1, 0, 6, 6));
    return m;
}

// #M7：定位真实 nn-kernelgen 库（PATH 目录 / exe 邻近目录），找不到返回空串
//（空串 = 交回系统搜索顺序）。测试不硬编码构建布局。
std::string find_kernelgen_(const char* argv0)
{
#ifdef _WIN32
    const char* libname = "nn-kernelgen.dll";
    const char sep = ';';
#else
    const char* libname = "libnn-kernelgen.so";
    const char sep = ':';
#endif
    const auto exists = [](const std::string& p)
    {
        std::ifstream f(p, std::ios::binary);
        return f.good();
    };
    const auto join = [](const std::string& d, const char* n)
    { return d.empty() ? std::string(n) : d + "/" + n; };
    // 1) PATH 目录
    if (const char* penv = std::getenv("PATH"))
    {
        const std::string p = penv;
        std::size_t pos = 0;
        for (;;)
        {
            const std::size_t next = p.find(sep, pos);
            const std::string dir =
                p.substr(pos, next == std::string::npos ? std::string::npos : next - pos);
            const std::string cand = join(dir, libname);
            if (exists(cand)) return cand;
            if (next == std::string::npos) break;
            pos = next + 1;
        }
    }
    // 2) exe 邻近（build/<...>/test → build/<...>/engine 是当前布局）
    if (argv0 != nullptr)
    {
        const std::string exe = argv0;
        const std::size_t slash = exe.find_last_of("/\\");
        const std::string dir = slash == std::string::npos ? std::string{} : exe.substr(0, slash);
        for (const std::string& d :
             {dir, dir + "/..", dir + "/../engine", dir + "/../../engine"})
        {
            const std::string cand = join(d, libname);
            if (exists(cand)) return cand;
        }
    }
    return {};
}

// 环境变量写入（_putenv_s / setenv 断言不可失败）
void set_env_(const char* name, const std::string& value)
{
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

} // namespace
#endif // NN_GPU_ENGINE

int main(int argc, char** argv)
{
    (void)argc;
    // [1] CPU 透传
    {
        auto eng = nn::create_cpu_engine();
        CHECK(!eng->batch_active(), "CPU batch_active 恒 false");
        nn::InferenceScope scope(*eng);
        CHECK(scope.ok(), "CPU InferenceScope 应 ok");
        CHECK(scope.active(), "CPU：batch_active()==false ⇒ 自开（引擎侧 no-op）");
        CHECK(!eng->batch_active(), "CPU batch_active 仍 false");
    }

#ifdef NN_GPU_ENGINE
    // #M7：隔离 .nnk 缓存（kernel_cache_path() 进程内只读一次，须在任何引擎
    // 触碰前落定）——否则用户缓存里已预热的 Linear 结构会让 JIT 不触发，
    // 用例无法构造"生成器缺失"的失败路径。
    TempCacheFile temp_cache;
    {
        const char* tmp = std::getenv("TEMP");
        if (tmp == nullptr) tmp = std::getenv("TMPDIR");
        if (tmp == nullptr) tmp = ".";
        temp_cache.path = std::string(tmp) + "/nn_m7_gpu_scope_";
        temp_cache.path +=
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
        temp_cache.path += ".nnk";
        set_env_("NN_KERNEL_CACHE_PATH", temp_cache.path);
    }

    auto engine_r = nn::cli::create_engine(nn::cli::EngineConfig{.use_gpu = true});
    if (!engine_r)
    {
        std::cout << "[SKIP] GPU 初始化失败：" << engine_r.error().message << "\n";
        return 77;
    }
    nn::ComputeEngine& gpu = **engine_r;

    // [2] GPU：不手动包 batch 直接 forward（issue #28 主诉路径）
    {
        auto model_r = make_conv_model(gpu);
        CHECK(model_r.has_value(), "建多 Conv 模型");
        if (model_r)
        {
            // 输入 (C·H·W, B) = (64, 1)：conv3→(288,1)→conv3→(256,1)
            std::vector<nn::Scalar> img(64, 0.5f);
            auto in = nn::upload_span(gpu, 64, 1, nn::Precision::F32, img);
            CHECK(in.has_value(), "upload 输入");
            if (in)
            {
                // 曾在此路径（无 begin_batch）第二次 Conv 分配报
                // VK_ERROR_OUT_OF_DEVICE_MEMORY；Model::forward 现自动作用域
                auto out = model_r->forward(*in);
                CHECK(out.has_value(), "GPU Model::forward 无手动 batch 应成功");
            }
        }
        CHECK(!gpu.batch_active(), "forward 返回后 batch 已收口");
    }

    // [3] 外层已持 batch：InferenceScope 被动透传
    {
        nn::InferenceScope outer(gpu);
        CHECK(outer.ok() && outer.active(), "外层作用域自开");
        nn::InferenceScope inner(gpu);
        CHECK(inner.ok(), "内层作用域 ok");
        CHECK(!inner.active(), "内层：外层已持 batch ⇒ 透传（不重复 begin）");
    }
    CHECK(!gpu.batch_active(), "外层作用域析构后 batch 已收口");

    // [4] 手动 begin_batch + InferenceScope 组合（模拟训练循环内评估）
    {
        auto br = gpu.begin_batch();
        CHECK(br.has_value(), "手动 begin_batch");
        nn::InferenceScope scope(gpu);
        CHECK(scope.ok() && !scope.active(), "手动 batch 下作用域透传");
        auto er = gpu.end_batch();
        CHECK(er.has_value(), "手动 end_batch");
    }

    // [5] #M7：model_warmup 失败可回滚、二次可用
    {
        // 清掉 [2]~[4] 已落盘的 kernel（同进程内共享 .nnk 文件），保证下面
        // 独立引擎必定 JIT miss —— 否则 Conv 的 "matmul+行广播" 结构与 Linear
        // 同 key，会直接命中缓存而绕过生成器加载。
        std::remove(temp_cache.path.c_str());
        std::remove((temp_cache.path + ".tmp").c_str());

        // 独立引擎：全新的 JIT 懒加载状态（不继承 [2]~[4] 已加载的生成器）。
        auto e2 = nn::create_gpu_engine(nullptr);
        CHECK(static_cast<bool>(e2), "创建独立 GPU 引擎");
        if (e2)
        {
            nn::Model m(*e2);
            CHECK(static_cast<bool>(m.add<nn::Linear>(4, 3)), "建 Linear 模型（表达式需 JIT）");
            std::vector<nn::Scalar> vals(8, 0.5f);
            auto in = nn::upload_span(*e2, 4, 2, nn::Precision::F32, vals);
            CHECK(in.has_value(), "上传 warmup 输入");
            if (in)
            {
                const char* old_path = std::getenv("NN_KERNELGEN_PATH");
                const std::string saved = old_path != nullptr ? old_path : std::string{};
                // 生成器指向不存在的文件 → 首次 warmup 必失败（失败可回滚）
                set_env_("NN_KERNELGEN_PATH", "__nn_m7_missing_kernelgen__");
                auto first = nn::model_warmup(m, *in);
                CHECK(!first.has_value(), "生成器缺失 → 首次 warmup 失败");
                CHECK(!e2->batch_active(), "首次失败后引擎已退出 batch（无残留）");

                // 同一坏路径再跑：旧实现"已尝试"latch 会让二次静默成功并解引用
                // 空 generate 指针（崩溃）；修正后应再次干净报错。
                auto second = nn::model_warmup(m, *in);
                CHECK(!second.has_value(), "二次 warmup 干净报错（失败未被 latch 固化）");
                CHECK(!e2->batch_active(), "二次失败后 batch 仍无残留");

                // 修正 NN_KERNELGEN_PATH → 第三次应真正可用
                const std::string real = find_kernelgen_(argv[0]);
                set_env_("NN_KERNELGEN_PATH", real);
                if (!real.empty())
                {
                    auto third = nn::model_warmup(m, *in);
                    CHECK(third.has_value(), "修正 NN_KERNELGEN_PATH 后 warmup 成功");
                    CHECK(!e2->batch_active(), "成功后 batch 已收口");
                }
                else
                {
                    std::cout << "[SKIP] 未定位 nn-kernelgen 库，跳过\"修正后可成功\"分支\n";
                }
                set_env_("NN_KERNELGEN_PATH", saved);
            }
        }
    }

    std::cout << (g_fail == 0 ? "\n全部通过 ✅\n" : "\n存在失败 ❌\n");
    return g_fail == 0 ? 0 : 1;
#else
    std::cout << "[SKIP] 纯 CPU 构建（无 Vulkan）\n";
    return 77;
#endif
}
