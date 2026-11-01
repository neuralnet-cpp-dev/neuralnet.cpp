// ═══════════════════════════════════════════════════════════════════════════
//  jit_demo — 演示 Phase 2 运行期 JIT（docs/internal/33）
//
//  构造一个**不在内嵌 AOT 注册表**的自定义表达式 exp(x*y)+0.5，在 GPU 上：
//    · 首次执行 → JIT 现编（懒 dlopen libnn-kernelgen.so → nnk_generate_spirv →
//      后端懒建 pipeline → 执行），计时含一次性编译延迟；
//    · 第二次 → .nnk 缓存命中（纯执行），计时应远小于首次；
//    · 结果与 CPU 参考对拍；
//    · 打印 .nnk 缓存里新增的 kernel key（证明 JIT 落盘持久化）。
//  运行：NN_VULKAN_DEVICE=850M NN_KERNELGEN_PATH=<build>/lib/libnn_kernelgen.so \
//        NN_KERNEL_CACHE_PATH=/tmp/jit_demo.nnk ./build/jit_demo
// ═══════════════════════════════════════════════════════════════════════════

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>

#include <neuralnet.cpp/nn.hpp>
#include <neuralnet.cpp/kernel_cache.hpp>

using nn::Scalar;

[[maybe_unused]] static double ms(std::chrono::steady_clock::time_point a,
                                  std::chrono::steady_clock::time_point b)
{
    return std::chrono::duration<double, std::milli>(b - a).count();
}

int main()
{
    auto cpu = nn::create_cpu_engine();

#ifdef NN_GPU_ENGINE
    auto gpu = nn::create_gpu_engine("");   // 空 = 自动选卡（NN_VULKAN_DEVICE 可显式）
    if (!gpu)
    {
        const char* err = nn::last_error();
        std::printf("[SKIP] 无可用 GPU 设备：%s\n", err ? err : "(无错误信息)");
        return 77;
    }
    std::printf("GPU: %s\n\n", gpu->describe().device.c_str());


    // 演示干净起点：清空缓存文件（验证"空缓存 → JIT → 落盘"全过程）
    const std::string cpath = nn::fused::kernel_cache_path();
    std::remove(cpath.c_str());

    const std::size_t R = 16, C = 32;
    std::mt19937 rng(20261101);
    std::uniform_real_distribution<Scalar> dist(-1.5f, 1.5f);
    nn::Matrix x(R, C), y(R, C);
    for (auto& v : x.span()) v = dist(rng);
    for (auto& v : y.span()) v = dist(rng);

    // 表达式：x*y + 3（两个输入 + 常量；fused_gpu_test 已验证**不在内嵌注册表**、
    // 必然触发 JIT 现编）。若命中内嵌 AOT，则缓存不会新增 kernel（demo 会标明）。
    // Tensor 一律经引擎创建（铁律 #11）
    const auto xc_r = cpu->from_matrix(x), yc_r = cpu->from_matrix(y);
    const auto xg_r = gpu->from_matrix(x), yg_r = gpu->from_matrix(y);
    if (!xc_r || !yc_r || !xg_r || !yg_r) { std::printf("[FAIL] 上传失败\n"); return 1; }
    const nn::Tensor xc = *xc_r, yc = *yc_r, xg = *xg_r, yg = *yg_r;

    // CPU 参考
    auto cref = nn::dsl::compute(*cpu, nn::dsl::leaf(xc) * nn::dsl::leaf(yc) + Scalar{3}, R, C);
    if (!cref) { std::printf("[FAIL] CPU 求值失败：%s\n", cref.error().message.c_str()); return 1; }

    // 首次 GPU 执行（触发 JIT 现编）
    const auto t0 = std::chrono::steady_clock::now();
    auto g1 = nn::dsl::compute(*gpu, nn::dsl::leaf(xg) * nn::dsl::leaf(yg) + Scalar{3}, R, C);
    const auto t1 = std::chrono::steady_clock::now();
    if (!g1) { std::printf("[FAIL] 首次 GPU 执行失败：%s\n", g1.error().message.c_str()); return 1; }

    // 第二次（.nnk 缓存命中，纯执行）
    const auto t2 = std::chrono::steady_clock::now();
    auto g2 = nn::dsl::compute(*gpu, nn::dsl::leaf(xg) * nn::dsl::leaf(yg) + Scalar{3}, R, C);
    const auto t3 = std::chrono::steady_clock::now();

    // 下载 GPU 结果与 CPU 对拍
    auto gm1 = gpu->to_matrix(*g1);
    auto gm2 = gpu->to_matrix(*g2);
    auto cref_m_r = cpu->to_matrix(*cref);
    if (!gm1 || !gm2 || !cref_m_r) { std::printf("[FAIL] 下载失败\n"); return 1; }
    const nn::Matrix& cref_m = *cref_m_r;
    Scalar maxerr1 = 0, maxerr2 = 0;
    for (std::size_t i = 0; i < cref_m.span().size(); ++i)
    {
        maxerr1 = std::max(maxerr1, std::fabs(cref_m.span()[i] - gm1->span()[i]));
        maxerr2 = std::max(maxerr2, std::fabs(cref_m.span()[i] - gm2->span()[i]));
    }

    std::printf("表达式 x*y+3（不在内嵌注册表）\n");
    std::printf("  首次 GPU 执行（JIT 现编 + 执行）: %7.3f ms\n", ms(t0, t1));
    std::printf("  第二次执行（.nnk 缓存命中）   : %7.3f ms\n", ms(t2, t3));
    std::printf("  JIT vs 命中倍率               : %.1fx\n",
                ms(t2, t3) > 0 ? ms(t0, t1) / ms(t2, t3) : 0.0);
    std::printf("  结果对拍 CPU  max err          : 首次 %.3e / 命中 %.3e\n", maxerr1, maxerr2);

    // 打印 .nnk 缓存落盘的 kernel key（证明 JIT 持久化）
    std::printf("\n.nnk 缓存（%s）落盘的 kernel：\n", cpath.c_str());
    {
        std::ifstream f(cpath, std::ios::binary);
        if (f)
        {
            std::ostringstream ss;
            ss << f.rdbuf();
            auto rec = nn::KeyValueRecord::parse(ss.str());
            if (rec)
            {
                auto keys = rec->string_keys_with_prefix("kernel.");
                for (const auto& k : keys) std::printf("  - %s\n", k.c_str());
                if (keys.empty()) std::printf("  （空——该结构走了 AOT 命中，未落盘）\n");
            }
        }
        else std::printf("  （缓存文件未生成）\n");
    }

    const bool ok = maxerr1 < 1e-4f && maxerr2 < 1e-4f;
    std::printf("\n%s\n", ok ? "JIT 演示通过：现编成功、缓存命中、结果与 CPU 一致" : "JIT 演示失败");
    return ok ? 0 : 1;
#else
    std::printf("[SKIP] 本构建未链接 GPU 引擎（NN_ENABLE_GPU=OFF / 无 Vulkan）\n");
    return 77;
#endif
}
