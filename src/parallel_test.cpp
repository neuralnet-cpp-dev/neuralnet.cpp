// ── parallel_test — core_parallel.hpp：顶层并行算法（阈值回落语义）─────────
// 覆盖：
//   1. for_each / transform（一元/二元）/ transform_reduce（一元/二元）在
//      小输入（< PARALLEL_THRESHOLD，串行回落）下与串行参考逐元素一致
//   2. 大输入（≥ PARALLEL_THRESHOLD，真并行路径）结果与串行参考一致
//   3. 空区间 / 单元素边界
//   4. parallel_for_samples 的 0/1 样本串行回落
//   5. 并行 vs 串行对拍的确定性（transform_reduce 用整数求和，精确对拍）
// 退出码：0 = 通过，1 = 失败。
// ───────────────────────────────────────────────────────────────────────────

#include <atomic>
#include <cstddef>
#include <cstdio>
#include <numeric>
#include <vector>

#include <neuralnet.cpp/core_parallel.hpp>

int g_fail = 0;

#include "test_common.hpp"

namespace
{

// PARALLEL_THRESHOLD = 524288（core_config.hpp）。测试用两档规模：
//   小 = 阈值/16（必然串行回落）、大 = 阈值 + 阈值/8（必然真并行）
constexpr std::size_t kSmall = nn::PARALLEL_THRESHOLD / 16;
constexpr std::size_t kLarge = nn::PARALLEL_THRESHOLD + nn::PARALLEL_THRESHOLD / 8;

void test_for_each()
{
    std::printf("for_each（串行回落 + 真并行）:\n");
    for (std::size_t kN : {kSmall, kLarge})
    {
        std::vector<int> hits(kN, 0);
        nn::for_each(hits.begin(), hits.end(), [](int& v) { ++v; });
        bool full = true;
        for (std::size_t i = 0; i < kN; ++i)
            if (hits[i] != 1) full = false;
        CHECK(full, (kN >= nn::PARALLEL_THRESHOLD ? "大输入全并行覆盖" : "小输入串行覆盖"));
    }
    // 空 / 单元素
    std::vector<int> empty;
    nn::for_each(empty.begin(), empty.end(), [](int& v) { ++v; });
    CHECK(true, "空区间不崩溃");
    std::vector<int> one{0};
    nn::for_each(one.begin(), one.end(), [](int& v) { ++v; });
    CHECK(one[0] == 1, "单元素");
}

void test_transform()
{
    std::printf("transform（一元/二元）:\n");
    for (std::size_t kN : {kSmall, kLarge})
    {
        const std::vector<float> in(kN, 2.5f);
        std::vector<float> out(kN, 0.0f);
        nn::transform(in.begin(), in.end(), out.begin(), [](float x) { return x + 1.0f; });
        bool ok = true;
        for (std::size_t i = 0; i < kN; ++i)
            if (out[i] != 3.5f) ok = false;
        CHECK(ok, "一元 transform 逐元素一致");

        std::vector<float> out2(kN, 0.0f);
        nn::transform(in.begin(), in.end(), out.begin(), out2.begin(),
                      [](float a, float b) { return a * b; });
        bool ok2 = true;
        for (std::size_t i = 0; i < kN; ++i)
            if (out2[i] != 8.75f) ok2 = false;   // in(2.5) * out(3.5)
        CHECK(ok2, "二元 transform 逐元素一致");
    }
}

void test_transform_reduce()
{
    std::printf("transform_reduce（整数精确对拍）:\n");
    for (std::size_t kN : {kSmall, kLarge})
    {
        std::vector<int> v(kN);
        std::iota(v.begin(), v.end(), 1);                 // 1..kN
        const long long ref = static_cast<long long>(kN) * (kN + 1) / 2;

        const long long sum = nn::transform_reduce(
            v.begin(), v.end(), 0LL, [](long long a, long long b) { return a + b; },
            [](int x) { return static_cast<long long>(x); });
        CHECK(sum == ref, kN >= nn::PARALLEL_THRESHOLD ? "一元 transform_reduce 大输入（并行）"
                                                       : "一元 transform_reduce 小输入（串行）");

        // 二元：v 与自身点积的和
        const long long sq = nn::transform_reduce(
            v.begin(), v.end(), v.begin(), 0LL, [](long long a, long long b) { return a + b; },
            [](int a, int b) { return static_cast<long long>(a) * b; });
        // Σ i² = n(n+1)(2n+1)/6
        const long long ref_sq = static_cast<long long>(kN) * (kN + 1) * (2 * kN + 1) / 6;
        CHECK(sq == ref_sq, "二元 transform_reduce 与 Σi² 一致");
    }
}

void test_parallel_for_samples_fallback()
{
    std::printf("parallel_for_samples 串行回落:\n");
    std::atomic<int> calls{0};
    nn::parallel_for_samples(0, [&](std::size_t) { ++calls; });
    CHECK(calls.load() == 0, "0 样本不调用回调");
    nn::parallel_for_samples(1,
                             [&](std::size_t i)
                             {
                                 if (i == 0) ++calls;
                             });
    CHECK(calls.load() == 1, "1 样本串行回落且索引 0");
    std::vector<int> hits(16, 0);
    nn::parallel_for_samples(16, [&](std::size_t i) { ++hits[i]; });
    bool full = true;
    for (int h : hits)
        if (h != 1) full = false;
    CHECK(full, "16 样本每样本恰好一次");
}

void test_parallel_for_blocks()
{
    std::printf("parallel_for_blocks:\n");
    // 无阈值保护：任何规模直接并行
    for (std::size_t kN : {std::size_t{3}, kSmall, kLarge})
    {
        std::vector<int> hits(kN, 0);
        nn::parallel_for_blocks(hits.begin(), hits.end(), [](int& v) { ++v; });
        bool full = true;
        for (std::size_t i = 0; i < kN; ++i)
            if (hits[i] != 1) full = false;
        CHECK(full, (std::to_string(kN) + " 块全覆盖").c_str());
    }
}

}  // namespace

int main()
{
    std::puts("=== parallel algorithms (core_parallel.hpp) ===");
    test_for_each();
    test_transform();
    test_transform_reduce();
    test_parallel_for_samples_fallback();
    test_parallel_for_blocks();
    std::printf("\nparallel_test: %d failure(s)\n", g_fail);
    return g_fail != 0 ? 1 : 0;
}
