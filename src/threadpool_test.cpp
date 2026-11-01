// ── threadpool_test — core_threadpool.hpp：ThreadPool 本体 ──────────────────
// 覆盖：
//   1. 构造（指定线程数 / 0 → 回落 1）与析构（任务排空后线程退出）
//   2. parallel_for_each：全覆盖（每元素恰好访问一次）、空区间、单元素
//   3. parallel_for_each_indexed：worker 索引 ∈ [0, n_threads]（调用者 = size()）
//   4. parallel_for_blocks / parallel_for_samples：全覆盖 + 0/1 样本回落
//   5. 并行 transform / transform_reduce 结果与串行参考一致
//   6. 重复调用稳定性（同一池上数千次空并行区，防 latch/cv 死锁回归）
// 退出码：0 = 通过，1 = 失败。
// ───────────────────────────────────────────────────────────────────────────

#include <atomic>
#include <cstddef>
#include <cstdio>
#include <numeric>
#include <vector>

#include <neuralnet.cpp/core_threadpool.hpp>

int g_fail = 0;

#include "test_common.hpp"

namespace
{

void test_construction()
{
    std::printf("构造/析构:\n");
    {
        nn::ThreadPool pool(4);
        CHECK(true, "指定线程数构造不崩溃");
    }
    {
        nn::ThreadPool pool(0);
        CHECK(true, "0 线程回落 1 不崩溃");
    }
    CHECK(true, "析构（线程 join）正常");
}

void test_parallel_for_each_full_coverage()
{
    std::printf("parallel_for_each 全覆盖:\n");
    nn::ThreadPool pool(4);
    // 200k 元素 > MIN_CHUNK(1024)*2 → 必然分多块
    constexpr std::size_t kN = 200000;
    std::vector<int> hits(kN, 0);
    pool.parallel_for_each(hits.begin(), hits.end(), [](int& v) { ++v; });
    bool full = true;
    for (std::size_t i = 0; i < kN; ++i)
        if (hits[i] != 1) full = false;
    CHECK(full, "每元素恰好访问一次（无丢失/无重复）");

    // 空区间与单元素
    std::vector<int> empty;
    pool.parallel_for_each(empty.begin(), empty.end(), [](int& v) { ++v; });
    CHECK(true, "空区间不崩溃");
    std::vector<int> one{0};
    pool.parallel_for_each(one.begin(), one.end(), [](int& v) { ++v; });
    CHECK(one[0] == 1, "单元素被访问");

    // 小区间（< MIN_CHUNK*2）内部串行回落，结果仍全覆盖
    std::vector<int> small(100, 0);
    pool.parallel_for_each(small.begin(), small.end(), [](int& v) { ++v; });
    bool small_ok = true;
    for (int v : small)
        if (v != 1) small_ok = false;
    CHECK(small_ok, "小区间（串行回落）全覆盖");
}

void test_parallel_for_each_indexed()
{
    std::printf("parallel_for_each_indexed worker 索引:\n");
    nn::ThreadPool pool(4);
    constexpr std::size_t kN = 200000;
    constexpr std::size_t kThreads = 4;
    std::vector<int> hits(kN, 0);
    std::atomic<std::size_t> bad_index{0};
    pool.parallel_for_each_indexed(hits.begin(), hits.end(),
                                   [&](int& v, std::size_t worker_index)
                                   {
                                       ++v;
                                       // worker = 0..3，调用者线程 = 4（= pool size）
                                       if (worker_index > kThreads) ++bad_index;
                                   });
    CHECK(bad_index.load() == 0, "worker 索引 ∈ [0, size()]");
    bool full = true;
    for (std::size_t i = 0; i < kN; ++i)
        if (hits[i] != 1) full = false;
    CHECK(full, "全覆盖");

    // 单线程回落：index 恒 0
    std::vector<int> one_hits(100, 0);
    nn::ThreadPool single(1);
    std::atomic<std::size_t> nonzero{0};
    single.parallel_for_each_indexed(one_hits.begin(), one_hits.end(),
                                     [&](int&, std::size_t wi)
                                     {
                                         if (wi != 0) ++nonzero;
                                     });
    CHECK(nonzero.load() == 0, "单线程回落 index = 0");
}

void test_parallel_for_blocks_and_samples()
{
    std::printf("parallel_for_blocks / parallel_for_samples:\n");
    nn::ThreadPool pool(4);

    // blocks：块数 <= 线程数，但每块仍完整覆盖
    std::vector<int> hits(64, 0);
    pool.parallel_for_blocks(hits.begin(), hits.end(), [](int& v) { ++v; });
    bool full = true;
    for (int v : hits)
        if (v != 1) full = false;
    CHECK(full, "for_blocks 全覆盖");

    // samples：8 样本 × 大工作量
    constexpr std::size_t kSamples = 8;
    std::vector<int> sample_hits(kSamples, 0);
    pool.parallel_for_samples(kSamples, [&](std::size_t i) { ++sample_hits[i]; });
    bool sfull = true;
    for (int v : sample_hits)
        if (v != 1) sfull = false;
    CHECK(sfull, "for_samples 每样本恰好一次");

    // 0/1 样本串行回落
    std::atomic<int> zero_calls{0};
    pool.parallel_for_samples(0, [&](std::size_t) { ++zero_calls; });
    CHECK(zero_calls.load() == 0, "0 样本不调用回调");
    std::atomic<int> one_calls{0};
    pool.parallel_for_samples(1,
                              [&](std::size_t i)
                              {
                                  if (i == 0) ++one_calls;
                              });
    CHECK(one_calls.load() == 1, "1 样本串行回落且索引 0");
}

void test_transform_and_reduce()
{
    std::printf("parallel transform / transform_reduce:\n");
    nn::ThreadPool pool(4);
    // 200k 触发并行路径（chunk_count 多块）
    constexpr std::size_t kN = 200000;
    std::vector<float> in(kN), out(kN, 0.0f);
    for (std::size_t i = 0; i < kN; ++i)
        in[i] = static_cast<float>(i) * 0.5f;

    pool.parallel_transform(in.begin(), in.end(), out.begin(), [](float x) { return x * 2.0f; });
    bool tok = true;
    for (std::size_t i = 0; i < kN; ++i)
        if (out[i] != in[i] * 2.0f) tok = false;
    CHECK(tok, "一元 transform 逐元素一致");

    // 二元 transform
    std::vector<float> out2(kN, 0.0f);
    pool.parallel_transform(in.begin(), in.end(), out.begin(), out2.begin(),
                            [](float a, float b) { return a + b; });
    bool tok2 = true;
    for (std::size_t i = 0; i < kN; ++i)
        if (out2[i] != in[i] + out[i]) tok2 = false;
    CHECK(tok2, "二元 transform 逐元素一致");

    // transform_reduce：int 求和（无浮点结合律噪声，可精确对拍）
    std::vector<int> ints(kN);
    std::iota(ints.begin(), ints.end(), 0);          // 0..199999
    const long long ref_sum = static_cast<long long>(kN) * (kN - 1) / 2;  // 精确闭式
    const long long sum = pool.parallel_transform_reduce(
        ints.begin(), ints.end(), 0LL, [](long long a, long long b) { return a + b; },
        [](int x) { return static_cast<long long>(x); });
    CHECK(sum == ref_sum, "一元 transform_reduce 与闭式和一致");

    const long long dot = pool.parallel_transform_reduce(
        ints.begin(), ints.end(), in.begin(), 0LL, [](long long a, long long b) { return a + b; },
        [](int x, float y) { return static_cast<long long>(x) * static_cast<long long>(y); });
    CHECK(dot >= 0, "二元 transform_reduce 返回有限值（不崩）");
}

void test_repeated_empty_regions()
{
    // 历史回归：latch 归零 + notify 不持 queue_mutex_ 会丢唤醒 → 必死锁。
    // 空并行区压测（持锁 notify 修复后应稳定通过）。
    std::printf("重复空并行区压测（latch/cv 回归）:\n");
    nn::ThreadPool pool(4);
    for (int iter = 0; iter < 10000; ++iter)
    {
        pool.parallel_for_samples(2, [](std::size_t) {});   // 多块 → 走 latch 路径
        pool.parallel_for_samples(3, [](std::size_t) {});
    }
    CHECK(true, "2×10000 次并行区无死锁");
}

}  // namespace

int main()
{
    std::puts("=== ThreadPool (core_threadpool.hpp) ===");
    test_construction();
    test_parallel_for_each_full_coverage();
    test_parallel_for_each_indexed();
    test_parallel_for_blocks_and_samples();
    test_transform_and_reduce();
    test_repeated_empty_regions();
    std::printf("\nthreadpool_test: %d failure(s)\n", g_fail);
    return g_fail != 0 ? 1 : 0;
}
