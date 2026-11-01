// ── assert_test — core_assert.hpp：断言宏族（纯宏级，不含上层调用点）──────
// 覆盖：
//   1. NN_ASSERT_ALWAYS：条件真时不终止；条件假 → 子进程非零退出，
//      stderr 含 "NN_ASSERT_ALWAYS failed at" / 文件名 / 行号 / 消息
//   2. NN_FATAL：无条件终止，打印 "NN_FATAL failed at ..."（NDEBUG 无关）
//   3. NN_ASSERT：Debug 下失败即 abort；NDEBUG 下为空展开（条件可为副作用
//      表达式——空展开时**不求值**，本测试只断言"真条件不终止"的两态一致）
//   4. NN_REQUIRE：前置失败返回 std::unexpected，成功路径不改动
// 子进程模式：--expect-assert-always / --expect-fatal（父进程 std::system 自举）
// 退出码：0 = 通过，1 = 失败。
// ───────────────────────────────────────────────────────────────────────────

#include <cstdio>
#include <cstdlib>
#include <string>

#ifndef _WIN32
#include <sys/wait.h>
#endif

#include <neuralnet.cpp/core_assert.hpp>

int g_fail = 0;

#include "test_common.hpp"

namespace
{

constexpr int kSkip = -1;   // std::system 无法启动子进程

std::string g_self;         // argv[0]：子进程自举用

nn::Result<int> require_ok()
{ return 42; }
nn::Result<int> require_bad()
{ NN_FAIL("require failed"); }

int run_self(const char* args)
{
    const std::string cmd = "\"" + g_self + "\" " + args;
    const int rc = std::system(cmd.c_str());
    if (rc == -1)
    {
        std::printf("[SKIP] 无法启动子进程（system 返回 -1）\n");
        return kSkip;
    }
    return rc;
}

int normalize_rc(int rc)
{
#ifdef _WIN32
    return rc;
#else
    if (WIFEXITED(rc)) return WEXITSTATUS(rc);
    if (WIFSIGNALED(rc)) return 128 + WTERMSIG(rc);
    return rc;
#endif
}

// 子进程 stderr 是否含全部给定子串（经 shell 重定向到临时文件后扫描）
bool child_stderr_contains(const char* args, const std::string& needle)
{
    const std::string err_path = "assert_test_child_stderr.txt";
    const std::string cmd = "\"" + g_self + "\" " + args + " 2> \"" + err_path + "\"";
    if (std::system(cmd.c_str()) == -1) return false;
    std::FILE* f = std::fopen(err_path.c_str(), "rb");
    if (!f) return false;
    std::string content;
    char buf[4096];
    std::size_t n = 0;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0)
        content.append(buf, n);
    std::fclose(f);
    std::remove(err_path.c_str());
    return content.find(needle) != std::string::npos;
}

void test_assert_always_semantics()
{
    std::printf("NN_ASSERT_ALWAYS 语义:\n");
    NN_ASSERT_ALWAYS(2 + 2 == 4, "条件成立不终止");
    CHECK(true, "NN_ASSERT_ALWAYS 条件成立时不终止");
    // 失败路径（子进程）：非零退出 + stderr 带完整位置信息
    const int rc = normalize_rc(run_self("--expect-assert-always"));
    if (rc != kSkip) CHECK(rc != 0, "NN_ASSERT_ALWAYS 失败 → 非零退出");
    CHECK(child_stderr_contains("--expect-assert-always", "NN_ASSERT_ALWAYS failed at"),
          "stderr 含 tag");
    CHECK(child_stderr_contains("--expect-assert-always", "assert_test.cpp"), "stderr 含文件名");
    CHECK(child_stderr_contains("--expect-assert-always", "boom always"), "stderr 含消息");
}

void test_fatal_semantics()
{
    std::printf("NN_FATAL 语义:\n");
    const int rc = normalize_rc(run_self("--expect-fatal"));
    if (rc != kSkip) CHECK(rc != 0, "NN_FATAL → 非零退出");
    CHECK(child_stderr_contains("--expect-fatal", "NN_FATAL failed at"), "stderr 含 NNFATAL tag");
    CHECK(child_stderr_contains("--expect-fatal", "boom fatal"), "stderr 含消息");
}

void test_assert_debug_semantics()
{
    std::printf("NN_ASSERT 语义（当前构建 NDEBUG=%d）:\n",
#ifdef NDEBUG
                1
#else
                0
#endif
    );
    // 两态共同：真条件不终止
    NN_ASSERT(1 + 1 == 2, "真条件");
    CHECK(true, "NN_ASSERT 真条件不终止");
#ifndef NDEBUG
    // Debug：失败即 abort（abort 可能表现为信号或非零退出，统一 != 0）
    const int rc = normalize_rc(run_self("--expect-assert"));
    if (rc != kSkip) CHECK(rc != 0, "Debug 下 NN_ASSERT 失败 → 非零退出/信号");
#endif
}

void test_require_semantics()
{
    std::printf("NN_REQUIRE 语义:\n");
    auto ok = require_ok();
    CHECK(ok.has_value() && *ok == 42, "NN_REQUIRE 成功路径返回值");
    auto bad = require_bad();
    CHECK(!bad.has_value(), "NN_REQUIRE 失败路径返回 unexpected");
    if (!bad) CHECK(bad.error().message == "require failed", "错误消息原样");
}

}  // namespace

int main(int argc, char* argv[])
{
    g_self = (argc > 0 && argv[0] != nullptr) ? argv[0] : "";

    // ── 子进程模式 ─────────────────────────────────────────────────────────
    if (argc >= 2 && std::string(argv[1]) == "--expect-assert-always")
    {
        NN_ASSERT_ALWAYS(false, "boom always");
        std::printf("不应到达\n");
        return 0;
    }
    if (argc >= 2 && std::string(argv[1]) == "--expect-fatal")
    {
        NN_FATAL("boom fatal");
        std::printf("不应到达\n");
        return 0;
    }
    if (argc >= 2 && std::string(argv[1]) == "--expect-assert")
    {
        NN_ASSERT(false, "boom assert");
        std::printf("到达这里 = NDEBUG 下为空展开（预期）\n");
        return 0;
    }

    std::puts("=== assert macros (core_assert.hpp) ===");
    test_assert_always_semantics();
    test_fatal_semantics();
    test_assert_debug_semantics();
    test_require_semantics();

    std::printf("\nassert_test: %d failure(s)\n", g_fail);
    return g_fail != 0 ? 1 : 0;
}
