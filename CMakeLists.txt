# CMake 3.30+ required for C++26 standard support (stable in 3.30+)
cmake_minimum_required(VERSION 3.30)

# ── 编译器与标准库：一律使用默认，不由本文件指定 ──────────────────────────
# 编译器：用 CMake 默认探测结果，或调用方显式 -DCMAKE_CXX_COMPILER / CXX 环境变量。
# 标准库：用编译器默认（Linux clang/gcc 默认 libstdc++）。如需 libc++，由调用方
# 通过 -DCMAKE_CXX_FLAGS / -DCMAKE_EXE_LINKER_FLAGS 统一指定（编译与链接必须一致，
# 否则链接期会出现 std 符号 ABI 不匹配）。
# 本文件不再隐式挑选编译器或 stdlib，避免与调用方/CI 的设置冲突。

# VERSION 以 git tag 为准（本仓无 CONFIGURE_FILE 消费该字段，
# 它只是元数据——发版时同步到最新 tag，实际版本以 git tag / gh release 为准）
project(neuralnet.cpp VERSION 1.14.1 LANGUAGES CXX)

# GNUInstallDirs：定义 CMAKE_INSTALL_LIBDIR/BINDIR/INCLUDEDIR（install 布局用，
# docs/internal/34 §5 与 33 §4.3 的 bin/ + lib/ + include/ 干净产物）。
include(GNUInstallDirs)

# ── IDE / clangd：导出 compile_commands.json（仅顶层工程时）───────────────
# include 路径（include/ + extern/*/include + build/generated）全部来自
# target_include_directories，clangd 没有编译数据库就解析不到
# <neuralnet.cpp/xxx.hpp>。Ninja/Makefile 生成器有效，MSVC/Xcode 忽略；
# 嵌入消费方是否导出由其顶层工程决定，这里不越权。
if(PROJECT_IS_TOP_LEVEL)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

# ══════════════════════════════════════════════════════════════════════════
#  作为库被消费（add_subdirectory / FetchContent 嵌入 3rd_party）的门控
# ══════════════════════════════════════════════════════════════════════════
# PROJECT_IS_TOP_LEVEL：本文件是顶层工程（正常开发/CI）时 ON；被消费方
# add_subdirectory 嵌入时 OFF。据此默认值翻转，让嵌入方**只拿到库目标**，
# 不被迫编译本仓 12 个 app/探针/基准（可显式 -DNN_BUILD_APPS=ON 打开）。
option(NN_BUILD_APPS "Build neuralnet.cpp application/bench executables"
       ${PROJECT_IS_TOP_LEVEL})

# ── GPU（Vulkan）开关：消费方在自己工程里一行控制 NN_HAS_VULKAN ───────────
#   AUTO（默认）= 探测到 Vulkan+glslc 就启用，否则纯 CPU（本仓历史行为）
#   ON          = 强制要求，缺 Vulkan 或 glslc 直接配置失败（fail-fast）
#   OFF         = 完全跳过：不探测 Vulkan、不生成 shader、不编 scan_exprs
# 嵌入方写法：在 add_subdirectory **之前** set(NN_ENABLE_GPU OFF)（或 -D）。
set(NN_ENABLE_GPU "AUTO" CACHE STRING "GPU support: AUTO (detect) / ON (require) / OFF (disable)")
set_property(CACHE NN_ENABLE_GPU PROPERTY STRINGS AUTO ON OFF)
if(NOT NN_ENABLE_GPU MATCHES "^(AUTO|ON|OFF)$")
    message(FATAL_ERROR "NN_ENABLE_GPU must be AUTO/ON/OFF, got '${NN_ENABLE_GPU}'")
endif()

# ── 本仓源/构建根（CACHE，全局可见）─────────────────────────────────────
# CMake 函数体在**调用方作用域**求值：消费方调用 nn_enable_gpu_fusion 时，
# CMAKE_SOURCE_DIR/CMAKE_BINARY_DIR 会解析到消费方的目录（实测报错/路径错）。
# 因此把本仓的根路径烘焙进 CACHE（INTERNAL 隐含 FORCE，每次配置刷新），
# 函数体内一律引用这两个变量，与调用方是谁无关。
set(NN_LIB_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}" CACHE INTERNAL
    "neuralnet.cpp source root (stable under add_subdirectory)")
set(NN_LIB_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}" CACHE INTERNAL
    "neuralnet.cpp binary root (stable under add_subdirectory)")

# ── C++ 模块：粒度导入路径，显式接线，不开 CMake 模块扫描 ────────────────
# 本仓自 2026-11-01 起提供粒度 C++20 模块（nn.core / nn.kvrecord /
# nn.dataset / nn.tokenizer，见下方 nn_add_module 块与 research/modules-import-nn）。
# 但**不**为此打开 CMake 默认的模块依赖扫描：C++20 起 CMake 默认给每个源文件
# 追加一次 `clang-scan-deps` 依赖扫描（build.ninja 里的 *.obj.ddi / CXX.dd /
# *.obj.modmap 三条边），实测每个 TU 约 2.1s（占单文件编译时间 ~12%），而本仓
# 的模块路径走**显式 BMI 自定义命令 + -fmodule-file 传播**（nn_add_module /
# nn_use_modules），只有模块单元与它们的消费方参与，其余 ~270 个 TU 零扫描。
# 若将来改用 CMake 原生 FILE_SET CXX_MODULES，删掉这一行即可恢复默认行为。
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# ── C++26 标准 ──────────────────────────────────────────────────────────
# CMake 4.3 尚未内置 MSVC 的 CXX26 方言标志映射（VS2026 cl 19.51 也不认
# /std:c++26，C++26 特性走 /std:c++latest），故对 MSVC 用 CMAKE_CXX_STANDARD=23
# 让方言探测通过，再显式传 /std:c++latest。Clang/GCC 走标准 CMAKE_CXX_STANDARD=26。
if(MSVC)
    set(CMAKE_CXX_STANDARD 23)
    set(CMAKE_CXX_STANDARD_REQUIRED OFF)
    set(CMAKE_CXX_EXTENSIONS OFF)
    add_compile_options(/std:c++latest)
else()
    set(CMAKE_CXX_STANDARD 26)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
endif()

# ── 默认 Release 模式，可通过 -DCMAKE_BUILD_TYPE=Debug 切换 ────────────────
# 仅顶层工程时设置：嵌入（add_subdirectory）时不得强改消费方的构建类型。
if(PROJECT_IS_TOP_LEVEL AND NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ── 公共编译标志 ─────────────────────────────────────────────────────────
# 项目规范：禁止异常、开启严格警告。按编译器选择对应标志：
# - Clang/GCC：-fno-exceptions -Wall -Wextra -Wpedantic -Werror
# - MSVC：     /W4 /permissive- /Zc:preprocessor（严格一致性）；保留 /EHsc
#              （ppltasks.h 需要），不强制 /WX（MSVC 与 clang 警告集不同，
#              代码未按 MSVC 零警告编写）
#
# /Zc:preprocessor 是**必需**项，不是加固项（2026-10-10 四编译器矩阵实测）：
#   cl.exe 默认走传统预处理器，它在把 `__VA_ARGS__` 代入**嵌套宏调用**时
#   会把整个实参表当成**一个**实参（core_errors.hpp 的 NN_DETAIL_PICK2/3 实参
#   计数分发因此全部错位）。后果对两种语义都致命：
#     · `NN_UNWRAP(cond, "ctx")` → 错选 NN_UNWRAP_1 → 展开成逗号表达式
#       `check_value((cond, "ctx"), …)` → 条件恒为字符串指针（恒真）
#       → **检查静默失效且编译通过**（实测：`NN_UNWRAP(false, "x")` 不 abort）；
#     · `NN_EXIT(cond, code, "ctx")` → 错选 NN_PICK_END → C3861 编译失败。
#   打开 /Zc:preprocessor 后两种形态均按 C++ 标准展开（实测 `NN_UNWRAP(false,…)`
#   正常 abort），本仓 500+ 处宏调用点的语义在 MSVC 下才与 clang/GCC 一致。
#   该开关同时也是**接口**选项（见下方 nn_core），否则库外消费者在 cl.exe 下
#   会拿到失效的 NN_UNWRAP。
if(MSVC)
    # ⚠ 字符集只用 **/utf-8 一种写法**（= /source-charset:utf-8 +
    #   /execution-charset:utf-8）。此前这里写的是那对分开的开关，而
    #   `neuralnet_cpp` 的 **INTERFACE** 选项又给链接它的目标追加了 /utf-8 →
    #   同一条命令行同时出现两者 ⇒ cl 直接报
    #   `D8016: "/source-charset:utf-8"和"/utf-8"命令行选项不兼容`（实测
    #   MSVC 19.51 / VS18：凡是链接本库的目标全数编译失败，MSVC 配置整体不可用）。
    #   实测（D:\probe.bat）：`/utf-8 /utf-8` 与「同一对开关重复两遍」都合法，
    #   只有"混用两种写法"才报冲突——故统一成 /utf-8，目录级与接口级重复无害。
    # /bigobj：Debug 下 text_train.cpp（header-only 全量展开）实测触发
    # C1128「节数超过对象文件格式限制」（P1-5，MSVC 19.51 / VS18，2026-10-18）；
    # Release 未触发，但该限制与优化级别相关，统一打开以避免配置相关失败。
    add_compile_options(/W4 /permissive- /Zc:preprocessor /utf-8 /bigobj)
    # /STACK：MSVC 默认 1 MB 栈。Debug 下多个模型/序列化测试实测
    # STATUS_STACK_OVERFLOW (0xC00000FD)（P1-5，2026-10-18：t1_58 / bitnet_struct /
    # model_roles / batchnorm / model_spec_validation / text_train_modules /
    # train_state）；Linux 默认 8 MB。统一抬到 8 MB，与 Linux 口径对齐。
    add_link_options(/STACK:8388608)
else()
    add_compile_options(-fno-exceptions -Wall -Wextra -Wpedantic -Werror)
    # GCC 15.2 在 -O3 + PCH 下对 std::optional / std::unique_ptr 的**内联析构**
    # 报 -Wmaybe-uninitialized 假阳性（诊断里的内存位置全是 <unknown>；实测
    # src/layer_lifetime_test.cpp 的 `unique_ptr<nn::Model>` 析构路径，成员已由
    # make_unique/默认初始化，无真实未初始化读取）。Clang 不产生该告警，因此
    # 只对 GNU 关闭此子告警；-Wuninitialized 保持开启。
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        add_compile_options(-Wno-maybe-uninitialized)
    endif()
endif()

# ── LTO（链接期优化，2026-11：Clang thin LTO）──────────────────────────
# 用 Clang thin LTO（并行分片 + 惰性合并，性能优于 GCC fat LTO）+ LLD 链接器。
# 仅 Clang（非 MSVC 分支）生效；GCC 构建不自动开（避免 GCC LTO 与 -Werror
# 的组合噪音）。NN_ENABLE_LTO=OFF 可关闭（A/B 或排障）。需 clang + lld。
option(NN_ENABLE_LTO "Clang thin LTO（需 LLD）" ON)
if(NN_ENABLE_LTO AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND NOT MSVC)
    add_compile_options(-flto=thin)
    add_link_options(-flto=thin -fuse-ld=lld)
    message(STATUS "LTO: Clang thin LTO + LLD 已启用（NN_ENABLE_LTO=OFF 可关闭）")
endif()

# Windows 上的 clang++（GNU-like 驱动）不是 CMake 的 MSVC 分支，但链接仍走
# link.exe，默认栈同样是 1 MB。实测（102 / clang 23.1 + MSVC STL / Release）
# t1_58_test、bitnet_struct_test、model_roles_test 在 ctest 中 SegFault；
# 与 cl.exe 的 0xC00000FD 同源。用 -Wl, 把 /STACK 透传给 link.exe。
if(WIN32 AND CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT MSVC)
    add_link_options(-Wl,/STACK:8388608)
endif()

# ── 编译器特有选项 ───────────────────────────────────────────────────────
# -fexperimental-library 仅 Clang 识别；GCC 不识别会产生警告
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    add_compile_options(-fexperimental-library)
    # f16 矩阵乘法（MatrixT<F16>）无原生 x86 SIMD → 优化器无法向量化是预期行为
    # （§7.2 兼容路径：f32 参考计算 + f16 舍入），抑制 -Wpass-failed 避免 -Werror 下编译失败
    add_compile_options(-Wno-pass-failed)
endif()
# C++ 标准库由编译器默认决定（见文件开头的说明），此处不再显式指定 -stdlib。
#
# ── 编译器缓存（可选，强烈建议装了就用）──────────────────────────────────
# 自动探测 ccache / sccache（Windows 上 ccache 常缺席、sccache 更常见），也可用
# `-DNN_COMPILER_LAUNCHER=<path>` 显式指定（显式指定为空串 = 禁用）。
# 对本 header-only 工程收益很大：改动只影响单个 TU 时，未变 TU 直接命中缓存。
# ⚠ 调用方若已给 CMake 原生的 `-DCMAKE_CXX_COMPILER_LAUNCHER=...`（CI 就是这么
#   配的），这里**不再重复包一层** RULE_LAUNCH_COMPILE——两套机制叠加会变成
#   `ccache ccache clang++`。
# ⚠ RULE_LAUNCH_* 是 **GLOBAL 属性**，会作用于整个构建树——嵌入（add_subdirectory）
#   消费方的构建时**自动探测一律不生效**（避免污染消费方）；消费方想要就自己设
#   CMAKE_CXX_COMPILER_LAUNCHER，或显式传 -DNN_COMPILER_LAUNCHER=<path>（显式 =
#   调用方主动要求，允许生效于整个构建树）。
set(NN_COMPILER_LAUNCHER "" CACHE STRING
    "Compiler launcher prefix (e.g. ccache / sccache); empty = auto-detect (top-level only)")
set(NN_COMPILER_LAUNCHER_EXPLICIT "OFF" CACHE INTERNAL "NN_COMPILER_LAUNCHER passed by caller")
if(NN_COMPILER_LAUNCHER)
    set(NN_COMPILER_LAUNCHER_EXPLICIT "ON")
endif()
if(NOT NN_COMPILER_LAUNCHER AND NOT CMAKE_CXX_COMPILER_LAUNCHER AND PROJECT_IS_TOP_LEVEL)
    find_program(NN_CCACHE_PROGRAM ccache)
    find_program(NN_SCCACHE_PROGRAM sccache)
    if(NN_CCACHE_PROGRAM)
        set(NN_COMPILER_LAUNCHER "${NN_CCACHE_PROGRAM}")
    elseif(NN_SCCACHE_PROGRAM)
        set(NN_COMPILER_LAUNCHER "${NN_SCCACHE_PROGRAM}")
    endif()
endif()
if(NN_COMPILER_LAUNCHER AND (NN_COMPILER_LAUNCHER_EXPLICIT OR PROJECT_IS_TOP_LEVEL))
    # ── ccache × PCH：把 sloppiness 钉进编译命令 ──────────────────────────
    # ccache 手册《Precompiled headers》要求 Clang 侧**三项同时满足**才能缓存
    # "用了 PCH 的 TU"：① sloppiness 含 `pch_defines,time_macros`；② 以
    # `-include-pch` 使用 PCH（CMake 就是这么传的）；③ 编译带
    # `-fno-pch-timestamp`（见文件末尾 PCH 落地块）。缺 ①/③ 时 ccache 对**每个**
    # 用 PCH 的 TU 都记 "Could not use precompiled header" → 全部不可缓存，
    # 而本仓 42+ 个目标的编译**全部**走 PCH，等于缓存形同虚设。
    # 这里用 `cmake -E env` 把 sloppiness 钉在命令前缀里，**不依赖用户全局
    # ccache 配置**（避免"换个机器/换个人就悄悄失效，还看不出慢在哪"）。
    # 两项 sloppiness 对本仓安全：全仓无 `__DATE__` / `__TIME__` / `__TIMESTAMP__`
    # （`pch_defines` 只在"用 PCH + 源内 #define 影响 PCH 内容"的写法下才有风险，
    # 而 PCH 是强制前置包含、源内 #define 影响不到它）。
    if(NN_COMPILER_LAUNCHER MATCHES "ccache")
        # ⚠ 必须写成**单个**字符串：写成两个相邻的带引号参数会变成 CMake 列表
        #   （分号连接）→ 命令行里被 `sh` 当成第二个命令，实测报
        #   `-E env ;CCACHE_SLOPPINESS=...`。
        set(NN_COMPILER_LAUNCHER "${CMAKE_COMMAND} -E env CCACHE_SLOPPINESS=pch_defines,time_macros ${NN_COMPILER_LAUNCHER}")
    endif()
    message(STATUS "Compiler launcher: ${NN_COMPILER_LAUNCHER}")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${NN_COMPILER_LAUNCHER}")
    # 不挂 RULE_LAUNCH_LINK：本项目不做 LTO，ccache 对普通链接只有进程开销与
    # "Called for linking" 统计噪声，没有收益。
elseif(CMAKE_CXX_COMPILER_LAUNCHER)
    message(STATUS "Compiler launcher: ${CMAKE_CXX_COMPILER_LAUNCHER} "
                   "(来自 CMAKE_CXX_COMPILER_LAUNCHER，用 CMake 原生机制)")
else()
    message(STATUS "Compiler launcher: (none found; install ccache or sccache for faster rebuilds)")
endif()

# ── 按构建类型设置优化标志 ───────────────────────────────────────────────
# Release: -O3 -funroll-loops -march=native（最大性能）
# Debug:   -O0 -g（调试优先）
# 注意：不使用 -ffast-math，因为它会禁用 NaN/Infinity 支持，影响训练稳定性
# MSVC 对应 /fp:fast 同样禁用 NaN 语义（实测：base 子仓 precision_type_test 的
# "NaN != NaN" / "NaN 比较均为 false" 断言在 /fp:fast 下被编译器折叠为
# false → 假失败；/fp:precise 下全绿）→ 按同一铁律用 /fp:precise。
# Clang/GCC 用 -O3 + 向量化（自带 -fno-math-errno/-fno-trapping-math）。
#
# -march=native 会让二进制只能在编译机同代 CPU 上运行，分发/换机可能 SIGILL。
# 默认开启（本地追求最大性能）；发布/CI 构建请用 -DNN_ENABLE_NATIVE=OFF 生成
# 通用可移植基线（见 .github/workflows 的 Configure 步骤）。
option(NN_ENABLE_NATIVE "Enable -march=native (max perf, non-portable binary)" ON)
if(MSVC)
    set(RELEASE_FLAGS /O2 /fp:precise)
    set(DEBUG_FLAGS /Od /Zi)
else()
    if(NN_ENABLE_NATIVE)
        set(RELEASE_FLAGS -O3 -fno-math-errno -fno-trapping-math -funroll-loops -march=native)
    else()
        set(RELEASE_FLAGS -O3 -fno-math-errno -fno-trapping-math -funroll-loops)
    endif()
    set(DEBUG_FLAGS -O0 -g)
endif()

add_compile_options(
    "$<$<CONFIG:Release>:${RELEASE_FLAGS}>"
    "$<$<CONFIG:Debug>:${DEBUG_FLAGS}>"
)

# ── 线程库（Linux 需要 pthread） ─────────────────────────────────────────
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# ── std::stacktrace 运行时库探测 (GCC & Clang) ──────────────────────────
# GCC 14+ 和 Clang (使用 libstdc++) 都需要链接实验性库才能使用 std::stacktrace。
# GCC 使用 stdc++exp，Clang 可能需要 stdc++exp 或 stdc++_libbacktrace。
# 探测一次，成功后通过 nn_core INTERFACE 传播给所有目标。
# ⚠ **只有 `tools/scan_exprs.cpp` 用 std::stacktrace**（库头文件零使用，已 grep
#   确认）；嵌入（add_subdirectory）消费方编译头文件根本不需要它。因此 GNU 下
#   探测失败时：顶层工程仍 FATAL_ERROR（保住既有行为），嵌入方降级为警告 +
#   留空——否则消费方会为了一个用不到的库在配置期被卡死。

set(NN_STACKTRACE_FATAL ${PROJECT_IS_TOP_LEVEL})
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # ── GCC 探测逻辑 ──────────────────────────────────────────────
    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_LIBRARIES stdc++exp)
    check_cxx_source_compiles("
        #include <stacktrace>
        int main() { return std::stacktrace::current().empty() ? 1 : 0; }
    " NN_HAS_STDCXXEXP)
    unset(CMAKE_REQUIRED_LIBRARIES)

    if(NN_HAS_STDCXXEXP)
        set(NN_STACKTRACE_LIB stdc++exp)
    else()
        set(CMAKE_REQUIRED_LIBRARIES stdc++_libbacktrace)
        check_cxx_source_compiles("
            #include <stacktrace>
            int main() { return std::stacktrace::current().empty() ? 1 : 0; }
        " NN_HAS_STDCXX_LIBBACKTRACE)
        unset(CMAKE_REQUIRED_LIBRARIES)
        if(NN_HAS_STDCXX_LIBBACKTRACE)
            set(NN_STACKTRACE_LIB stdc++_libbacktrace)
        elseif(NN_STACKTRACE_FATAL)
            message(FATAL_ERROR "std::stacktrace 需要 stdc++exp 或 stdc++_libbacktrace")
        else()
            message(WARNING "std::stacktrace 库未找到（嵌入模式降级：仅 tools/scan_exprs 受影响）")
            set(NN_STACKTRACE_LIB "")
        endif()
    endif()

elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # ── Clang 探测逻辑 ─────────────────────────────────────────────
    # Clang 可能链接 libstdc++ 或 libc++。我们通过尝试链接来探测。
    # 如果 Clang 使用 libc++ 且其版本尚不支持 stacktrace，则此处会失败。
    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_LIBRARIES stdc++exp)
    check_cxx_source_compiles("
        #include <stacktrace>
        int main() { return std::stacktrace::current().empty() ? 1 : 0; }
    " NN_CLANG_HAS_STDCXXEXP)

    if(NN_CLANG_HAS_STDCXXEXP)
        set(NN_STACKTRACE_LIB stdc++exp)
    else()
        # 回退尝试旧版 libbacktrace 库
        set(CMAKE_REQUIRED_LIBRARIES stdc++_libbacktrace)
        check_cxx_source_compiles("
            #include <stacktrace>
            int main() { return std::stacktrace::current().empty() ? 1 : 0; }
        " NN_CLANG_HAS_STDCXX_LIBBACKTRACE)
        unset(CMAKE_REQUIRED_LIBRARIES)
        if(NN_CLANG_HAS_STDCXX_LIBBACKTRACE)
            set(NN_STACKTRACE_LIB stdc++_libbacktrace)
        else()
            # Clang 可能在使用 libc++，其 stacktrace 支持可能不完整。
            message(WARNING "Clang 无法链接 std::stacktrace。如果使用 libc++，请确保版本 >= 18。CI 构建可能会失败。")
            set(NN_STACKTRACE_LIB "")
        endif()
    endif()
endif()

if(NN_STACKTRACE_LIB)
    message(STATUS "std::stacktrace link library: ${NN_STACKTRACE_LIB}")
else()
    message(STATUS "std::stacktrace link library: (none found,可能链接失败)")
endif()

# ── 公共 INTERFACE 库：所有目标共享 include 目录和链接库 ──────────────
# 通过 INTERFACE 库消除 7 个目标的重复设置
add_library(nn_core INTERFACE)
# 引擎抽象层（submodule engine/abstraction，MIT）排在**主仓库 include 之前**：
# 主仓库已删除与抽象层重复的接口/值类型头（algebra_* / compute_engine /
# compute_tensor / core_* / expr_spec 等），同名头一律解析到抽象层 —— 与
# libnn-cpu.so / libnn-gpu.so（LGPL）按同一份头编译，跨界 ABI 一致。
target_include_directories(nn_core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/engine/abstraction/include/neuralnet.cpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/engine/abstraction/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(nn_core INTERFACE Threads::Threads)
if(NN_STACKTRACE_LIB)
    target_link_libraries(nn_core INTERFACE ${NN_STACKTRACE_LIB})
endif()

# ── 诊断探针编译门（NN_ENABLE_PROBES，见下方 option；默认关）─────────────
# 定义放在 nn_core 的 **INTERFACE**：库头里的探针门（`nn_dbg_scan`、
# `NN_PREC_TRACE`、`env_flag` 等，core_config.hpp 的 NN_PROBES_ENABLED）在
# **所有 TU 上必须取值一致**，否则同一头文件两种语义（ODR 隐患）；随 nn_core
# 传播可同时覆盖库内目标、PCH 目标、收集器与嵌入消费方。
if(NN_ENABLE_PROBES)
    target_compile_definitions(nn_core INTERFACE NN_ENABLE_PROBES=1)
endif()

# ── C++ 标准（消费方使用需求，实测于 CMake 4.3）─────────────────────────────
# Clang/GNU（GNU 前端）认识 cxx_std_26 → 传 -std=c++26；
# **MSVC cl 不认识 cxx_std_26**（实测报 "not known to CXX compiler"，其
# CMAKE_CXX23_STANDARD_COMPILE_OPTION 就映射为 /std:c++latest = C++26 预览
# 全开，与本文件顶部 MSVC 分支同策略）→ 降级请求 cxx_std_23；
# clang-cl 认识 cxx_std_26（映射 /std:c++latest）→ 走 26。
# 判据用「特征列表实测」而非 if(MSVC)，因为 MSVC 变量对 clang-cl 也为真。
if("cxx_std_26" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    target_compile_features(nn_core INTERFACE cxx_std_26)
elseif("cxx_std_23" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    target_compile_features(nn_core INTERFACE cxx_std_23)
endif()

# ── 消费门面目标：下游只写 target_link_libraries(app PRIVATE neuralnet::nn) ──
# 嵌入（add_subdirectory / FetchContent）用法：
#     add_subdirectory(3rd_party/neuralnet.cpp)
#     target_link_libraries(app PRIVATE neuralnet::nn)
#     #include <neuralnet.cpp/nn.hpp>
# 门面 = nn_core（include/标准/线程）+ **编译本库头文件所需**的选项（见下）
#      + **GPU 开启时**的四项附加（在下方 `if(Vulkan_FOUND)` 块内补：
#      NN_HAS_VULKAN 宏 + generated/ 嵌入头目录 + Vulkan::Vulkan + shader
#      生成依赖）。
# ⚠ 刻意与 nn_core 分离的原因：
#   ① `scan_exprs` / 收集器必须保持**无 NN_HAS_VULKAN**（AOT 闭合世界，
#      AGENTS §7 铁律），它们只链 nn_core；
#   ② 下方这些「编译选项」只该给**真正的库使用者**（库内目标已从本文件的
#      目录级 add_compile_options 拿到，重复传递会让命令行出现两份相同 flag）。
add_library(neuralnet_cpp INTERFACE)
target_link_libraries(neuralnet_cpp INTERFACE nn_core)

# ── 编译本库头文件所需的消费方选项（目录级 add_compile_options 传不过去）──
# 实测缺口：下游 Release 构建库头文件时出现 6 个 `-Wpass-failed` 警告
# （`#pragma clang loop` 向量化未达成），库内靠 -Wno-pass-failed 压住；
# 消费方一旦开 -Werror 就直接编译失败——必须随目标传播。
target_compile_options(neuralnet_cpp INTERFACE
    # Clang：库内同款（-fexperimental-library 与 -Wno-pass-failed 都在本文件
    # 顶部的 Clang 分支加给库内目标；作用域限定 CXX 避免影响消费方 C 代码）
    $<$<COMPILE_LANG_AND_ID:CXX,Clang,AppleClang>:-fexperimental-library>
    $<$<COMPILE_LANG_AND_ID:CXX,Clang,AppleClang>:-Wno-pass-failed>
    # MSVC：库头文件是 UTF-8 + 大量中文注释/字符串，不指定字符集时 cl 按
    # 系统 codepage 读 → C4819/C4828 甚至误解析；/utf-8 = source+execution
    # 都按 UTF-8（库内目录级选项**同款写法**，见上方公共标志处的 D8016 实测：
    # 目录级与接口级必须同用 /utf-8，混用分开的两个开关会让 cl 直接拒绝命令行）。
    # ⚠ 这同时要求消费方源码也是 UTF-8（现代工程默认如此）。
    # /Zc:preprocessor 必须随接口传出去（见上方公共标志处的实测说明：
    # 传统预处理器会让 NN_UNWRAP/NN_EXIT 的变参分发错位）。
    $<$<CXX_COMPILER_ID:MSVC>:/utf-8>
    $<$<CXX_COMPILER_ID:MSVC>:/Zc:preprocessor>
    # /bigobj 也随接口传：消费方在大 TU 里展开 nn.hpp 同样可能触发 C1128
    # （P1-5 实测 text_train.cpp；对 ABI 无影响）。
    $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)

add_library(neuralnet::nn ALIAS neuralnet_cpp)

# ── 预编译头（PCH）：本项目编译时间的最大头 ──────────────────────────────
# 库是 header-only，每个 TU 都要把 `nn.hpp`（→ 全部 Layer/Loss/Optimizer/
# Engine + 构建期嵌入的 36 份原语 SPIR-V + AOT 融合注册表，合计 ~7MB 头文本）
# 从头 parse 一遍。实测（Debug / clang / 本机）单 TU：
#   无 PCH 17.5s → 有 PCH 9.8s（-44%）。
# 所有候选目标的编译条件完全相同 → 只编**一份** PCH 供它们复用（REUSE_FROM）：
# 31 份逐目标 PCH 合计约 780s CPU，共享后只剩 1 份。见文件末尾的落地块。
# 关掉：-DNN_ENABLE_PCH=OFF。
# ⚠ **收集器目标不套 PCH**：`scan_exprs.cpp` 是在**源文件内**（include 之前）
#   自行 `#define NN_EXPR_SCAN` 的——PCH 会在该宏生效前把 nn.hpp 预编译进 AST，
#   导致 DSL 收集钩子整体失效（注册表空 → 构建硬失败）。故只有经
#   `nn_add_executable` 建的应用/测试目标进入候选。
option(NN_ENABLE_PCH "Precompile nn.hpp once and reuse it across targets" ON)

# ── 便捷函数：统一添加可执行目标 ────────────────────────────────────────
# PCH **不在这里挂**：目标编译条件（NN_HAS_VULKAN / include 目录）要到文件
# 末尾的 GPU 循环才定型，提前挂会与共享 PCH 目标的属性对不上（REUSE_FROM
# 要求两边完全一致）。这里只登记候选，末尾统一落地。
function(nn_add_executable name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE nn_core)
    # GPU 块按这份名单统一挂 NN_HAS_VULKAN / Vulkan::Vulkan。名单与创建点
    # 同源，避免"测试目标只在 NN_ENABLE_TESTS=ON 时才进 GPU 名单"这类漏配：
    # 编译门是 NN_BUILD_APPS OR NN_ENABLE_TESTS（默认 Release 也会编出全部
    # 测试程序），漏挂时共享 PCH 里的 NN_HAS_VULKAN 会泄漏进 TU，表现为
    # 链接期整屏 `undefined reference to vk*`。
    list(APPEND NN_EXECUTABLE_TARGETS ${name})
    set(NN_EXECUTABLE_TARGETS ${NN_EXECUTABLE_TARGETS} PARENT_SCOPE)
    if(NN_ENABLE_PCH)
        set(NN_PCH_CANDIDATES ${NN_PCH_CANDIDATES} ${name} PARENT_SCOPE)
    endif()
endfunction()

# ══════════════════════════════════════════════════════════════════════════
#  C++ 模块：粒度导入路径（nn.core / nn.kvrecord / nn.dataset / nn.tokenizer）
# ══════════════════════════════════════════════════════════════════════════
# 形态 = 「薄接口」模块（research/modules-import-nn 实证，2026-11-01）：
#   全局模块片段文本包含域头文件（STL 与库声明留在全局模块，实体仍挂全局
#   模块 → 与 `#include` 消费方是**同一实体**，import 与 include 可混合链接，
#   无双轨 ODR 问题），`export module nn.<域>` 后经
#   `export namespace nn { using nn::X; ... }` 逐符号 re-export；
#   `export import nn.<依赖域>` 让消费方只 import 最顶域即可连带拿到依赖域
#   名字（C++ 模块的 import **不可传递**，必须显式 re-export）。
# 约束（实证，README §4）：
#   · 宏（NN_TRY / NN_UNWRAP / NN_ASSERT_* 等）**不可导出** → 消费者双轨：
#     `import nn.<域>` + 按需 `#include <neuralnet.cpp/core_errors.hpp>` 宏头；
#   · STL 留在全局片段 → 消费者自带用到的 STL 头；
#   · 仅 GNU 驱动 Clang（GCC / cl.exe / clang-cl 无等效显式 -fmodule-file
#     姿势，NN_MODULES_ENABLED 自动为 OFF，消费方回落 include 分支）。
# 构建 = 显式 BMI 自定义命令（clang --precompile）+ -fmodule-file 经
#   INTERFACE 目标传播，不开 CMake 模块扫描（见文件头注释的成本说明）。
# 嵌入消费方默认 OFF（PROJECT_IS_TOP_LEVEL = ON 才默认开）。
option(NN_ENABLE_MODULES
    "Build granular C++20 modules (nn.core/nn.kvrecord/nn.dataset/nn.tokenizer) \
     and switch leaf consumers to import (clang only; embedders default OFF)"
    ${PROJECT_IS_TOP_LEVEL})
if(NN_ENABLE_MODULES AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND NOT MSVC)
    set(NN_MODULES_ENABLED ON)
else()
    set(NN_MODULES_ENABLED OFF)
endif()
message(STATUS "C++ modules (import nn.<domain>): "
               "${NN_MODULES_ENABLED} (NN_ENABLE_MODULES=${NN_ENABLE_MODULES}, "
               "compiler=${CMAKE_CXX_COMPILER_ID})")

# ── 各域头文件闭包（nn_add_module 的 DEPENDS；跨域头改动触发对应 BMI 重编）──
# 闭包 = 模块单元文本包含的全部 nn 头（含传递）。宁多勿少：多列只会偶发
# 重编，少列会在头改动后**静默用旧 BMI**（比编译慢更糟）。
# ⚠ 引擎 .so 化后 core_*（除 core_file/core_sha256）与 precision 位于
#   submodule 抽象层（engine/abstraction/include）——闭包路径须指向实际位置。
#   nn_add_module 会把 HEADERS 拼到 ${NN_LIB_SOURCE_DIR}/ 前 → 这里用相对路径。
set(NN_ABST_INC engine/abstraction/include/neuralnet.cpp)
set(NN_MOD_CORE_HEADERS
    ${NN_ABST_INC}/core_assert.hpp
    ${NN_ABST_INC}/core_config.hpp
    ${NN_ABST_INC}/core_errors.hpp
    include/neuralnet.cpp/core_file.hpp
    ${NN_ABST_INC}/core_parallel.hpp
    include/neuralnet.cpp/core_sha256.hpp
    ${NN_ABST_INC}/core_threadpool.hpp
    ${NN_ABST_INC}/precision.hpp)
# kvrecord 闭包 = 自身 + core（model_keyvalue_record → core_errors）
set(NN_MOD_KVRECORD_HEADERS
    include/neuralnet.cpp/model_keyvalue_record.hpp
    ${NN_MOD_CORE_HEADERS})
# dataset 闭包 = dataset/dataset_format + kvrecord + core_file/core_sha256/core_errors
#   （dataset 不依赖 threadpool/parallel/config/precision/assert，无需全 core）
set(NN_MOD_DATASET_HEADERS
    include/neuralnet.cpp/dataset.hpp
    include/neuralnet.cpp/dataset_format.hpp
    include/neuralnet.cpp/model_keyvalue_record.hpp
    ${NN_ABST_INC}/core_errors.hpp
    include/neuralnet.cpp/core_file.hpp
    include/neuralnet.cpp/core_sha256.hpp)
# tokenizer 闭包 = 4 分词器头 + base 的传递（core_errors/core_parallel/
#   core_threadpool/dataset_format → core_config/precision/core_assert/kvrecord）
#   —— 恰好与 spike 实测的「12 头闭包」一致
set(NN_MOD_TOKENIZER_HEADERS
    include/neuralnet.cpp/domain_tokenizer.hpp
    include/neuralnet.cpp/domain_tokenizer_base.hpp
    include/neuralnet.cpp/domain_tokenizer_bpe.hpp
    include/neuralnet.cpp/domain_tokenizer_charbpe.hpp
    ${NN_ABST_INC}/core_errors.hpp
    ${NN_ABST_INC}/core_parallel.hpp
    ${NN_ABST_INC}/core_threadpool.hpp
    ${NN_ABST_INC}/core_config.hpp
    ${NN_ABST_INC}/core_assert.hpp
    ${NN_ABST_INC}/precision.hpp
    include/neuralnet.cpp/dataset_format.hpp
    include/neuralnet.cpp/model_keyvalue_record.hpp)

# ── 模块注册：nn_add_module(<短名> <模块源> DEPS <依赖短名...> HEADERS <闭包头...>) ──
# 短名 = "core" / "kvrecord" / "dataset" / "tokenizer"；C++ 模块全名 = nn.<短名>。
# 产出：
#   · <build>/modules/nn.<短名>.pcm —— BMI（自定义命令，DEPENDS 头闭包 + 依赖 BMI）
#   · nn_mod_<短名>（INTERFACE）—— 消费方 target_link_libraries 即得
#     -fmodule-file=nn.<短名>=<bmi>（依赖域的标志经 INTERFACE 传递）+ NN_USE_MODULES
#   · nn_<短名>_bmi（custom target）—— BMI 产物聚合，nn_use_modules 挂到消费方
function(nn_add_module modname source)
    cmake_parse_arguments(NNM "" "" "DEPS;HEADERS" ${ARGN})
    if(NOT NN_MODULES_ENABLED)
        return()
    endif()
    set(_cpp "nn.${modname}")
    set(_bmi "${NN_LIB_BINARY_DIR}/modules/${_cpp}.pcm")
    # 编译标志 = 库内目标同款（目录级 add_compile_options 不作用于自定义命令，
    # 必须在此显式列出；优化标志与消费方同档，避免"同实体两种 -O"的度量失真；
    # 探针门保持与 nn_core INTERFACE 同值，见该变量注释）。
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(_opt ${DEBUG_FLAGS})
    else()
        set(_opt ${RELEASE_FLAGS})
    endif()
    set(_flags
        "-std=c++${CMAKE_CXX_STANDARD}"
        -fno-exceptions -Wall -Wextra -Wpedantic -Werror
        -fexperimental-library -Wno-pass-failed
        ${_opt}
        -I "${NN_LIB_SOURCE_DIR}/include"
        -I "${NN_LIB_SOURCE_DIR}/engine/abstraction/include"
        -I "${NN_LIB_SOURCE_DIR}/engine/abstraction/include/neuralnet.cpp")
    if(NN_ENABLE_PROBES)
        list(APPEND _flags -DNN_ENABLE_PROBES=1)
    endif()
    set(_file_deps "")
    foreach(d IN LISTS NNM_DEPS)
        list(APPEND _flags "-fmodule-file=nn.${d}=${NN_LIB_BINARY_DIR}/modules/nn.${d}.pcm")
        list(APPEND _file_deps "${NN_LIB_BINARY_DIR}/modules/nn.${d}.pcm")
    endforeach()
    foreach(h IN LISTS NNM_HEADERS)
        list(APPEND _file_deps "${NN_LIB_SOURCE_DIR}/${h}")
    endforeach()
    add_custom_command(
        OUTPUT ${_bmi}
        COMMAND ${CMAKE_CXX_COMPILER} ${_flags} --precompile
                -o ${_bmi} -x c++-module ${NN_LIB_SOURCE_DIR}/${source}
        DEPENDS ${NN_LIB_SOURCE_DIR}/${source} ${_file_deps}
        COMMENT "[nn] module BMI: ${_cpp}"
        VERBATIM
    )
    add_library(nn_mod_${modname} INTERFACE)
    target_compile_options(nn_mod_${modname} INTERFACE "-fmodule-file=${_cpp}=${_bmi}")
    target_compile_definitions(nn_mod_${modname} INTERFACE NN_USE_MODULES)
    foreach(d IN LISTS NNM_DEPS)
        target_link_libraries(nn_mod_${modname} INTERFACE nn_mod_${d})
    endforeach()
    target_link_libraries(nn_mod_${modname} INTERFACE nn_core)
    # BMI 产物挂到消费方的编译边：INTERFACE 目标的依赖会传递到消费方
    #（与 compile_gpu_shaders 同机制，见 GPU 块注释），nn_use_modules 另加
    # 一层显式 add_dependencies 兜底。
    add_custom_target(nn_${modname}_bmi DEPENDS ${_bmi})
    add_dependencies(nn_mod_${modname} nn_${modname}_bmi)
endfunction()

# ── 消费方接入：nn_use_modules(<目标> <模块...>) ──────────────────────────
# 链接模块 INTERFACE（-fmodule-file 链 + NN_USE_MODULES 宏）并把目标从共享
# PCH 候选剔除——模块消费方的 TU 同时带 nn.hpp 的 PCH 与模块 BMI 时，同一
# 实体两处声明（PCH 按 NN_HAS_VULKAN 编、BMI 不带）存在分歧风险，叶子目标
# 编译量小，直接各自解析更稳。
function(nn_use_modules target)
    if(NOT NN_MODULES_ENABLED)
        return()
    endif()
    foreach(m IN LISTS ARGN)
        if(TARGET nn_mod_${m})
            target_link_libraries(${target} PRIVATE nn_mod_${m})
            add_dependencies(${target} nn_${m}_bmi)
        endif()
    endforeach()
    list(REMOVE_ITEM NN_PCH_CANDIDATES ${target})
    set(NN_PCH_CANDIDATES ${NN_PCH_CANDIDATES} PARENT_SCOPE)
endfunction()

# ── 四个叶子域模块（DAG：core → kvrecord → dataset → tokenizer）──────────
# 域边界 = 合并回主仓的四个子仓域；全部不依赖 GPU/AOT 扫描（NN_EXPR_SCAN 的
# 宏收集无法进模块，engine/layers/model/train 域本轮不模块化，见研究 README）。
nn_add_module(core        modules/nn.core.cpp
    HEADERS ${NN_MOD_CORE_HEADERS})
nn_add_module(kvrecord    modules/nn.kvrecord.cpp
    DEPS core
    HEADERS ${NN_MOD_KVRECORD_HEADERS})
nn_add_module(dataset     modules/nn.dataset.cpp
    DEPS core kvrecord
    HEADERS ${NN_MOD_DATASET_HEADERS})
nn_add_module(tokenizer   modules/nn.tokenizer.cpp
    DEPS core kvrecord dataset
    HEADERS ${NN_MOD_TOKENIZER_HEADERS})

# ── 测试/验证目标开关（默认关闭） ──────────────────────────────────────
# 默认只构建应用与基准工具目标，避免生成大量验证程序；
# 需要时用 -DNN_ENABLE_TESTS=ON 开启全部 gradcheck / *_test / *_probe / *_debug。
option(NN_ENABLE_TESTS "Build test/verification executables" OFF)

# ── 诊断探针开关（默认关闭）─────────────────────────────────────────────
# "探针" = 会在**热路径**采样/打印/下载/扫描的诊断代码 + 只服务于排障的可执行
# 目标。打开后：① 库头里的探针门打开（`nn_dbg_scan`/`NN_F16_DEBUG` 中间量扫描、
# `NN_PREC_TRACE` 变体与边界 cast 归因、`text_train` 的 `NN_NAN_TRACE` /
# `NN_MEM_STATS`）；② 创建探针可执行目标（mem_probe / f16_cpu_probe /
# gpu_stability_probe）。
# 关闭（默认）时这些代码**不参与编译**：没有额外分支、没有字符串、没有 host
# 下载路径，相关环境变量一律无效——探针只在排障时按需打开，不允许"设了环境
# 变量就悄悄拖慢正常训练/推理"。运行时**行为开关**（NN_VULKAN_DEVICE 设备选择、
# NN_VULKAN_* 设备/特性逃生阀、NN_POOL_* 池 A/B）不在本门内：它们读取一次并
# 缓存，不产生热路径开销（新增同类开关必须遵守"读一次 + 缓存"）。
# ⚠ 打开会改变库头语义 → 探针构建的二进制/缓存与正常构建不可混用；
#   覆盖率/归因类验收（如 docs 里 `NN_PREC_TRACE` 的 V1 覆盖实测）需用本开关配置。
option(NN_ENABLE_PROBES "Compile diagnostic probes (hot-path trace hooks + probe executables)" OFF)

# 收集实际构建的目标，供 GPU 统一启用（避免引用未定义目标）
set(NN_APP_TARGETS "")
set(NN_TEST_TARGETS "")
set(NN_EXECUTABLE_TARGETS "")   # 所有经 nn_add_executable 创建的可执行目标
set(NN_BENCH_TARGETS "")   # 性能基准已移除（bench 工具不注册 ctest 且无法在 -Werror 下编译）

# ── libnn.so 外部消费者验证（旧 C ABI 门面）已随引擎 .so 化移除 ─────────
# 新架构（docs/internal/34 §10）：双引擎 .so（libnn-cpu.so / libnn-gpu.so）
# C++ 直出（nn::create_cpu_engine / create_gpu_engine），旧 nn_api C ABI 门面
# 与其验证目标 nn_consumer_test 一并删除（src/nn_api.cpp / src/nn_consumer_test.cpp）。

# ── 应用 / 探针 / 基准目标（NN_BUILD_APPS，默认 = 是否顶层工程）───────────
# 嵌入消费方（add_subdirectory）默认 OFF：只拿库目标（neuralnet::nn），
# 不被迫编译本仓 12 个可执行文件。需要复现应用时显式 -DNN_BUILD_APPS=ON。
# ⚠ f16_writeback_probe 同时是 ctest 用例（见 NN_ENABLE_TESTS 块），故它
#   单独按「任一开关」创建；其余 app 只看 NN_BUILD_APPS。
if(NN_BUILD_APPS)
    # ── MNIST 目标（标准构建核心应用） ─────────────────────────────────────
    nn_add_executable(mnist_train src/mnist_train.cpp)
    nn_add_executable(mnist_infer src/mnist_infer.cpp)
    list(APPEND NN_APP_TARGETS mnist_train mnist_infer)

    # ── 文本目标（GPT 训练/推理 + BPE 分词器训练/推理，标准构建核心应用） ──
    nn_add_executable(text_train src/text_train.cpp)
    nn_add_executable(text_infer src/text_infer.cpp)
    nn_add_executable(tokenizer_train src/tokenizer_train.cpp)
    nn_add_executable(tokenizer_infer src/tokenizer_infer.cpp)
    list(APPEND NN_APP_TARGETS text_train text_infer tokenizer_train tokenizer_infer)
    nn_use_modules(tokenizer_train tokenizer)
    nn_use_modules(tokenizer_infer tokenizer)

    # ── 数据集工具（生成器 + 旧格式转换器）────────────────────────────────
    nn_add_executable(dataset_gen src/dataset_gen.cpp)
    nn_add_executable(dataset_convert src/dataset_convert.cpp)
    list(APPEND NN_APP_TARGETS dataset_gen dataset_convert)
    nn_use_modules(dataset_gen tokenizer)

    # ── 运行期 JIT 演示（Phase 2；docs/internal/33）───────────────
    # 展示"不在内嵌注册表的表达式 → JIT 现编 + .nnk 缓存命中 + CPU 对拍"。
    # 仅演示用，不注册 ctest、不 install。
    nn_add_executable(jit_demo src/jit_demo.cpp)
    nn_use_modules(dataset_convert tokenizer)

    # ── 层吞吐量基准（CPU/Vulkan GPU；--all 或 --layer <name> 指定） ────────
    nn_add_executable(layer_bench src/layer_bench.cpp)
    list(APPEND NN_APP_TARGETS layer_bench)

    # ── 探针目标（诊断专用；需 -DNN_ENABLE_PROBES=ON，默认不构建）───────────
    # 这三个只服务排障，不进任何正常流程：正常构建里连可执行文件都不存在，
    # 文档里引用它们的复现步骤须先配置 NN_ENABLE_PROBES=ON。
    if(NN_ENABLE_PROBES)
        # 运行时显存分项探针（双池采样 + nvidia-smi + 逻辑分项归因）
        # ⚠ mem_probe 依赖引擎实现头（已被引擎 .so 化移除），已随迁新仓库
        # （compute.neuralnet.cpp）——需要时在新仓库构建。

        # CPU f16 组件级诊断探针（A-I 阶段定位发散；NN_F16_DEBUG=1 开逐中间量扫描）
        nn_add_executable(f16_cpu_probe src/f16_cpu_probe.cpp)
        list(APPEND NN_APP_TARGETS f16_cpu_probe)

        # CPU/GPU 训练 run-to-run 确定性探针（ComputeEngine Refresh P0 / D9 实测；
        # 手动运行不注册 ctest，退出码 0=逐位一致 1=非确定 2=运行错误）
        nn_add_executable(gpu_stability_probe src/gpu_stability_probe.cpp)
        list(APPEND NN_APP_TARGETS gpu_stability_probe)
    endif()
endif()

# ── GPU f16 参数写回探针（issue #13 P0-②：MSVC 下权重冻结定位）──────────────
# app 与 ctest 双身份 → 任一开关打开就创建（目标重复创建会报错，故统一在此）。
# ⚠ 刻意**不**并入 NN_ENABLE_PROBES：它是注册在 ctest 里的**回归用例**
#   （断言 in-place f16 写回值正确 + 8 步 Adam 后参数必须移动），去掉会削弱
#   测试门禁；它自身不含热路径探针，不参与 NN_ENABLE_PROBES 的语义。
if(NN_BUILD_APPS OR NN_ENABLE_TESTS)
    nn_add_executable(f16_writeback_probe src/f16_writeback_probe.cpp)
    if(NN_BUILD_APPS)
        list(APPEND NN_APP_TARGETS f16_writeback_probe)
    endif()
endif()

# cli 安装到 bin/（D8 完整布局：bin/ + lib/ + include/，无散落 .o/.obj）。
# cli 仍用 header-only C++ 面（模板 + 引擎内联；libnn.so 是为纯 C ABI 消费方 /
# LGPL 场景）——编译慢的根治（引擎实现抽 .so）见 docs/internal/34 §6，双轨保留。
# 放此处是因为 NN_APP_TARGETS 要到上面才填满（install 在生成期解析列表）。
install(TARGETS ${NN_APP_TARGETS} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# ── 测试 / 验证目标 ────────────────────────────────────────────────────
# 构建门 = (NN_BUILD_APPS OR NN_ENABLE_TESTS)：默认（顶层工程 NN_BUILD_APPS=ON）
#   就把全部回归可执行文件编出来，方便本地**直接跑**（无需每次重配）；
#   `-DNN_ENABLE_TESTS=OFF -DNN_BUILD_APPS=OFF` 才是"纯库、零测试产物"。
# （ctest 注册另算，见下方 NN_ENABLE_TESTS 门——只编不注册 = 本地手跑。）
if(NN_BUILD_APPS OR NN_ENABLE_TESTS)

    # ══════════════════════════════════════════════════════════════════════
    # 测试目标（19 个，按功能域分组合并）
    # 合并策略：wrapper .cpp 通过 #define main + #include 聚合原测试源文件
    # ══════════════════════════════════════════════════════════════════════

    # ── GPU 集成测试（需要 Vulkan）────────────────────────────────────────
    # ⚠ gpu_test 等引擎实现级测试（gpu_test / gpu_f16_test / scan_bitwise_test /
    #   gpt/rapt_offload_test / expr_graph_test / fused_gpu_test）已随引擎 .so 化
    #   迁往 compute.neuralnet.cpp（新仓库 tests/）——主仓库不再编引擎实现头。

    # ── GPU f16 参数写回回归（issue #13 P0-②：MSVC 下权重冻结）───────────
    # 复用 app 目标 f16_writeback_probe（已在 NN_APP_TARGETS 构建）：
    # 断言 in-place f16 写回值正确 + 8 步 Adam 后参数必须移动（防"静默冻结"）
    list(APPEND NN_TEST_TARGETS f16_writeback_probe)

    # ── 多精度类型系统（f16 转换 / 舍入 / 张量精度 / f16 计算）──────────
    nn_add_executable(precision_test src/precision_test.cpp)
    list(APPEND NN_TEST_TARGETS precision_test)

    # ── 多精度适配层（f16 存储真正生效：PrecisionEngine + DSL + 端到端）──
    nn_add_executable(f16_precision_test src/f16_precision_test.cpp)
    list(APPEND NN_TEST_TARGETS f16_precision_test)

    # ── 三值（T1_58）精度 + BitLinear（层能力校验 / 量化规则 / STE / 端到端）──
    nn_add_executable(t1_58_test src/t1_58_test.cpp)
    list(APPEND NN_TEST_TARGETS t1_58_test)

    # ── BitNet b1.58 2B4T 的结构开关（SubLN / GQA / tied embedding，docs 22）──
    nn_add_executable(bitnet_struct_test src/bitnet_struct_test.cpp)
    list(APPEND NN_TEST_TARGETS bitnet_struct_test)

    # ── 写版本（write epoch）契约 + BitLinear 量化缓存（docs 21 §4.11）──
    nn_add_executable(write_epoch_test src/write_epoch_test.cpp)
    list(APPEND NN_TEST_TARGETS write_epoch_test)

    # ── 三值打包权重 + 量化 GEMM（P2/F3，docs 21 §5 的 P2 验收）──
    nn_add_executable(ternary_gemm_test src/ternary_gemm_test.cpp)
    list(APPEND NN_TEST_TARGETS ternary_gemm_test)

    # ── int8 激活量化（W1.58A8，P3，docs 21 §4.16）：quantize/unpack/matmul_q_i8
    #    + BitLinear int8-act 接线（前向界 / backward STE / 端到端收敛）──
    nn_add_executable(int8_act_test src/int8_act_test.cpp)
    list(APPEND NN_TEST_TARGETS int8_act_test)

    # ── 模型文件角色（v7：.nnmodel / .nnckpt）+ 参数形态协议 ──
    nn_add_executable(model_roles_test src/model_roles_test.cpp)
    list(APPEND NN_TEST_TARGETS model_roles_test)

    # ── 量化状态优化器 AdamWQuant（F5 第一步，docs 21 §4.13）──
    nn_add_executable(adam_quant_test src/adam_quant_test.cpp)
    list(APPEND NN_TEST_TARGETS adam_quant_test)

    # ── int8 打包状态优化器 AdamWInt8（docs 21 §4.19）：融合原语逐位对拍 +
    #    收敛 A/B + 状态账 + save/load 往返 ──
    nn_add_executable(adam_int8_test src/adam_int8_test.cpp)
    list(APPEND NN_TEST_TARGETS adam_int8_test)

    # ── Embedding 层（issue #32：查表 / 伴随散射 / padding / 整数入口）──
    nn_add_executable(embedding_test src/embedding_test.cpp)
    list(APPEND NN_TEST_TARGETS embedding_test)

    # ── Attention（梯度检查 + forward 一致性 + batch 行号回归）──────────
    nn_add_executable(attn_test src/attn_test.cpp)
    list(APPEND NN_TEST_TARGETS attn_test)

    # ── CrossAttention（issue #33：双输入 / 矩形 / 双路梯度）─────────────
    nn_add_executable(cross_attention_test src/cross_attention_test.cpp)
    list(APPEND NN_TEST_TARGETS cross_attention_test)

    # ── 基础层数值梯度检查（SwiGLU / RMSNorm / Softmax）─────────────────
    nn_add_executable(layer_gradcheck_test src/layer_gradcheck_test.cpp)
    list(APPEND NN_TEST_TARGETS layer_gradcheck_test)

    # ── BatchNorm（工厂不回落 + 训练/推理双态 fwd/bwd + running EMA
    #    + Model::set_training 转发 + save/load 往返逐位一致）────────────
    nn_add_executable(batchnorm_test src/batchnorm_test.cpp)
    list(APPEND NN_TEST_TARGETS batchnorm_test)

    # ── CNN（MaxPool2D 独立参考比对 + 缓存/checkpoint 契约 + 端到端冒烟）──
    nn_add_executable(cnn_test src/cnn_test.cpp)
    list(APPEND NN_TEST_TARGETS cnn_test)

    # ── GPT 模型（整链梯度检查 + 梯度检查点一致性）──────────────────────
    nn_add_executable(gpt_test src/gpt_test.cpp)
    list(APPEND NN_TEST_TARGETS gpt_test)

    # ── 采样器（P0-5：top-k / top-p / 重复惩罚 / 种子；默认配置 ≡ 旧路径）──
    nn_add_executable(sampling_test src/sampling_test.cpp)
    list(APPEND NN_TEST_TARGETS sampling_test)

    # ── 文档感知注意力（端到端不变性 + 灵敏度/跨样本隔离对照；掩码构建
    #    单测已随 build_attention_mask 删除）────────────────────────────────
    nn_add_executable(doc_attn_test src/doc_attn_test.cpp)
    list(APPEND NN_TEST_TARGETS doc_attn_test)

    # ── ZiPT / AttnZip 已移除（保留在 legacy/zipt 分支）──
    #     恢复前提见 docs/history/history.md「ZiPT 移除」条。

    # ── RAPT / ReLU 线性注意力（梯度 + 冒烟 + 梯度检查点）──────────────
    nn_add_executable(rapt_test src/rapt_test.cpp)
    list(APPEND NN_TEST_TARGETS rapt_test)

    # ── 表达式/IR 纯 CPU（DSL + 归约 + matmul + 优化 pass + 图 IR）────
    nn_add_executable(expr_cpu_test src/expr_cpu_test.cpp)
    list(APPEND NN_TEST_TARGETS expr_cpu_test)

    # ── 表达式/IR GPU（图融合端到端 + AOT shader 验证 + 表达式求值）────
    nn_add_executable(expr_gpu_test src/expr_gpu_test.cpp)
    list(APPEND NN_TEST_TARGETS expr_gpu_test)

    # ── DSL 无形状重载（dsl::compute / compute_reduce 的推导口径，P4-4）──
    # 纯 CPU（推理规则与后端无关）：推导形状 == 显式形状、逐位值一致、
    # 显式 P 透传、负例（冲突 / 无网格贡献者 → 明确报错、不猜形状）
    nn_add_executable(dsl_shape_infer_test src/dsl_shape_infer_test.cpp)
    list(APPEND NN_TEST_TARGETS dsl_shape_infer_test)

    # ── ModelSpec 架构校验（load_model 文件头 vs 模型规格比对）──────────
    nn_add_executable(model_spec_validation_test src/model_spec_validation_test.cpp)
    list(APPEND NN_TEST_TARGETS model_spec_validation_test)

    # ── 稀疏交叉熵融合（loss/grad vs 参考，CPU+GPU）─────────────────────
    nn_add_executable(ce_fusion_test src/ce_fusion_test.cpp)
    list(APPEND NN_TEST_TARGETS ce_fusion_test)

    # ── 复合层逐子层默认广播（issue #29：set_precision_profile / set_training /
    # clear_cache 按 param_children() 默认转发）
    nn_add_executable(layer_broadcast_test src/layer_broadcast_test.cpp)
    list(APPEND NN_TEST_TARGETS layer_broadcast_test)

    # ── GPU 单次推理 batch 作用域（issue #28：InferenceScope / Model::forward
    # 自动作用域 / batch_active 查询）。无 Vulkan/无卡 → 77 = SKIP。
    nn_add_executable(gpu_batch_scope_test src/gpu_batch_scope_test.cpp)
    list(APPEND NN_TEST_TARGETS gpu_batch_scope_test)

    # ── 宿主桥公开 API（issue #30：nn::upload_span / download_span / download_vector）
    # 覆盖：F32/F16 往返（RHE 与 from_matrix 逐位同口径）、形状不匹配报错、
    # nn::detail:: 旧名过渡别名向后兼容。
    nn_add_executable(host_bridge_test src/host_bridge_test.cpp)
    list(APPEND NN_TEST_TARGETS host_bridge_test)

    # ── RV-01：前置条件校验（NN_ASSERT_ALWAYS / NN_FATAL）在 NDEBUG 下也终止 ──
    # 负例经子进程断言：越界 at / 未绑定 Tensor::engine() / 空 logits 采样
    # 必须非零退出、打印 `file:line`、且 tag 是新宏（不是 NDEBUG 下为空的 NN_ASSERT）
    nn_add_executable(assert_always_test src/assert_always_test.cpp)
    list(APPEND NN_TEST_TARGETS assert_always_test)

    # ── 生命周期 v3（设备域 / Layer 状态机 / 引擎先析构路径）────────────
    # 覆盖：域模型与跨域拒绝、import 唯一入口、backend weak 缓存回收、
    # init 失败回滚、跨引擎重绑拒绝、Tensor 活过引擎。
    nn_add_executable(layer_lifetime_test src/layer_lifetime_test.cpp)
    list(APPEND NN_TEST_TARGETS layer_lifetime_test)

    # ── P0 生命周期契约负例（docs/internal/28 §5 P0）──────────────────────────
    # 覆盖：ExprSegment 强引用（录制段内丢外部引擎句柄仍可 end）、
    # ActivationOffloader 状态机（未 import 的重复 export = fail-fast；
    # discard 作废通道）与精度门（slab 恒 f32 ⇒ 非 F32 激活拒绝）、
    # optimizer 先于 Model 析构的正常顺序。
    nn_add_executable(lifetime_contract_test src/lifetime_contract_test.cpp)
    list(APPEND NN_TEST_TARGETS lifetime_contract_test)

    # -- P2a: TrainingSession (Model + Optimizer 同生共死, docs/internal/29) --
    nn_add_executable(training_session_test src/training_session_test.cpp)
    list(APPEND NN_TEST_TARGETS training_session_test)

    # ── RL 公共基础：有序 tensor schema / 具名张量 / 梯度包（docs/internal/31 §3）
    nn_add_executable(tensor_schema_test src/tensor_schema_test.cpp)
    list(APPEND NN_TEST_TARGETS tensor_schema_test)

    # ── RL-0：named outputs / OutputBundleLayer / RowBandPacking / ModelSpec ─
    nn_add_executable(rl_output_test src/rl_output_test.cpp)
    list(APPEND NN_TEST_TARGETS rl_output_test)

    # ── RL-1a：experience schema / GAE / return/group normalization ───────
    nn_add_executable(rl_experience_test src/rl_experience_test.cpp)
    list(APPEND NN_TEST_TARGETS rl_experience_test)

    # ── RL-1b：categorical distribution / objective components / REINFORCE ─
    nn_add_executable(rl_policy_test src/rl_policy_test.cpp)
    list(APPEND NN_TEST_TARGETS rl_policy_test)

    # ── RL-1c：PPO policy/value clip + 标准 preset 组合等价 ────────────────
    nn_add_executable(rl_ppo_test src/rl_ppo_test.cpp)
    list(APPEND NN_TEST_TARGETS rl_ppo_test)

    # ── RL-1d：group advantage + GRPO preset ──────────────────────────────
    nn_add_executable(rl_grpo_test src/rl_grpo_test.cpp)
    list(APPEND NN_TEST_TARGETS rl_grpo_test)

    # ── RL Vulkan smoke：AOT 闭合世界 + GPU 设备域（无卡时 77=SKIP）────────
    nn_add_executable(rl_gpu_smoke_test src/rl_gpu_smoke_test.cpp)
    list(APPEND NN_TEST_TARGETS rl_gpu_smoke_test)

    # ── RL-2a：通用多模型 TrainingGroupSession ────────────────────────────
    nn_add_executable(training_group_session_test
                      src/training_group_session_test.cpp)
    list(APPEND NN_TEST_TARGETS training_group_session_test)

    # ── RL-2b：rollout/buffer/sampler/metrics/checkpoint 编排 ─────────────
    nn_add_executable(rl_pipeline_test src/rl_pipeline_test.cpp)
    list(APPEND NN_TEST_TARGETS rl_pipeline_test)

    # ── RL end-to-end：named PPO gradients → optimizer update ─────────────
    nn_add_executable(rl_training_integration_test
                      src/rl_training_integration_test.cpp)
    list(APPEND NN_TEST_TARGETS rl_training_integration_test)

    # ── F1：危险精度组合启动期告警（仅警告，不拒绝；docs/internal/24 §2.3）────
    # [1][2] 头文件函数组合矩阵（含 all=f16）单测；[3] 子进程跑 text_train/
    # mnist_train，捕获 stderr 断言告警出现 + 退出码不变；二进制缺失 → 77 = SKIP
    nn_add_executable(cli_precision_warning_test src/cli_precision_warning_test.cpp)
    list(APPEND NN_TEST_TARGETS cli_precision_warning_test)

    # ── P1-1：text_train 模块化（recovery / checkpoint 的恢复路径单测）──────
    # [1] 断点路径派生；[2] --resume 角色门禁；[3] 断点往返（参数逐位一致）；
    # [4] 断点记账 + 收尾退出码语义（P1-1 第一刀回归锁）；[5] 推理模型角色。
    # 全 CPU、秒级、无外部依赖（临时目录自建自清）。
    nn_add_executable(text_train_modules_test src/text_train_modules_test.cpp)
    list(APPEND NN_TEST_TARGETS text_train_modules_test)

    # ── P1-2 + P2-1：v8 训练状态段 + 整文件校验和（role 门禁 / 往返 / 截断）──
    # [1] TrainState ⇄ kvrec 往返；[2] Checkpoint 往返（状态 + 优化器张量逐位）；
    # [3] role 门禁；[4] Model 角色状态段长度为 0；[5] 校验和（翻字节/截断报错）；
    # [6] load_model 跳过状态段后参数逐位一致；[7] 数据集身份差异说明；
    # [8] 剥成 v7 后仍可加载（零破坏）。全 CPU、秒级、临时目录自建自清。
    nn_add_executable(train_state_test src/train_state_test.cpp)
    list(APPEND NN_TEST_TARGETS train_state_test)

    # ── TrainingSession 断点编排（issue #37：save_checkpoint/resume）──
    nn_add_executable(session_checkpoint_test src/session_checkpoint_test.cpp)
    list(APPEND NN_TEST_TARGETS session_checkpoint_test)

    # ── issue #39 RFC 32 阶段 B：BatchSource 数据供给契约 ──────────────────
    nn_add_executable(batch_source_test src/batch_source_test.cpp)
    list(APPEND NN_TEST_TARGETS batch_source_test)

    # ── issue #39 RFC 32 阶段 C：RunDriver 领域无关训练骨架 ────────────────
    nn_add_executable(run_driver_test src/run_driver_test.cpp)
    list(APPEND NN_TEST_TARGETS run_driver_test)

    # ── issue #31：epoch/step 学习率调度（multistep / step_cosine / 统一 warmup）─
    # [1] constant 兜底；[2] cosine warmup+退火+越界钳；[3] multistep ×gamma；
    # [4] step_cosine 分段重启（SGDR）+ 末段吸收余数；[5] warmup 对所有 schedule；
    # [6] lr_per_epoch 逃生舱优先级；[7] compute_step_lr 回归。
    nn_add_executable(lr_scheduler_test src/lr_scheduler_test.cpp)
    list(APPEND NN_TEST_TARGETS lr_scheduler_test)

    # ── P0-5 尾项：多轮对话闭环（prompt 构建 + CLI --turn/--reply-out 接线）──
    # [1] 对话模板结构；[2] ★ 上下文不丢（第 N 轮 prompt 以第 N-1 轮为前缀）；
    # [3] 助手轮 tokens 原样回灌；[4] 整轮截断（system + 末轮恒保留）；
    # [5] 纯文本词表回退；[6] 真实 text_infer 子进程端到端（prompt 与库逐 token
    # 一致、--reply-out == 生成 token 解码）；[7] CLI 负例（--turn 形状/互斥）。
    # app 缺失 → 77 = SKIP（CLI 半边跳过）。
    nn_add_executable(chat_prompt_test src/chat_prompt_test.cpp)
    list(APPEND NN_TEST_TARGETS chat_prompt_test)

    # ── P1-2：优化器断点状态（save_state/load_state + 重启轨迹连续）────────
    # 覆盖 = kOptimizerTable 全表：状态存在性/非零 + 步计数器；中途存状态→重启续跑
    # 与直跑参数**逐位一致**（含冷启动负例做判别力检查）；张量数/形状不符报错。
    nn_add_executable(optimizer_state_test src/optimizer_state_test.cpp)
    list(APPEND NN_TEST_TARGETS optimizer_state_test)
    # ── RV-04 优化器元数据表（唯一真源 ↔ 工厂）+ RV-03 同目标写序回归 ──────
    # [1] 表内每个 name 都能被 create_optimizer 构造、名单文本与表一致；
    # [2] AdamW（wd≠0）单步 vs 手写参考 + 顺序对调的负例；
    # [3] 同目标两次写必须建顺序边、不合组，反序时 validate_dst_order 必须报错
    nn_add_executable(optimizer_test src/optimizer_test.cpp)
    list(APPEND NN_TEST_TARGETS optimizer_test)

    # ── P2b：训练样本概念（数据集的训练单位 = 窗口样本；step 是派生量）───────
    # [1] 切窗/样本视图；[2] make_batch 逐位约定（PAD/目标/掩码/doc_ids/
    # batch-major 扁平化）；[3] assistant 掩码按目标位置；[4] 缺 test 子集与
    # 非法窗口参数报错；[5] 断点状态段旧键 dataset_windows 兼容读。
    nn_add_executable(dataset_samples_test src/dataset_samples_test.cpp)
    list(APPEND NN_TEST_TARGETS dataset_samples_test)

    # ── P2b：训练工厂 TextTrainer（数据集直接接入 + 训练状态 + 断点）─────────
    # [1] create(dataset) 建样本集/词表/数据身份；[2] fit 端到端（步数/参数移动/
    # 输出锚）；[3] max-steps/start_step；[4] 断点钩子 → TrainState → .nnckpt；
    # [5] 前置条件（batch=0 / 裁剪配置不一致）。
    nn_add_executable(text_trainer_test src/text_trainer_test.cpp)
    list(APPEND NN_TEST_TARGETS text_trainer_test)

    # ── 基础件（原 base 子仓下沉测试回归主仓 ctest）─────────────────────────
    nn_add_executable(error_macro_test src/error_macro_test.cpp)
    list(APPEND NN_TEST_TARGETS error_macro_test)
    nn_add_executable(assert_test src/assert_test.cpp)
    list(APPEND NN_TEST_TARGETS assert_test)
    nn_add_executable(file_test src/file_test.cpp)
    list(APPEND NN_TEST_TARGETS file_test)
    nn_add_executable(sha256_test src/sha256_test.cpp)
    list(APPEND NN_TEST_TARGETS sha256_test)
    nn_add_executable(precision_type_test src/precision_type_test.cpp)
    list(APPEND NN_TEST_TARGETS precision_type_test)

    # ── 可移植算子缓存（kernels.nnk）round-trip 逐位锁 ────────────────────
    nn_add_executable(kernel_cache_test src/kernel_cache_test.cpp)
    list(APPEND NN_TEST_TARGETS kernel_cache_test)

    # ── 并行（原 threadpool 子仓）───────────────────────────────────────────
    nn_add_executable(threadpool_test src/threadpool_test.cpp)
    list(APPEND NN_TEST_TARGETS threadpool_test)
    nn_add_executable(parallel_test src/parallel_test.cpp)
    list(APPEND NN_TEST_TARGETS parallel_test)

    # ── KVRecord（原 kvrecord 子仓）─────────────────────────────────────────
    nn_add_executable(kvrec_test src/kvrec_test.cpp)
    list(APPEND NN_TEST_TARGETS kvrec_test)

    # ── 数据集 / 分词器（原 dataset / tokenizer 子仓）────────────────────────
    nn_add_executable(dataset_test src/dataset_test.cpp)
    list(APPEND NN_TEST_TARGETS dataset_test)
    nn_add_executable(tokenizer_test src/tokenizer_test.cpp)
    list(APPEND NN_TEST_TARGETS tokenizer_test)
    nn_add_executable(nnvocab_test src/nnvocab_test.cpp)
    list(APPEND NN_TEST_TARGETS nnvocab_test)
    nn_add_executable(dataset_integration_test src/dataset_integration_test.cpp)
    list(APPEND NN_TEST_TARGETS dataset_integration_test)
    nn_use_modules(dataset_test dataset)
    nn_use_modules(tokenizer_test tokenizer)
    nn_use_modules(nnvocab_test tokenizer)
    nn_use_modules(dataset_integration_test tokenizer)

endif()

# ── 测试目标统一处理：输出到 <构建根>/test ──────────────────────────────
# 编译门（上面）与注册门（下方 NN_ENABLE_TESTS）解耦，故输出目录在此统一设置。
# 可执行文件位置不硬编码生成器差异：
#   - 单配置生成器（Ninja / Make）：默认可执行目录为 build 根目录 → build/test
#   - 多配置生成器（MSVC）：CMake 自动追加配置子目录 → build/test/Release(或 Debug)
# add_test 直接传目标名，CTest 会按当前配置自动解析出真实可执行文件路径。
foreach(TARGET ${NN_TEST_TARGETS})
    set_target_properties(${TARGET} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test)
endforeach()

# ── ctest 注册（NN_ENABLE_TESTS 门；编译门见上方 NN_BUILD_APPS 块）──────────
# 注册门与编译门解耦：默认（NN_BUILD_APPS=ON）测试程序**已编译**，但只有
# NN_ENABLE_TESTS=ON 才把 NN_TEST_TARGETS 逐一 add_test —— 用于「本地直接跑、
# 不注册 ctest」的场景。输出目录已在上方统一设为 <构建根>/test。
if(NN_ENABLE_TESTS)
    enable_testing()
    foreach(TARGET ${NN_TEST_TARGETS})
        add_test(NAME ${TARGET} COMMAND ${TARGET})
        # 退出码 77 = 依约定"跳过"（如纯 CPU 构建下需要 Vulkan 的测试）：
        # 记 Skipped 而非 FAILED，否则 CPU-only 构建/CI 出现假红。
        set_tests_properties(${TARGET} PROPERTIES SKIP_RETURN_CODE 77)
    endforeach()
    # CNN 的 GPU 变体：DSL 表达式校验只在 GPU/scan 路径发生（CPU 模板路径不校验），
    # 默认的纯 CPU cnn_test 测不出"只在 GPU 触发"的表达式结构回归
    # （如 MaxPool2D 裸视图根 → 空指令表）。无 Vulkan 构建下返回 77 = SKIP。
    if(TARGET cnn_test)
        add_test(NAME cnn_test_gpu COMMAND cnn_test --gpu)
        set_tests_properties(cnn_test_gpu PROPERTIES SKIP_RETURN_CODE 77)
    endif()
    # CNN / RAPT 的 f16 端到端变体（P2-3）：Conv2D 权重/梯度按 param、融合算子
    # 按 compute 走 f16 → 断言"loss 有限且下降 / 梯度无 NaN"。CPU 与 GPU
    # 各一条（GPU 才暴露精度变体的闭合世界 miss 与 f16/f32 槽位错配）。
    if(TARGET cnn_test)
        add_test(NAME cnn_test_f16 COMMAND cnn_test --f16)
        add_test(NAME cnn_test_f16_gpu COMMAND cnn_test --f16 --gpu)
        set_tests_properties(cnn_test_f16 cnn_test_f16_gpu
                             PROPERTIES SKIP_RETURN_CODE 77)
    endif()
    if(TARGET rapt_test)
        add_test(NAME rapt_test_f16 COMMAND rapt_test --f16)
        add_test(NAME rapt_test_f16_gpu COMMAND rapt_test --f16 --gpu)
        set_tests_properties(rapt_test_f16 rapt_test_f16_gpu
                             PROPERTIES SKIP_RETURN_CODE 77)
    endif()
    # BatchNorm 的 GPU 变体：推理态/EMA 表达式的 ExprSpec 校验只在 GPU
    # 路径发生（CPU 模板路径不校验），且闭合世界 miss 只在 GPU 运行期暴露。
    # 无 Vulkan 构建下返回 77 = SKIP。
    if(TARGET batchnorm_test)
        add_test(NAME batchnorm_test_gpu COMMAND batchnorm_test --gpu)
        set_tests_properties(batchnorm_test_gpu PROPERTIES SKIP_RETURN_CODE 77)
    endif()
    # 基准工具已移除（2026 清理：无法在 -Werror 下编译，且无 ctest 覆盖）
    # 三值（T1_58）BitLinear 的 GPU 变体：BitLinear 的表达式必须命中 AOT 融合
    # 注册表（闭合世界）+ 走一遍 f16 运行期精度分派；纯 CPU 构建/无设备 → 77。
    if(TARGET t1_58_test)
        add_test(NAME t1_58_test_gpu COMMAND t1_58_test --gpu)
        set_tests_properties(t1_58_test_gpu PROPERTIES SKIP_RETURN_CODE 77)
    endif()
    # BitNet 结构开关的 GPU 变体：SubLN 的 norm 表达式 / GQA 的 reshape+gather
    # 必须命中 AOT 融合注册表（闭合世界）。无 Vulkan 构建/无设备 → 77。
    if(TARGET bitnet_struct_test)
        add_test(NAME bitnet_struct_test_gpu COMMAND bitnet_struct_test --gpu)
        set_tests_properties(bitnet_struct_test_gpu PROPERTIES SKIP_RETURN_CODE 77)
    endif()
    # 写版本 + BitLinear 量化缓存的 GPU 变体：写版本记在 GpuBuffer 上（别名
    # 安全），且缓存命中路径不得绕过 AOT 闭合世界。无 Vulkan 构建/无设备 → 77。
    if(TARGET write_epoch_test)
        add_test(NAME write_epoch_test_gpu COMMAND write_epoch_test --gpu)
        set_tests_properties(write_epoch_test_gpu PROPERTIES SKIP_RETURN_CODE 77)
    endif()
    # 三值打包 GEMM 的 GPU 变体：打包/解码在设备侧（GpuEngine 接打包路径后生效），
    # 未接的后端上 BitLinear 自动退回 f16 稠密缓冲（本用例会打印跳过）。无设备 → 77。
    if(TARGET ternary_gemm_test)
        add_test(NAME ternary_gemm_test_gpu COMMAND ternary_gemm_test --gpu)
        set_tests_properties(ternary_gemm_test_gpu PROPERTIES SKIP_RETURN_CODE 77)
    endif()
    # int8 激活量化 GEMM 的 GPU 变体（quantize_i8/matmul_q_i8 设备侧 pipeline +
    # 2B4T 量级形状对拍）。无设备 → 77。
    if(TARGET int8_act_test)
        add_test(NAME int8_act_test_gpu COMMAND int8_act_test --gpu)
        set_tests_properties(int8_act_test_gpu PROPERTIES SKIP_RETURN_CODE 77)
    endif()
    # 量化状态优化器的 GPU 变体（状态读写全在设备侧；GPU 上表达式须命中闭合世界）。
    if(TARGET adam_quant_test)
        add_test(NAME adam_quant_test_gpu COMMAND adam_quant_test --gpu)
        set_tests_properties(adam_quant_test_gpu PROPERTIES SKIP_RETURN_CODE 77)
    endif()
    # int8 状态优化器的 GPU 变体（quantized_adamw_step 设备侧 pipeline）。无设备 → 77。
    if(TARGET adam_int8_test)
        add_test(NAME adam_int8_test_gpu COMMAND adam_int8_test --gpu)
        set_tests_properties(adam_int8_test_gpu PROPERTIES SKIP_RETURN_CODE 77)
    endif()
    foreach(TARGET ${NN_BENCH_TARGETS})
        set_target_properties(${TARGET} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test)
    endforeach()
endif()

# ══════════════════════════════════════════════════════════════════════════
#  引擎 .so（submodule engine/，LGPL）：libnn-cpu.so / libnn-gpu.so
# ══════════════════════════════════════════════════════════════════════════
# 引擎实现（CpuEngine/GpuEngine + 原语 + Vulkan backend + shader + AOT 融合 +
# kernelgen JIT）全部编进 compute.neuralnet.cpp（submodule）的两个 .so，经 MIT
# 抽象层接口（abstraction/include/neuralnet.cpp/compute_engine.hpp）C++ 直出
# （nn::create_cpu_engine / nn::create_gpu_engine / nn::last_error）。
# 本仓（MIT）只保留模板面（Layer/DSL/训练/数据集/分词器 + cli 业务），所有
# 可执行目标链接 .so；不再自编引擎、不再生成 shader / fused_registry、不再
# 需要 Vulkan SDK（GPU 能力全部经 libnn-gpu.so）。
#
# NN_ENABLE_GPU 语义（与旧版一致，直接透传给 submodule）：
#   OFF  = 纯 CPU：不编 libnn-gpu.so（cli --gpu 链接不到 → 明确报错）
#   AUTO / ON = 编 libnn-gpu.so（submodule 内部 find Vulkan，缺则跳过并提示；
#               ON 的"强制要求"语义由 submodule 自行处理）
# submodule 自带测试（consumer_test / gpu_consumer_test / kernelgen_smoke）
# 不并入本仓 ctest（NN_BUILD_TESTS=OFF）；需要时在新仓库单独构建。
set(NN_BUILD_TESTS OFF CACHE BOOL "Build compute.neuralnet.cpp self-tests" FORCE)
add_subdirectory(engine)

# 所有可执行目标（应用 + 测试）统一链接引擎 .so：
#   nn_cpu 恒有；nn_gpu 仅 Vulkan 可用时存在（if(TARGET) 守卫）。
# 模板面（Layer/DSL）在本仓 TU 展开，原语执行落在 .so 虚方法分派。
# NN_GPU_ENGINE 宏 = 本构建链接了 GPU 引擎（cli_engine_factory 据此决定
# 是否引用 create_gpu_engine 符号；GPU OFF 时引用会链接失败）。
foreach(TARGET ${NN_EXECUTABLE_TARGETS})
    target_link_libraries(${TARGET} PRIVATE nn_cpu)
    if(TARGET nn_gpu)
        target_link_libraries(${TARGET} PRIVATE nn_gpu)
    endif()
endforeach()
if(TARGET nn_gpu)
    target_compile_definitions(nn_core INTERFACE NN_GPU_ENGINE=1)
endif()

# ══════════════════════════════════════════════════════════════════════════
#  PCH 落地（所有目标创建完、编译条件定型之后再挂）
# ══════════════════════════════════════════════════════════════════════════
# 候选目标的编译条件逐项相同（nn_core 的 include：主仓库 include/ +
# submodule abstraction/），因此**一份 PCH 全目标共享**（REUSE_FROM）。
# 引擎 .so 化后 nn.hpp 不再含引擎实现/shader 嵌入头 → PCH 体积大幅缩小。
# 代价对比（本机 Debug 全量构建，旧版）：31 份逐目标 PCH ≈ 780s CPU → 1 份。
if(NN_ENABLE_PCH AND NN_PCH_CANDIDATES)
    add_library(nn_pch OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/cmake/nn_pch_stub.cpp)
    target_link_libraries(nn_pch PRIVATE nn_core)
    target_precompile_headers(nn_pch PRIVATE [["neuralnet.cpp/nn.hpp"]])

    # ── ccache × PCH 的第三项要求：`-fno-pch-timestamp` ────────────────────
    # ccache 手册（Precompiled headers）明确：Clang 侧必须用该选项编 PCH，否则
    # ccache 拒绝把"用了 PCH 的编译"入缓存（统计计数 Could not use precompiled
    # header）。它会让 PCH 不嵌写入时间戳 —— 顺带让 PCH 内容可复现。
    # ⚠ 这是 cc1 开关：驱动不认 `-fno-pch-timestamp`，必须 `-Xclang` 形式；ccache
    #   是逐 token 匹配该字面量，`-Xclang -fno-pch-timestamp` 满足它的检查
    #   （实测：开了之后同一 TU 从 30.2s 掉到 0.11s，见 docs/history/history.md）。
    # ⚠ PCH 生成方（nn_pch）与所有使用方必须带同一份选项：CMake 的 REUSE_FROM
    #   要求双方编译选项一致，故下面两处都加（对非同族的 sccache 也无害）。
    # ⚠ 已知未覆盖：GCC 侧要缓存"用了 PCH 的 TU"须另加 `-fpch-preprocess`（手册），
    #   它改变预处理输出、影响 depfile 口径，故本轮不擅自打开——GCC + PCH 的编译
    #   仍不可缓存（CI 用 Clang，不受影响）。要用就显式 -DNN_COMPILER_LAUNCHER 自己加。
    set(NN_PCH_COMPAT_OPTIONS "")
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        set(NN_PCH_COMPAT_OPTIONS "-Xclang" "-fno-pch-timestamp")
    endif()
    target_compile_options(nn_pch PRIVATE ${NN_PCH_COMPAT_OPTIONS})

    set(NN_PCH_OWN_TARGETS "")
    foreach(_pch_tgt IN LISTS NN_PCH_CANDIDATES)
        target_compile_options(${_pch_tgt} PRIVATE ${NN_PCH_COMPAT_OPTIONS})
        target_precompile_headers(${_pch_tgt} REUSE_FROM nn_pch)
    endforeach()
    list(LENGTH NN_PCH_CANDIDATES _nn_pch_n)
    list(LENGTH NN_PCH_OWN_TARGETS _nn_pch_own)
    message(STATUS "PCH: nn.hpp precompiled once, shared by "
                   "${_nn_pch_n} targets (${_nn_pch_own} own PCH)")
elseif(NN_ENABLE_PCH)
    message(STATUS "PCH: no candidate targets")
endif()
