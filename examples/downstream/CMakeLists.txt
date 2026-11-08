# ══════════════════════════════════════════════════════════════════════════
#  examples/downstream — 最小下游消费工程（P0-4 验收样例）
#
#  模拟"第三方把 neuralnet.cpp 丢进 3rd_party/ 后只写两行 CMake"的形态：
#
#      add_subdirectory(3rd_party/neuralnet.cpp)
#      target_link_libraries(app PRIVATE neuralnet::nn)
#
#  外加源码侧一行：#include <neuralnet.cpp/nn.hpp>
#
#  本工程是**独立 project**（不被仓库根 CMakeLists 引用），单独配置构建：
#      cmake -S examples/downstream -B <builddir>
#  也由仓库 ctest 的 downstream_smoke 用例驱动（cmake/downstream_smoke.cmake）。
#
#  可调开关（全部沿用库侧命名）：
#      -DNN_ENABLE_GPU=OFF|AUTO|ON   GPU 开关（smoke 用 OFF 保快速且无 Vulkan 依赖）
#      -DNEURALNET_CPP_DIR=<path>    库位置（默认 ../..，可指向 3rd_party/neuralnet.cpp）
#
#  自研 dsl::compute 结构无需任何 CMake 步骤：GPU 首次执行时走运行期 JIT
#  （libnn-kernelgen → SPIR-V → .nnk 缓存），可用 model_warmup() 预热。
# ══════════════════════════════════════════════════════════════════════════
cmake_minimum_required(VERSION 3.30)
project(neuralnet_downstream LANGUAGES CXX)

# ① 嵌入库（真实场景这里是 3rd_party/neuralnet.cpp；路径可覆盖）
set(NEURALNET_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." CACHE PATH
    "Path to the neuralnet.cpp repository")
add_subdirectory("${NEURALNET_CPP_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/neuralnet_cpp")

# ② 一行链接——C++ 标准 / include / 线程库 / （GPU 时）NN_HAS_VULKAN +
#    SPIR-V 嵌入头目录 + Vulkan 链接 + shader 生成顺序，全部由目标传递。
add_executable(downstream_demo main.cpp)
target_link_libraries(downstream_demo PRIVATE neuralnet::nn)

# ctest 约定：77 = 跳过（本样例在 GPU 编译进但无设备时返回 77）
enable_testing()
add_test(NAME downstream_demo COMMAND downstream_demo)
set_tests_properties(downstream_demo PROPERTIES SKIP_RETURN_CODE 77)
