// ── nn_module.cpp — header wrapping 直接尝试（**会编译失败,保留作为证据**）────
// 本文件演示"直接把 nn.hpp 塞进模块 purview"的失败形态,对应 README §4 第 1 条。
// 失败原因：`export { #include <nn.hpp> }` 会把 nn.hpp 递归引入的全部系统头
// （本仓实测 418 个）纳入模块 purview,其中 glibc/libstdc++ 的私有头
// （bits/libc-header-start.h 等）带 `#error ... do not include directly` 守卫 → 硬失败。
// 唯一可行形态 = 薄接口模块,见同目录 nn_module_thin.cpp。
export module nn;

export {
#include <neuralnet.cpp/nn.hpp>
}
