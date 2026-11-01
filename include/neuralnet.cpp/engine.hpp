#pragma once

// ── engine.hpp — 计算引擎（按需加载薄转发头）───────────────────────────────
//
// 与 `nn.hpp` 的关系：**`nn.hpp` 仍是唯一聚合头**（一次拿齐全部模块）；本头
// 是「按需加载」的转发，只为了 `#include <neuralnet.cpp/engine.hpp>` 这种
// 好记的写法，不必去猜 `compute_*.hpp` 的命名。
//
// 引擎实现（CpuEngine/GpuEngine 具体类体）自 2026-11 起编进独立 .so
// （libnn-cpu.so / libnn-gpu.so，LGPL，见 docs/internal/34 §10）；本头只提供
// 抽象层接口（ComputeEngine）+ 工厂声明（nn::create_cpu_engine /
// nn::create_gpu_engine / nn::last_error），消费方链接对应 .so 拿到实现。
// 用法：
//     #include <neuralnet.cpp/engine.hpp>
//     auto engine = nn::create_cpu_engine();   // shared_ptr，RAII 管理

#include <neuralnet.cpp/compute_engine.hpp>
