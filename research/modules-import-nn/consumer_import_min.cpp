// ── consumer_import_min.cpp — 最小 import 消费（BMI 已导出库符号；STL 由消费者自带）──
// 说明：薄接口模块把 STL 留在全局模块，故消费者需自行 include 用到的 STL 头（如 <cstddef>）。
// 这是 header-wrapping 模块的真实限制，记入 README 边界。
import nn;
#include <cstddef>

#include "consumer_body.inc"

int main()
{
    return run_workload() > 0.0 ? 0 : 1;
}
