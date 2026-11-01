// ── consumer_include.cpp — 基线：传统 #include 消费 ─────────────────────────
#include <neuralnet.cpp/nn.hpp>

#include "consumer_body.inc"

int main()
{
    return run_workload() > 0.0 ? 0 : 1;
}
