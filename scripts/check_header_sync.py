#!/usr/bin/env python3
# ── check_header_sync.py — 双仓头一致性门禁 ──────────────────────────────
# 层/模型/数据集/域/损失/优化器头在两个仓库各存一份（AGENTS §1「层/模型头副本」：
# 主仓 include/neuralnet.cpp 消费侧 + engine 子模块 engine/engine/include/neuralnet.cpp
# 引擎侧），必须逐字一致；聚合/转发头与 cli/ 允许有意差异（两侧各自表述）。
# 退出码：0 = 副本头全部一致；1 = 存在非白名单漂移。
#
# 用法：python scripts/check_header_sync.py [--verbose]
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN = os.path.join(ROOT, 'include', 'neuralnet.cpp')
ENG = os.path.join(ROOT, 'engine', 'engine', 'include', 'neuralnet.cpp')

# 允许有意差异的文件（相对 include/neuralnet.cpp/ 的路径）：
#   聚合/转发头（nn/engine/train）两侧各自聚合面不同；
#   model_serialization / compute_layer_bitlinear 两侧 include 路径形式不同
#   （"..." 消费侧相对包含 vs <neuralnet.cpp/...> 引擎侧全路径）；
#   cli/cli_engine_factory 两侧描述各自的 CLI 工厂语义。
ALLOW_DIFF = {
    'nn.hpp',
    'engine.hpp',
    'train.hpp',
    'model_serialization.hpp',
    'compute_layer_bitlinear.hpp',
    'cli/cli_engine_factory.hpp',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    diffs = []
    notes = 0
    for root, _dirs, files in os.walk(MAIN):
        for fn in sorted(files):
            if not fn.endswith('.hpp'):
                continue
            rel = os.path.relpath(os.path.join(root, fn), MAIN)
            eng_path = os.path.join(ENG, rel)
            if not os.path.isfile(eng_path):
                notes += 1
                if args.verbose:
                    print(f'  [note] 仅主仓有 {rel}')
                continue
            with open(os.path.join(root, fn), encoding='utf-8') as f:
                main_text = f.read()
            with open(eng_path, encoding='utf-8') as f:
                eng_text = f.read()
            if main_text == eng_text:
                continue
            if rel in ALLOW_DIFF:
                notes += 1
                if args.verbose:
                    print(f'  [note] 白名单差异（有意） {rel}')
                continue
            diffs.append(rel)
            print(f'  [DIFF] {rel}')

    for root, _dirs, files in os.walk(ENG):
        for fn in sorted(files):
            if not fn.endswith('.hpp'):
                continue
            rel = os.path.relpath(os.path.join(root, fn), ENG)
            if not os.path.isfile(os.path.join(MAIN, rel)):
                notes += 1
                if args.verbose:
                    print(f'  [note] 仅 engine 有 {rel}')

    if diffs:
        print(f'❌ FAIL：{len(diffs)} 个副本头漂移（主仓与 engine 侧不一致，需双仓同步或加入 ALLOW_DIFF）')
        return 1
    print(f'✅ 副本头全部一致（{notes} 项豁免/单侧提示，--verbose 查看）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
