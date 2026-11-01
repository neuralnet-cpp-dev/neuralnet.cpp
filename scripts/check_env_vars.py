#!/usr/bin/env python3
# ── check_env_vars.py — 环境变量三处对照审计 ─────────────────────────────
# 代码读取点 ↔ AGENTS.md ↔ docs/ 三方对照：
#   · 文档记载但代码已无读取点（死变量，如 NN_GPU_PROFILE / NN_MM_NO_KSPLIT）→ FAIL
#   · 代码有读取点但 AGENTS.md 未记载（漏记，如 NN_MM_NO_FINE 曾长期如此）→ 提示
#   · 编译开关 / CMake 变量（NN_ENABLE_* 等，非环境变量）自动排除
# 退出码：0 = 无死变量（漏记只提示）；1 = 存在死变量。
#
# 用法：python scripts/check_env_vars.py [--verbose]
import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 代码目录（主仓 + engine 子模块）
CODE_DIRS = [
    'include', 'src',
    'engine/engine/include', 'engine/engine/tools',
    'engine/abstraction/include',
]

# 平台/系统变量（kernel cache 目录定位用），不参与对照
PLATFORM_VARS = {'HOME', 'XDG_CACHE_HOME', 'USERPROFILE', 'LOCALAPPDATA'}

# AGENTS/docs 中出现的编译开关 / CMake 变量前缀（非环境变量），提取时排除
NON_ENV_PREFIX = re.compile(
    r'^NN_ENABLE_|^NN_BUILD_|^NN_HAS_|^NN_USE_|^NN_MOD_|^NN_EXPR_SCAN|'
    r'^NN_SHADER_|^NN_ETYPE|^NN_PROBES_ENABLED|^NN_FUSED_|^NN_COMPILER_LAUNCHER|'
    r'^NN_GPU_ENGINE'
)

# 宏 / CMake 变量 / 截断 token（文档中出现但非环境变量），提取时排除
MACRO_OR_CMAKE = {
    'NN_TRY', 'NN_TRY_CHECK', 'NN_TRY_MSG', 'NN_UNWRAP', 'NN_UNWRAP_1',
    'NN_EXIT', 'NN_FAIL', 'NN_FATAL', 'NN_ASSERT', 'NN_ASSERT_ALWAYS',
    'NN_ASSERT_', 'NN_CHECK', 'NN_EASY_CHECK', 'NN_REQUIRE', 'NN_BIND_DEBUG',
    'NN_DETAIL_PICK2', 'NN_RD', 'NN_WR', 'NN_V4', 'NN_PACK4', 'NN_LOADV4',
    'NN_STDLIB', 'NN_VECTORIZE_PRAGMA', 'NN_PROF_OP', 'NN_PROF_SCOPE',
    'NN_PROFILE_EVERY', 'NN_PROBE_FOLD_NOMM', 'NN_PROBE_FOLD_NOVECACC',
    'NN_MM_FINE', 'NN_ATTN_MASK_KIND_LIST', 'NN_SCAN_F32_ONLY',
    'NN_SCAN_NO_DRYRUN', 'NN_GPU_LIFETIME_FULL', 'NN_LIB_SOURCE_DIR',
    'NN_LIB_BINARY_DIR', 'NN_APP_TARGETS', 'NN_TEST_TARGETS',
    'NN_SPV_FN_', 'NN_SPV_FN_MISSING', 'NN_SPV_GETTER', 'NN_SPV_GETTER_LIST',
    'NN_X_SPV_EMBEDDED', 'NN_POOL_', 'NN_VULKAN_', 'NN_PROBE_FOLD_',
}

# 历史文档（归档性质，不参与"死变量"判定——它们记载的变量可能已被正当删除）
HISTORY_DIRS = ('docs/history', 'docs/release-notes', 'docs/benchmarks', 'research')

READ_PAT = re.compile(
    r'(?:env_flag_raw|env_string_raw|env_int_raw|probe_flag_raw|probe_string_raw|'
    r'tool_env_flag|tool_env_int|env_flag|get_env|prec_env_flag)'
    r'\("([A-Z][A-Z0-9_]{2,})"'
    r'|getenv\("([A-Z][A-Z0-9_]{2,})"'
    r'|_dupenv_s\(&buf, &len, "([A-Z][A-Z0-9_]{2,})"'
)

DOC_VAR_PAT = re.compile(r'\b(NN_[A-Z0-9_]{2,})\b')


def code_env_vars():
    found = set()
    for d in CODE_DIRS:
        base = os.path.join(ROOT, d)
        if not os.path.isdir(base):
            continue
        for root, _dirs, files in os.walk(base):
            for fn in files:
                if not (fn.endswith('.hpp') or fn.endswith('.cpp')):
                    continue
                with open(os.path.join(root, fn), encoding='utf-8', errors='replace') as f:
                    text = f.read()
                for m in READ_PAT.finditer(text):
                    for g in m.groups():
                        if g:
                            found.add(g)
    return {v for v in found if v not in PLATFORM_VARS}


def doc_env_vars():
    found = set()
    targets = [os.path.join(ROOT, 'AGENTS.md')]
    for sub in ('docs',):
        base = os.path.join(ROOT, sub)
        for root, _dirs, files in os.walk(base):
            for fn in files:
                if fn.endswith('.md'):
                    p = os.path.join(root, fn)
                    if any(p.startswith(os.path.join(ROOT, h)) for h in HISTORY_DIRS):
                        continue
                    targets.append(p)
    for p in targets:
        if not os.path.isfile(p):
            continue
        with open(p, encoding='utf-8', errors='replace') as f:
            text = f.read()
        for m in DOC_VAR_PAT.finditer(text):
            v = m.group(1)
            if NON_ENV_PREFIX.match(v) or v in MACRO_OR_CMAKE:
                continue
            found.add(v)
    return {v for v in found if v not in PLATFORM_VARS}


# 已知死变量（文档中的 token 是历史 A/B 记录、已标注删除，不算 FAIL）
KNOWN_DEAD = {
    'NN_MM_NO_KSPLIT', 'NN_POOL_BLOCK_MB', 'NN_POOL_LADDER_MAX_MB',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    code = code_env_vars()
    docs = doc_env_vars()

    dead = sorted(docs - code)          # 文档有、代码读取点无 → 死变量
    known = sorted(set(dead) & KNOWN_DEAD)
    dead = [v for v in dead if v not in KNOWN_DEAD]
    missing = sorted(code - docs)       # 代码有、文档无 → 漏记

    print(f'代码读取点：{len(code)} 个；文档记载：{len(docs)} 个\n')

    fails = 0
    if dead:
        fails = len(dead)
        print('❌ 死变量（文档记载但代码已无读取点）：')
        for v in dead:
            print(f'  · {v}')
        print('   处理：从 AGENTS.md / docs 删除；若代码读取确已删除则同步清理注释\n')
    else:
        print('✅ 无死变量')

    if known:
        print(f'ℹ 已标注的历史记录（{len(known)} 个，文档中带"已删除"标注，不阻塞）：')
        for v in known:
            print(f'  · {v}')

    if missing:
        print('⚠ 漏记（代码有读取点但 AGENTS.md/docs 未记载，需人工甄别）：')
        for v in missing:
            print(f'  · {v}')
    else:
        print('✅ 无漏记')

    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
