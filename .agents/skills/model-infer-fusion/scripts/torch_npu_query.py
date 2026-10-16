#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Query torch_npu API documentation from the local installation.

Data source: _op_plugin_docs.py inside the installed torch_npu package. It is
copied in at build time from the op-plugin repository codegen template and
holds the operator entries whose docstrings were generated; the entry count
varies with the version.

Results are bound to the torch_npu version actually installed, need no network
access, and cannot drift from the runtime.

op-plugin codegen coverage is partial: some operators are callable through
torch_npu yet have no entry in _op_plugin_docs.py. The built-in _FALLBACK_DOCS
dict backfills the fusion operators inside model-infer-fusion scope. For an
operator that exists at runtime but appears in neither source, the tool reports
that it is callable without a docstring.

Subcommands:
  show   details of a single API
  search reverse lookup, matching a keyword against API names and doc text
  list   enumerate every operator name

Examples:
  python3 torch_npu_query.py show  npu_fusion_attention
  python3 torch_npu_query.py search attention
  python3 torch_npu_query.py search "MoE" --max 20
  python3 torch_npu_query.py list
  python3 torch_npu_query.py list --prefix npu_

Without a torch_npu environment, point at the doc file explicitly:
  python3 torch_npu_query.py --docs-path /path/to/_op_plugin_docs.py list
"""

from __future__ import annotations

import argparse
import os
import re
import sys


DOC_ENTRY_PATTERN = re.compile(
    r'_add_torch_npu_docstr\(\s*"([^"]+)"\s*,\s*"""(.+?)"""',
    re.DOTALL,
)


class QueryError(Exception):
    """Raised on an unrecoverable error; main catches it and exits."""


def _emit(text: str = "", *, file=sys.stdout) -> None:
    """Write one line of result data to the given stream."""
    file.write(text + "\n")


# Fallback: operators for which hasattr is True in the current torch_npu build
# but for which _op_plugin_docs.py registered no docstring. Each entry is a
# hand-maintained one-line summary plus an online URL.
#
# Scope: only the fusion operators the model-infer-fusion skill cares about.
# Cross-scope operators such as stream/device control stay out of this table and
# are maintained by their own skills.
#
# URL choice: the hiascend.com 26.0.0 documentation site.
#
# Staleness check: once op-plugin fills in the missing docstrings, entries parsed
# from _op_plugin_docs.py override this table automatically.
_HIASCEND_URL_BASE = (
    "https://www.hiascend.com/document/detail/zh/Pytorch/2600/"
    "apiref/torchnpuCustomsapi/docs/zh/custom_APIs/torch_npu/"
)


def _hiascend_url(api_name: str) -> str:
    return f"{_HIASCEND_URL_BASE}torch_npu-{api_name}.md"


_FALLBACK_DOCS: dict[str, tuple[str, str]] = {
    # ===== Category C: key LLM inference fusion operators whose docstring op-plugin codegen missed =====
    "npu_kv_rmsnorm_rope_cache": (
        "融合 MLA 结构中 RMSNorm + RoPE + ScatterUpdate(KVCache 写入) 三步骤",
        _hiascend_url("npu_kv_rmsnorm_rope_cache"),
    ),
    "npu_interleave_rope": (
        "针对单输入 x 的 interleave 模式 RoPE 旋转位置编码",
        _hiascend_url("npu_interleave_rope"),
    ),
    "npu_gelu": (
        "GELU 激活函数",
        _hiascend_url("npu_gelu"),
    ),
    "npu_group_quant": (
        "对输入张量进行分组量化操作",
        _hiascend_url("npu_group_quant"),
    ),
    "npu_moe_distribute_dispatch": (
        "MoE EP 并行的 token dispatch（quant+EP AllToAll+TP AllGather），与 combine 配套",
        _hiascend_url("npu_moe_distribute_dispatch"),
    ),
    "npu_moe_distribute_dispatch_v2": (
        "MoE EP 并行的 token dispatch v2（MC2），与 combine_v2 / combine_add_rms_norm 配套",
        _hiascend_url("npu_moe_distribute_dispatch_v2"),
    ),
    "npu_moe_distribute_combine": (
        "MoE EP 并行的 token combine（reduce_scatterv + alltoallv），与 dispatch 配套",
        _hiascend_url("npu_moe_distribute_combine"),
    ),
    "npu_moe_distribute_combine_v2": (
        "MoE EP 并行的 token combine v2（MC2），与 dispatch_v2 配套",
        _hiascend_url("npu_moe_distribute_combine_v2"),
    ),
    "npu_moe_distribute_combine_add_rms_norm": (
        "MoE combine + add_rms_norm 融合（MC2），与 dispatch_v2 配套",
        _hiascend_url("npu_moe_distribute_combine_add_rms_norm"),
    ),
    "npu_moe_re_routing": (
        "MoE 网络 AlltoAll 后将 token 按专家顺序重新排列",
        _hiascend_url("npu_moe_re_routing"),
    ),

    # ===== Category B: runtime __doc__ has content but _op_plugin_docs.py registered no entry =====
    # show already returns the full document from the runtime __doc__; these entries
    # exist so that list and search can see the operators too.
    "erase_stream": (
        "移除张量被 record_stream 添加的已被某 stream 使用的标记",
        _hiascend_url("erase_stream"),
    ),
    "matmul_checksum": (
        "基于 torch.matmul 的 AIcore 错误硬件故障检测：校验矩阵计算结果是否超过门限",
        _hiascend_url("matmul_checksum"),
    ),
}


def _fallback_doc_text(api_name: str) -> str | None:
    """Render a fallback entry for display: summary, online link, and a warning."""
    entry = _FALLBACK_DOCS.get(api_name)
    if entry is None:
        return None
    summary, url = entry
    return (
        f"{summary}\n\n"
        f"[!] 当前 torch_npu build 未生成本地 docstring。请查在线文档：\n"
        f"    {url}"
    )


_OFFLINE_LIST_HINT = (
    "[!] 无 torch_npu 环境且未指定 --docs-path，已降级到 _FALLBACK_DOCS 兜底集。\n"
    "    完整算子目录请查离线总表：references/torch_npu_API/torch_npu_list.md\n"
    "    （该表为版本快照，可能与实际安装版本 drift，签名以本地 docstring 或在线文档为准）"
)


def resolve_docs_path(explicit_path: str | None) -> str | None:
    """Locate _op_plugin_docs.py. Return None when torch_npu is absent and no path
    was given, so the caller can fall back to _FALLBACK_DOCS.
    """
    if explicit_path:
        if not os.path.isfile(explicit_path):
            raise QueryError(f"指定的 --docs-path 不存在: {explicit_path}")
        return explicit_path

    try:
        import torch_npu
    except ImportError:
        return None

    candidate = os.path.join(torch_npu.__path__[0], "_op_plugin_docs.py")
    if not os.path.isfile(candidate):
        raise QueryError(
            f"未在 torch_npu 包内找到 _op_plugin_docs.py: {candidate}\n"
            "可能 torch_npu 版本过旧或非官方 build，请用 --docs-path 指定文档源。"
        )
    return candidate


def parse_doc_entries(docs_path: str | None) -> list[tuple[str, str]]:
    """Parse _op_plugin_docs.py and merge in the fallback entries.

    A real docstring from _op_plugin_docs.py takes precedence over _FALLBACK_DOCS,
    so entries filled in later by op-plugin override the fallback automatically.
    When docs_path is None, only _FALLBACK_DOCS is returned.
    """
    entries: list[tuple[str, str]] = []
    if docs_path is not None:
        with open(docs_path, encoding="utf-8") as f:
            content = f.read()
        entries = DOC_ENTRY_PATTERN.findall(content)

    existing = {name for name, _ in entries}
    for name in _FALLBACK_DOCS:
        if name not in existing:
            entries.append((name, _fallback_doc_text(name) or ""))
    return entries


def cmd_show(api_name: str, docs_path: str | None) -> None:
    """Print the details of a single API.

    Lookup order:
      1. Runtime torch_npu.<api>.__doc__, covering Category B and normal operators
      2. parse_doc_entries, i.e. _op_plugin_docs.py plus the _FALLBACK_DOCS dict
      3. hasattr is True at runtime but neither source matched: report that the
         operator is callable without a docstring
      4. Nothing matched: report that the API was not found
    """
    api_exists_at_runtime = False
    try:
        import torch_npu

        if hasattr(torch_npu, api_name):
            api_exists_at_runtime = True
            obj = getattr(torch_npu, api_name)
            if obj is not None and obj.__doc__:
                _emit(obj.__doc__)
                return
    except ImportError:
        pass

    entries = parse_doc_entries(docs_path)
    for name, doc in entries:
        if name == api_name:
            _emit(doc.strip())
            return

    if api_exists_at_runtime:
        raise QueryError(
            f"算子 {api_name} 在当前 torch_npu 环境中可调用，"
            f"但 op-plugin codegen 未生成 docstring 条目，"
            f"且未被 _FALLBACK_DOCS 字典覆盖。\n"
            f"请查在线文档：\n"
            f"  {_hiascend_url(api_name)}\n"
            f"如该算子属于 model-infer-fusion scope 的关键融合算子，"
            f"建议补充到本脚本的 _FALLBACK_DOCS 字典。"
        )

    if docs_path is None:
        raise QueryError(
            f"未找到 API: {api_name}\n{_OFFLINE_LIST_HINT}"
        )
    raise QueryError(f"未找到 API: {api_name}")


def cmd_search(keyword: str, max_results: int, docs_path: str | None) -> None:
    """Reverse lookup: match a keyword against API names and doc text."""
    if docs_path is None:
        _emit(_OFFLINE_LIST_HINT, file=sys.stderr)
    entries = parse_doc_entries(docs_path)
    keyword_lower = keyword.lower()

    results = []
    for name, doc in entries:
        if keyword_lower in name.lower() or keyword_lower in doc.lower():
            first_line = next(
                (line.strip() for line in doc.splitlines() if line.strip()), ""
            )
            if len(first_line) > 150:
                first_line = first_line[:147] + "..."
            # Tag fallback entries so the caller can tell a stub apart from a full docstring
            if name in _FALLBACK_DOCS:
                first_line = f"[fallback] {first_line}"
            results.append((name, first_line))
            if len(results) >= max_results:
                break

    if not results:
        raise QueryError(f"未找到匹配 '{keyword}' 的算子。")

    _emit(f"搜索 '{keyword}' 找到 {len(results)} 个结果（上限 {max_results}）:\n")
    for i, (name, summary) in enumerate(results, 1):
        _emit(f"{i}. {name}")
        if summary:
            _emit(f"   {summary}")
        _emit()


def cmd_list(prefix: str | None, docs_path: str | None) -> None:
    """Enumerate every operator name, optionally filtered by prefix."""
    if docs_path is None:
        _emit(_OFFLINE_LIST_HINT, file=sys.stderr)
    entries = parse_doc_entries(docs_path)
    names = sorted({name for name, _ in entries})

    if prefix:
        names = [n for n in names if n.startswith(prefix)]

    for name in names:
        _emit(name)
    sys.stdout.flush()
    _emit(f"\n总计 {len(names)} 个算子。", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Query torch_npu API documentation locally",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Subcommands:", 1)[-1] if __doc__ else "",
    )
    parser.add_argument(
        "--docs-path",
        default=None,
        help="explicit path to _op_plugin_docs.py, for use without a torch_npu environment",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_show = sub.add_parser("show", help="details of a single API")
    p_show.add_argument("api_name", help="operator name, for example npu_fusion_attention")

    p_search = sub.add_parser("search", help="reverse lookup")
    p_search.add_argument("keyword", help="keyword matched against API names and doc text")
    p_search.add_argument(
        "--max", type=int, default=10, dest="max_results", help="maximum number of results (default 10)"
    )

    p_list = sub.add_parser("list", help="enumerate every operator name")
    p_list.add_argument("--prefix", default=None, help="list only operators starting with this prefix")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        docs_path = resolve_docs_path(args.docs_path)
        if args.cmd == "show":
            cmd_show(args.api_name, docs_path)
        elif args.cmd == "search":
            cmd_search(args.keyword, args.max_results, docs_path)
        elif args.cmd == "list":
            cmd_list(args.prefix, docs_path)
    except QueryError as err:
        _emit(str(err), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
