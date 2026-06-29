#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
多进程批量：将 DeepSeek-V4-Flash 原生 MXFP4 的多个 MoE 层分别转为独立 GGUF。

每层专家独占一个 safetensors shard（见 handoff §3），所以单层转换只读一个文件，
层间用 ProcessPoolExecutor 并行。输出 ``{output_dir}/{prefix}{L}{suffix}.gguf``
（默认 ``dsv4_layer{L}_mxfp4.gguf``）。无损 nibble repack，详见
``convert_mxfp4_layer_to_gguf.py``。

容量：MXFP4 ~3.42 GiB/layer ×43 = ~147 GiB（Q8_0 是 ~295 GiB，正好一半）。

示例（全量 43 层）::

  /usr/local/python3.11.14/bin/python3 tools/batch_convert_mxfp4_layers_mp.py \\
    --input /workspace/models/DeepSeekV4/DeepSeek-V4-Flash \\
    --output-dir /workspace/models/cache \\
    --layer-start 0 --layer-end 42 --jobs 4 --skip-existing --verify-sample 3
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import subprocess
import sys
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

logger = logging.getLogger("kt.tools.batch_convert_mxfp4")

# MoE per-layer dims, grouped so converter calls stay under the arg-count limit.
MoeDims = namedtuple("MoeDims", ["num_experts", "hidden_size", "moe_intermediate_size"])


def _setup_logging() -> None:
    """INFO -> stdout, WARNING+ -> stderr (preserves the original print stream split)."""
    out = logging.StreamHandler(sys.stdout)
    out.addFilter(lambda r: r.levelno < logging.WARNING)
    err = logging.StreamHandler(sys.stderr)
    err.setLevel(logging.WARNING)
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[out, err])


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _convert_script() -> Path:
    return _repo_root() / "tools" / "convert_mxfp4_layer_to_gguf.py"


def _run_one_layer(py, model_dir, layer_idx, output_path, dims):
    cmd = [
        py, str(_convert_script()),
        "--input", model_dir,
        "--layer-idx", str(layer_idx),
        "--output", output_path,
        "--num-experts", str(dims.num_experts),
        "--hidden-size", str(dims.hidden_size),
        "--moe-intermediate-size", str(dims.moe_intermediate_size),
    ]
    env = os.environ.copy()
    env.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")  # pure numpy/torch CPU, skip torch_npu autoload
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    tail = (proc.stdout or "")[-3000:]
    if proc.stderr:
        tail += "\n--- stderr ---\n" + proc.stderr[-3000:]
    return layer_idx, proc.returncode, tail


def _verify_sample_paths(paths: list[Path]) -> None:
    gguf_py = _repo_root() / "third_party" / "llama.cpp" / "gguf-py"
    # vendored gguf-py (pinned b3173) must shadow any installed gguf; insert(0) is intentional
    sys.path.insert(0, str(gguf_py))  # pylint: disable=no-use-sys-path-insert
    from gguf import GGUFReader
    for p in paths:
        if not p.is_file():
            logger.info(f"[verify-sample] SKIP missing: {p}")
            continue
        reader = GGUFReader(str(p))
        logger.info(f"[verify-sample] {p.name} ({p.stat().st_size/1e9:.3f} GB) tensors={len(reader.tensors)}")
        for t in reader.tensors:
            tt = t.tensor_type
            logger.info(f"    {t.name} type={getattr(tt,'name',tt)} shape={list(t.shape)}")


def main() -> int:
    _setup_logging()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True, help="原生 MXFP4 模型目录（含 index.json）")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--layer-start", type=int, default=0)
    ap.add_argument("--layer-end", type=int, default=42, help="含端点")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--python", type=Path, default=Path(sys.executable))
    ap.add_argument("--num-experts", type=int, default=256)
    ap.add_argument("--hidden-size", type=int, default=4096)
    ap.add_argument("--moe-intermediate-size", type=int, default=2048)
    ap.add_argument("--name-prefix", type=str, default="dsv4_layer")
    ap.add_argument("--name-suffix", type=str, default="_mxfp4")
    ap.add_argument("--skip-existing", action="store_true", help="目标 >=3GiB（接近完整层大小）才跳过")
    ap.add_argument("--verify-sample", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    model_dir = args.input.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()
    if not model_dir.is_dir():
        logger.error(f"ERROR: --input 不是目录: {model_dir}")
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)
    # 完整 MXFP4 层约 3.42 GiB；阈值设 3 GiB，截断/不完整文件(<3GiB)不会被当"已完成"跳过。
    # 配合 convert 的原子写（.tmp+rename），最终文件名只会是完整文件。
    min_skip = 3 << 30
    py = str(args.python.expanduser())

    dims = MoeDims(args.num_experts, args.hidden_size, args.moe_intermediate_size)
    layers = list(range(args.layer_start, args.layer_end + 1))
    tasks = []
    for lid in layers:
        outp = out_dir / f"{args.name_prefix}{lid}{args.name_suffix}.gguf"
        if args.skip_existing and outp.is_file() and outp.stat().st_size > min_skip:
            logger.info(f"[batch] skip existing {outp.name}")
            continue
        tasks.append((py, str(model_dir), lid, str(outp), dims))

    if not tasks:
        logger.info("[batch] 无待转换任务（均 skip）")
    else:
        logger.info(f"[batch] model={model_dir} layers={args.layer_start}..{args.layer_end} "
                    f"pending={len(tasks)} jobs={args.jobs}")
        failed = []
        with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as ex:
            futures = {ex.submit(_run_one_layer, *t): t[2] for t in tasks}
            for fut in as_completed(futures):
                lid = futures[fut]
                try:
                    layer_idx, rc, tail = fut.result()
                except Exception as e:
                    failed.append((lid, repr(e)))
                    logger.info(f"[batch] layer {lid} worker exception: {e!r}")
                    continue
                if rc != 0:
                    failed.append((layer_idx, f"exit {rc}"))
                    logger.info(f"[batch] layer {layer_idx} FAILED rc={rc}\n{tail[-1500:]}")
                else:
                    logger.info(f"[batch] layer {layer_idx} OK")
        if failed:
            logger.error(f"[batch] 失败 {len(failed)} 层: {failed[:10]}")
            return 1

    if args.verify_sample > 0:
        rnd = random.Random(args.seed)
        k = min(args.verify_sample, len(layers))
        sample = sorted(rnd.sample(layers, k)) if k > 0 else []
        paths = [out_dir / f"{args.name_prefix}{lid}{args.name_suffix}.gguf" for lid in sample]
        logger.info(f"[batch] verify-sample k={k} layers={sample}")
        _verify_sample_paths(paths)

    logger.info("[batch] 全部结束。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
