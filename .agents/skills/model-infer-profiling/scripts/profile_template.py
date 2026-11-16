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
"""
NPU profiling collection template.

Environment:
    Requires torch and torch_npu on an Ascend NPU, with the toolkit sourced:
        source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash

How to use:
    1. Replace `build_model()` / `run_prefill()` / `warm_one_decode()` / `run_one_decode()`
       with your own model loading and inference code
    2. Choose PHASE: "decode" | "prefill" | "both" | "separate"
    3. Run: python profile_template.py
    4. View: tensorboard --logdir=./prof

Collection settings:
    Level1 + PipeUtilization collects pipeline utilization metrics.
    The four modes use warmup=0 and advance the schedule once per inference step.
    Compile / JIT warmup runs outside the profiler for steady-state collection.
"""

import os
import time
import torch
import torch_npu

# ============ Configuration ============
SAVE_PATH = "./prof"                  # Directory the profiler writes results to
PHASE = "decode"                       # "prefill" | "decode" | "both" | "separate"
DECODE_STEPS = 30                      # Decode steps actually run in the loop when PHASE includes decode


# ============ Profiler setup ============

def create_profiler(save_path, warmup, active):
    """Create a profiler with pipeline utilization metrics."""
    os.makedirs(save_path, exist_ok=True)
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
    )
    return torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.NPU,
            torch_npu.profiler.ProfilerActivity.CPU,
        ],
        with_stack=False,
        record_shapes=False,
        profile_memory=False,
        experimental_config=experimental_config,
        schedule=torch_npu.profiler.schedule(
            wait=0, warmup=warmup, active=active, repeat=1, skip_first=0,
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(save_path),
    )


# ============ Replace the four functions below with your own code ============

def build_model():
    """Load or initialize the model; return it together with whatever state the run needs."""
    device = torch.device("npu:0")
    model = torch.nn.Linear(1000, 1000).to(device)
    x = torch.randn(32, 1000, device=device)
    return model, x, device


def run_prefill(model, x, device):
    """One prefill step. In a real workload this covers the full-sequence forward, KVCache init and so on."""
    with torch.no_grad():
        y = model(x)
    torch.npu.synchronize()
    return y


def warm_one_decode(model, x, device):
    """Warm up the target decode path before steady-state collection."""
    with torch.no_grad():
        y = model(x)
    torch.npu.synchronize()
    return y


def run_one_decode(model, x, device):
    """One steady-state decode step."""
    with torch.no_grad():
        y = model(x)
    torch.npu.synchronize()
    return y


# ============ Main flow ============

def profile_decode_only(model, x, device):
    """Mode A: capture the decode steps only, in a single trace."""
    active = DECODE_STEPS
    prof_ctx = create_profiler(SAVE_PATH, warmup=0, active=active)
    print(f"\n[PHASE=decode] schedule(warmup=0, active={active}); "
          f"prof.step() total = {DECODE_STEPS}")
    with prof_ctx as prof:
        for step in range(DECODE_STEPS):
            t0 = time.time()
            run_one_decode(model, x, device)
            print(f"  decode step {step+1}/{DECODE_STEPS}: {(time.time()-t0)*1000:.2f} ms")
            prof.step()


def profile_prefill_only(model, x, device):
    """Mode B: capture the single prefill step only."""
    active = 1
    prof_ctx = create_profiler(SAVE_PATH, warmup=0, active=active)
    print(f"\n[PHASE=prefill] schedule(warmup=0, active={active}); "
          "prof.step() total = 1")
    with prof_ctx as prof:
        t0 = time.time()
        run_prefill(model, x, device)
        print(f"  prefill step 1/1: {(time.time()-t0)*1000:.2f} ms")
        prof.step()


def profile_both_in_one_trace(model, x, device):
    """Mode C: capture prefill and decode together, on one timeline in one trace."""
    active = 1 + DECODE_STEPS
    total_steps = 1 + DECODE_STEPS
    prof_ctx = create_profiler(SAVE_PATH, warmup=0, active=active)
    print(f"\n[PHASE=both] schedule(warmup=0, active={active}); "
          f"prof.step() total = {total_steps}")
    with prof_ctx as prof:
        t0 = time.time()
        run_prefill(model, x, device)
        print(f"  prefill: {(time.time()-t0)*1000:.2f} ms")
        prof.step()
        for step in range(DECODE_STEPS):
            t0 = time.time()
            run_one_decode(model, x, device)
            print(f"  decode step {step+1}/{DECODE_STEPS}: {(time.time()-t0)*1000:.2f} ms")
            prof.step()


def profile_separate(model, x, device):
    """Mode D: capture prefill and decode in two separate runs, each into its own save_path."""
    # prefill
    prefill_path = os.path.join(SAVE_PATH, "prefill")
    with create_profiler(prefill_path, warmup=0, active=1) as prof:
        run_prefill(model, x, device)
        prof.step()
    print(f"[PHASE=separate] prefill trace -> {prefill_path}")

    # decode
    decode_path = os.path.join(SAVE_PATH, "decode")
    with create_profiler(decode_path, warmup=0, active=DECODE_STEPS) as prof:
        for step in range(DECODE_STEPS):
            run_one_decode(model, x, device)
            prof.step()
    print(f"[PHASE=separate] decode trace  -> {decode_path}")


PHASES = {
    "decode": profile_decode_only,
    "prefill": profile_prefill_only,
    "both": profile_both_in_one_trace,
    "separate": profile_separate,
}


def main():
    print("=" * 60)
    print("NPU Profiling 采集")
    print("=" * 60)

    if PHASE not in PHASES:
        raise ValueError(f"PHASE 必须是 {list(PHASES)} 之一，得到: {PHASE}")

    # 1) Build the model and prepare inputs, outside the profiler
    print("\n[1/3] 构建模型...")
    model, x, device = build_model()

    # 2) Warm up the selected paths outside the profiler for steady-state collection.
    print("[2/3] 在采集窗口外准备输入并预热目标执行路径...")
    run_prefill(model, x, device)
    if PHASE != "prefill":
        warm_one_decode(model, x, device)

    # 3) Enter the profiler and collect
    print(f"[3/3] 开始采集 (PHASE={PHASE})...")
    PHASES[PHASE](model, x, device)

    print("\n" + "=" * 60)
    print("采集完成")
    print("=" * 60)
    print(f"结果保存在: {os.path.abspath(SAVE_PATH)}")
    print("按 SKILL.md「产物自检清单」核对目标阶段、rank、所需字段和 trace 完整性。")


if __name__ == "__main__":
    main()
