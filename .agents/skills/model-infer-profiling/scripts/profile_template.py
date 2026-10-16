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
    Verified on torch_npu 2.8.0.post2 with CANN 8.x/9.x on Atlas A2 / A3.
    On torch_npu <= 2.7.x the trailing prof.step() is mandatory, otherwise
    trace_view.json is truncated and the CSV parsers fail.

How to use:
    1. Replace `build_model()` / `run_prefill()` / `warm_one_decode()` / `run_one_decode()`
       with your own model loading and inference code
    2. Choose PHASE: "decode" | "prefill" | "both" | "separate"
    3. Run: python profile_template.py
    4. View: tensorboard --logdir=./prof

Three core rules, matching the three key points in SKILL.md:
    1. ExperimentalConfig(Level1 + PipeUtilization) is mandatory; without it kernel_details has only 9 columns
    2. Total prof.step() calls must be >= schedule_warmup + active + 1, so call it one extra time at the end
    3. Keep the first torch.compile / torchair / JIT warmup outside the profiler; whether prefill and decode
       themselves are captured is up to the phase you pick
"""

import os
import time
import torch
import torch_npu

# ============ Configuration ============
SAVE_PATH = "./prof"                  # Directory the profiler writes results to
PHASE = "decode"                       # "prefill" | "decode" | "both" | "separate"
DECODE_STEPS = 30                      # Decode steps actually run in the loop when PHASE includes decode
SCHEDULE_WARMUP = 0                    # Warmup inside the profiler schedule; keep 0 and warm the graph outside
EXTRA_STEP_AT_END = True               # True: one extra prof.step() after the loop to flush the last step


# ============ Profiler setup ============

def create_profiler(save_path, warmup, active):
    """Create the NPU profiler. Level1 + PipeUtilization is what yields 47-column kernel_details."""
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
    """One decode step used as warmup: this is where torchair / torch.compile actually builds the graph."""
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

def _assert_step_budget(n_active, n_steps_called):
    required = SCHEDULE_WARMUP + n_active + 1
    if n_steps_called < required:
        raise RuntimeError(
            f"prof.step() 总次数 {n_steps_called} < {required} "
            f"(schedule_warmup+active+1) -- 会触发 Incorrect schedule warning。"
            f"请把 EXTRA_STEP_AT_END 设为 True 或提高步数。"
        )


def profile_decode_only(model, x, device):
    """Mode A: capture the decode steps only, in a single trace."""
    active = DECODE_STEPS
    _assert_step_budget(active, DECODE_STEPS + (1 if EXTRA_STEP_AT_END else 0))

    prof_ctx = create_profiler(SAVE_PATH, SCHEDULE_WARMUP, active)
    print(f"\n[PHASE=decode] schedule(warmup={SCHEDULE_WARMUP}, active={active}); "
          f"prof.step() total = {DECODE_STEPS + (1 if EXTRA_STEP_AT_END else 0)}")
    with prof_ctx as prof:
        for step in range(DECODE_STEPS):
            t0 = time.time()
            run_one_decode(model, x, device)
            print(f"  decode step {step+1}/{DECODE_STEPS}: {(time.time()-t0)*1000:.2f} ms")
            prof.step()
        if EXTRA_STEP_AT_END:
            prof.step()


def profile_prefill_only(model, x, device):
    """Mode B: capture the single prefill step only."""
    active = 1
    _assert_step_budget(active, 1 + (1 if EXTRA_STEP_AT_END else 0))

    prof_ctx = create_profiler(SAVE_PATH, SCHEDULE_WARMUP, active)
    print(f"\n[PHASE=prefill] schedule(warmup={SCHEDULE_WARMUP}, active={active}); "
          f"prof.step() total = {1 + (1 if EXTRA_STEP_AT_END else 0)}")
    with prof_ctx as prof:
        t0 = time.time()
        run_prefill(model, x, device)
        print(f"  prefill step 1/1: {(time.time()-t0)*1000:.2f} ms")
        prof.step()
        if EXTRA_STEP_AT_END:
            prof.step()


def profile_both_in_one_trace(model, x, device):
    """Mode C: capture prefill and decode together, on one timeline in one trace."""
    active = 1 + DECODE_STEPS
    total_steps = 1 + DECODE_STEPS + (1 if EXTRA_STEP_AT_END else 0)
    _assert_step_budget(active, total_steps)

    prof_ctx = create_profiler(SAVE_PATH, SCHEDULE_WARMUP, active)
    print(f"\n[PHASE=both] schedule(warmup={SCHEDULE_WARMUP}, active={active}); "
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
        if EXTRA_STEP_AT_END:
            prof.step()


def profile_separate(model, x, device):
    """Mode D: capture prefill and decode in two separate runs, each into its own save_path."""
    # prefill
    prefill_path = os.path.join(SAVE_PATH, "prefill")
    with create_profiler(prefill_path, SCHEDULE_WARMUP, 1) as prof:
        run_prefill(model, x, device)
        prof.step()
        if EXTRA_STEP_AT_END:
            prof.step()
    print(f"[PHASE=separate] prefill trace -> {prefill_path}")

    # decode
    decode_path = os.path.join(SAVE_PATH, "decode")
    with create_profiler(decode_path, SCHEDULE_WARMUP, DECODE_STEPS) as prof:
        for step in range(DECODE_STEPS):
            run_one_decode(model, x, device)
            prof.step()
        if EXTRA_STEP_AT_END:
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

    # 2) Graph compile and JIT warmup, outside the profiler. This is what pollutes a
    #    trace, not prefill itself
    print("[2/3] 图编译 / JIT 预热（跑一次 prefill + 一次 decode 暖图；不进 profiler 窗口）...")
    run_prefill(model, x, device)
    warm_one_decode(model, x, device)

    # 3) Enter the profiler and collect
    print(f"[3/3] 开始采集 (PHASE={PHASE})...")
    PHASES[PHASE](model, x, device)

    print("\n" + "=" * 60)
    print("采集完成")
    print("=" * 60)
    print(f"结果保存在: {os.path.abspath(SAVE_PATH)}")
    print("建议跑 SKILL.md「产物自检清单」确认 kernel_details 是 47 列、"
          "trace_view 以 `]` 结尾、op/api_statistic 存在。")


if __name__ == "__main__":
    main()
