# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

"""Profiler context manager for NPU profiling using torch_npu.profiler.

This module provides a context manager for profiling NPU operations during
inference. When profiling is disabled, the context manager becomes a no-op.
"""

import os
from enum import Enum

import torch_npu
from executor.core.config.inference_config import DEFAULT_PROFILER_CONFIG


class ProfilerPhase(Enum):
    MM_ENCODE = "mm_encode"
    PREFILL = "prefill"
    DECODE = "decode"


class FakeContextManager:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    @staticmethod
    def step():
        return

    @staticmethod
    def stop():
        return


def create_profiler(
    enable_profiler=False,
    profile_save_path="prof",
    active=DEFAULT_PROFILER_CONFIG["decode"]["active"],
    repeat=1,
    skip_first=DEFAULT_PROFILER_CONFIG["decode"]["skip_first"],
):
    if enable_profiler:
        os.makedirs(profile_save_path, exist_ok=True)
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization
        )
        profiler = torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.NPU,
                torch_npu.profiler.ProfilerActivity.CPU,
            ],
            with_stack=False,
            record_shapes=False,
            profile_memory=False,
            experimental_config=experimental_config,
            schedule=torch_npu.profiler.schedule(
                wait=0,
                warmup=0,
                active=active,
                repeat=repeat,
                skip_first=skip_first
            ),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profile_save_path)
        )
    else:
        profiler = FakeContextManager()

    return profiler


class ProfilerManager:
    """
    Profiler manager for profiling using torch_npu.profiler.
    This class provides a manager for profiling NPU operations during inference.
    """
    def __init__(self, enable_profiler, profile_save_path, profiler_config=None):
        self.enable_profiler = enable_profiler
        self.profile_save_path = profile_save_path
        self.profiler_config = profiler_config
        self.profiler_status = None
        self.current_profiler = FakeContextManager()
        self.decode_profile_steps = 0
        self.decode_profile_complete = not enable_profiler

    def check_if_update(self, cur_status: ProfilerPhase) -> bool:
        """Return whether the requested phase should replace the active profiler."""
        # Reuse the profiler across consecutive batches in the same phase.
        if not self.enable_profiler or self.profiler_status == cur_status:
            return False
        # Preserve decode-only behavior: profiling must start in an earlier phase.
        if self.profiler_status is None and cur_status == ProfilerPhase.DECODE:
            return False
        # Once decode profiling starts, do not switch to another phase.
        if self.profiler_status == ProfilerPhase.DECODE:
            return False
        return True

    def set_status(self, cur_status: ProfilerPhase):
        if not isinstance(cur_status, ProfilerPhase):
            raise TypeError("cur_status must be a ProfilerPhase")
        if not self.check_if_update(cur_status):
            return
        if self.profiler_status is not None:
            self.current_profiler.stop()
        profiler_kwargs = {"repeat": 1, **self.profiler_config[cur_status.value]}
        self.current_profiler = create_profiler(
            enable_profiler=True,
            profile_save_path=os.path.join(self.profile_save_path, "prof", cur_status.value),
            **profiler_kwargs,
        )
        self.current_profiler.start()
        self.profiler_status = cur_status
        if cur_status == ProfilerPhase.DECODE:
            self.decode_profile_steps = 0
            self.decode_profile_complete = False

    def step(self):
        self.current_profiler.step()
        if self.enable_profiler and self.profiler_status == ProfilerPhase.DECODE:
            self.decode_profile_steps += 1
            profile_window_steps = sum(self.profiler_config["decode"].values())
            # Timing starts after the skipped and active profiling steps.
            self.decode_profile_complete = (
                self.decode_profile_steps >= profile_window_steps
            )

    @property
    def timing_ready(self) -> bool:
        """Whether decode timing is outside the profiling window."""
        return self.decode_profile_complete

    def __del__(self):
        if self.enable_profiler:
            self.current_profiler.stop()
