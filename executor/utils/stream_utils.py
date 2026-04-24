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

import torch


class FakeContextManager:
    def __init__(self) -> None:
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass


def npu_stream_switch(switch_flag: bool, stream: torch.npu.Stream):
    if switch_flag:
        return torch.npu.stream(stream)
    else:
        return FakeContextManager()


def record_event(switch_flag: bool, events: tuple[torch.npu.Event], idx: int):
    """Records the specified NPU event if switch_flag is True."""
    if switch_flag:
        events[idx].record()


def wait_event(switch_flag: bool, events: tuple[torch.npu.Event], idx: int):
    """
    Waits for the specified NPU event to complete if switch_flag is True.

    Note: torch.npu.Event.wait() does NOT support passing a stream explicitly now.
    Internally, it uses torch.npu.current_stream().
    """
    if switch_flag:
        events[idx].wait()


def record_stream(switch_flag: bool, out: torch.Tensor, stream: torch.npu.Stream):
    """
    Conditionally tracks the tensor's lifecycle on a specific NPU stream
    to prevent premature memory deallocation during asynchronous operations.
    """
    if switch_flag:
        out.record_stream(stream)
