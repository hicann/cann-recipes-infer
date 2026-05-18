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
# pylint: disable=protected-access  # tests inspect scheduler internals (_bootstrap_failures)

from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from executor.core.config import SchedulerConfig
from executor.online.kv_transfer import KVPoll
from executor.online.scheduler.prefill import (
    PrefillDisaggScheduler,
)


class MockTokenizer:
    def __call__(self, text, **kwargs):
        del kwargs, text
        return SimpleNamespace(input_ids=torch.tensor([[1, 2, 3]]))


@pytest.fixture
def tokenizer():
    return MockTokenizer()


@pytest.fixture
def config():
    return SchedulerConfig(batch_size_per_dp_rank=1, max_new_tokens=8)


def _make_scheduler(tokenizer, config):
    return PrefillDisaggScheduler(
        tokenizer=tokenizer,
        config=config,
        kv_transfer_manager=Mock(attn_cp_rank=0),
        kv_cache_manager=Mock(),
    )


def test_bootstrapped_request_moves_to_waiting_queue(tokenizer, config):
    scheduler = _make_scheduler(tokenizer, config)
    req = Mock()
    req.disagg_kv_sender.poll_and_all_reduce.return_value = KVPoll.WaitingForInput
    scheduler.bootstrap_queue = deque([req])
    scheduler.advance_queues_consensus(engine=Mock())
    assert list(scheduler.waiting_queue) == [req]
    assert len(scheduler.bootstrap_queue) == 0
    req.disagg_kv_sender.init.assert_called_once()


def test_failed_bootstrap_request_does_not_return_to_waiting_queue(tokenizer, config):
    scheduler = _make_scheduler(tokenizer, config)
    req = Mock()
    req.request_id = 7
    req.disagg_kv_sender.poll_and_all_reduce.return_value = KVPoll.Failed
    req.disagg_kv_sender.conclude_state = None
    scheduler.bootstrap_queue = deque([req])
    scheduler.advance_queues_consensus(engine=Mock())
    assert len(scheduler.waiting_queue) == 0
    assert 7 in scheduler.finished_requests
    assert scheduler._bootstrap_failures == [7]
