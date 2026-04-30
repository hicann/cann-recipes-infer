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

"""Unit tests for scheduler batching behavior."""
# pylint: disable=protected-access  # tests inspect scheduler internals (_request_counter, _schedule_batch)

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from executor.core.scheduler.scheduler import Scheduler
from executor.core.config import SchedulerConfig
from executor.core.types_.types import Request, Batch


class MockTokenizer:
    """Mock tokenizer for testing."""
    
    def __init__(self):
        self.eos_token_id = 2
        self.pad_token_id = 0
        
    def __call__(self, text, **kwargs):
        """Mock tokenize call."""
        # Simple mock: return token IDs based on text length
        tokens = list(range(1, len(text) + 1))
        attention_mask = [1] * len(tokens)
        
        if kwargs.get('return_tensors') == 'pt':
            if kwargs.get('padding') == 'max_length':
                max_len = kwargs.get('max_length', 10)
                valid_len = min(len(tokens), max_len)
                tokens = tokens[:max_len] + [0] * (max_len - len(tokens))
                attention_mask = [1] * valid_len + [0] * (max_len - valid_len)

            return SimpleNamespace(
                input_ids=torch.tensor([tokens]),
                attention_mask=torch.tensor([attention_mask]),
            )
        return tokens
    
    def encode(self, text):
        """Mock encode."""
        return list(range(1, len(text) + 1))
    
    def decode(self, token_ids, skip_special_tokens=True):
        """Mock decode."""
        return f"decoded_{len(token_ids)}_tokens"


class MockKVCacheManager:
    """Mock KV cache manager for testing."""

    def allocate_slots(self, request_id, computed_tokens, num_new_tokens, lookahead_tokens=0):
        del request_id, computed_tokens, num_new_tokens, lookahead_tokens
        return True

    def free(self, request_id):
        del request_id

    def format_usage(self):
        return "0/0"

    def __bool__(self):
        return True


class MockEngine:
    """Mock engine for testing."""

    def __init__(self):
        self.tokenizer = MockTokenizer()
        self.kvcache_manager = MockKVCacheManager()
        self.next_n = 0

    def forward_batch(self, batch):
        """Mock forward pass."""
        next_tokens = {}
        for i, req in enumerate(batch.requests):
            # Return different token for each request
            next_tokens[req.request_id] = i + 10

        return {
            'next_tokens': next_tokens,
            'logits': torch.randn(len(batch.requests), 10, 100),
            'inference_time': 0.0,
        }


class TestSchedulerBatching:
    """Test suite for scheduler batching functionality."""
    
    @pytest.fixture
    def config(self):
        """Create scheduler config."""
        return SchedulerConfig(
            batch_size_per_dp_rank=4,
            max_new_tokens=32,
            max_prefill_tokens=64,
        )

    @pytest.fixture
    def scheduler(self, config):
        """Create scheduler instance."""
        tokenizer = MockTokenizer()
        return Scheduler(tokenizer, config)
    
    @pytest.fixture
    def engine(self):
        """Create mock engine."""
        return MockEngine()
    
    def test_scheduler_initialization(self, scheduler, config):
        """Test scheduler initializes correctly."""
        assert scheduler.config == config
        assert len(scheduler.waiting_queue) == 0
        assert len(scheduler.running_requests) == 0
        assert len(scheduler.finished_requests) == 0
        assert scheduler._request_counter == 0
    
    def test_add_request(self, scheduler):
        """Test adding requests to scheduler."""
        request_id = scheduler.add_request("test prompt")
        
        assert request_id == 0
        assert len(scheduler.waiting_queue) == 1
        assert scheduler._request_counter == 1
        
        # Add another request
        request_id2 = scheduler.add_request("another prompt")
        assert request_id2 == 1
        assert len(scheduler.waiting_queue) == 2
    
    def test_has_work(self, scheduler):
        """Test has_work method."""
        assert not scheduler.has_work()
        
        scheduler.add_request("prompt")
        assert scheduler.has_work()
        
        # Simulate request in running state
        request = Request(request_id=1, prompt="test")
        scheduler.running_requests[1] = request
        assert scheduler.has_work()
    
    def test_schedule_batch_prefill_only(self, scheduler, engine):
        """Test scheduling prefill-only batch."""
        scheduler.add_request("prompt1")
        scheduler.add_request("prompt2")
        batch = scheduler._schedule_batch(engine)

        assert batch is not None
        assert len(batch.requests) == 2
        assert batch.is_prefill == True

    def test_schedule_batch_decode_only(self, scheduler, engine):
        """Test scheduling decode-only batch."""
        # Add running requests (already in decode phase)
        req1 = Request(request_id=0, prompt="test1")
        req1.is_prefill_done = True
        req1.output_id_list = [10]

        req2 = Request(request_id=1, prompt="test2")
        req2.is_prefill_done = True
        req2.output_id_list = [11]

        scheduler.running_requests[0] = req1
        scheduler.running_requests[1] = req2
        batch = scheduler._schedule_batch(engine)

        assert batch is not None
        assert len(batch.requests) == 2
        assert batch.is_prefill == False
    
    def test_run_step_prefill(self, scheduler, engine):
        """Test running a prefill step."""
        scheduler.add_request("prompt1")
        scheduler.add_request("prompt2")
        
        step_output = scheduler.run_step(engine)
        
        assert step_output is not None
        assert len(step_output.next_tokens) == 2
        assert 0 in step_output.next_tokens
        assert 1 in step_output.next_tokens
        
        # Check requests moved to running
        assert len(scheduler.running_requests) == 2
    
    def test_run_step_decode(self, scheduler, engine):
        """Test running a decode step."""
        # Setup: requests already in running state
        req1 = Request(request_id=0, prompt="test1")
        req1.input_ids = torch.tensor([1, 2, 3])
        req1.is_prefill_done = True
        req1.output_id_list = [10]
        
        scheduler.running_requests[0] = req1
        
        step_output = scheduler.run_step(engine)
        
        assert step_output is not None
        assert 0 in step_output.next_tokens
    
    def test_request_completion(self, scheduler, engine):
        """Test request completion and result dispatch."""
        # Online mode finishes a request the moment output_id_list
        # crosses max_new_tokens; offline mode instead waits until the
        # scheduler has run max_new_tokens decode steps, which can't
        # happen in a single run_step call.
        scheduler.mode = "online"

        # Setup request already at completion length: max_new_tokens=32.
        req = Request(request_id=0, prompt="test")
        req.input_ids = torch.tensor([1, 2, 3])
        req.is_prefill_done = True
        req.output_id_list = list(range(32))

        scheduler.running_requests[0] = req

        step_output = scheduler.run_step(engine)

        # Request should be finished after this decode
        assert len(step_output.finished_requests) == 1
        assert 0 in step_output.finished_requests

        # Check request moved to finished
        assert 0 in scheduler.finished_requests
        assert 0 not in scheduler.running_requests

    def test_run_step_forced_prefill_builds_dummy_batch(self, scheduler):
        """Forced prefill phase should build a dummy batch without post-processing."""
        engine = Mock()
        engine.next_n = 0
        engine.kvcache_manager = MockKVCacheManager()
        engine.forward_batch.return_value = {
            'next_tokens': {},
            'logits': torch.randn(1, 1, 10),
            'inference_time': 0.0,
        }

        step_output = scheduler.run_step(engine, phase="prefill")

        assert step_output is None
        engine.forward_batch.assert_called_once()
        dummy_batch = engine.forward_batch.call_args.args[0]
        assert dummy_batch.is_dummy is True
        assert dummy_batch.is_prefill is True

    def test_run_step_forced_decode_builds_dummy_batch(self, scheduler):
        """Forced decode phase should build a dummy batch without post-processing."""
        engine = Mock()
        engine.next_n = 0
        engine.kvcache_manager = MockKVCacheManager()
        engine.forward_batch.return_value = {
            'next_tokens': {},
            'logits': torch.randn(1, 1, 10),
            'inference_time': 0.0,
        }

        step_output = scheduler.run_step(engine, phase="decode")

        assert step_output is None
        engine.forward_batch.assert_called_once()
        dummy_batch = engine.forward_batch.call_args.args[0]
        assert dummy_batch.is_dummy is True
        assert dummy_batch.is_prefill is False
    
    def test_prefill_priority_over_decode_batch(self, scheduler, engine):
        """Test scheduler prioritizes prefill over decode when both exist."""
        # Add waiting request (prefill)
        scheduler.add_request("new_prompt")

        # Add running request (decode)
        req = Request(request_id=1, prompt="old_prompt")
        req.input_ids = torch.tensor([1, 2, 3])
        req.is_prefill_done = True
        req.output_id_list = [10]
        scheduler.running_requests[1] = req

        batch = scheduler._schedule_batch(engine)

        assert batch is not None
        assert batch.is_prefill is True
        assert len(batch.requests) == 1

    def test_batch_size_limit(self, scheduler, engine):
        """Decode batches must respect ``batch_size_per_dp_rank``.

        Prefill is bounded by ``max_prefill_tokens`` (token budget) rather
        than request count, so this constraint only applies to decode.
        """
        config = scheduler.config
        for i in range(config.batch_size_per_dp_rank * 2):
            req = Request(request_id=i, prompt=f"prompt_{i}")
            req.is_prefill_done = True
            req.output_id_list = [10]
            scheduler.running_requests[i] = req

        batch = scheduler._schedule_batch(engine)

        assert batch is not None
        assert batch.is_prefill is False
        assert len(batch.requests) == config.batch_size_per_dp_rank
    
    def test_get_stats(self, scheduler):
        """Test scheduler statistics."""
        stats = scheduler.get_stats()
        
        assert stats['pending'] == 0
        assert stats['running'] == 0
        assert stats['finished'] == 0
        assert stats['total'] == 0
        
        scheduler.add_request("prompt1")
        scheduler.add_request("prompt2")
        
        stats = scheduler.get_stats()
        assert stats['pending'] == 2
        assert stats['total'] == 2
    
    def test_reset(self, scheduler):
        """Test scheduler reset."""
        scheduler.add_request("prompt1")
        req = Request(request_id=1, prompt="test")
        scheduler.running_requests[1] = req
        
        scheduler.reset()
        
        assert len(scheduler.waiting_queue) == 0
        assert len(scheduler.running_requests) == 0
        assert len(scheduler.finished_requests) == 0
        assert scheduler._request_counter == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
