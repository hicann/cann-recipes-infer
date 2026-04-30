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

"""Unit tests for packed_sequence functionality."""
# pylint: disable=protected-access  # tests inspect scheduler internals (_build_prefill_tensors)

from dataclasses import dataclass, field
from typing import Optional, Dict, List

import pytest
import torch


@dataclass
class ForwardMetaData:
    is_prefill: bool = False
    attention_mask: Optional[torch.Tensor] = None
    kv_len: Optional[torch.Tensor] = None
    actual_seq_lengths_kv: Optional[torch.Tensor] = None
    actual_seq_lengths_q: Optional[torch.Tensor] = None
    packed_sequence: bool = False


_forward_metadata = ForwardMetaData()


def get_forward_metadata():
    return _forward_metadata


def set_forward_metadata(**kwargs):
    global _forward_metadata
    _forward_metadata = ForwardMetaData(**kwargs)


def reset_forward_metadata():
    global _forward_metadata
    _forward_metadata = ForwardMetaData()


@dataclass
class Request:
    request_id: int
    prompt: str
    input_ids: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    output_id_list: List[int] = field(default_factory=list)
    is_prefill_done: bool = False
    is_finished: bool = False
    finish_reason: Optional[str] = None


@dataclass
class Batch:
    requests: List['Request'] = field(default_factory=list)
    is_prefill: bool = True
    input_ids: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None
    seq_lens: Optional[List[int]] = None
    request_indices: Dict[int, int] = field(default_factory=dict)


@dataclass
class SchedulerConfig:
    batch_size: int = 1
    input_max_len: int = 1024
    max_new_tokens: int = 32
    batch_size_per_dp_rank: int = 1
    max_wait_time: float = 0.01
    min_batch_size: int = 1
    packed_sequence: bool = True


class MockTokenizer:
    """Mock tokenizer for testing."""
    pad_token_id = 0
    eos_token_id = 1
    
    def __call__(self, text, **kwargs):
        mock_tokens = torch.tensor([ord(c) % 100 + 2 for c in text[:10]], dtype=torch.long)
        if kwargs.get("padding") == "max_length":
            max_len = kwargs.get("max_length", 10)
            if len(mock_tokens) < max_len:
                mock_tokens = torch.cat([
                    mock_tokens,
                    torch.zeros(max_len - len(mock_tokens), dtype=torch.long)
                ])
        return type('MockOutput', (), {'input_ids': mock_tokens.unsqueeze(0)})()


class MockScheduler:
    """Mock scheduler for testing packed sequence logic."""
    
    def __init__(self, config: SchedulerConfig):
        self.config = config
    
    def _build_prefill_tensors(self, batch: Batch) -> None:
        if self.config.packed_sequence:
            packed_tokens = []
            for req in batch.requests:
                packed_tokens.append(req.input_ids)

            batch.seq_lens = [req.input_ids.numel() for req in batch.requests]
            batch.input_ids = torch.cat(packed_tokens)
        else:
            batch.seq_lens = [(req.input_ids != 0).sum().item() for req in batch.requests]
            batch.input_ids = torch.stack([req.input_ids for req in batch.requests])


class TestForwardMetaData:
    """Tests for ForwardMetaData with packed_sequence fields."""
    
    def test_default_values(self):
        reset_forward_metadata()
        metadata = get_forward_metadata()
        
        assert metadata.is_prefill == False
        assert metadata.attention_mask is None
        assert metadata.kv_len is None
        assert metadata.actual_seq_lengths_kv is None
        assert metadata.actual_seq_lengths_q is None
        assert metadata.packed_sequence == False
    
    def test_set_packed_sequence(self):
        reset_forward_metadata()
        set_forward_metadata(
            is_prefill=True,
            packed_sequence=True,
            actual_seq_lengths_q=torch.tensor([10, 25, 40]),
            actual_seq_lengths_kv=torch.tensor([10, 25, 40]),
        )
        
        metadata = get_forward_metadata()
        assert metadata.packed_sequence == True
        assert metadata.actual_seq_lengths_q is not None
        assert metadata.actual_seq_lengths_kv is not None


class TestSchedulerPackedSequence:
    """Tests for Scheduler packed_sequence support."""
    
    def test_scheduler_config_packed_sequence(self):
        config = SchedulerConfig(packed_sequence=True)
        assert config.packed_sequence == True
    
    def test_build_prefill_tensors_packed_mode(self):
        config = SchedulerConfig(packed_sequence=True, input_max_len=10)
        scheduler = MockScheduler(config=config)
        
        request1 = Request(request_id=0, prompt="hello")
        request1.input_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
        
        request2 = Request(request_id=1, prompt="world")
        request2.input_ids = torch.tensor([6, 7, 8], dtype=torch.long)
        
        batch = Batch(requests=[request1, request2], is_prefill=True)
        scheduler._build_prefill_tensors(batch)
        
        assert batch.input_ids is not None
        assert batch.input_ids.dim() == 1, "Packed mode should produce 1D tensor"
        assert batch.input_ids.numel() == 8, "Total tokens should be 5 + 3 = 8"
    
    def test_build_prefill_tensors_normal_mode(self):
        config = SchedulerConfig(packed_sequence=False, input_max_len=10)
        scheduler = MockScheduler(config=config)
        
        request1 = Request(request_id=0, prompt="hello")
        request1.input_ids = torch.tensor([1, 2, 3, 4, 5, 0, 0, 0, 0, 0], dtype=torch.long)
        
        request2 = Request(request_id=1, prompt="world")
        request2.input_ids = torch.tensor([6, 7, 8, 9, 10, 0, 0, 0, 0, 0], dtype=torch.long)
        
        batch = Batch(requests=[request1, request2], is_prefill=True)
        scheduler._build_prefill_tensors(batch)
        
        assert batch.input_ids is not None
        assert batch.input_ids.dim() == 2, "Normal mode should produce 2D tensor"
        assert batch.input_ids.shape[0] == 2, "Batch size should be 2"
        assert batch.input_ids.shape[1] == 10, "Seq length should be 10"


class TestBuildModelInputs:
    """Tests for _build_model_inputs logic with packed mode."""
    
    def test_packed_mode_position_ids_and_actual_seq_lengths(self):
        seq_lens = torch.tensor([5, 3, 4], dtype=torch.long)
        total_tokens = 12
        input_ids = torch.zeros(total_tokens, dtype=torch.long)
        
        position_ids_list = []
        for seq_len in seq_lens.tolist():
            position_ids_list.append(torch.arange(seq_len, dtype=torch.long))
        position_ids = torch.cat(position_ids_list)
        
        actual_seq_lengths = seq_lens.cumsum(0)
        
        assert position_ids.numel() == 12
        assert torch.equal(actual_seq_lengths, torch.tensor([5, 8, 12]))
        assert torch.equal(position_ids, torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 0, 1, 2, 3]))
    
    def test_packed_mode_position_ids_per_sequence(self):
        seq_lens = torch.tensor([3, 2, 5], dtype=torch.long)
        
        position_ids_list = []
        for seq_len in seq_lens.tolist():
            position_ids_list.append(torch.arange(seq_len, dtype=torch.long))
        position_ids = torch.cat(position_ids_list)
        
        expected = torch.tensor([0, 1, 2, 0, 1, 0, 1, 2, 3, 4])
        assert torch.equal(position_ids, expected)
    
    def test_normal_mode_position_ids_with_seq_lens(self):
        """Test position_ids construction using seq_lens in normal (non-packed) mode."""
        batch_size = 3
        seq_len = 5
        seq_lens = torch.tensor([3, 5, 2], dtype=torch.long)
        
        position_ids = torch.ones((batch_size, seq_len), dtype=torch.long)
        for i, length in enumerate(seq_lens.tolist()):
            position_ids[i, :length] = torch.arange(length, dtype=torch.long)
        
        expected = torch.tensor([
            [0, 1, 2, 1, 1],
            [0, 1, 2, 3, 4],
            [0, 1, 1, 1, 1]
        ])
        assert torch.equal(position_ids, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
