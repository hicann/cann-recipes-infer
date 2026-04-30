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
# pylint: disable=protected-access  # tests exercise _get_next_dp_rank round-robin counter

from executor.online.dispatcher import DPDispatcher


def _make(disaggregation_mode):
    return DPDispatcher(
        dp_size=4,
        router_port=0,
        pull_port=0,
        disaggregation_mode=disaggregation_mode,
    )


def test_get_next_dp_rank_round_robin_in_pd_mode():
    dispatcher = _make("PREFILL")
    first = dispatcher._get_next_dp_rank()
    second = dispatcher._get_next_dp_rank()
    assert first != second


def test_get_next_dp_rank_round_robin_in_normal_mode():
    dispatcher = _make("NONE")
    first = dispatcher._get_next_dp_rank()
    second = dispatcher._get_next_dp_rank()
    assert first != second


def test_get_next_dp_rank_advances_independently_of_payload():
    dispatcher = _make("PREFILL")
    first = dispatcher._get_next_dp_rank()
    second = dispatcher._get_next_dp_rank()
    assert first != second
