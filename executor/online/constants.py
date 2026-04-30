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

"""Shared constants for online inference modules."""

# ── Role names ──
ROLE_PREFILL = "PREFILL"
ROLE_DECODE = "DECODE"
ROLE_NONE = "NONE"

# ── HTTP ports ──
PREFILL_HTTP_PORT = 8001
DECODE_HTTP_PORT = 8100
ROUTER_HTTP_PORT = 8000
BOOTSTRAP_PORT = 18800

# ── ZMQ ports (server <-> DP Leader) ──
ZMQ_ROUTER_PORT = 5555
ZMQ_PULL_PORT = 5556

# When prefill and decode share the same host (single-machine PD disaggregation),
# decode must bind different ports to avoid conflicting with prefill.
_ZMQ_DECODE_PORT_OFFSET = 2


def zmq_ports_for_role(disaggregation_mode: str) -> tuple:
    """Return (router_port, pull_port) for the given disaggregation mode.

    Decode uses a fixed +2 offset so prefill and decode can coexist on one host.
    Both server.py (bind side) and online_inference.py (connect side) must call
    this function — never duplicate the offset logic inline.
    """
    offset = _ZMQ_DECODE_PORT_OFFSET if disaggregation_mode == ROLE_DECODE else 0
    return ZMQ_ROUTER_PORT + offset, ZMQ_PULL_PORT + offset
