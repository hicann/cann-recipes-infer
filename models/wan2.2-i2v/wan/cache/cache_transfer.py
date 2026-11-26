# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

"""Request-scoped Taylor factor ownership and bounded Host/NPU transfers.

The model adapter owns scheduling. This module never guesses the next expert,
CFG branch or diffusion step. A lease is read-only and valid until release;
publish transfers ownership of the supplied factors to the manager. Callers
must obtain an update credit *before* allocating the new residual/derivatives.
CPU supports the resident path for logic tests, never emulated NPU transfers.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import ctypes
import math
import threading
import time

import torch


CacheKey = tuple[str, str, int, str, int]
Factors = dict[int, torch.Tensor]
MAX_CACHE_BYTES = (1 << 63) - 1


def cache_byte_requirements(feature_bytes, bundle_bytes, resident_bytes, offload):
    """Return the minimum ordinary Host, pinned and NPU cache bytes."""
    pinned_bytes = 3 * bundle_bytes if offload else 0
    host_bytes = resident_bytes + bundle_bytes if offload else 0
    device_factors = pinned_bytes if offload else resident_bytes + bundle_bytes
    device_bytes = device_factors + 4 * feature_bytes
    if max(resident_bytes, host_bytes, pinned_bytes, device_bytes) > MAX_CACHE_BYTES:
        raise ValueError("Taylor cache byte requirement exceeds signed 64-bit range")
    return host_bytes, pinned_bytes, device_bytes


@dataclass(eq=False)
class FactorsLease:
    key: CacheKey
    generation: int
    factors: Factors
    slot_id: int | None = None
    _owner: object = field(default=None, repr=False)
    _released: bool = field(default=False, repr=False)


@dataclass(eq=False)
class UpdateCredit:
    _owner: object = field(repr=False)
    _submitted: bool = False
    _cancelled: bool = False


@dataclass
class _Entry:
    generation: int
    factors: Factors


@dataclass
class _Slot:
    slot_id: int
    host: torch.Tensor
    device: torch.Tensor
    key: CacheKey | None = None
    generation: int = 0
    count: int = 0
    ready: object = None
    ready_ticket: int | None = None
    ready_handle: int | None = None
    last_use: object = None
    leased: bool = False
    prefetched: bool = False


class WanCacheTransfer:
    """Bounded factor store; use from one inference thread and one device.

    offload=True uses two H2D slots, one D2H staging bundle and one producer
    credit. A single Host writer retains D2H source tensors until completion,
    then copies staging into ordinary Host memory before returning the credit.
    Retaining the credit through Host memcpy is deliberately conservative.
    """

    def __init__(
        self, *, request_id, device, feature_shape, num_groups, max_order,
        offload=False, prefetch=False, max_inflight=2,
        host_cache_budget_bytes=None, pinned_budget_bytes=None,
        device_cache_budget_bytes=None,
    ):
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a nonempty string")
        if type(offload) is not bool or type(prefetch) is not bool:
            raise ValueError("offload and prefetch must be bool")
        if prefetch and not offload:
            raise ValueError("prefetch requires offload")
        if type(max_inflight) is not int or max_inflight != 2:
            raise ValueError("max_inflight must be 2")
        if type(max_order) is not int or not 0 <= max_order <= 3:
            raise ValueError("max_order must be an integer in [0, 3]")
        if type(num_groups) is not int or num_groups < 1:
            raise ValueError("num_groups must be a positive integer")
        if len(feature_shape) != 3 or any(type(x) is not int or x < 1 for x in feature_shape):
            raise ValueError("feature_shape must contain three positive integers")
        if str(device).startswith("npu") and not hasattr(torch, "npu"):
            import torch_npu  # noqa: F401
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "npu") or (offload and self.device.type != "npu"):
            raise ValueError("offload requires NPU; resident logic supports CPU or NPU")
        self._npu = torch.npu if self.device.type == "npu" else None
        if self._npu is not None and self.device.index is None:
            self.device = torch.device("npu", self._npu.current_device())
        self.request_id = request_id
        self.feature_shape = tuple(feature_shape)
        self.num_groups = num_groups
        self.max_order = max_order
        self.offload = offload
        self.prefetch_enabled = prefetch
        self.feature_bytes = math.prod(self.feature_shape) * 4
        self.bundle_bytes = (max_order + 1) * self.feature_bytes
        self.canonical_bytes = 2 * num_groups * self.bundle_bytes
        required_host, required_pinned, required_device = cache_byte_requirements(
            self.feature_bytes, self.bundle_bytes, self.canonical_bytes, offload
        )
        requirements = dict(
            host_cache_budget_bytes=required_host,
            pinned_budget_bytes=required_pinned,
            device_cache_budget_bytes=required_device,
        )
        supplied = dict(
            host_cache_budget_bytes=host_cache_budget_bytes,
            pinned_budget_bytes=pinned_budget_bytes,
            device_cache_budget_bytes=device_cache_budget_bytes,
        )
        self.budgets = {}
        for name, needed in requirements.items():
            value = supplied[name]
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be null or a positive integer")
            if value is not None and value < needed:
                raise ValueError(f"{name}: need {needed} bytes, configured {value}")
            self.budgets[name] = needed if value is None else value
        self._entries = {}
        self._pending = {}
        self._generations = {}
        self._leases = {}
        self._retired = []
        self._slots = []
        self._d2h_host = None
        self._h2d_stream = None
        self._d2h_stream = None
        self._writer = None
        self._future = None
        self._credit = None
        self._current_key = None
        self._closed = False
        self._error = None
        self._drain_failure = None
        self._failed_sources = []
        self._timings = []
        self._ready_ticket = 0
        self._lock = threading.RLock()
        self._thread_id = threading.get_ident()
        self._stats = dict(
            h2d_bytes=0, d2h_bytes=0, h2d_count=0, d2h_count=0,
            prefetch_count=0, demand_count=0, history_hit=0, prefetch_hit=0, h2d_time_ms=0.0,
            d2h_time_ms=0.0, device_wait_time_ms=0.0, host_wait_seconds=0.0,
            peak_host_cache_bytes=0, peak_pinned_bytes=0,
            peak_device_factor_bytes=0, updates=0, failed=False,
            event_identity_checks=0, event_identity_unsupported=0,
        )
        try:
            if offload:
                self._allocate_transfer_pool()
        except BaseException:
            self.close(error=True)
            raise

    def _allocate_transfer_pool(self):
        shape = (self.max_order + 1,) + self.feature_shape
        with self._npu.device(self.device):
            self._h2d_stream = self._npu.Stream(device=self.device)
            self._d2h_stream = self._npu.Stream(device=self.device)
            for index in range(2):
                self._slots.append(_Slot(
                    index, torch.empty(shape, dtype=torch.float32, pin_memory=True),
                    torch.empty(shape, dtype=torch.float32, device=self.device),
                ))
            self._d2h_host = torch.empty(shape, dtype=torch.float32, pin_memory=True)
            # Pool allocations belong to the current stream. A cached device
            # allocation may still have earlier current-stream work queued.
            self._h2d_stream.wait_event(self._event(self._npu.current_stream(self.device)))
        self._writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="wan-cache-host")
        self._stats["peak_pinned_bytes"] = 3 * self.bundle_bytes
        self._stats["peak_device_factor_bytes"] = 2 * self.bundle_bytes

    def _check_open(self):
        if threading.get_ident() != self._thread_id:
            raise RuntimeError("cache manager must be used by its inference thread")
        if self._closed:
            raise RuntimeError("cache manager is closed")
        if self._error is not None:
            raise RuntimeError("cache transfer failed") from self._error

    def _check_key(self, key):
        self._check_open()
        if not isinstance(key, tuple) or len(key) != 5:
            raise ValueError("cache key must be (request, expert, epoch, CFG, group)")
        request, expert, epoch, cfg, group = key
        if request != self.request_id or expert not in ("high", "low") or cfg not in ("cond", "uncond"):
            raise ValueError("cache key has a foreign request or unsupported expert/CFG")
        if type(epoch) is not int or epoch < 0 or type(group) is not int or not 0 <= group < self.num_groups:
            raise ValueError("cache epoch/group out of range")

    def _event(self, stream, timing=False):
        event = self._npu.Event(enable_timing=timing)
        event.record(stream)
        return event

    def _range(self, operation, key, generation):
        return torch.profiler.record_function(
            f"WanTaylor/{operation}/{key[0]}/{key[1]}:{key[2]}/{key[3]}/g{key[4]}/v{generation}"
        )

    @staticmethod
    def _native_ready_handle(event):
        # npu_event is the Host-side opaque ACL handle, not a device event ID.
        # Reading it after record returns does not query or synchronize NPU work.
        try:
            handle = event.npu_event
        except (AttributeError, NotImplementedError):
            return None
        if type(handle) is not int or handle <= 0:
            raise RuntimeError("cache ready event has an invalid native handle")
        return handle

    def _ready_range(self, operation, key, generation, slot_id, ticket, handle=None):
        name = (
            f"WanTaylorEvent/{operation}/{key[0]}/{key[1]}:{key[2]}/{key[3]}"
            f"/g{key[4]}/v{generation}/slot{slot_id}/ticket{ticket}"
        )
        if operation != "ReadyRecord":
            name += f"/acl{hex(handle) if handle is not None else 'unresolved'}"
        return torch.profiler.record_function(name)

    def _record_ready(self, slot, key, generation):
        self._ready_ticket += 1
        ticket = self._ready_ticket
        ready = self._npu.Event(enable_timing=False)
        with self._ready_range("ReadyRecord", key, generation, slot.slot_id, ticket):
            ready.record(self._h2d_stream)
        # Publish the new record lifecycle before inspecting its native handle.
        # If inspection fails, close still owns the event and drains its stream.
        slot.ready, slot.ready_ticket, slot.ready_handle = ready, ticket, None
        slot.ready_handle = self._native_ready_handle(ready)
        with self._ready_range(
            "ReadyBind", key, generation, slot.slot_id, ticket, slot.ready_handle
        ):
            pass

    def _wait_ready(self, compute, slot, key):
        if slot.ready_ticket is None:
            raise RuntimeError("cache ready event has no record ticket")
        handle = self._native_ready_handle(slot.ready)
        supported = handle is not None and slot.ready_handle is not None
        if supported and handle != slot.ready_handle:
            raise RuntimeError("cache ready event native handle changed after record")
        # A backend without npu_event remains usable, but cannot establish a
        # native identity match. Never substitute Python object identity.
        if not supported:
            handle = None
        with self._ready_range(
            "ConsumerWait", key, slot.generation, slot.slot_id, slot.ready_ticket, handle
        ):
            compute.wait_event(slot.ready)
        # Counts successful waits with verified or unresolved native identity;
        # native addresses appear only in local profiler ranges, never stats.
        self._stats["event_identity_checks" if supported else "event_identity_unsupported"] += 1

    def _wait_host(self, wait):
        start = time.perf_counter()
        try:
            return wait()
        finally:
            self._stats["host_wait_seconds"] += time.perf_counter() - start

    def _finish_future(self, future):
        if future is None:
            return
        try:
            self._wait_host(future.result)
        except BaseException as error:
            self._error = error
            raise
        for key in list(self._pending):
            if self._pending[key] is future:
                del self._pending[key]

    def has_history(self, key):
        self._check_key(key)
        return key in self._entries or key in self._pending

    def reserve_update(self):
        """Acquire before allocating *any* new residual/derivative tensor."""
        self._check_open()
        if self._credit is not None and not self._credit._submitted:
            raise RuntimeError("an update credit is already reserved")
        self._finish_future(self._future)
        self._future = None
        for factors, event in self._retired:
            if event is not None:
                self._wait_host(event.synchronize)
        self._retired.clear()
        self._credit = UpdateCredit(self)
        return self._credit

    def cancel_update(self, credit):
        """Release an unsubmitted credit after forward/math raises."""
        self._check_open()
        self._check_credit(credit)
        credit._cancelled = True
        self._credit = None

    def _check_credit(self, credit):
        if credit is not self._credit or credit._owner is not self or credit._submitted or credit._cancelled:
            raise RuntimeError("invalid, cancelled or already submitted update credit")

    def _check_factors(self, factors):
        if not isinstance(factors, dict) or not factors or len(factors) > self.max_order + 1:
            raise ValueError("factors must contain 1..max_order+1 tensors")
        if set(factors) != set(range(len(factors))) or any(type(x) is not int for x in factors):
            raise ValueError("factor orders must be contiguous integers starting at zero")
        for tensor in factors.values():
            if not isinstance(tensor, torch.Tensor):
                raise ValueError("factor must be a tensor")
            if tensor.dtype != torch.float32 or tuple(tensor.shape) != self.feature_shape:
                raise ValueError("factor dtype/shape differs from the FP32 cache contract")
            if tensor.device != self.device or not tensor.is_contiguous():
                raise ValueError("factor must be contiguous on the configured device")
            if tensor.untyped_storage().nbytes() > self.feature_bytes:
                raise ValueError("factor must own compact storage, not retain a larger tensor view")
            if tensor.requires_grad:
                raise ValueError("factor must not retain an autograd graph")

    def publish(self, key, factors, credit):
        """Transfer ownership; callers must not mutate factors after this call."""
        self._check_key(key)
        self._check_credit(credit)
        self._check_factors(factors)
        if key not in self._entries and len(self._entries) >= 2 * self.num_groups:
            raise ValueError("retire the previous expert epoch before publishing more groups")
        if any(lease.key == key for lease in self._leases.values()):
            raise RuntimeError("release the old factor lease before publishing its replacement")
        generation = self._generations.get(key, 0) + 1
        self._generations[key] = generation
        owned = {order: tensor.detach() for order, tensor in factors.items()}
        credit._submitted = True
        size = len(owned) * self.feature_bytes
        if not self.offload:
            old = self._entries.get(key)
            if old is not None:
                event = self._event(self._npu.current_stream(self.device)) if self._npu else None
                self._retired.append((old.factors, event))
            self._entries[key] = _Entry(generation, owned)
            self._credit = None
            live = sum(len(entry.factors) * self.feature_bytes for entry in self._entries.values())
            self._stats["peak_device_factor_bytes"] = max(
                self._stats["peak_device_factor_bytes"], live + (len(old.factors) * self.feature_bytes if old else 0)
            )
            self._stats["updates"] += 1
            return
        try:
            compute = self._npu.current_stream(self.device)
            produced = self._event(compute)
            with self._range("D2H", key, generation), self._npu.stream(self._d2h_stream):
                self._d2h_stream.wait_event(produced)
                start = self._event(self._d2h_stream, timing=True)
                for order, tensor in owned.items():
                    self._d2h_host[order].copy_(tensor, non_blocking=True)
                end = self._event(self._d2h_stream, timing=True)
                done = self._event(self._d2h_stream)
            self._timings.append(("d2h_time_ms", start, end))
            self._stats["d2h_bytes"] += size
            self._stats["d2h_count"] += 1
            self._stats["peak_device_factor_bytes"] = max(
                self._stats["peak_device_factor_bytes"], 2 * self.bundle_bytes + size
            )
            self._future = self._writer.submit(self._write_host, key, generation, owned, done, produced, credit)
            self._pending[key] = self._future
        except BaseException as error:
            self._failed_sources.append(owned)
            self._error = error
            raise

    def _write_host(self, key, generation, sources, done, produced, credit):
        # sources and produced are intentionally held through DMA completion.
        with self._npu.device(self.device):
            done.synchronize()
        count = len(sources)
        sources.clear()
        self._publish_host(key, generation, self._d2h_host, count, credit)

    def _copy_host(self, target, source):
        """Copy private, non-autograd CPU storage after D2H completion.

        Tensor.copy_ is valid here, but dispatches through ATen's CPU copy
        path. On the tested A3 host, a fixed-buffer diagnostic with a fresh
        writer per request accumulated allocator RSS with Tensor.copy_;
        the same writer lifecycle with memmove did not. Use a synchronous
        byte copy for this private publication step without changing global
        Torch thread settings or keeping a writer alive across requests.
        H2D/D2H and inference-thread Host staging still use Tensor.copy_.

        These buffers never enter autograd or escape through offloaded leases.
        Only the payload copy bypasses ATen; update Torch's mutation metadata
        explicitly. Keep both tensors alive until the synchronous copy returns.
        """
        tensor = tensor_refs = None
        try:
            tensor_refs = (target, source)
            for tensor in tensor_refs:
                if type(tensor) is not torch.Tensor or tensor.layout != torch.strided:
                    raise ValueError("Host copy requires plain strided tensors")
                if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
                    raise ValueError("Host copy requires CPU float32 tensors")
                if tuple(tensor.shape) != self.feature_shape or not tensor.is_contiguous():
                    raise ValueError("Host copy requires the contiguous feature shape")
                if tensor.requires_grad or tensor.grad_fn is not None or tensor.is_conj() or tensor.is_neg():
                    raise ValueError("Host copy requires non-autograd tensors without logical view flags")
                offset = tensor.storage_offset() * tensor.element_size()
                if (tensor.numel() * tensor.element_size() != self.feature_bytes
                        or offset < 0 or offset + self.feature_bytes > tensor.untyped_storage().nbytes()):
                    raise ValueError("Host copy exceeds its tensor storage extent")
            target_ptr, source_ptr = target.data_ptr(), source.data_ptr()
            if not target_ptr or not source_ptr or target_ptr % 4 or source_ptr % 4:
                raise ValueError("Host copy requires valid aligned CPU pointers")
            if not (target_ptr + self.feature_bytes <= source_ptr
                    or source_ptr + self.feature_bytes <= target_ptr):
                raise ValueError("Host copy source and target must not overlap")
            ctypes.memmove(target_ptr, source_ptr, self.feature_bytes)
            torch.autograd.graph.increment_version(target)
        finally:
            # Do not retain our tensors in a failed copy/validation traceback.
            target = source = tensor = tensor_refs = None

    def _publish_host(self, key, generation, staging, count, credit):
        """Publish ordinary CPU factors after D2H completion.

        Canonical CPU buffers are private: offloaded leases borrow device slots.
        publish rejects a live lease; _load waits for this key's pending future
        before inspecting its entry or device slot. Reuse is therefore restricted
        to the same key while the single writer owns its pending publication.
        """
        host, reusable = {}, {}
        previous = target = None
        allocated = 0
        try:
            with self._lock:
                previous = self._entries.get(key)
                reusable = dict(previous.factors) if previous is not None else {}
                current = sum(len(entry.factors) * self.feature_bytes for entry in self._entries.values())
            for order in range(count):
                target = reusable.get(order)
                if target is None:
                    target = torch.empty(self.feature_shape, dtype=torch.float32, device="cpu", pin_memory=False)
                    allocated += self.feature_bytes
                    # Account each successful allocation, including a later
                    # allocation/copy failure. Reused storage is not a new copy.
                    with self._lock:
                        self._stats["peak_host_cache_bytes"] = max(
                            self._stats["peak_host_cache_bytes"], current + allocated
                        )
                host[order] = target
            with self._range("HostPublish", key, generation):
                for order in range(count):
                    self._copy_host(host[order], staging[order])
            with self._lock:
                self._entries[key] = _Entry(generation, host)
                self._stats["updates"] += 1
                if self._credit is credit:
                    self._credit = None
        except BaseException as error:
            # An in-place copy may have overwritten only part of the old
            # generation. Poison the manager and quarantine this key; no reader
            # may fall back to its old entry. The failed future remains pending
            # until close drains it and clears all request ownership.
            with self._lock:
                self._error = error
                self._stats["failed"] = True
                self._entries.pop(key, None)
            # Do not retain our Host tensors via this exception's Python frame.
            # Nested backend/caller tracebacks remain intact for diagnosis.
            host.clear()
            reusable.clear()
            previous = target = staging = None
            raise

    def _find_slot(self, key, generation):
        return next((s for s in self._slots if s.key == key and s.generation == generation), None)

    def _load(self, key, speculative):
        if key in self._pending:
            if speculative:
                return None
            self._finish_future(self._pending[key])
        entry = self._entries.get(key)
        if entry is None:
            return None
        found = self._find_slot(key, entry.generation)
        if found is not None:
            return found
        candidates = [slot for slot in self._slots if not slot.leased]
        candidates.sort(key=lambda slot: slot.key is not None)
        if speculative:
            candidates = [slot for slot in candidates if slot.ready is None or slot.ready.query()]
        if not candidates:
            if speculative:
                return None
            raise RuntimeError("all cache slots are leased; release the prior group")
        slot = candidates[0]
        try:
            # Host staging cannot be overwritten before its H2D completes.
            if slot.ready is not None:
                self._wait_host(slot.ready.synchronize)
            count = len(entry.factors)
            with self._range("HostStage", key, entry.generation):
                for order, tensor in entry.factors.items():
                    slot.host[order].copy_(tensor)
            with (
                self._range("PrefetchH2D" if speculative else "DemandH2D", key, entry.generation),
                self._npu.stream(self._h2d_stream),
            ):
                if slot.last_use is not None:
                    self._h2d_stream.wait_event(slot.last_use)
                start = self._event(self._h2d_stream, timing=True)
                slot.device[:count].copy_(slot.host[:count], non_blocking=True)
                end = self._event(self._h2d_stream, timing=True)
                self._record_ready(slot, key, entry.generation)
            slot.key, slot.generation, slot.count = key, entry.generation, count
            slot.prefetched = speculative
            slot.last_use = None
        except BaseException as error:
            # A copy may already have overwritten the previous generation even
            # when a later Host event/profiler operation fails. Never reuse it.
            self._error = error
            self._stats["failed"] = True
            raise
        self._timings.append(("h2d_time_ms", start, end))
        self._stats["h2d_bytes"] += count * self.feature_bytes
        self._stats["h2d_count"] += 1
        self._stats["prefetch_count" if speculative else "demand_count"] += 1
        return slot

    def prefetch(self, key):
        self._check_key(key)
        if not self.prefetch_enabled:
            return False
        if self._current_key is None or key[:4] != self._current_key[:4] or key[4] != self._current_key[4] + 1:
            raise ValueError("prefetch is restricted to the next group in the current expert/CFG")
        return self._load(key, speculative=True) is not None

    def acquire(self, key):
        self._check_key(key)
        if any(lease.key == key for lease in self._leases.values()):
            raise RuntimeError("cache group is already leased")
        if self.offload:
            slot = self._load(key, speculative=False)
            if slot is None:
                raise KeyError(key)
            try:
                compute = self._npu.current_stream(self.device)
                with self._range("AcquireWait", key, slot.generation):
                    start = self._event(compute, timing=True)
                    self._wait_ready(compute, slot, key)
                    end = self._event(compute, timing=True)
            except BaseException as error:
                self._error = error
                self._stats["failed"] = True
                raise
            self._timings.append(("device_wait_time_ms", start, end))
            self._stats["prefetch_hit"] += int(slot.prefetched)
            slot.prefetched = False
            slot.leased = True
            lease = FactorsLease(
                key, slot.generation, {i: slot.device[i] for i in range(slot.count)}, slot.slot_id, self
            )
        else:
            entry = self._entries.get(key)
            if entry is None:
                raise KeyError(key)
            lease = FactorsLease(key, entry.generation, dict(entry.factors), _owner=self)
        self._leases[id(lease)] = lease
        self._stats["history_hit"] += 1
        self._current_key = key
        return lease

    def release(self, lease):
        self._check_open()
        if lease._owner is not self or lease._released or id(lease) not in self._leases:
            raise RuntimeError("foreign or already released factor lease")
        if lease.slot_id is not None:
            slot = self._slots[lease.slot_id]
            if slot.key != lease.key or slot.generation != lease.generation:
                raise RuntimeError("cache slot generation changed while leased")
            slot.last_use = self._event(self._npu.current_stream(self.device))
            slot.leased = False
        lease.factors.clear()
        lease._released = True
        del self._leases[id(lease)]

    def drain(self):
        """Finish this branch before changing CFG/expert or stopping a timer."""
        self._check_open()
        self._finish_future(self._future)
        self._future = None
        if self._npu is not None:
            self._npu.current_stream(self.device).synchronize()
        for stream in (self._h2d_stream, self._d2h_stream):
            if stream is not None:
                stream.synchronize()
        for name, start, end in self._timings:
            self._stats[name] += start.elapsed_time(end)
        self._timings.clear()
        self._retired.clear()
        return self.statistics()

    def retire(self, expert_epoch):
        """Remove one (expert_id, expert_epoch) after all its consumers finish."""
        self._check_open()
        if not isinstance(expert_epoch, tuple) or len(expert_epoch) != 2:
            raise ValueError("retire requires (expert_id, expert_epoch)")
        if self._leases:
            raise RuntimeError("release factor leases before retiring an expert")
        self.drain()
        for key in list(self._entries):
            if key[1:3] == expert_epoch:
                del self._entries[key]
                self._generations.pop(key, None)
        for slot in self._slots:
            if slot.key is not None and slot.key[1:3] == expert_epoch:
                slot.key = None
                slot.ready = slot.last_use = None
                slot.ready_ticket = slot.ready_handle = None
        self._current_key = None

    def statistics(self):
        with self._lock:
            result = dict(self._stats)
            result.update(
                budgets=dict(self.budgets), live_cache_entries=len(self._entries),
                live_cache_bytes=sum(len(e.factors) * self.feature_bytes for e in self._entries.values()),
                outstanding_transfers=int(self._future is not None and not self._future.done()),
                active_leases=len(self._leases), active_update_credit=int(self._credit is not None),
                pinned_bytes=3 * self.bundle_bytes if self._d2h_host is not None else 0,
                device_slot_bytes=len(self._slots) * self.bundle_bytes,
            )
            return result

    def close(self, error=None):
        """Drain before releasing ownership, including an interrupted update.

        A device drain failure is propagated: the caller must terminate this
        process instead of reusing a partially failed runtime.
        """
        if self._closed:
            if self._drain_failure is not None:
                raise RuntimeError("cache drain failed; restart the inference process") from self._drain_failure
            return self.statistics()
        publication_failure = None
        drain_failure = None
        self._stats["failed"] = bool(error) or self._error is not None
        try:
            if self._future is not None:
                self._future.result()
        except BaseException as caught:
            publication_failure = caught
            self._stats["failed"] = True
        try:
            if self._npu is not None:
                self._npu.synchronize(self.device)
            for stream in (self._h2d_stream, self._d2h_stream):
                if stream is not None:
                    stream.synchronize()
            for name, start, end in self._timings:
                self._stats[name] += start.elapsed_time(end)
        except BaseException as caught:
            drain_failure = caught
            self._drain_failure = caught
            self._stats["failed"] = True
        finally:
            if self._writer is not None:
                self._writer.shutdown(wait=True)
            self._writer = None
            self._closed = True
        if drain_failure is not None:
            # Retain tensors if DMA completion could not be established.
            raise RuntimeError("cache drain failed; restart the inference process") from drain_failure
        for lease in self._leases.values():
            lease.factors.clear()
            lease._released = True
        self._leases.clear()
        self._entries.clear()
        self._pending.clear()
        self._generations.clear()
        self._retired.clear()
        self._failed_sources.clear()
        for slot in self._slots:
            slot.ready_ticket = slot.ready_handle = None
        self._slots.clear()
        self._d2h_host = None
        self._h2d_stream = self._d2h_stream = None
        self._future = self._credit = self._current_key = None
        self._timings.clear()
        if publication_failure is not None:
            raise RuntimeError("cache Host publication failed") from publication_failure
        return self.statistics()
