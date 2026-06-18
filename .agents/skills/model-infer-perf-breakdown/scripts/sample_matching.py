"""Sample-driven layer detection core.

无 IO 的纯函数 + dataclass。调用方（detect_structure.py）负责 load
raw_ops.json / structure_spec.json，把数据传进来；本模块只算。

两条装配路径（按入口区分；detect_structure.py 实际走 stream 模式）：
  · stream 模式（**活跃路径**）：
        run_stream_sample_mode(ops, structure_spec, sample_ack) -> StreamSampleDraft
    装配走 _assemble_stream_components -> _expand_schedule，**强依赖 structure_spec
    的 layer_range / layer_indices** 逐层 schedule（交错/逐层交替结构用 layer_indices；
    同层被多个 composition 覆盖时 _expand_schedule 直接 hard error）。
  · 非 stream 模式（row-order，legacy）：
        run_sample_mode(ops, structure_spec, samples, strict_composition=False)
            -> SampleDraft
    装配走 assemble_layers（按 row 升序 + lookahead，**忽略 layer_range**）。

下面"子函数"列的是非 stream 路径（run_sample_mode）的调用链；stream 路径的装配见
run_stream_sample_mode / _assemble_stream_components / _expand_schedule。

子函数（run_sample_mode 路径，按调用顺序）：
    extract_fingerprint(ops, lo, hi, k=3) -> Fingerprint
    build_prefix_counts(ops) -> 用于 O(distinct) Jaccard
    scan_seeds(prefix, fingerprint, direction) -> [(direction, pos)]
    match_instances(ops, prefix, fingerprint, seeds, expected_len)
                                              -> [InstanceCandidate]
    endpoint_adjust(ops, prefix, fingerprint, candidate) -> InstanceCandidate
    nms_instances(candidates, overlap=0.30) -> [Instance]
    arbitrate_overlaps(instances_by_component) -> [Instance]
    classify_unmatched(instances, total_ops, repeat_threshold=2)
                                              -> [UnmatchedRegion]
    assemble_layers(instances, structure_spec) -> (layers, warnings)
    build_validation(structure_spec, instances) -> dict

阈值常量集中在文件顶部，便于回归时调。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

K = 3                          # head / tail 指纹窗口大小
SEED_THRESHOLD = 0.8           # head/tail Jaccard 种子阈值
ACCEPT_THRESHOLD = 0.85        # body_ms Jaccard 验收阈值
MICROADJUST_LOWER = 0.70       # body_ms 落在 [0.70, 0.85) 触发端点微调
SHAPE_CONFIDENCE_THRESHOLD = 0.75    # body_shape_ms < 0.75 → low_shape_confidence
LENGTH_BAND = (0.85, 1.15)     # tail / head 窗口落在 [L*0.85, L*1.15]
ENDPOINT_RADIUS = 3            # 端点微调 ±K
NMS_OVERLAP = 0.30             # 区间重叠 > 30% 丢弃后者
ARBITRATE_SCORE_TOLERANCE = 0.05      # 两个 component 差 < 0.05 进入 shape 仲裁
ARBITRATE_SHAPE_TOLERANCE = 0.02      # shape 也差 < 0.02 → ambiguous_match
SUSPECTED_UNDECLARED_MIN_REGIONS = 2  # 末尾 ≥ N 个相似 region 报警
SUSPECTED_UNDECLARED_SIMILARITY = 0.80  # 区间长度互相 ≥ 80% → 视为重复 region
LARGE_UNMATCHED_REGION_OPS = 30       # 单段 inter_layer_region > N ops → 升级为 suspected
# aux 时间脱节体检：形态无关，只度量"并入辅流是否撑大 component 包络"。
AUX_DISPLACEMENT_RATIO_LIMIT = 1.0    # 单 occurrence 膨胀比 (total_span-primary_span)/primary_span 超此 → 显著
AUX_DISPLACEMENT_FRACTION = 0.5       # 一条 aux 流多数(>此比例)occurrence 脱节 → 判流级 displaced
# 判据全用相对/结构量（inflation 比、流级多数），不设绝对时间阈值——避免量级过拟合。

AUXILIARY_NAMES = {
    "Cast", "Reshape", "Transpose", "Contiguous",
    "DequantSwiglu", "DequantBmm", "Slice", "Concat",
}


@dataclass(frozen=True)
class Fingerprint:
    """单个 sample 的指纹"""
    component: str
    op_range: tuple[int, int]
    length: int
    head_ms: dict[tuple[str, str], int]   # Counter on (normalized_name, canon_shape)
    tail_ms: dict[tuple[str, str], int]
    body_ms: dict[str, int]               # Counter on normalized_name
    body_shape_ms: dict[tuple[str, str], int]


@dataclass
class InstanceCandidate:
    component: str
    start: int
    end: int
    score_body: float
    score_body_shape: float
    score_head: float
    score_tail: float
    seed_direction: str
    endpoint_adjusted: bool


@dataclass
class Instance:
    """通过 NMS 仲裁后的最终 instance"""
    component: str
    start: int
    end: int
    layer_idx: int = -1
    phase: str = ""
    scores: dict[str, float] = field(default_factory=dict)
    seed_direction: str = "head"
    endpoint_adjusted: bool = False


@dataclass
class UnmatchedRegion:
    start: int
    end: int
    classification: str    # pre_arch / post_arch / inter_layer_region / suspected_undeclared_component / intra_layer_gap / intra_layer_outlier


@dataclass
class Warning:
    code: str
    message: str
    extra: dict = field(default_factory=dict)


@dataclass
class SampleDraft:
    samples_used: list[dict]
    components: list[Instance]
    unmatched_regions: list[UnmatchedRegion]
    warnings: list[Warning]
    validation: dict


@dataclass
class StreamSegment:
    stream_id: str
    role: str
    op_indices: list[int]
    scores: dict[str, float] = field(default_factory=dict)


@dataclass
class StreamComponent:
    component_id: str
    type: str
    phase: str
    layer_idx: int
    occurrence_idx: int
    primary_stream_id: str
    op_indices: list[int]
    stream_segments: list[StreamSegment]
    scores: dict[str, float] = field(default_factory=dict)


@dataclass
class StreamSampleDraft:
    structure_spec: dict
    samples_used: list[dict]
    components: list[StreamComponent]
    op_to_component: dict[str, str]
    unmatched_op_indices: list[int]
    unmatched_stream_segments: list[dict]
    warnings: list[Warning]
    validation: dict
    displaced_streams: list = field(default_factory=list)   # [(component_type, stream_id)]


# 其余函数将在 Task 5+ 添加
def canon_shape(op: dict) -> str:
    """规范化 input_shapes + output_shapes 为单字符串。

    跨 run 稳定指纹的关键：去空格、统一定界符、合并 in/out。
    """
    ins = (op.get("input_shapes") or "").replace(" ", "").rstrip(";")
    outs = (op.get("output_shapes") or "").replace(" ", "").rstrip(";")
    return f"{ins}|{outs}"


def _name_of(op: dict) -> str:
    return op.get("normalized_name") or op.get("name", "")


def _op_index(op: dict, fallback: int) -> int:
    idx = op.get("index", fallback)
    return fallback if idx is None else int(idx)


def _stream_id_of(op: dict) -> str:
    sid = op.get("stream_id")
    return "unknown" if sid in (None, "") else str(sid)


def _start_of(op: dict, fallback: int) -> float:
    st = op.get("start_time_us")
    if st is None:
        return float(fallback)
    try:
        return float(st)
    except (TypeError, ValueError):
        return float(fallback)


def build_stream_index(ops: list[dict]) -> tuple[dict[str, list[int]], dict[int, dict]]:
    """Build stream-local op order.

    Returns:
      streams: stream_id -> global op indices sorted by (start_time_us, index)
      op_to_stream_pos: op index -> {stream_id, pos}
    """
    streams: dict[str, list[int]] = {}
    for pos, op in enumerate(ops):
        idx = _op_index(op, pos)
        streams.setdefault(_stream_id_of(op), []).append(idx)

    pos_by_idx = {_op_index(op, pos): pos for pos, op in enumerate(ops)}
    for sid, indices in streams.items():
        indices.sort(key=lambda i: (_start_of(ops[pos_by_idx[i]], pos_by_idx[i]), i))

    op_to_stream_pos: dict[int, dict] = {}
    for sid, indices in streams.items():
        for p, idx in enumerate(indices):
            op_to_stream_pos[idx] = {"stream_id": sid, "pos": p}
    return streams, op_to_stream_pos


def _ops_by_index(ops: list[dict]) -> dict[int, dict]:
    return {_op_index(op, pos): op for pos, op in enumerate(ops)}


def _counts_for_indices(ops_by_idx: dict[int, dict], indices: list[int]) -> Counter:
    return Counter(_name_of(ops_by_idx[i]) for i in indices)


def _shape_counts_for_indices(ops_by_idx: dict[int, dict], indices: list[int]) -> Counter:
    return Counter((_name_of(ops_by_idx[i]), canon_shape(ops_by_idx[i])) for i in indices)


def _sequence_score(a: list[str], b: list[str]) -> float:
    if not a and not b:
        return 1.0
    if len(a) != len(b) or not a or not b:
        return 0.0
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def _stream_window_scores(ops_by_idx: dict[int, dict],
                          window: list[int],
                          sample_names: list[str],
                          sample_counts: Counter,
                          sample_shape_counts: Counter) -> dict[str, float]:
    names = [_name_of(ops_by_idx[i]) for i in window]
    return {
        "stream_sequence": _sequence_score(names, sample_names),
        "stream_body": multiset_jaccard(
            _counts_for_indices(ops_by_idx, window), sample_counts),
        "stream_shape": multiset_jaccard(
            _shape_counts_for_indices(ops_by_idx, window), sample_shape_counts),
    }


def _scan_stream_segment(ops_by_idx: dict[int, dict],
                         stream_indices: list[int],
                         sample_indices: list[int]) -> list[StreamSegment]:
    if not sample_indices or len(sample_indices) > len(stream_indices):
        return []
    sample_names = [_name_of(ops_by_idx[i]) for i in sample_indices]
    sample_counts = _counts_for_indices(ops_by_idx, sample_indices)
    sample_shape_counts = _shape_counts_for_indices(ops_by_idx, sample_indices)
    L = len(sample_indices)
    out: list[StreamSegment] = []
    for start in range(0, len(stream_indices) - L + 1):
        window = stream_indices[start:start + L]
        scores = _stream_window_scores(
            ops_by_idx, window, sample_names, sample_counts, sample_shape_counts)
        if (scores["stream_sequence"] >= 0.75
                and scores["stream_body"] >= 0.85
                and scores["stream_shape"] >= SHAPE_CONFIDENCE_THRESHOLD):
            out.append(StreamSegment(
                stream_id=_stream_id_of(ops_by_idx[window[0]]),
                role="",
                op_indices=list(window),
                scores=scores,
            ))
    return out


def _low_shape_stream_matches(ops_by_idx: dict[int, dict],
                              stream_indices: list[int],
                              sample_indices: list[int]) -> list[dict]:
    if not sample_indices or len(sample_indices) > len(stream_indices):
        return []
    sample_names = [_name_of(ops_by_idx[i]) for i in sample_indices]
    sample_counts = _counts_for_indices(ops_by_idx, sample_indices)
    sample_shape_counts = _shape_counts_for_indices(ops_by_idx, sample_indices)
    L = len(sample_indices)
    out = []
    for start in range(0, len(stream_indices) - L + 1):
        window = stream_indices[start:start + L]
        scores = _stream_window_scores(
            ops_by_idx, window, sample_names, sample_counts, sample_shape_counts)
        if (scores["stream_sequence"] >= 0.75
                and scores["stream_body"] >= 0.85
                and scores["stream_shape"] < SHAPE_CONFIDENCE_THRESHOLD):
            out.append({"op_indices": list(window), "scores": scores})
    return out


def _segment_time_bounds(seg: StreamSegment,
                         ops_by_idx: dict[int, dict]) -> tuple[float, float]:
    starts = []
    ends = []
    for idx in seg.op_indices:
        op = ops_by_idx[idx]
        st = _start_of(op, idx)
        dur = float(op.get("duration_us") or 0.0)
        starts.append(st)
        ends.append(st + dur)
    if not starts:
        return 0.0, 0.0
    return min(starts), max(ends)


def _segment_pair_metrics(primary_seg: StreamSegment,
                          aux_seg: StreamSegment,
                          ops_by_idx: dict[int, dict]) -> dict[str, float]:
    p_start, p_end = _segment_time_bounds(primary_seg, ops_by_idx)
    a_start, a_end = _segment_time_bounds(aux_seg, ops_by_idx)
    overlap = max(0.0, min(p_end, a_end) - max(p_start, a_start))
    if overlap > 0:
        gap = 0.0
    else:
        gap = min(abs(a_start - p_end), abs(p_start - a_end))
    center_delta = abs(((p_start + p_end) / 2.0) - ((a_start + a_end) / 2.0))
    return {
        "pair_overlap_us": overlap,
        "pair_gap_us": gap,
        "pair_center_delta_us": center_delta,
    }


def _match_auxiliary_segments(primary_segs: list[StreamSegment],
                              aux_cands: list[StreamSegment],
                              ops_by_idx: dict[int, dict]
                              ) -> tuple[dict[int, tuple[int, StreamSegment, dict]],
                                         dict[int, dict]]:
    """全局最优一一匹配 primary occurrence ↔ aux segment。

    对每条 aux 流，在 primary occurrence 与该流的 aux segment 之间求一组一一匹配，使匹配数
    最多、总时间距离最小。当辅流步距小于主流、逐层领先时，最近的 aux 会先被前段 occurrence
    占走、后段只能配到远期 segment——按全局最优分配可避免这种错配，逐层单调时自然退化为序号
    对齐（occ k ↔ aux 第 k 段）。

    算法：代价 = 两段中心时间距离 pair_center_delta_us（一维点匹配）。一维点匹配的最优解
    非交叉（保序）只在两序列各自按"中心时间"排序时成立；滑窗产出的 segment 按 *起点* 升序、
    duration 非单调时起点序 ≠ 中心序，会破坏该前提。因此入口先按中心时间稳定排序 primary/aux，
    再跑非交叉 DP，回溯后映射回原始 occurrence/cand 位置。目标按 (-matched_count, total_cost)
    字典序最小化：先最大化匹配数，平局再比总代价。

    cost 用 center_delta：等长窗口下它单调代表 overlap，且辅流通常短于 primary（重叠的 aux
    中心必落在 primary 跨度内 → center_delta 小），故并发优先自然成立；仅当 aux 段反常地长于
    primary 时，一个中心更近的脱节段可能赢过重叠段（aux 不长于 primary 时不出现）。

    返回:
      assignment: {occ_idx -> (cand_pos, seg, pair_metrics)}
      ambiguous:  {occ_idx -> ambiguity_dict}（全局最优非唯一时，该 occ 的配对可替换）
    """
    n, m = len(primary_segs), len(aux_cands)
    if n == 0 or m == 0:
        return {}, {}

    def _center(seg):
        a, b = _segment_time_bounds(seg, ops_by_idx)
        return (a + b) / 2.0

    # 按中心时间稳定排序，使一维点距离严格满足 Monge（非交叉 DP 才是全局最优）。
    p_order = sorted(range(n), key=lambda i: (_center(primary_segs[i]), i))
    a_order = sorted(range(m), key=lambda j: (_center(aux_cands[j]), j))
    P = [primary_segs[i] for i in p_order]
    A = [aux_cands[j] for j in a_order]
    metrics = [[_segment_pair_metrics(P[i], A[j], ops_by_idx)
                for j in range(m)] for i in range(n)]
    cost = [[metrics[i][j]["pair_center_delta_us"] for j in range(m)] for i in range(n)]

    def _rank(state):
        return (-state[0], state[1])

    def _solve(blocked_col):
        """非交叉 DP 最优匹配；blocked_col(已排序列索引) 被禁用，用于唯一性检验。"""
        dp = [[(0, 0.0)] * (m + 1) for _ in range(n + 1)]
        choice = [[""] * (m + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                best, ch = dp[i - 1][j], "skip_occ"
                if _rank(dp[i][j - 1]) < _rank(best):
                    best, ch = dp[i][j - 1], "skip_seg"
                if j - 1 != blocked_col:
                    cand = (dp[i - 1][j - 1][0] + 1,
                            dp[i - 1][j - 1][1] + cost[i - 1][j - 1])
                    if _rank(cand) < _rank(best):
                        best, ch = cand, "match"
                dp[i][j], choice[i][j] = best, ch
        pairs = {}
        i, j = n, m
        while i > 0 and j > 0:
            if choice[i][j] == "match":
                pairs[i - 1] = j - 1
                i -= 1
                j -= 1
            elif choice[i][j] == "skip_occ":
                i -= 1
            else:
                j -= 1
        return dp[n][m], pairs

    opt, pairs = _solve(blocked_col=-1)

    # 映射回原始下标
    assignment: dict[int, tuple[int, StreamSegment, dict]] = {}
    for si, sj in pairs.items():
        occ, pos = p_order[si], a_order[sj]
        assignment[occ] = (pos, aux_cands[pos], metrics[si][sj])

    # ambiguity: 禁用某 occ 选中的列后重跑，最优 (count,cost) 不变 → 该配对全局可替换。
    ambiguous: dict[int, dict] = {}
    for si, sj in pairs.items():
        opt2, _ = _solve(blocked_col=sj)
        if opt2[0] == opt[0] and abs(opt2[1] - opt[1]) < 1e-9:
            occ, pos = p_order[si], a_order[sj]
            alt_sj = next((k for k in sorted(range(m), key=lambda k: (cost[si][k], k))
                           if k != sj), None)
            ambiguous[occ] = {
                "selected_op_indices": assignment[occ][1].op_indices,
                "alternative_op_indices": (
                    aux_cands[a_order[alt_sj]].op_indices if alt_sj is not None else []),
                "selected_pair_metrics": assignment[occ][2],
            }
    return assignment, ambiguous


def _normalize_stream_ack(sample_ack: dict) -> dict:
    if sample_ack.get("schema_version") == "stream_sample_ack.v1":
        return sample_ack
    raise SystemExit("stream sample mode requires sample_ack.json schema_version=stream_sample_ack.v1")


def extract_fingerprint(ops: list[dict], component: str,
                        lo: int, hi: int, k: int = K) -> Fingerprint:
    """从 [lo, hi] 闭区间提取 4 个指纹（head_ms / tail_ms / body_ms / body_shape_ms）。"""
    if lo < 0 or hi >= len(ops) or lo > hi:
        raise ValueError(f"bad sample range [{lo}, {hi}] for ops of len {len(ops)}")
    if hi - lo + 1 < 2 * k:
        raise ValueError(f"sample length {hi - lo + 1} < 2K ({2 * k})")

    window = ops[lo:hi + 1]
    head_ms = Counter((_name_of(o), canon_shape(o)) for o in window[:k])
    tail_ms = Counter((_name_of(o), canon_shape(o)) for o in window[-k:])
    body_ms = Counter(_name_of(o) for o in window)
    body_shape_ms = Counter((_name_of(o), canon_shape(o)) for o in window)

    return Fingerprint(
        component=component,
        op_range=(lo, hi),
        length=hi - lo + 1,
        head_ms=dict(head_ms),
        tail_ms=dict(tail_ms),
        body_ms=dict(body_ms),
        body_shape_ms=dict(body_shape_ms),
    )


def build_prefix_counts(ops: list[dict]) -> dict[str, list[int]]:
    """每个 normalized_name 一个 prefix-sum 数组。

    prefix[name][i] = count of `name` in ops[0:i]，i ∈ [0, N]。
    单次 multiset Jaccard 取 [s, e+1] - [s] 即得窗口内 count。
    """
    names = set(_name_of(o) for o in ops)
    out: dict[str, list[int]] = {n: [0] * (len(ops) + 1) for n in names}
    for i, op in enumerate(ops):
        name = _name_of(op)
        for n in names:
            out[n][i + 1] = out[n][i] + (1 if n == name else 0)
    return out


def window_name_counts(prefix: dict[str, list[int]],
                       s: int, e: int) -> dict[str, int]:
    """O(distinct_names) 取 [s, e] 闭区间内的 name multiset。"""
    return {
        n: prefix[n][e + 1] - prefix[n][s]
        for n in prefix
        if prefix[n][e + 1] - prefix[n][s] > 0
    }


def multiset_jaccard(a, b) -> float:
    """两个 dict-like multiset 的 Jaccard = sum(min) / sum(max)。

    支持 a / b 是 dict 或 Counter。空多集对空多集 → 1.0；
    一方非空一方空 → 0.0。
    """
    if not a and not b:
        return 1.0
    keys = set(a) | set(b)
    inter = sum(min(a.get(k, 0), b.get(k, 0)) for k in keys)
    union = sum(max(a.get(k, 0), b.get(k, 0)) for k in keys)
    return inter / union if union else 0.0


def window_shape_counts(ops: list[dict], s: int, e: int) -> dict[tuple[str, str], int]:
    """[s, e] 闭区间内 (name, canon_shape) 多集——给 head_ms/tail_ms/body_shape_ms 用。"""
    c: Counter = Counter()
    for i in range(s, e + 1):
        c[(_name_of(ops[i]), canon_shape(ops[i]))] += 1
    return dict(c)


def scan_head_seeds(ops: list[dict], fp: Fingerprint, k: int = K) -> list[int]:
    """主方向：扫所有起点 s 使 window[s:s+k] 的 (name, shape) 多集与 head_ms Jaccard ≥ SEED_THRESHOLD。

    返回 s 列表（升序），每个 s 是候选 instance 起点。
    """
    n = len(ops)
    seeds = []
    for s in range(n - k + 1):
        wc = window_shape_counts(ops, s, s + k - 1)
        if multiset_jaccard(wc, fp.head_ms) >= SEED_THRESHOLD:
            seeds.append(s)
    return seeds


def scan_tail_seeds(ops: list[dict], fp: Fingerprint, k: int = K) -> list[int]:
    """兜底方向：扫所有终点 e 使 window[e-k+1:e+1] 与 tail_ms Jaccard ≥ SEED_THRESHOLD。

    返回 e 列表（升序）。只在 head seeds 全空时调用。
    """
    n = len(ops)
    seeds = []
    for e in range(k - 1, n):
        wc = window_shape_counts(ops, e - k + 1, e)
        if multiset_jaccard(wc, fp.tail_ms) >= SEED_THRESHOLD:
            seeds.append(e)
    return seeds


def _body_jaccard(prefix, fp: Fingerprint, s: int, e: int) -> float:
    return multiset_jaccard(window_name_counts(prefix, s, e), fp.body_ms)


def _body_shape_jaccard(ops, fp: Fingerprint, s: int, e: int) -> float:
    return multiset_jaccard(window_shape_counts(ops, s, e), fp.body_shape_ms)


def scan_body_seeds(ops, prefix, fp: Fingerprint, step: int = 1,
                    threshold: float = MICROADJUST_LOWER) -> list[int]:
    """body_ms Jaccard 滑窗：固定窗长 L = fp.length，按 step 扫所有起点。

    body 多集是 instance validity 的真信号（与 head/tail boundary 是否被
    fused kernel 吞掉无关）。head/tail seed 卡掉的 instance（典型 case：
    pre-norm 架构入口 RmsNorm 被前一层 InplaceAddRmsNorm 吸收）由此路径
    找回，再由 endpoint_adjust 精化 (s, e)。

    threshold 取 MICROADJUST_LOWER 与 head-seed 路径一致：seed 宽 →
    endpoint_adjust 收紧 → 验收 ≥ ACCEPT_THRESHOLD。
    """
    n = len(ops)
    L = fp.length
    if L > n:
        return []
    return [
        s for s in range(0, n - L + 1, step)
        if _body_jaccard(prefix, fp, s, s + L - 1) >= threshold
    ]


def _best_tail_in_band(ops, fp: Fingerprint, start: int, n: int,
                      k: int = K) -> Optional[tuple[int, float]]:
    """在 [start + L*0.85 - k + 1, start + L*1.15 - k + 1) 内找 tail Jaccard 最高的 e。"""
    lo_end = start + int(fp.length * LENGTH_BAND[0]) - 1
    hi_end = min(n - 1, start + int(fp.length * LENGTH_BAND[1]) - 1)
    best: Optional[tuple[int, float]] = None
    for e in range(max(lo_end, start + k - 1), hi_end + 1):
        wc = window_shape_counts(ops, e - k + 1, e)
        j = multiset_jaccard(wc, fp.tail_ms)
        if j >= SEED_THRESHOLD and (best is None or j > best[1]):
            best = (e, j)
    return best


def _best_head_in_band(ops, fp: Fingerprint, end: int,
                      k: int = K) -> Optional[tuple[int, float]]:
    """tail-seeded 兜底：给定 e，在 [e - L*1.15 + 1, e - L*0.85 + 1] 内找 head 最佳起点。"""
    lo_start = max(0, end - int(fp.length * LENGTH_BAND[1]) + 1)
    hi_start = max(0, end - int(fp.length * LENGTH_BAND[0]) + 1)
    best: Optional[tuple[int, float]] = None
    for s in range(lo_start, hi_start + 1):
        if s + k - 1 > end:
            continue
        wc = window_shape_counts(ops, s, s + k - 1)
        j = multiset_jaccard(wc, fp.head_ms)
        if j >= SEED_THRESHOLD and (best is None or j > best[1]):
            best = (s, j)
    return best


def endpoint_adjust(ops, prefix, fp: Fingerprint, s: int, e: int,
                    radius: int = ENDPOINT_RADIUS) -> tuple[int, int, float]:
    """在 (s, e) 周围 ±radius 内枚举 (2*r+1)² 组合，返回 body_ms 最高的 (s, e, score)。

    用于 body_ms ∈ [MICROADJUST_LOWER, ACCEPT_THRESHOLD) 时的救援。
    """
    n = len(ops)
    best = (s, e, _body_jaccard(prefix, fp, s, e))
    for ds in range(-radius, radius + 1):
        for de in range(-radius, radius + 1):
            ns, ne = s + ds, e + de
            if ns < 0 or ne >= n or ns >= ne:
                continue
            score = _body_jaccard(prefix, fp, ns, ne)
            if score > best[2]:
                best = (ns, ne, score)
    return best


def match_component(ops, prefix, fp: Fingerprint) -> list[InstanceCandidate]:
    """主入口：对一个 component 的指纹扫全文，输出候选 instance 列表（已 NMS）。

    body_ms Jaccard 是 instance validity 的真信号——multiset 与 boundary
    kernel 是否被融合无关。head/tail 都只是加速 anchor，可能被 fused boundary
    kernel（如入口 RmsNorm 被前一层 residual-add 融合吃掉）卡掉。

    路径（执行顺序，并行附加候选，最后 NMS 去重）：
      1. head-seed（快，head 完整时给出精确起点 + length-band tail；
         与 body-seed 重叠时由 NMS 按 body_j 合并，相同分时插入序在前的胜出）
      2. body-seed 滑窗（O(N) step=1）：扫所有 s，body_j ≥ MICROADJUST_LOWER
         即 seed，endpoint_adjust 精化 (s, e)，验收 body_j ≥ ACCEPT_THRESHOLD。
         覆盖 head 被融合吞掉的 instance。
      3. tail-seed（兜底；仅 head 和 body 都空时尝试）
    """
    n = len(ops)
    accepted: list[InstanceCandidate] = []

    head_seeds = scan_head_seeds(ops, fp, k=K)
    for s in head_seeds:
        head_j = multiset_jaccard(window_shape_counts(ops, s, s + K - 1), fp.head_ms)
        tail = _best_tail_in_band(ops, fp, s, n, k=K)
        if tail is None:
            continue
        e, tail_j = tail
        body_j = _body_jaccard(prefix, fp, s, e)
        adjusted = False
        if body_j < ACCEPT_THRESHOLD:
            if body_j < MICROADJUST_LOWER:
                continue
            ns, ne, body_j = endpoint_adjust(ops, prefix, fp, s, e)
            if body_j < ACCEPT_THRESHOLD:
                continue
            s, e = ns, ne
            adjusted = True
            head_j = multiset_jaccard(window_shape_counts(ops, s, min(s + K - 1, e)), fp.head_ms)
            tail_j = multiset_jaccard(window_shape_counts(ops, max(e - K + 1, s), e), fp.tail_ms)
        body_shape_j = _body_shape_jaccard(ops, fp, s, e)
        accepted.append(InstanceCandidate(
            component=fp.component, start=s, end=e,
            score_body=body_j, score_body_shape=body_shape_j,
            score_head=head_j, score_tail=tail_j,
            seed_direction="head", endpoint_adjusted=adjusted,
        ))

    body_seeds = scan_body_seeds(ops, prefix, fp, step=1)
    for s in body_seeds:
        e = s + fp.length - 1
        if e >= n:
            continue
        ns, ne, body_j = endpoint_adjust(ops, prefix, fp, s, e)
        if body_j < ACCEPT_THRESHOLD:
            continue
        head_j = multiset_jaccard(window_shape_counts(ops, ns, min(ns + K - 1, ne)), fp.head_ms)
        tail_j = multiset_jaccard(window_shape_counts(ops, max(ne - K + 1, ns), ne), fp.tail_ms)
        body_shape_j = _body_shape_jaccard(ops, fp, ns, ne)
        accepted.append(InstanceCandidate(
            component=fp.component, start=ns, end=ne,
            score_body=body_j, score_body_shape=body_shape_j,
            score_head=head_j, score_tail=tail_j,
            seed_direction="body", endpoint_adjusted=(ns != s or ne != e),
        ))

    if accepted:
        return nms_candidates(accepted)

    tail_seeds = scan_tail_seeds(ops, fp, k=K)
    for e in tail_seeds:
        tail_j = multiset_jaccard(window_shape_counts(ops, e - K + 1, e), fp.tail_ms)
        head = _best_head_in_band(ops, fp, e, k=K)
        if head is None:
            continue
        s, head_j = head
        body_j = _body_jaccard(prefix, fp, s, e)
        adjusted = False
        if body_j < ACCEPT_THRESHOLD:
            if body_j < MICROADJUST_LOWER:
                continue
            ns, ne, body_j = endpoint_adjust(ops, prefix, fp, s, e)
            if body_j < ACCEPT_THRESHOLD:
                continue
            s, e = ns, ne
            adjusted = True
            head_j = multiset_jaccard(window_shape_counts(ops, s, min(s + K - 1, e)), fp.head_ms)
            tail_j = multiset_jaccard(window_shape_counts(ops, max(e - K + 1, s), e), fp.tail_ms)
        body_shape_j = _body_shape_jaccard(ops, fp, s, e)
        accepted.append(InstanceCandidate(
            component=fp.component, start=s, end=e,
            score_body=body_j, score_body_shape=body_shape_j,
            score_head=head_j, score_tail=tail_j,
            seed_direction="tail", endpoint_adjusted=adjusted,
        ))
    return accepted


def _overlap_ratio(a: InstanceCandidate, b: InstanceCandidate) -> float:
    lo = max(a.start, b.start)
    hi = min(a.end, b.end)
    if lo > hi:
        return 0.0
    inter = hi - lo + 1
    shorter = min(a.end - a.start + 1, b.end - b.start + 1)
    return inter / shorter


def nms_candidates(cands: list[InstanceCandidate],
                   overlap: float = NMS_OVERLAP) -> list[InstanceCandidate]:
    """按 score_body 降序贪心，与已选区间重叠 > overlap 的丢弃。"""
    ordered = sorted(cands, key=lambda c: -c.score_body)
    kept: list[InstanceCandidate] = []
    for c in ordered:
        if any(_overlap_ratio(c, k) > overlap for k in kept):
            continue
        kept.append(c)
    return sorted(kept, key=lambda c: c.start)


def arbitrate_overlaps(all_cands_by_component: dict[str, list[InstanceCandidate]]
                       ) -> tuple[list[Instance], list[Warning]]:
    """跨 component 仲裁：

    1. 把所有候选混进同一池，按 score_body 降序贪心
    2. 与已选区间重叠 > NMS_OVERLAP 时：
       - 如果是同 component → 直接丢弃后者（NMS 已做完，理论上不会进来）
       - 如果是异 component → 比 score_body：差 < 0.05 比 score_body_shape；
         差 < 0.02 → ambiguous_match 警告 + 都保留？不，选高的，
         同时记 warning 让 AI 提示用户重挑 sample。
    """
    pool = []
    for cands in all_cands_by_component.values():
        pool.extend(cands)
    pool.sort(key=lambda c: -c.score_body)

    kept: list[InstanceCandidate] = []
    warnings: list[Warning] = []
    for c in pool:
        conflict = next((k for k in kept if _overlap_ratio(c, k) > NMS_OVERLAP), None)
        if conflict is None:
            kept.append(c)
            continue
        if conflict.component == c.component:
            continue
        # 异 component 冲突：比较 body / body_shape
        d_body = abs(conflict.score_body - c.score_body)
        if d_body < ARBITRATE_SCORE_TOLERANCE:
            d_shape = abs(conflict.score_body_shape - c.score_body_shape)
            if d_shape < ARBITRATE_SHAPE_TOLERANCE:
                warnings.append(Warning(
                    code="ambiguous_match",
                    message=(f"component {conflict.component} vs {c.component} "
                             f"at rows [{c.start},{c.end}] body 差 {d_body:.3f} "
                             f"shape 差 {d_shape:.3f}"),
                    extra={"component_a": conflict.component,
                           "component_b": c.component,
                           "row_range": [c.start, c.end]},
                ))
        # 选 score 更高的；conflict 已经在 kept 里且分数 ≥ c（因为降序），保留 conflict

    instances = [
        Instance(component=c.component, start=c.start, end=c.end,
                 scores={"body_ms": c.score_body, "body_shape_ms": c.score_body_shape,
                         "head_ms": c.score_head, "tail_ms": c.score_tail},
                 seed_direction=c.seed_direction,
                 endpoint_adjusted=c.endpoint_adjusted)
        for c in sorted(kept, key=lambda x: x.start)
    ]
    return instances, warnings


def _are_lengths_similar(lens: list[int]) -> bool:
    if len(lens) < SUSPECTED_UNDECLARED_MIN_REGIONS:
        return False
    mn, mx = min(lens), max(lens)
    return mn / mx >= SUSPECTED_UNDECLARED_SIMILARITY if mx else False


def classify_unmatched(instances: list[Instance], n: int
                       ) -> tuple[list[UnmatchedRegion], list[Warning]]:
    """把 instance 之间的空隙分类为 pre_arch / post_arch / inter_layer_region.

    intra_layer_gap / intra_layer_outlier 在 assemble_layers 阶段才能算（需要先拼
    layer），这里只处理 component 间的全局空隙。
    """
    if not instances:
        return [], []
    sorted_inst = sorted(instances, key=lambda i: i.start)
    regions: list[UnmatchedRegion] = []
    warnings: list[Warning] = []

    if sorted_inst[0].start > 0:
        regions.append(UnmatchedRegion(0, sorted_inst[0].start - 1, "pre_arch"))

    for a, b in zip(sorted_inst, sorted_inst[1:]):
        if a.end + 1 <= b.start - 1:
            regions.append(UnmatchedRegion(a.end + 1, b.start - 1, "inter_layer_region"))

    if sorted_inst[-1].end < n - 1:
        regions.append(UnmatchedRegion(sorted_inst[-1].end + 1, n - 1, "post_arch"))

    # 末尾未匹配的"重复 region"检测：把 post_arch + 末尾连续若干 inter_layer_region
    # 合并看是否长度相似且 ≥ 2 个 → suspected_undeclared_component
    # 必须先于 LARGE_UNMATCHED_REGION_OPS 检测：尾部重复模式比"单段巨大"更有信息量。
    tail_regions = []
    for r in reversed(regions):
        if r.classification in ("post_arch", "inter_layer_region"):
            tail_regions.append(r)
            if len(tail_regions) > 4:
                break
        else:
            break
    tail_regions.reverse()
    lens = [r.end - r.start + 1 for r in tail_regions]
    if _are_lengths_similar(lens):
        for r in tail_regions:
            r.classification = "suspected_undeclared_component"
        warnings.append(Warning(
            code="suspected_undeclared_component",
            message=(f"末尾发现 {len(tail_regions)} 个长度 ~{lens[0]} 的重复区间未匹配，"
                     f"可能漏声明 MTP / lm_head 等 component"),
            extra={"row_ranges": [[r.start, r.end] for r in tail_regions]},
        ))

    # 余下的 inter_layer_region 单段 > LARGE_UNMATCHED_REGION_OPS 视为漏声明的
    # component（sampler / lm_head / draft head 等被当 gap 吞掉的常见症状）：
    # 升级 classification + 攒一条 large_unmatched_region 警告，render.py 据此硬停。
    large_regions = [
        r for r in regions
        if r.classification == "inter_layer_region"
        and (r.end - r.start + 1) > LARGE_UNMATCHED_REGION_OPS
    ]
    if large_regions:
        for r in large_regions:
            r.classification = "suspected_undeclared_component"
        warnings.append(Warning(
            code="large_unmatched_region",
            message=(f"{len(large_regions)} 段 inter_layer_region > "
                     f"{LARGE_UNMATCHED_REGION_OPS} ops 被升级为 "
                     f"suspected_undeclared_component；可能漏声明 component "
                     f"(sampler / lm_head / draft head 等)，回 Phase 0a/0b 补"),
            extra={
                "threshold": LARGE_UNMATCHED_REGION_OPS,
                "regions": [
                    {"op_range": [r.start, r.end], "length": r.end - r.start + 1}
                    for r in large_regions
                ],
            },
        ))

    return regions, warnings


def _pick_composition_by_lookahead(
    candidates: list[dict], by_start: list[Instance],
    consumed: list[bool], inst_idx: int, n: int,
) -> tuple[dict, bool]:
    """多 composition 共享 first.component 时，扫接下来 K 条未消费 instance
    的 type 多集，挑能被该 multiset 最大覆盖的 composition。

    返回 (chosen, tie)：tie=True 表示最高分仍有多个 composition 并列。
    K 取当前 candidate 的 components 数；若 candidate 间长度不同，分别用各自长度
    取 lookahead 切片。
    """
    if len(candidates) == 1:
        return candidates[0], False
    scores = []
    for comp in candidates:
        need = list(comp["components"])
        pending_types: list[str] = []
        for j in range(inst_idx, n):
            if consumed[j]:
                continue
            pending_types.append(by_start[j].component)
            if len(pending_types) >= len(need):
                break
        remaining = list(pending_types)
        s = 0
        for t in need:
            if t in remaining:
                remaining.remove(t)
                s += 1
        scores.append(s)
    top = max(scores)
    winners = [c for c, s in zip(candidates, scores) if s == top]
    return winners[0], len(winners) > 1


def assemble_layers(instances: list[Instance], structure_spec: dict
                    ) -> tuple[list[Instance], list[Warning]]:
    """row-order composition 推断.

    遍历每个 phase 的声明 layer 数 N：
      - pending = phase row 范围内未消费 instance（按 row 升序）
      - 对每个 layer slot：看 pending[0].component，找匹配的 declared composition
        （component ∈ comp.components 且 comp 与已选 composition 之间可区分）
      - 选中 composition → 从 pending 顺序消费一个该 composition 包含的每个 type
      - 给消费的 instance 填 layer_idx / phase

    返回填好 layer_idx / phase 的 instances + warnings。
    """
    if not structure_spec.get("phases"):
        return instances, []
    sorted_inst = sorted(instances, key=lambda i: i.start)
    consumed = [False] * len(sorted_inst)
    by_start: list[Instance] = sorted_inst
    warnings: list[Warning] = []

    inst_idx = 0
    next_layer_id = 0
    n = len(sorted_inst)
    for phase in structure_spec["phases"]:
        declared = phase["layer_compositions"]
        layers_expected = phase["layers"]
        layers_emitted = 0
        for _ in range(layers_expected):
            if inst_idx >= n:
                warnings.append(Warning(
                    code="layer_count_mismatch",
                    message=(f"phase {phase['name']} 期望 {layers_expected} 层，"
                             f"实测仅拼出 {layers_emitted}"),
                    extra={"phase": phase["name"],
                           "declared": layers_expected,
                           "detected": layers_emitted},
                ))
                break
            first = by_start[inst_idx]
            candidates = [c for c in declared if first.component in c["components"]]
            if not candidates:
                warnings.append(Warning(
                    code="composition_mismatch",
                    message=(f"phase {phase['name']} layer {next_layer_id}: "
                             f"first instance type={first.component} "
                             f"不属于任何声明 composition"),
                    extra={"phase": phase["name"],
                           "layer_idx": next_layer_id,
                           "unexpected_type": first.component},
                ))
                inst_idx += 1
                continue
            # 多 composition 共享 first.component（如 main phase 同时声明
            # [mla,dense] 和 [mla,moe]）时，靠 lookahead 看接下来 K 条未消费
            # instance 的 type 多集，选与之最匹配的那个 composition。
            comp, tie = _pick_composition_by_lookahead(
                candidates, by_start, consumed, inst_idx, n)
            if tie:
                warnings.append(Warning(
                    code="ambiguous_composition",
                    message=(f"phase {phase['name']} layer {next_layer_id}: "
                             f"type={first.component} 同时属于多个声明 composition，"
                             f"lookahead 仍无法区分"),
                    extra={"phase": phase["name"],
                           "layer_idx": next_layer_id,
                           "type": first.component},
                ))
            need = list(comp["components"])
            for t in need:
                consumed_here = False
                for j in range(inst_idx, n):
                    if consumed[j]:
                        continue
                    if by_start[j].component == t:
                        by_start[j].layer_idx = next_layer_id
                        by_start[j].phase = phase["name"]
                        consumed[j] = True
                        consumed_here = True
                        if j == inst_idx:
                            while inst_idx < n and consumed[inst_idx]:
                                inst_idx += 1
                        break
                if not consumed_here:
                    warnings.append(Warning(
                        code="composition_mismatch",
                        message=(f"phase {phase['name']} layer {next_layer_id}: "
                                 f"composition {comp['components']} 缺 type={t}"),
                        extra={"phase": phase["name"],
                               "layer_idx": next_layer_id,
                               "missing_type": t},
                    ))
            next_layer_id += 1
            layers_emitted += 1
        if layers_emitted != layers_expected:
            pass  # 已 warn

    return [i for i in by_start if i.layer_idx >= 0], warnings


def build_validation(structure_spec: dict, instances: list[Instance]
                     ) -> tuple[dict, list[Warning]]:
    """component_count_mismatch 校验.

    某 type 实测 instance 数 ≠ Σ phase.layers × count(type ∈ composition)。
    """
    warnings: list[Warning] = []
    expected_per_type: Counter = Counter()
    for phase in structure_spec.get("phases", []):
        for comp in phase["layer_compositions"]:
            for t in comp["components"]:
                expected_per_type[t] += phase["layers"] // len(phase["layer_compositions"])
                # 上面是粗估；精确期望需要知道每种 composition 在 phase 内的比例，
                # 但 row-order 推断本身就决定了比例，所以这里只能给一个上界
    detected_per_type = Counter(i.component for i in instances)

    declared_layers = sum(p["layers"] for p in structure_spec.get("phases", []))
    detected_layers = len({(i.phase, i.layer_idx) for i in instances})

    per_phase_match = {}
    for phase in structure_spec.get("phases", []):
        decl = phase["layers"]
        det = len({i.layer_idx for i in instances if i.phase == phase["name"]})
        per_phase_match[phase["name"]] = [decl, det]
        if det != decl:
            warnings.append(Warning(
                code="layer_count_mismatch",
                message=f"phase {phase['name']}: 声明 {decl} 层，实测 {det} 层",
                extra={"phase": phase["name"], "declared": decl, "detected": det},
            ))

    expected_components = set(structure_spec.get("expected_components", []))
    detected_components = set(detected_per_type)
    missing = expected_components - detected_components
    for t in missing:
        warnings.append(Warning(
            code="component_count_mismatch",
            message=f"component {t}: 期望 ≥ 1，实测 0",
            extra={"component": t, "detected": 0},
        ))

    validation = {
        "declared_layer_count": declared_layers,
        "detected_layer_count": detected_layers,
        "per_phase_layer_match": per_phase_match,
        "missing_samples": [],
        "ambiguous_matches": [],
    }
    return validation, warnings


def _low_shape_confidence_warnings(instances: list[Instance]) -> list[Warning]:
    out = []
    for i in instances:
        bs = i.scores.get("body_shape_ms")
        if bs is not None and bs < SHAPE_CONFIDENCE_THRESHOLD:
            out.append(Warning(
                code="low_shape_confidence",
                message=(f"layer {i.layer_idx} type={i.component}: "
                         f"body_shape_ms={bs:.2f} (<{SHAPE_CONFIDENCE_THRESHOLD})"),
                extra={"layer_idx": i.layer_idx, "component": i.component,
                       "body_shape_ms": bs},
            ))
    return out


def run_sample_mode(ops: list[dict], structure_spec: dict,
                    samples: dict[str, tuple[int, int]],
                    strict_composition: bool = False) -> SampleDraft:
    """主入口：从 ops + spec + samples 跑全流程.

    Args:
        ops: raw_ops.json 的 operators 列表
        structure_spec: 0a 解析结果 dict
        samples: {component_name: (lo, hi)}（闭区间 row 号）
        strict_composition: 若 True，遇到 composition_mismatch warning 抛 SystemExit

    Returns:
        SampleDraft（结构见 dataclass）
    """
    expected = list(structure_spec.get("expected_components", []))
    missing = [c for c in expected if c not in samples]
    if missing:
        raise SystemExit(
            f"missing samples for components: {missing}; "
            f"expected_components={expected}"
        )

    prefix = build_prefix_counts(ops)
    fingerprints: dict[str, Fingerprint] = {
        comp: extract_fingerprint(ops, comp, lo, hi, k=K)
        for comp, (lo, hi) in samples.items()
    }

    candidates_by_comp: dict[str, list[InstanceCandidate]] = {}
    for comp, fp in fingerprints.items():
        cands = match_component(ops, prefix, fp)
        cands = nms_candidates(cands)
        candidates_by_comp[comp] = cands

    instances, arb_warnings = arbitrate_overlaps(candidates_by_comp)

    layers_filled, asm_warnings = assemble_layers(instances, structure_spec)

    unmatched, unmatched_warnings = classify_unmatched(layers_filled, n=len(ops))

    validation, val_warnings = build_validation(structure_spec, layers_filled)
    shape_warnings = _low_shape_confidence_warnings(layers_filled)

    all_warnings = (arb_warnings + asm_warnings + unmatched_warnings
                    + val_warnings + shape_warnings)

    if strict_composition and any(w.code == "composition_mismatch" for w in all_warnings):
        for w in all_warnings:
            if w.code == "composition_mismatch":
                print(f"strict-composition: {w.message}", flush=True)
        raise SystemExit(1)

    samples_used = [
        {
            "component": comp,
            "op_range": list(samples[comp]),
            "length": fingerprints[comp].length,
            "head_size": K,
            "tail_size": K,
        }
        for comp in samples
    ]

    return SampleDraft(
        samples_used=samples_used,
        components=layers_filled,
        unmatched_regions=unmatched,
        warnings=all_warnings,
        validation=validation,
    )


def _component_expected_samples(structure_spec: dict, sample_ack: dict) -> list[str]:
    expected = list(structure_spec.get("expected_components", []))
    if not expected:
        expected = sorted(sample_ack.get("components", {}).keys())
    return expected


def _stream_samples_for_component(sample_ack: dict, component: str) -> tuple[str, list[dict]]:
    entry = (sample_ack.get("components") or {}).get(component)
    if not entry:
        raise SystemExit(f"missing stream sample for component {component}")
    stream_samples = entry.get("stream_samples") or []
    if not stream_samples:
        raise SystemExit(f"component {component} has no stream_samples")
    primary_entries = [s for s in stream_samples if s.get("role") == "primary"]
    if len(primary_entries) != 1:
        raise SystemExit(
            f"component {component} must have exactly one primary stream sample, "
            f"got {len(primary_entries)}"
        )
    primary_stream_id = str(entry.get("primary_stream_id") or primary_entries[0]["stream_id"])
    return primary_stream_id, stream_samples


def _component_candidates_for_stream_ack(
    ops_by_idx: dict[int, dict],
    streams: dict[str, list[int]],
    component: str,
    primary_stream_id: str,
    stream_samples: list[dict],
) -> tuple[list[dict], list[Warning], dict]:
    warnings: list[Warning] = []
    stream_cands: dict[str, list[StreamSegment]] = {}
    role_by_stream: dict[str, str] = {}
    sample_echo = {
        "component": component,
        "primary_stream_id": primary_stream_id,
        "stream_samples": [],
    }

    for sample in stream_samples:
        sid = str(sample["stream_id"])
        role = sample.get("role") or "unknown"
        role_by_stream[sid] = role
        indices = [int(i) for i in sample.get("op_indices") or []]
        sample_echo["stream_samples"].append({
            "stream_id": sid,
            "role": role,
            "op_indices": indices,
        })
        if indices:
            actual_head = _name_of(ops_by_idx[indices[0]]) if indices[0] in ops_by_idx else None
            actual_tail = _name_of(ops_by_idx[indices[-1]]) if indices[-1] in ops_by_idx else None
            expected_head = sample.get("head_op")
            expected_tail = sample.get("tail_op")
            if expected_head and actual_head and str(expected_head) != str(actual_head):
                warnings.append(Warning(
                    code="sample_ack_mismatch",
                    message=(f"component {component}: stream {sid} head_op "
                             f"{expected_head!r} != actual {actual_head!r}"),
                    extra={"component": component, "stream_id": sid,
                           "field": "head_op", "expected": expected_head,
                           "actual": actual_head, "op_index": indices[0]},
                ))
            if expected_tail and actual_tail and str(expected_tail) != str(actual_tail):
                warnings.append(Warning(
                    code="sample_ack_mismatch",
                    message=(f"component {component}: stream {sid} tail_op "
                             f"{expected_tail!r} != actual {actual_tail!r}"),
                    extra={"component": component, "stream_id": sid,
                           "field": "tail_op", "expected": expected_tail,
                           "actual": actual_tail, "op_index": indices[-1]},
                ))
        if role == "unknown":
            continue
        cands = _scan_stream_segment(ops_by_idx, streams.get(sid, []), indices)
        if not cands:
            low_shape = _low_shape_stream_matches(ops_by_idx, streams.get(sid, []), indices)
            for match in low_shape[:3]:
                warnings.append(Warning(
                    code="stream_shape_mismatch",
                    message=(f"component {component}: stream {sid} has name/body "
                             f"match but shape score "
                             f"{match['scores']['stream_shape']:.2f} below "
                             f"{SHAPE_CONFIDENCE_THRESHOLD:.2f}"),
                    extra={"component": component, "stream_id": sid,
                           "op_indices": match["op_indices"],
                           "scores": match["scores"]},
                ))
        for c in cands:
            c.role = role
        stream_cands[sid] = cands

    primary = stream_cands.get(primary_stream_id) or []
    if not primary:
        warnings.append(Warning(
            code="primary_stream_missing",
            message=f"component {component}: primary stream {primary_stream_id} matched no segment",
            extra={"component": component, "stream_id": primary_stream_id},
        ))
        return [], warnings, sample_echo

    candidates = []
    # 每条 aux 流先做一次全局最优匹配（occurrence ↔ segment），再按 occurrence 组装。
    aux_sids = [sid for sid in stream_cands
                if sid != primary_stream_id
                and role_by_stream.get(sid, "auxiliary") != "unknown"]
    aux_match: dict[str, dict[int, tuple[int, StreamSegment, dict]]] = {}
    aux_ambig: dict[str, dict[int, dict]] = {}
    for sid in aux_sids:
        assignment, ambiguous = _match_auxiliary_segments(
            primary, stream_cands[sid], ops_by_idx)
        aux_match[sid] = assignment
        aux_ambig[sid] = ambiguous

    for occ, primary_seg in enumerate(primary):
        segments = [primary_seg]
        for sid in aux_sids:
            entry = aux_match.get(sid, {}).get(occ)
            if entry is None:
                warnings.append(Warning(
                    code="auxiliary_stream_missing",
                    message=(f"component {component} occurrence {occ}: "
                             f"auxiliary stream {sid} has no matching segment"),
                    extra={"component": component, "occurrence_idx": occ,
                           "stream_id": sid},
                ))
                continue
            _pos, seg, metrics = entry
            seg.scores.update(metrics)
            segments.append(seg)
            if occ in aux_ambig.get(sid, {}):
                warnings.append(Warning(
                    code="auxiliary_stream_ambiguous",
                    message=(f"component {component} occurrence {occ}: "
                             f"auxiliary stream {sid} has multiple time-near matches"),
                    extra={"component": component, "occurrence_idx": occ,
                           "stream_id": sid, **aux_ambig[sid][occ]},
                ))
        op_indices = sorted({idx for seg in segments for idx in seg.op_indices})
        scores = {
            "primary_stream_sequence": primary_seg.scores.get("stream_sequence", 0.0),
            "stream_body": min((s.scores.get("stream_body", 0.0) for s in segments), default=0.0),
            "stream_shape": min((s.scores.get("stream_shape", 0.0) for s in segments), default=0.0),
            "auxiliary_stream_coverage": (
                (len(segments) - 1) /
                max(1, sum(1 for sid, role in role_by_stream.items()
                           if sid != primary_stream_id and role == "auxiliary"))
            ) if any(role == "auxiliary" for role in role_by_stream.values()) else 1.0,
        }
        candidates.append({
            "type": component,
            "occurrence_idx": occ,
            "primary_stream_id": primary_stream_id,
            "op_indices": op_indices,
            "stream_segments": segments,
            "scores": scores,
        })
    return candidates, warnings, sample_echo


def _arbitrate_stream_candidates(candidates: list[dict]) -> tuple[list[dict], list[Warning]]:
    kept: list[dict] = []
    occupied: dict[int, dict] = {}
    warnings: list[Warning] = []
    ordered = sorted(
        candidates,
        key=lambda c: (
            -c["scores"].get("primary_stream_sequence", 0.0),
            -c["scores"].get("stream_body", 0.0),
            c["type"],
            c["occurrence_idx"],
        ),
    )
    for cand in ordered:
        conflicts = [idx for idx in cand["op_indices"] if idx in occupied]
        if not conflicts:
            kept.append(cand)
            for idx in cand["op_indices"]:
                occupied[idx] = cand
            continue
        warnings.append(Warning(
            code="op_membership_conflict",
            message=(f"component {cand['type']} occurrence {cand['occurrence_idx']} "
                     f"shares {len(conflicts)} ops with another candidate"),
            extra={"component": cand["type"],
                   "occurrence_idx": cand["occurrence_idx"],
                   "op_indices": conflicts[:20]},
        ))
    return sorted(kept, key=lambda c: (c["type"], c["occurrence_idx"])), warnings


def _expand_schedule(structure_spec: dict) -> tuple[list[dict], list[Warning]]:
    warnings: list[Warning] = []
    schedule: list[dict] = []
    global_base = 0
    for phase in structure_spec.get("phases", []):
        phase_name = phase.get("name", "")
        layers = int(phase.get("layers", 0))
        per_layer: dict[int, list[str]] = {}
        for comp in phase.get("layer_compositions", []):
            local_indices: list[int] = []
            if "layer_range" in comp:
                lo, hi = comp["layer_range"]
                local_indices = list(range(int(lo), int(hi) + 1))
            elif "layer_indices" in comp:
                local_indices = [int(i) for i in comp["layer_indices"]]
            else:
                warnings.append(Warning(
                    code="composition_schedule_missing",
                    message=(f"phase {phase_name}: layer_compositions item "
                             f"{comp.get('components')} has no layer_range/layer_indices"),
                    extra={"phase": phase_name, "components": comp.get("components", [])},
                ))
                continue
            for local_idx in local_indices:
                if local_idx < 0 or local_idx >= layers:
                    warnings.append(Warning(
                        code="composition_mismatch",
                        message=f"phase {phase_name}: layer index {local_idx} out of range",
                        extra={"phase": phase_name, "layer_idx": local_idx},
                    ))
                    continue
                if local_idx in per_layer:
                    raise SystemExit(
                        f"composition overlap: phase {phase_name} layer {local_idx} "
                        f"同时属于多个 composition（已分配 {per_layer[local_idx]}，又被 "
                        f"{list(comp.get('components', []))} 覆盖）。相邻层用不同 composition "
                        f"的交错结构请用 layer_indices 显式枚举各 composition 的层号，"
                        f"不要用相互重叠的 layer_range。"
                    )
                per_layer[local_idx] = list(comp.get("components", []))
        for local_idx in range(layers):
            comps = per_layer.get(local_idx)
            if comps is None:
                warnings.append(Warning(
                    code="composition_schedule_missing",
                    message=f"phase {phase_name}: layer {local_idx} has no composition schedule",
                    extra={"phase": phase_name, "layer_idx": local_idx},
                ))
                continue
            schedule.append({
                "phase": phase_name,
                "layer_idx": global_base + local_idx,
                "local_layer_idx": local_idx,
                "components": comps,
            })
        global_base += layers
    return schedule, warnings


def _assemble_stream_components(candidates: list[dict], structure_spec: dict
                                ) -> tuple[list[StreamComponent], list[Warning]]:
    schedule, warnings = _expand_schedule(structure_spec)
    if any(w.code == "composition_schedule_missing" for w in warnings):
        return [], warnings

    by_type: dict[str, list[dict]] = {}
    for cand in candidates:
        by_type.setdefault(cand["type"], []).append(cand)
    for cands in by_type.values():
        cands.sort(key=lambda c: c["occurrence_idx"])

    cursors: Counter = Counter()
    components: list[StreamComponent] = []
    for layer in schedule:
        for ctype in layer["components"]:
            idx = cursors[ctype]
            available = by_type.get(ctype, [])
            if idx >= len(available):
                warnings.append(Warning(
                    code="composition_mismatch",
                    message=(f"phase {layer['phase']} layer {layer['layer_idx']}: "
                             f"missing component {ctype} occurrence {idx}"),
                    extra={"phase": layer["phase"], "layer_idx": layer["layer_idx"],
                           "missing_type": ctype, "occurrence_idx": idx},
                ))
                continue
            cand = available[idx]
            cursors[ctype] += 1
            component_id = (
                f"{layer['phase']}.L{int(layer['layer_idx']):03d}."
                f"{ctype}.{cand['occurrence_idx']}"
            )
            components.append(StreamComponent(
                component_id=component_id,
                type=ctype,
                phase=layer["phase"],
                layer_idx=layer["layer_idx"],
                occurrence_idx=cand["occurrence_idx"],
                primary_stream_id=cand["primary_stream_id"],
                op_indices=list(cand["op_indices"]),
                stream_segments=list(cand["stream_segments"]),
                scores=dict(cand["scores"]),
            ))

    for ctype, available in by_type.items():
        if cursors[ctype] < len(available):
            warnings.append(Warning(
                code="composition_mismatch",
                message=(f"component {ctype}: {len(available) - cursors[ctype]} "
                         f"matched occurrences were not consumed by schedule"),
                extra={"component": ctype, "unused_count": len(available) - cursors[ctype]},
            ))
    return components, warnings


def _unmatched_stream_segments(unmatched: list[int],
                               op_to_stream_pos: dict[int, dict]) -> list[dict]:
    by_stream: dict[str, list[int]] = {}
    for idx in unmatched:
        info = op_to_stream_pos.get(idx, {"stream_id": "unknown", "pos": idx})
        by_stream.setdefault(str(info["stream_id"]), []).append(idx)
    out = []
    for sid, indices in sorted(by_stream.items()):
        indices.sort(key=lambda i: op_to_stream_pos.get(i, {}).get("pos", i))
        current = []
        last_pos = None
        for idx in indices:
            pos = op_to_stream_pos.get(idx, {}).get("pos", idx)
            if current and last_pos is not None and pos != last_pos + 1:
                out.append({"stream_id": sid, "op_indices": current})
                current = []
            current.append(idx)
            last_pos = pos
        if current:
            out.append({"stream_id": sid, "op_indices": current})
    return out


def detect_displaced_aux_streams(candidates: list[dict],
                                 ops_by_idx: dict[int, dict]
                                 ) -> tuple[list[Warning], set[tuple[str, str]]]:
    """并入膨胀体检——形态无关，只度量"撑大包络"这一危害本身。

    对每个 component 实例量"并入某条 aux 流让 component 时间包络相对主流自身膨胀多少"：
        primary_span = max_end(primary) − min_start(primary)
        total_span   = max_end(primary∪aux) − min_start(primary∪aux)
        inflation    = (total_span − primary_span) / primary_span
    按 (component_type, aux_stream) 聚合：若多数 occurrence 满足 overlap==0 且
    inflation > AUX_DISPLACEMENT_RATIO_LIMIT，则该流与主流时间脱节，并入会撑大 bubble。

    不识别任何具体流形态，只看包络是否被撑大；并发辅流（overlap>0）天然不进判定。
    只发 hard warning（detect_structure 会 block），不改 membership —— displaced 流的 op
    默认仍保留在 component（单列 displaced_op_indices 供 render 从 cluster/bubble/TOTAL 剔除
    并标注）；用户仅在确认该流不属于该 component 时才于 Phase 0b 改判为 unmatched。

    口径/边界：overlap 与 total 用 segment 时间包络（min_start/max_end），非 op 级 union；
    对内部时间稀疏的 aux seg 是近似，per-layer 紧凑 aux 下足够。退化主流（p_len<=0 或
    primary/aux 段为空）信息不足，跳过。判据全是相对/结构量（inflation 比 + 流级多数），
    无绝对时间阈值——避免量级过拟合（prefill/小模型/不同芯片量级各异）；偶发单点膨胀由
    流级多数（FRACTION）过滤，不靠绝对地板。
    """
    by_comp_aux: dict[tuple[str, str], list[dict]] = {}
    for cand in candidates:
        segs = cand.get("stream_segments") or []
        primary = next((s for s in segs if s.role == "primary"), None)
        if primary is None or not primary.op_indices:
            continue
        p_start, p_end = _segment_time_bounds(primary, ops_by_idx)
        p_len = p_end - p_start
        if p_len <= 0:        # 主流跨度为 0：膨胀比无意义，不纳入判定
            continue
        for s in segs:
            if s.role == "primary" or not s.op_indices:
                continue
            a_start, a_end = _segment_time_bounds(s, ops_by_idx)
            overlap = max(0.0, min(p_end, a_end) - max(p_start, a_start))
            displacement = (max(p_end, a_end) - min(p_start, a_start)) - p_len
            by_comp_aux.setdefault((cand["type"], s.stream_id), []).append({
                "overlap": overlap,
                "inflation": displacement / p_len,
                "occ": cand.get("occurrence_idx"),
                "op_indices": list(s.op_indices),
            })
    out: list[Warning] = []
    displaced_set: set[tuple[str, str]] = set()
    for (ctype, sid), recs in sorted(by_comp_aux.items()):
        displaced = [r for r in recs
                     if r["overlap"] <= 0.0
                     and r["inflation"] > AUX_DISPLACEMENT_RATIO_LIMIT]
        if recs and len(displaced) / len(recs) > AUX_DISPLACEMENT_FRACTION:
            displaced_set.add((ctype, sid))
            ratios = sorted(r["inflation"] for r in displaced)
            median_ratio = ratios[len(ratios) // 2]
            out.append(Warning(
                code="auxiliary_stream_temporally_displaced",
                message=(
                    f"component {ctype}: auxiliary stream {sid} 与主流时间脱节"
                    f"（{len(displaced)}/{len(recs)} occurrence overlap=0 且膨胀比 > "
                    f"{AUX_DISPLACEMENT_RATIO_LIMIT}）。op 仍保留在本 component，render 会"
                    f"自动从 cluster/bubble/TOTAL 剔除并标注；仅当确认该流不属于本 component "
                    f"时才在 Phase 0b 改判为 unmatched。"),
                extra={
                    "component": ctype,
                    "stream_id": sid,
                    "displaced_occurrence_count": len(displaced),
                    "total_occurrence_count": len(recs),
                    "displaced_median_inflation_ratio": round(median_ratio, 2),
                    "example_op_indices": displaced[0]["op_indices"][:20],
                },
            ))
    return out, displaced_set


def run_stream_sample_mode(ops: list[dict], structure_spec: dict,
                           sample_ack: dict
                           ) -> StreamSampleDraft:
    """Stream-aware Step 2 implementation.

    Consumes stream_sample_ack.v1. Row/time ranges must be converted during
    Phase 0b before this function runs; they are not accepted as structure
    facts.
    """
    ack = _normalize_stream_ack(sample_ack)
    streams, op_to_stream_pos = build_stream_index(ops)
    ops_by_idx = _ops_by_index(ops)
    expected = _component_expected_samples(structure_spec, ack)

    all_candidates: list[dict] = []
    all_warnings: list[Warning] = []
    samples_used: list[dict] = []
    for component in expected:
        primary, stream_samples = _stream_samples_for_component(ack, component)
        cands, warnings, sample_echo = _component_candidates_for_stream_ack(
            ops_by_idx, streams, component, primary, stream_samples)
        all_candidates.extend(cands)
        all_warnings.extend(warnings)
        samples_used.append(sample_echo)

    kept_candidates, arb_warnings = _arbitrate_stream_candidates(all_candidates)
    all_warnings.extend(arb_warnings)
    disp_warnings, displaced_streams = detect_displaced_aux_streams(kept_candidates, ops_by_idx)
    all_warnings.extend(disp_warnings)
    components, asm_warnings = _assemble_stream_components(kept_candidates, structure_spec)
    all_warnings.extend(asm_warnings)

    op_to_component: dict[str, str] = {}
    for comp in components:
        for idx in comp.op_indices:
            op_to_component[str(idx)] = comp.component_id
    all_indices = sorted(_op_index(op, pos) for pos, op in enumerate(ops))
    unmatched = [idx for idx in all_indices if str(idx) not in op_to_component]
    unmatched_segments = _unmatched_stream_segments(unmatched, op_to_stream_pos)

    declared = sum(int(p.get("layers", 0)) for p in structure_spec.get("phases", []))
    detected = len({(c.phase, c.layer_idx) for c in components})
    per_phase = {}
    for phase in structure_spec.get("phases", []):
        name = phase.get("name", "")
        per_phase[name] = [
            int(phase.get("layers", 0)),
            len({c.layer_idx for c in components if c.phase == name}),
        ]
    validation = {
        "declared_layer_count": declared,
        "detected_layer_count": detected,
        "per_phase_layer_match": per_phase,
        "missing_samples": [],
        "ambiguous_matches": [
            w.extra for w in all_warnings if w.code == "op_membership_conflict"
        ],
    }

    return StreamSampleDraft(
        structure_spec=structure_spec,
        samples_used=samples_used,
        components=components,
        op_to_component=op_to_component,
        unmatched_op_indices=unmatched,
        unmatched_stream_segments=unmatched_segments,
        warnings=all_warnings,
        validation=validation,
        displaced_streams=sorted(displaced_streams),
    )


def stream_draft_to_dict(draft: StreamSampleDraft,
                         structure_spec: Optional[dict] = None) -> dict:
    displaced = {(t, s) for t, s in draft.displaced_streams}

    def _component_dict(c):
        d = {
            "component_id": c.component_id,
            "type": c.type,
            "phase": c.phase,
            "layer_idx": c.layer_idx,
            "occurrence_idx": c.occurrence_idx,
            "primary_stream_id": c.primary_stream_id,
            "op_indices": c.op_indices,
            "stream_segments": [
                {"stream_id": s.stream_id, "role": s.role, "op_indices": s.op_indices}
                for s in c.stream_segments
            ],
            "scores": c.scores,
        }
        # 时间脱节辅流的 op 仍留在 op_indices 全集（matched 不丢、partition 不破），
        # 但单列出来供 render 从 TOTAL/bubble 排除并标注。
        disp_ops = sorted({i for s in c.stream_segments
                           if (c.type, s.stream_id) in displaced
                           for i in s.op_indices})
        if disp_ops:
            d["displaced_op_indices"] = disp_ops
        return d

    return {
        "mode": "stream_sample_driven",
        "schema_version": "structure_draft.stream.v1",
        "structure_spec": structure_spec if structure_spec is not None else draft.structure_spec,
        "samples_used": draft.samples_used,
        "components": [_component_dict(c) for c in draft.components],
        "op_to_component": draft.op_to_component,
        "unmatched_op_indices": draft.unmatched_op_indices,
        "unmatched_stream_segments": draft.unmatched_stream_segments,
        "warnings": [
            {"code": w.code, "message": w.message, **w.extra}
            for w in draft.warnings
        ],
        "validation": draft.validation,
    }
