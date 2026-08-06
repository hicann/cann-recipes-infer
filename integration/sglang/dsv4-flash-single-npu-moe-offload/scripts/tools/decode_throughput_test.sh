#!/usr/bin/env bash
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
# Long-sequence test: build prompts at several lengths, warm up, then loop to measure the
# steady-state decode rate AFTER warmup. Reports prefill TTFT per length, per-iteration decode
# (ms/token, tok/s) and the post-warmup averages.
#
# Prerequisites: the server is up, and CHUNKED_PREFILL_SIZE >= the actual prompt_tokens of the
# longest step (single chunk, to avoid the cross-chunk compressor bug).
#   context-length=65536 is a hard limit: prompt + output must be <= 65536.
#   A 64k single-chunk prefill peaks at ~20GB of activations and may OOM; if it does, lower
#   KT_NUM_GPU_EXPERTS/MEM_FRACTION or drop the 64k step.
#
# Usage:
#   PORT=8020 bash tools/decode_throughput_test.sh          # 130/1k/8k/16k/32k, 1 warmup + 3 iterations
#   TARGET_TOKENS_LIST="32000" PORT=8020 REPEAT=5 WARMUP=2 bash tools/decode_throughput_test.sh
# env:
#   TARGET_TOKENS_LIST  default "130 1000 8000 16000 32000" (space separated; all <=32768, one chunk)
#                       130 is a short-prompt smoke step (uses NAT_SHORT, a natural question, to
#                       check short sequences do not regress); read long-sequence performance
#                       from the 1000 step onwards
#   MAX_NEW             default 1000 (ignore_eos, runs to the limit)
#   REPEAT              default 3   -- measured iterations after warmup (each is one full max_new generation)
#   WARMUP              default 1   -- discarded runs before measuring, to warm the dynamic hot experts
#   CHARS_PER_TOK       default 4.6
#   PORT(8020) / HOST(127.0.0.1) / PY(python)
set -euo pipefail

PORT="${PORT:-8020}"
HOST="${HOST:-127.0.0.1}"
MAX_NEW="${MAX_NEW:-1000}"
REPEAT="${REPEAT:-3}"
WARMUP="${WARMUP:-1}"
CHARS_PER_TOK="${CHARS_PER_TOK:-4.6}"
TARGET_TOKENS_LIST="${TARGET_TOKENS_LIST:-130 1000 8000 16000 32000}"
PY="${PY:-python}"

CONTEXT_LIMIT=65536
GEN_URL="http://${HOST}:${PORT}/generate"

if ! curl -sf -m5 "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
  echo "[longseq][ERROR] Server is down or unhealthy: http://${HOST}:${PORT}/health (start it first)" >&2
  exit 1
fi

# Read one field out of meta_info
metaf() { $PY -c "import json,sys;print(json.load(open(sys.argv[1])).get('meta_info',{}).get(sys.argv[2],'') or '')" "$1" "$2"; }

# Private temp directory: fixed /tmp/longseq_* names would collide between concurrent runs on one
# host, and a pre-planted symlink of the same name could redirect the writes. Use mktemp -d and
# remove it on exit.
WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/longseq.XXXXXXXX")"
trap 'rm -rf "$WORKDIR"' EXIT

echo "[longseq] REPEAT=${REPEAT} WARMUP=${WARMUP} MAX_NEW=${MAX_NEW}  URL=${GEN_URL}"
SUMMARY=()   # one row per step: "TGT pt e1 warmed_dec_msmean warmed_dec_tpsmean"

for TGT in ${TARGET_TOKENS_LIST}; do
  if (( TGT + MAX_NEW > CONTEXT_LIMIT )); then
    echo "[longseq][WARN] step ${TGT}: prompt + output (${MAX_NEW}) may exceed ${CONTEXT_LIMIT}" \
         "and be truncated or rejected."
  fi
  PF="$WORKDIR/prompt_${TGT}.txt"; RQ1="$WORKDIR/req1_${TGT}.json"; RQ2="$WORKDIR/req2_${TGT}.json"

  # ---- build the prompt, a probe body (max_new=1) and the full body (max_new=MAX_NEW) ----
  "$PY" - "$TGT" "$PF" "$CHARS_PER_TOK" "$MAX_NEW" "$RQ1" "$RQ2" <<'PYEOF'
import sys, json
target=int(sys.argv[1]); out=sys.argv[2]; cpt=float(sys.argv[3]); maxn=int(sys.argv[4]); rq1=sys.argv[5]; rq2=sys.argv[6]
facts=[
 "The Roman aqueducts carried water across vast distances using only gravity",
 "Beethoven composed his ninth symphony while almost completely deaf",
 "The mitochondria is often called the powerhouse of the cell",
 "Mount Everest grows a few millimeters taller each year due to plate tectonics",
 "The printing press was invented by Johannes Gutenberg around the year 1440",
 "Honeybees communicate the location of flowers through an intricate waggle dance",
 "The Great Barrier Reef is the largest living structure on the entire Earth",
 "Light from the sun takes about eight minutes and twenty seconds to reach our planet",
 "The Rosetta Stone was the key that helped scholars decipher Egyptian hieroglyphs",
 "Antarctica is simultaneously the driest, the windiest, and the coldest continent",
 "Octopuses have three hearts and blue copper-based blood instead of red iron-based blood",
 "The Eiffel Tower can grow more than fifteen centimeters taller during a hot summer",
 "A single bolt of lightning is roughly five times hotter than the surface of the sun",
 "The human brain contains approximately eighty-six billion individual neurons",
 "Venus rotates so slowly that one of its days is longer than one of its years",
]
# Short steps (130/1k) use real natural questions; long steps (8k+) use synthetic filler, which is
# adequate since those steps only measure performance
NAT_SHORT = (
    "A logistics company ships orders from three warehouses. Warehouse A handles forty percent "
    "of all orders, warehouse B handles thirty-five percent, and warehouse C handles the remaining "
    "twenty-five percent. Historically two percent of A's shipments arrive late, three percent of "
    "B's, and five percent of C's. A customer has just reported that their shipment arrived late. "
    "Walk through the reasoning step by step to compute the probability that the late shipment "
    "originated from warehouse C, state the final numerical probability, and then suggest the single "
    "operational change that would most reduce the company's overall late-delivery rate. Explain why."
)
_nat1k = [
 "Over the past decade, electricity grids around the world have begun a difficult transition away from "
 "fossil-fuel generation toward variable renewable sources such as wind and solar. This shift creates a "
 "fundamental engineering problem: supply and demand must be balanced on the grid at every instant, but "
 "the sun does not always shine and the wind does not always blow, while electricity demand follows its "
 "own daily and seasonal rhythms that rarely align with renewable output. Traditional grids balanced "
 "supply by ramping controllable thermal plants up and down, but a grid dominated by renewables needs a "
 "different toolkit, and energy storage sits at the center of that toolkit.",
 "The most mature grid-scale storage technology today is the lithium-ion battery. Large installations, "
 "sometimes exceeding several hundred megawatt-hours, can absorb excess solar energy in the middle of the "
 "day and release it during the evening demand peak, a pattern operators call energy time-shifting. "
 "Batteries also respond within milliseconds, which makes them valuable for frequency regulation, the fast "
 "second-by-second balancing that keeps grid frequency near its target. However, lithium-ion systems "
 "typically provide only a few hours of storage before they are exhausted, and their cost, while falling "
 "rapidly, still makes multi-day storage impractical at scale.",
 "For longer durations, engineers look to other approaches. Pumped hydroelectric storage, the oldest and "
 "still the largest form of grid storage, pumps water uphill into a reservoir when energy is abundant and "
 "lets it flow back down through turbines when energy is scarce. It offers enormous capacity and long life, "
 "but it depends on suitable geography and large upfront construction. Emerging alternatives include flow "
 "batteries, which store energy in liquid electrolytes and can be scaled independently in power and "
 "capacity; compressed-air and liquid-air systems; gravity-based designs that lift and drop heavy masses; "
 "and green hydrogen, produced by electrolysis when surplus renewable power is available and later burned "
 "or run through fuel cells. Each technology occupies a different point on the trade-off surface between "
 "cost, round-trip efficiency, response time, and storage duration.",
 "The hardest problem is not any single technology but the mismatch of timescales. A grid needs storage "
 "that spans milliseconds for frequency response, hours for daily solar shifting, and, critically, days or "
 "even weeks to ride through long stretches of cloudy, windless weather. No single technology is cheapest "
 "across all of these durations, so future grids will likely combine several, dispatching each where it is "
 "most economical. Planners must also remember that storage is not free energy: every round trip loses some "
 "power to inefficiency, and building storage consumes materials and capital that could otherwise fund "
 "additional generation or transmission.",
 "Economics complicate these choices further. Storage earns revenue in several distinct ways: arbitrage "
 "between cheap and expensive hours, payments for fast frequency services, and capacity payments for being "
 "available during peak demand. A battery optimized purely for daily arbitrage may sit idle during the rare "
 "multi-day shortfalls that actually threaten reliability, while a technology sized for those long shortfalls "
 "may be too expensive to justify on arbitrage alone. Regulators therefore increasingly value storage not by "
 "a single number but by the specific services it can guarantee. Meanwhile, batteries degrade with every "
 "cycle, losing a fraction of their capacity each year, so an operator must decide whether to oversize a "
 "system at the start or plan to augment it later. These financial and lifecycle questions often matter more "
 "to the final decision than the raw physics of any one storage medium.",
 "A regional grid operator is now planning the next fifteen years of investment. The region has abundant "
 "midday solar, growing evening demand, occasional multi-day winter periods of low wind and sun, limited "
 "mountainous terrain, and a policy goal of reaching ninety percent renewable electricity.",
 "Question: Recommend a portfolio of storage technologies for this operator. For each technology you "
 "include, explain which balancing timescale it addresses and why it is well suited to that role, identify "
 "the single biggest risk or limitation of your overall plan, and describe one concrete measurement the "
 "operator should track over the next five years to know whether the plan is working. Be specific and "
 "justify your reasoning.",
]
NAT_1K = "\n\n".join(_nat1k)
if target <= 300:
    prompt = NAT_SHORT
elif target <= 2000:
    prompt = NAT_1K
else:
    chars_needed=int(target*cpt); buf,n,clen=[],0,0
    while clen<chars_needed:
        s=f"{facts[n%len(facts)]} (fact {n})."; buf.append(s); clen+=len(s)+1; n+=1
    prompt=" ".join(buf)+("\n\nQuestion: The passage above is very long and contains many numbered facts. "
        "Cite three distinct facts by their fact numbers, then write a detailed multi-paragraph summary of the different topics covered.\nAnswer:")
open(out,"w").write(prompt)
json.dump({"text":prompt,"sampling_params":{"max_new_tokens":1,"temperature":0}},open(rq1,"w"))
json.dump({"text":prompt,"sampling_params":{"max_new_tokens":maxn,"temperature":0,"ignore_eos":True}},open(rq2,"w"))
print(f"[longseq] step {target}: built {len(prompt)} chars", file=sys.stderr)
PYEOF

  echo "----- step ${TGT} -----"

  # ---- warmup: run WARMUP full generations and discard them, warming the dynamic hot experts ----
  for w in $(seq 1 "${WARMUP}"); do
    MW="$WORKDIR/warm_${TGT}_${w}.json"
    curl -s -m 3600 -X POST "$GEN_URL" -H 'Content-Type: application/json' -d @"${RQ2}" -o "${MW}" \
      -w "  [warmup ${w}/${WARMUP}] http %{http_code} wall %{time_total}s\n"
  done

  # ---- measurement: REPEAT streamed full generations, timing inter-token directly (TTFT is
  #      prefill, the gaps are decode). Immune to prefill jitter, no e2e-minus-probe subtraction. ----
  ANS="$WORKDIR/ans_${TGT}.txt"
  PF_LIST=(); DEC_TPS=(); PT=""
  for r in $(seq 1 "${REPEAT}"); do
    line="$($PY - "$GEN_URL" "$RQ2" "$ANS" <<'PYEOF'
import sys, json, time, os, urllib.request
os.environ["no_proxy"] = "127.0.0.1,localhost"; os.environ["NO_PROXY"] = "127.0.0.1,localhost"
url, reqf, ansf = sys.argv[1], sys.argv[2], sys.argv[3]
body = json.load(open(reqf)); body["stream"] = True
req = urllib.request.Request(url, data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
t0 = time.time(); tf = None; tl = None; pt = None; nc = 0; text = ""
try:
    with urllib.request.urlopen(req, timeout=3600) as rsp:
        for raw in rsp:
            s = raw.decode("utf-8", "ignore").strip()
            if not s:
                continue
            if s.startswith("data:"):
                s = s[5:].strip()
            if s == "[DONE]":
                break
            try:
                o = json.loads(s)
            except Exception:
                continue
            m = o.get("meta_info", {}) or {}
            if m.get("prompt_tokens"):
                pt = m["prompt_tokens"]
            if m.get("completion_tokens"):
                nc = m["completion_tokens"]
            t = o.get("text")
            if t is not None and len(t) > len(text):
                now = time.time()
                if tf is None:
                    tf = now
                tl = now; text = t
    open(ansf, "w").write(text)
except Exception:
    print("- - - 0 ?"); sys.exit()
pf = (tf - t0) if tf else 0.0
dec = (tl - tf) if (tf and tl and tl > tf) else 0.0
if dec > 0 and nc > 1:
    print(f"{pf:.1f} {dec / (nc - 1) * 1000:.1f} {(nc - 1) / dec:.2f} {nc} {pt}")
else:
    print(f"{pf:.1f} - - {nc} {pt}")
PYEOF
)"
    read -r pf decms dectps nc pt <<<"$line"
    printf "    iter %s: prefill %ss | decode %s ms/tok  %s tok/s  (compl %s)\n" "$r" "$pf" "$decms" "$dectps" "$nc"
    [ "$dectps" != "-" ] && DEC_TPS+=("$dectps")
    [ "$pf" != "-" ] && PF_LIST+=("$pf")
    PT="$pt"
  done

  # ---- print this step's model answer (temp=0, from the last run) ----
  echo "  ---- step ${TGT} model answer ----"
  sed 's/^/  | /' "$ANS" 2>/dev/null
  echo "  ---- (end of answer) ----"

  # ---- post-warmup statistics: median decode (immune to prefill jitter), mean prefill ----
  PFMEAN="$($PY -c "import sys;xs=[float(x) for x in sys.argv[1:] if x not in ('','-')];print(('%.1f'%(sum(xs)/len(xs))) if xs else '-')" "${PF_LIST[@]:-}")"
  DSTATS="$($PY -c "import sys;xs=sorted(float(x) for x in sys.argv[1:] if x not in ('','-'));print(('%.2f %.2f %.2f %.2f'%(sum(xs)/len(xs),xs[len(xs)//2],xs[0],xs[-1])) if xs else '- - - -')" "${DEC_TPS[@]:-}")"
  read -r smean smed smin smax <<<"$DSTATS"
  echo "  >> step ${TGT} warmed decode: mean ${smean} / median ${smed} tok/s" \
       "(min ${smin}/max ${smax}), prompt=${PT}, prefill-mean=${PFMEAN}s"
  SUMMARY+=("${TGT} ${PT} ${PFMEAN} ${smean} ${smed} ${smin} ${smax}")
  echo
done

echo "=============================== SUMMARY (post-warmup decode) ==============================="
printf "%8s %10s %12s %11s %11s %9s %9s\n" target prompt_tok warm-pf\(s\) dec-mean dec-med dec-min dec-max
printf -- "-------------------------------------------------------------------------------------------\n"
for row in "${SUMMARY[@]}"; do
  read -r tgt pt e1 mean med mn mx <<<"$row"
  printf "%8s %10s %12s %11s %11s %9s %9s\n" "$tgt" "$pt" "$e1" "$mean" "$med" "$mn" "$mx"
done
echo "==========================================================================================="
echo "Note: decode tok/s = (compl-1)/(e2e - warmed prefill); ${WARMUP} warmup run(s) then" \
     "${REPEAT} measured runs, reported as mean/median."
