#!/usr/bin/env bash
# Acceptance gates for a running GLM-5.3-Flash single-die offload server.
#   ./verify.sh            run every gate against $GLM53_PORT
#   ./verify.sh chat       interactive client
set -uo pipefail

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=./glm53_env.sh
source "${_here}/glm53_env.sh"

LOG="${GLM53_LOG_DIR}/serve.log"
case "${GLM53_HOST}" in
  0.0.0.0|'') _url_host=127.0.0.1 ;;
  ::|'[::]') _url_host='[::1]' ;;
  '['*']') _url_host="${GLM53_HOST}" ;;
  *:*) _url_host="[${GLM53_HOST}]" ;;
  *) _url_host="${GLM53_HOST}" ;;
esac
BASE="http://${_url_host}:${GLM53_PORT}"
# --fail is load-bearing: without it curl exits 0 on ANY status, and sglang's
# /health_generate answers 503 precisely when the generate loop is unhealthy or shutting
# down -- the condition this gate exists to catch.
CURL=(curl -sS --fail --noproxy '*')
FAIL=0
pass() { printf '  \033[32mPASS\033[0m  %s\n' "$1"; }
fail() { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; FAIL=1; }
info() { printf '  ....  %s\n' "$1"; }

case "${1:-}" in
  -h|--help)
    cat <<'USAGE'
usage: verify.sh [chat]

Acceptance gate for a server that is already running. Reads GLM53_PORT / GLM53_LOG_DIR
and the log the server wrote; starts nothing and occupies no die.

  chat   interactive client against the same server instead of the gate
USAGE
    exit 0 ;;
  chat|"") ;;
  *) echo "[verify] unknown argument: $1 (try --help)" >&2; exit 2 ;;
esac

if [ "${1:-}" = "chat" ]; then
  exec "${GLM53_PYTHON}" - "$BASE" <<'PY'
import json, sys, urllib.request
base = sys.argv[1]
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
msgs = []
print(f"chat against {base}  (empty line or Ctrl-D to quit)")
while True:
    try: q = input("\n>>> ").strip()
    except EOFError: break
    if not q: break
    msgs.append({"role": "user", "content": q})
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps({"model": "glm53", "messages": msgs, "temperature": 0.6,
                          "max_tokens": 1024, "stream": True}).encode(),
        headers={"Content-Type": "application/json"})
    out = []
    with opener.open(req) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: "): continue
            if line == "data: [DONE]": break
            d = json.loads(line[6:])["choices"][0].get("delta", {}).get("content")
            if d: out.append(d); print(d, end="", flush=True)
    print()
    msgs.append({"role": "assistant", "content": "".join(out)})
PY
fi

echo "== GLM-5.3-Flash single-die offload: acceptance =="
echo "   port ${GLM53_PORT}  die ${GLM53_NPU_DEVICE_ID}  experts resident ${GLM53_NUM_GPU_EXPERTS}  log ${LOG}"

# ---- 1. weights and pools, from the log -------------------------------------
if [ -r "${LOG}" ]; then
  _lw="$(grep -oP 'Load weight end.*mem usage=\K[0-9.]+' "${LOG}" | tail -1)"
  if [ -n "${_lw}" ]; then
    # 15.60 GiB of non-expert weights + 0.9925 GiB per resident expert index, from a fit
    # over three measured loads. An earlier 15.46 + 1.009 came from 43
    # layers -- it counted the unserved MTP layer 45 -- and is gone.
    #
    # Streaming prefill adds a 6.75 GiB reusable convert slot, reserved BEFORE the KV pool
    # is sized. Leaving it out of the prediction under-shoots by exactly that and fails a
    # correct server: a K=32 streaming load is 54.11 GiB, this predicted 47.36, and 6.75
    # is well outside the +/-2.5 band. The full fit lives in glm53_env.sh.
    _slot=0
    [ "${GLM53_PREFILL_STREAM:-0}" = "1" ] && _slot=6.75
    _want="$(awk "BEGIN{printf \"%.2f\", 15.60 + ${_slot} + 0.9925*${GLM53_NUM_GPU_EXPERTS}}")"
    info "Load weight end: ${_lw} GB on die (predicted ~${_want} GiB for ${GLM53_NUM_GPU_EXPERTS} resident experts)"
    # TWO-sided, and this matters more on the low side. Too high means the loader failed
    # to skip the offloaded experts. Too low means it skipped experts it should have
    # kept -- every map_logical_expert_id_for_gpu_load returning -1, a mis-sized table, a
    # wrong mask -- and the die then holds ~15.6 GiB and serves wrong experts silently.
    # The old one-sided +8 GiB bound passed at K=32, K=24 and K=0 alike, and it is the
    # only gate in this file that touches the resident-subset loader at all.
    awk "BEGIN{exit !(${_lw} < ${_want} + 2.5 && ${_lw} > ${_want} - 2.5)}" \
      && pass "resident-expert weight load matches the capacity model (+/-2.5 GiB)" \
      || fail "loaded ${_lw} GB against ${_want} GiB predicted for ${GLM53_NUM_GPU_EXPERTS} resident experts -- the resident subset is wrong"
  else
    fail "no 'Load weight end' in ${LOG}"
  fi
  grep -q "KV Cache is allocated" "${LOG}" && pass "KV cache allocated" || fail "no KV cache line"
  if grep -q "Capture target decode NPU graph end\|Capture decode.*graph end" "${LOG}"; then
    pass "decode NPU graph captured"
  else
    info "no decode-graph capture line (graph mode is worth ~5x on this path -- check GLM53_EAGER)"
  fi
  # Count TRACEBACKS, not lines. The old form counted `grep -A3` context lines matching
  # freeze_gc, and one freeze_gc traceback yields 2-3 of them -- so the allowance grew
  # faster than the count and a real crash passed. Verified: a log with one freeze_gc
  # traceback and one AssertionError gave _tb=2 _tb_real=3 -> PASS.
  _tb="$(grep -c "Traceback (most recent call last)" "${LOG}")"
  # awk over the file: a traceback is "excused" only if freeze_gc appears within its own
  # next 3 lines, and each traceback can excuse itself at most once.
  _tb_real="$(awk '/Traceback \(most recent call last\)/{t=NR; excused=0}
                   t && NR>t && NR<=t+3 && /freeze_gc/ && !excused {c++; excused=1}
                   END{print c+0}' "${LOG}")"
  if [ "${_tb}" -le "${_tb_real}" ]; then
    pass "no unexplained tracebacks (${_tb_real} freeze_gc races excused)"
  else
    fail "$(( _tb - _tb_real )) unexplained tracebacks in the log"
  fi
else
  info "no log at ${LOG}, skipping log-derived gates"
fi

# ---- 2. liveness ------------------------------------------------------------
"${CURL[@]}" -m 10 -o /dev/null -w '' "${BASE}/health" \
  && pass "/health" || { fail "/health"; echo; echo "server is not up; the rest cannot run"; exit 1; }
"${CURL[@]}" -m 120 -o /dev/null -w '' "${BASE}/health_generate" \
  && pass "/health_generate" || fail "/health_generate"

# ---- 3. generation, on a prompt long enough to mean something ---------------
# A short prompt proves very little here: GLM's DSA indexer selects everything when
# seq_len < index_topk (2048), so the sparse attention path never runs. "The capital of
# France is -> Paris" passes on a badly broken server. Build a >2048-token context.
"${GLM53_PYTHON}" - "$BASE" "${GLM53_LOG_DIR}" <<'PY'
import json, sys, urllib.request
base, logdir = sys.argv[1], sys.argv[2]
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

filler = ("The following is an inventory log. " +
          " ".join(f"Item {i:04d} is stored in bay {i % 37} on shelf {i % 11}." for i in range(900)))
question = ("\n\nAnswer with a single arabic numeral and nothing else. "
            "In which bay is Item 0123 stored?")
expected = str(123 % 37)

def gen(text, n=8):
    req = urllib.request.Request(
        base + "/generate",
        data=json.dumps({"text": text,
                          "sampling_params": {"max_new_tokens": n, "temperature": 0}}).encode(),
        headers={"Content-Type": "application/json"})
    with opener.open(req, timeout=600) as r:
        d = json.load(r)
    return d["text"], d.get("meta_info", {}).get("prompt_tokens")

prompt = filler + question
try:
    a, ntok = gen(prompt)
except Exception as e:
    print(f"  \033[31mFAIL\033[0m  long-prompt generate raised {e!r}")
    sys.exit(3)

ok = True
print(f"  ....  prompt_tokens={ntok} (need >2048 to exercise the DSA sparse path)")
if not ntok or ntok <= 2048:
    print(f"  \033[31mFAIL\033[0m  prompt was only {ntok} tokens; this gate did not test the sparse path")
    ok = False
if not a.strip():
    print("  \033[31mFAIL\033[0m  empty completion -- usually a --kt-weight-path template that "
          "resolved to nothing, or kt-kernel loaded no experts")
    ok = False
else:
    print(f"  \033[32mPASS\033[0m  non-empty completion: {a.strip()[:60]!r}")

# Greedy determinism, at a FIXED batch width of 1. That is the only width at which
# "identical output" is a legal criterion on this deployment; batch composition is
# scheduler-dependent and not reproducible across runs.
b, _ = gen(prompt)
if a == b:
    print("  \033[32mPASS\033[0m  greedy output reproducible at width 1")
else:
    print(f"  \033[31mFAIL\033[0m  greedy output differs across two identical requests:\n"
          f"          {a.strip()[:60]!r}\n          {b.strip()[:60]!r}")
    ok = False

# Retrieval from a long context. NOT a precision criterion -- it is a smoke test that
# the offloaded experts are contributing something coherent rather than noise. A
# correct answer is evidence; a wrong one warrants investigation, not a verdict.
got = "".join(c for c in a if c.isdigit())[:2]
print(f"  ....  long-context retrieval: answered {got!r}, expected {expected!r}"
      f" -- {'consistent' if got == expected else 'INCONSISTENT, investigate'}")
sys.exit(0 if ok else 3)
PY
[ $? -eq 0 ] || FAIL=1

# ---- 4. the CPU MoE actually carried load -----------------------------------
if [ -r "${LOG}" ]; then
  if grep -qiE "kt.?kernel|KTMoEWrapper|LLAMAFILE|cpuinfer" "${LOG}"; then
    pass "kt-kernel CPU MoE is in the log"
  else
    fail "no kt-kernel evidence in the log -- is --kt-weight-path set?"
  fi
  if [ "${GLM53_PREFILL_STREAM:-0}" = "1" ]; then
    # -a and head -1. serve.log can acquire NUL bytes (CANN writes
    # some), and then grep -c without -a prints nothing at all: _in becomes empty and
    # the arithmetic test below fails with a syntax error instead of a verdict. The
    # `|| true` form also produced "0\n0" once and took a JSON writer down with it.
    _in="$(grep -ac 'inline resident' "${LOG}" 2>/dev/null | head -1)"; _in="${_in:-0}"
    _fb="$(grep -acE 'streaming failed|hybrid fallback' "${LOG}" 2>/dev/null | head -1)"; _fb="${_fb:-0}"
    [ "${_in}" -gt 0 ] && pass "streaming prefill engaged (${_in})" || fail "streaming prefill never engaged"
    [ "${_fb}" -eq 0 ] && pass "no streaming fallbacks" || fail "${_fb} streaming fallbacks"
  fi
fi

echo
if [ "${FAIL}" -eq 0 ]; then
  echo -e "\033[32mALL CHECKS PASSED\033[0m"
  echo "This is a smoke gate, not an accuracy verdict. For that: run_gsm8k.py,"
  echo "and the sharper per-token criteria (perplexity / MMLU single-token)."
else
  echo -e "\033[31mCHECKS FAILED\033[0m"
fi
exit "${FAIL}"
