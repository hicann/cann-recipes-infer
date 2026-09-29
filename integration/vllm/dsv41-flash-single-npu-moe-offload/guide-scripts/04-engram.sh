#!/usr/bin/env bash
# Copy the layers.1 and layers.14 engram tables byte for byte from the weights into /dev/shm, then verify.
# Actions: stage | verify | release | all (default)
set -euo pipefail
N=engram
: "${MODEL:?先 source dsv41.env}"
ACTION=${1:-all}

do_stage() {
  python3 - <<'PY'
import json, os, struct, subprocess, time
MODEL = os.environ["MODEL"]; STAGE = os.environ["DSV41_STAGE_DIR"]
os.makedirs(STAGE, exist_ok=True)
wm = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
KEYS = ("layers.1.engram.embed.weight", "layers.1.engram.embed.scale",
        "layers.14.engram.embed.weight", "layers.14.engram.embed.scale")
used = lambda: int(subprocess.run(["df","-B1","--output=used",STAGE],
                                  capture_output=True,text=True).stdout.strip().split("\n")[-1])
t_all, base = time.time(), used()
for k in KEYS:
    t0 = time.time(); src = f"{MODEL}/{wm[k]}"
    with open(src, "rb") as f:
        (hl,) = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(hl)); beg, end = hdr[k]["data_offsets"]
        f.seek(8 + hl + beg)
        with open(f"{STAGE}/{k}.bin", "wb") as out:
            left = end - beg
            while left:
                b = f.read(min(left, 1 << 30))
                if not b:
                    raise SystemExit("[engram] FAIL %s 读到 %s 时提前 EOF，还差 %d B" % (src, k, left))
                out.write(b); left -= len(b)
    sz, dt = os.path.getsize(f"{STAGE}/{k}.bin"), time.time() - t0
    print("[engram] %-34s %15d B  %5.1fs  %5.0f MB/s" % (k, sz, dt, sz/dt/1e6), flush=True)
print("[engram] 暂存合计 %.0fs，/dev/shm 增长 %.1f GiB" % (time.time()-t_all, (used()-base)/2**30))
PY
}

do_verify() {
  python3 - <<'PY'
import importlib.util, os, sys
p = os.path.join(os.environ["ASCEND_TREE"],
                 "vllm_ascend/models/deepseek_v4_1/engram_staging.py")
s = importlib.util.spec_from_file_location("engram_staging", p)
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
KEYS = ["layers.1.engram.embed.weight", "layers.1.engram.embed.scale",
        "layers.14.engram.embed.weight", "layers.14.engram.embed.scale"]
STAGE = os.environ["DSV41_STAGE_DIR"]
sizes = m.verify_staged_tables(KEYS, os.environ["MODEL"], STAGE)
print("[engram] PASS verify_staged_tables，合计 %d B" % sum(sizes.values()))
m.verify_staged_content(m.fingerprint_staged_tables(KEYS, STAGE), STAGE)
print("[engram] PASS verify_staged_content")
PY
}

# PIDs that still map the staging directory.
holders() {
  local d out=""
  for d in /proc/[0-9]*; do
    grep -qs "$DSV41_STAGE_DIR" "$d/maps" 2>/dev/null && out="$out ${d#/proc/}"
  done
  echo "${out# }"
}

do_release() {
  local h
  h=$(holders)
  [ -z "$h" ] || { echo "[$N] FAIL 还有进程映射着 $DSV41_STAGE_DIR（PID: $h），unlink 不会释放内存。"
                   echo "[$N]      先 bash -l 05-serve.sh stop，再 release。"; exit 1; }
  local before after
  before=$(awk '/^Shmem:/{print $2}' /proc/meminfo)
  rm -rf "$DSV41_STAGE_DIR"
  after=$(awk '/^Shmem:/{print $2}' /proc/meminfo)
  echo "[$N] PASS 已释放 $DSV41_STAGE_DIR，Shmem 归还 $(awk -v b="$before" -v a="$after" 'BEGIN{printf "%.1f", (b-a)/1048576}') GiB"
}

case "$ACTION" in
  stage)   do_stage ;;
  verify)  do_verify ;;
  release) do_release ;;
  all)     do_stage; do_verify ;;
  *) echo "用法: bash 04-engram.sh [stage|verify|release|all]"; exit 1 ;;
esac
echo "[$N] /dev/shm $(df -h "$(dirname "$DSV41_STAGE_DIR")" | tail -1 | awk '{print $3" 已用 / "$4" 可用"}')"
echo "=== $N 完成 ==="
