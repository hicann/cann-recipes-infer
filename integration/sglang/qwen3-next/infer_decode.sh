source set_env.sh

python3 -m sglang.launch_server --model-path ${MODEL_PATH} \
--tp 16 \
--mem-fraction-static 0.85 \
--max-total-tokens 66000 \
--trust-remote-code \
--attention-backend ascend \
--device npu \
--watchdog-timeout 9000 \
--host 0.0.0.0 --port 30001 \
--disable-radix-cache \
--max-running-requests 16 \
--nnodes $nnodes \
--node-rank $node_rank \
--cuda-graph-bs 16 \
--skip-server-warmup \
--mamba-ssm-dtype bfloat16 \
--disaggregation-transfer-backend ascend \
--disaggregation-mode decode \
--dist-init-addr ${d0}:10000 \
--context-length 66000 \
--chunked-prefill-size 327680 \
--max-prefill-tokens 66000 \
--disable-overlap-schedule \
--moe-a2a-backend deepep --deepep-mode auto \