source set_env.sh

python3 -m sglang.launch_server --model-path ${MODEL_PATH} \
--tp 16 \
--enable-dp-attention --dp-size 8 \
--mem-fraction-static 0.85 \
--max-total-tokens 1126400 \
--trust-remote-code \
--attention-backend ascend \
--device npu \
--host 0.0.0.0 --port 30002 \
--max-running-requests 16 \
--cuda-graph-bs 16 \
--disable-overlap-schedule \
--mamba-ssm-dtype bfloat16 \
--context-length 262144 --chunked-prefill-size 71680 --max-prefill-tokens 262144 \
--skip-server-warmup \
--disable-radix-cache \
--nnodes $nnodes --node-rank $node_rank --dist-init-addr ${p0}:10000 \
--watchdog-timeout 9000 \
--moe-a2a-backend deepep --deepep-mode auto \