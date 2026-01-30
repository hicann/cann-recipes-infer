#!/usr/bin/env python3
"""
DeepSeek-OCR-2 NPU 性能基准测试
测试不同并发数下的吞吐量
"""
import sys
import os
import time
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, '/workspace/vllm_deepseek_ocr2_adapted')
os.chdir('/workspace/vllm_deepseek_ocr2_adapted')

import torch
os.environ["VLLM_USE_V1"] = "0"

from PIL import Image
from typing import List, Dict
import numpy as np
import json

try:
    import torch_npu
    print("[INFO] torch_npu loaded")
except ImportError:
    print("[WARN] torch_npu not available, running on CPU")
except Exception as e:
    print(f"[WARN] Failed to import torch_npu: {e}")

import deepseek_ocr2_npu

from vllm import LLM, SamplingParams
from vllm.model_executor.models.registry import ModelRegistry
from deepseek_ocr2 import DeepseekOCR2ForCausalLM
from process.image_process import DeepseekOCR2Processor
from config import MODEL_PATH, PROMPT, CROP_MODE

ModelRegistry.register_model("DeepseekOCR2ForCausalLM", DeepseekOCR2ForCausalLM)

# 测试图片路径
TEST_IMAGES = [
    '.images/test_image_1.png',
    './test_image_2.png',
]

def load_test_image(path: str) -> Image.Image:
    """加载测试图片"""
    img = Image.open(path).convert('RGB')
    return img

def prepare_requests(num_requests: int) -> List[Dict]:
    """准备多个请求，使用两张图片交替构造"""
    processor = DeepseekOCR2Processor()

    # 预处理两张图片
    image_features_list = []
    for img_path in TEST_IMAGES:
        img = load_test_image(img_path)
        features = processor.tokenize_with_images(
            images=[img], bos=True, eos=True, cropping=CROP_MODE
        )
        image_features_list.append(features)
        print(f"[INFO] Loaded image: {img_path}")

    # 交替构造请求
    requests = []
    for i in range(num_requests):
        # 交替使用两张图片: 0, 1, 0, 1, ...
        img_idx = i % len(TEST_IMAGES)
        requests.append({
            "prompt": PROMPT,
            "multi_modal_data": {"image": image_features_list[img_idx]}
        })

    # 统计图片分布
    img1_count = sum(1 for i in range(num_requests) if i % 2 == 0)
    img2_count = num_requests - img1_count
    print(f"[INFO] Request distribution: image_1={img1_count}, image_2={img2_count}")

    return requests

def run_benchmark(
    concurrent: int,
    tensor_parallel: int = 1,
    gpu_mem_util: float = 0.85,
    max_tokens: int = 2048,
    num_rounds: int = 5,
    warmup_rounds: int = 1
) -> Dict:
    """运行热启动基准测试（多轮取均值）"""
    print(f"\n{'='*60}")
    print(f"Hot-Start Benchmark: concurrent={concurrent}, tp={tensor_parallel}")
    print(f"Rounds: {num_rounds}, Warmup: {warmup_rounds}")
    print(f"Using images: {TEST_IMAGES}")
    print(f"{'='*60}")

    # 准备请求（请求数 = 并发数，交替使用两张图片）
    print("[INFO] Preparing requests...")
    requests = prepare_requests(concurrent)

    # 初始化模型（冷启动，只做一次）
    print("[INFO] Initializing LLM (cold start)...")
    init_start = time.time()

    llm = LLM(
        model=MODEL_PATH,
        hf_overrides={"architectures": ["DeepseekOCR2ForCausalLM"]},
        dtype="bfloat16",
        max_model_len=8192,
        enforce_eager=True,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel,
        gpu_memory_utilization=gpu_mem_util,
        max_num_seqs=concurrent,
        swap_space=0,
    )

    init_time = time.time() - init_start
    print(f"[INFO] Cold start init time: {init_time:.2f}s")

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        skip_special_tokens=False,
    )

    # Warmup 阶段（预热，让模型完全热启动）
    print(f"\n[INFO] Warmup phase ({warmup_rounds} round(s))...")
    for w in range(warmup_rounds):
        _ = llm.generate(requests, sampling_params=sampling_params)
        print(f"  Warmup round {w+1}/{warmup_rounds} done")

    # 多轮热启动测试
    print(f"\n[INFO] Running {num_rounds} rounds of hot-start benchmark...")

    round_results = []
    for r in range(num_rounds):
        infer_start = time.time()
        outputs = llm.generate(requests, sampling_params=sampling_params)
        infer_time = time.time() - infer_start

        # 统计本轮结果
        total_prompt_tokens = 0
        total_output_tokens = 0

        for output in outputs:
            total_prompt_tokens += len(output.prompt_token_ids)
            if output.outputs:
                total_output_tokens += len(output.outputs[0].token_ids)

        # 计算本轮吞吐量
        prompt_throughput = total_prompt_tokens / infer_time
        output_throughput = total_output_tokens / infer_time
        total_throughput = (total_prompt_tokens + total_output_tokens) / infer_time
        avg_latency = infer_time / concurrent

        round_results.append({
            "infer_time_s": infer_time,
            "total_prompt_tokens": total_prompt_tokens,
            "total_output_tokens": total_output_tokens,
            "prompt_throughput": prompt_throughput,
            "output_throughput": output_throughput,
            "total_throughput": total_throughput,
            "avg_latency": avg_latency,
        })

        print(f"  Round {r+1}/{num_rounds}: {infer_time:.2f}s, output_throughput={output_throughput:.2f} tokens/s, latency={avg_latency:.3f}s/req")

    # 计算多轮统计（均值、标准差）
    infer_times = np.array([r["infer_time_s"] for r in round_results])
    prompt_throughputs = np.array([r["prompt_throughput"] for r in round_results])
    output_throughputs = np.array([r["output_throughput"] for r in round_results])
    total_throughputs = np.array([r["total_throughput"] for r in round_results])
    avg_latencies = np.array([r["avg_latency"] for r in round_results])

    # 取第一轮的token数（每轮应该相同）
    total_prompt_tokens = round_results[0]["total_prompt_tokens"]
    total_output_tokens = round_results[0]["total_output_tokens"]

    results = {
        "concurrent": concurrent,
        "tensor_parallel": tensor_parallel,
        "num_rounds": num_rounds,
        "warmup_rounds": warmup_rounds,
        "init_time_s": round(init_time, 2),
        "total_prompt_tokens": total_prompt_tokens,
        "total_output_tokens": total_output_tokens,
        # 均值
        "infer_time_mean_s": round(float(np.mean(infer_times)), 3),
        "prompt_throughput_mean": round(float(np.mean(prompt_throughputs)), 2),
        "output_throughput_mean": round(float(np.mean(output_throughputs)), 2),
        "total_throughput_mean": round(float(np.mean(total_throughputs)), 2),
        "avg_latency_mean_s": round(float(np.mean(avg_latencies)), 3),
        # 标准差
        "infer_time_std_s": round(float(np.std(infer_times)), 3),
        "total_throughput_std": round(float(np.std(total_throughputs)), 2),
        "output_throughput_std": round(float(np.std(output_throughputs)), 2),
        "avg_latency_std_s": round(float(np.std(avg_latencies)), 3),
        # 最小/最大
        "infer_time_min_s": round(float(np.min(infer_times)), 3),
        "infer_time_max_s": round(float(np.max(infer_times)), 3),
        "output_throughput_min": round(float(np.min(output_throughputs)), 2),
        "output_throughput_max": round(float(np.max(output_throughputs)), 2),
    }

    print(f"\n[HOT-START RESULTS] ({num_rounds} rounds, concurrent={concurrent})")
    print(f"  Total Prompt Tokens: {total_prompt_tokens}")
    print(f"  Total Output Tokens: {total_output_tokens}")
    print(f"  ---")
    print(f"  Inference Time:    mean={np.mean(infer_times):.3f}s, std={np.std(infer_times):.3f}s")
    print(f"  Prompt Throughput: mean={np.mean(prompt_throughputs):.2f} tokens/s")
    print(f"  Output Throughput: mean={np.mean(output_throughputs):.2f} tokens/s, std={np.std(output_throughputs):.2f}, min={np.min(output_throughputs):.2f}, max={np.max(output_throughputs):.2f}")
    print(f"  Total Throughput:  mean={np.mean(total_throughputs):.2f} tokens/s")
    print(f"  Avg Latency:       mean={np.mean(avg_latencies):.3f}s/request, std={np.std(avg_latencies):.3f}s")

    # 清理
    del llm
    torch.npu.empty_cache()

    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DeepSeek-OCR-2 Hot-Start Benchmark")
    parser.add_argument('--concurrent', '-c', type=int, default=4, help='Concurrent requests (also batch size per round)')
    parser.add_argument('--tp', type=int, default=1, help='Tensor parallel size')
    parser.add_argument('--gpu-mem', type=float, default=0.85, help='GPU memory utilization')
    parser.add_argument('--max-tokens', type=int, default=2048, help='Max output tokens')
    parser.add_argument('--rounds', '-r', type=int, default=5, help='Number of hot-start test rounds')
    parser.add_argument('--warmup', '-w', type=int, default=1, help='Number of warmup rounds')
    args = parser.parse_args()

    results = run_benchmark(
        concurrent=args.concurrent,
        tensor_parallel=args.tp,
        gpu_mem_util=args.gpu_mem,
        max_tokens=args.max_tokens,
        num_rounds=args.rounds,
        warmup_rounds=args.warmup
    )

    print(f"\n[JSON] {json.dumps(results)}")
