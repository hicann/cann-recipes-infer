#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Benchmark script for DeepSeek-OCR-2 on Ascend NPU."""
import sys
import os
import time
import argparse
import warnings
import gc
from dataclasses import dataclass, field
from typing import List

warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ["VLLM_USE_V1"] = "0"

import deepseek_ocr2_npu  # noqa: F401

from PIL import Image, ImageOps
from vllm import LLM, SamplingParams
from vllm.model_executor.models.registry import ModelRegistry
from deepseek_ocr2 import DeepseekOCR2ForCausalLM
from process.image_process import DeepseekOCR2Processor
from config import (
    MODEL_PATH,
    CROP_MODE,
    ENFORCE_EAGER,
    MAX_MODEL_LEN,
    TENSOR_PARALLEL_SIZE,
    GPU_MEMORY_UTILIZATION,
    SWAP_SPACE,
    DISABLE_MM_PREPROCESSOR_CACHE,
    PROMPT,
)

ModelRegistry.register_model(
    "DeepseekOCR2ForCausalLM",
    DeepseekOCR2ForCausalLM,
)


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    image_path: str
    concurrent_levels: List[int] = field(
        default_factory=lambda: [1, 8, 16]
    )
    max_tokens: int = 8192
    gpu_mem: float = 0.85
    warmup: int = 2
    rounds: int = 5
    output_file: str = "benchmark_results.txt"


@dataclass
class InferenceMetrics:
    """Metrics collected from a single inference round."""
    time: float
    input_tokens: int
    output_tokens: int
    output_tps: float
    total_tps: float


@dataclass
class BenchmarkResult:
    """Aggregated result for one concurrency level."""
    concurrent: int
    init_time: float
    cold_tps: float
    warm_tps: float
    warm_total_tps: float
    avg_time: float


def load_images_from_file(image_path, count):
    """Load and replicate single image to match count."""
    image = ImageOps.exif_transpose(
        Image.open(image_path)
    ).convert("RGB")
    print(f"[INFO] Loaded single image, replicated to {count} copies")
    return [image] * count


def load_images_from_dir(image_path, count):
    """Load images from directory and cycle to match count."""
    image_files = sorted([
        os.path.join(image_path, fname)
        for fname in os.listdir(image_path)
        if fname.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    if not image_files:
        raise FileNotFoundError(f"No images found in {image_path}")

    loaded = [
        ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        for path in image_files
    ]
    images = [loaded[idx % len(loaded)] for idx in range(count)]
    print(
        f"[INFO] Loaded {len(loaded)} images from directory,"
        f" using {count} for benchmark"
    )
    return images


def load_images(image_path, count):
    """Load images from file or directory."""
    if os.path.isfile(image_path):
        return load_images_from_file(image_path, count)
    elif os.path.isdir(image_path):
        return load_images_from_dir(image_path, count)
    else:
        raise FileNotFoundError(f"Path not found: {image_path}")


def prepare_requests(images, processor):
    """Prepare inference requests from images."""
    requests = []
    for img in images:
        token_data = processor.tokenize_with_images(
            images=[img],
            bos=True,
            eos=True,
            cropping=CROP_MODE,
        )
        requests.append({
            "prompt": PROMPT,
            "multi_modal_data": {"image": token_data},
        })
    return requests


def run_inference(llm, requests, max_tokens):
    """Run single inference round and return metrics."""
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
    )
    start_time = time.time()
    outputs = llm.generate(requests, sampling_params)
    total_time = time.time() - start_time

    total_input = sum(
        len(out.prompt_token_ids) for out in outputs
    )
    total_output = sum(
        len(out.outputs[0].token_ids) for out in outputs
    )

    return InferenceMetrics(
        time=total_time,
        input_tokens=total_input,
        output_tokens=total_output,
        output_tps=total_output / total_time,
        total_tps=(total_input + total_output) / total_time,
    )


def init_llm(gpu_mem):
    """Initialize LLM engine with NPU config."""
    return LLM(
        model=MODEL_PATH,
        hf_overrides={
            "architectures": ["DeepseekOCR2ForCausalLM"],
        },
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=ENFORCE_EAGER,
        trust_remote_code=True,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=gpu_mem,
        swap_space=SWAP_SPACE,
        disable_mm_preprocessor_cache=DISABLE_MM_PREPROCESSOR_CACHE,
    )


def run_warmup(llm, requests, max_tokens, rounds):
    """Run warmup rounds."""
    print(f"[INFO] Running {rounds} warmup rounds...")
    for idx in range(rounds):
        run_inference(llm, requests, max_tokens)
        print(f"  Warmup {idx + 1}/{rounds} done")


def run_test_rounds(llm, requests, max_tokens, rounds):
    """Run test rounds and return results."""
    print(f"[INFO] Running {rounds} test rounds...")
    results = []
    for idx in range(rounds):
        result = run_inference(llm, requests, max_tokens)
        results.append(result)
        print(
            f"  Test {idx + 1}/{rounds}:"
            f" {result.output_tps:.2f} tok/s"
        )
    return results


def calc_averages(results):
    """Calculate average metrics from test results."""
    count = len(results)
    return {
        "output_tps": sum(
            item.output_tps for item in results
        ) / count,
        "total_tps": sum(
            item.total_tps for item in results
        ) / count,
        "time": sum(
            item.time for item in results
        ) / count,
    }


def run_benchmark(config, concurrent):
    """Run complete benchmark for given concurrency."""
    print(f"\n{'=' * 70}")
    print(
        f"Benchmark: concurrent={concurrent},"
        f" warmup={config.warmup}, rounds={config.rounds}"
    )
    print(f"{'=' * 70}")

    images = load_images(config.image_path, concurrent)
    requests = prepare_requests(images, DeepseekOCR2Processor())

    print("[INFO] Initializing LLM engine...")
    init_start = time.time()
    llm = init_llm(config.gpu_mem)
    init_time = time.time() - init_start
    print(f"[INFO] LLM initialized in {init_time:.2f}s")

    print("[INFO] Running cold start test...")
    cold = run_inference(llm, requests, config.max_tokens)
    print(f"  Cold start: {cold.output_tps:.2f} tok/s")

    run_warmup(llm, requests, config.max_tokens, config.warmup)
    test_results = run_test_rounds(
        llm, requests, config.max_tokens, config.rounds
    )
    avg = calc_averages(test_results)

    print(f"\n[RESULT] Concurrent: {concurrent}")
    print(
        f"  Cold: {cold.output_tps:.2f} tok/s,"
        f" Warm avg: {avg['output_tps']:.2f} tok/s"
    )

    del llm
    gc.collect()

    return BenchmarkResult(
        concurrent=concurrent,
        init_time=init_time,
        cold_tps=cold.output_tps,
        warm_tps=avg['output_tps'],
        warm_total_tps=avg['total_tps'],
        avg_time=avg['time'],
    )


TABLE_HEADER = (
    f"{'Concurrent':>10} | {'Cold(tok/s)':>12}"
    f" | {'Warm(tok/s)':>12} | {'Time(s)':>10}"
)


def format_result_row(result):
    """Format a single benchmark result as a table row."""
    return (
        f"{result.concurrent:>10}"
        f" | {result.cold_tps:>12.2f}"
        f" | {result.warm_tps:>12.2f}"
        f" | {result.avg_time:>10.2f}"
    )


def print_summary(results):
    """Print benchmark summary table."""
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(TABLE_HEADER)
    print("-" * 70)
    for result in results:
        print(format_result_row(result))
    print("=" * 70)


def save_results(results, config):
    """Save results to file."""
    with open(config.output_file, "w", encoding="utf-8") as out_fp:
        out_fp.write("DeepSeek-OCR-2 Benchmark Results (Ascend NPU)\n")
        out_fp.write(
            f"Warmup: {config.warmup},"
            f" Test rounds: {config.rounds}\n"
        )
        out_fp.write("=" * 70 + "\n")
        out_fp.write(TABLE_HEADER + "\n")
        out_fp.write("-" * 70 + "\n")
        for result in results:
            out_fp.write(format_result_row(result) + "\n")
        out_fp.write("=" * 70 + "\n")
    print(f"[INFO] Results saved to {config.output_file}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark DeepSeek-OCR-2 on NPU",
    )
    parser.add_argument(
        "--image", required=True, help="Image file or directory",
    )
    parser.add_argument(
        "--concurrent", default="1,8,16", help="Concurrency levels",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=8192,
        help="Max output tokens",
    )
    parser.add_argument(
        "--gpu-mem", type=float, default=0.85,
        help="GPU memory utilization",
    )
    parser.add_argument(
        "--warmup", type=int, default=2, help="Warmup rounds",
    )
    parser.add_argument(
        "--rounds", type=int, default=5, help="Test rounds",
    )
    parser.add_argument(
        "--output", default="benchmark_results.txt",
        help="Output file",
    )
    args = parser.parse_args()

    return BenchmarkConfig(
        image_path=args.image,
        concurrent_levels=[
            int(val) for val in args.concurrent.split(",")
        ],
        max_tokens=args.max_tokens,
        gpu_mem=args.gpu_mem,
        warmup=args.warmup,
        rounds=args.rounds,
        output_file=args.output,
    )


def main():
    """Main entry point."""
    config = parse_args()

    print("=" * 70)
    print("DeepSeek-OCR-2 Benchmark on Ascend NPU")
    print("=" * 70)
    print(
        f"Image: {config.image_path},"
        f" Concurrency: {config.concurrent_levels}"
    )
    print(f"Warmup: {config.warmup}, Rounds: {config.rounds}")
    print("=" * 70)

    results = []
    for level in config.concurrent_levels:
        try:
            result = run_benchmark(config, level)
            results.append(result)
        except Exception as err:
            print(f"[ERROR] concurrent={level}: {err}")
            import traceback
            traceback.print_exc()

    if results:
        print_summary(results)
        save_results(results, config)


if __name__ == "__main__":
    main()
