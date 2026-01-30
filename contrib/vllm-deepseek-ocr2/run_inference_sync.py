#!/usr/bin/env python3
import sys
import os
import warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
os.environ["VLLM_USE_V1"] = "0"

from PIL import Image, ImageOps
try:
    import torch_npu
    print("[INFO] torch_npu loaded")
except ImportError:
    print("[WARN] torch_npu not available, running on CPU")
except Exception as e:
    print(f"[WARN] Failed to import torch_npu: {e}")

# Import NPU-optimized module (applies patches on import)
import deepseek_ocr2_npu

from vllm import LLM, SamplingParams
from vllm.model_executor.models.registry import ModelRegistry
from deepseek_ocr2 import DeepseekOCR2ForCausalLM
from process.image_process import DeepseekOCR2Processor
from config import MODEL_PATH, INPUT_PATH, OUTPUT_PATH, PROMPT, CROP_MODE, DEVICE, ENFORCE_EAGER, MAX_MODEL_LEN, TENSOR_PARALLEL_SIZE, GPU_MEMORY_UTILIZATION, MAX_CONCURRENCY, SWAP_SPACE, DISABLE_MM_PREPROCESSOR_CACHE

ModelRegistry.register_model("DeepseekOCR2ForCausalLM", DeepseekOCR2ForCausalLM)
print(f"[INFO] Device: {DEVICE}, Model: {MODEL_PATH}")

def load_image(path):
    try:
        return ImageOps.exif_transpose(Image.open(path))
    except ImportError:
        print("[WARN] PIL.ImageOps.exif_transpose not available, running on CPU")
        return None
    except Exception as e:
        print(f"[WARN] Failed to load image: {path}, error: {e}")
        return None


def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    image = load_image(INPUT_PATH)
    if not image: sys.exit(1)
    image = image.convert("RGB")

    image_features = DeepseekOCR2Processor().tokenize_with_images(images=[image], bos=True, eos=True, cropping=CROP_MODE) if "<image>" in PROMPT else ""

    print("[INFO] Initializing model...")
    llm = LLM(model=MODEL_PATH, hf_overrides={"architectures": ["DeepseekOCR2ForCausalLM"]}, tensor_parallel_size=TENSOR_PARALLEL_SIZE, gpu_memory_utilization=GPU_MEMORY_UTILIZATION, max_model_len=MAX_MODEL_LEN, max_num_seqs=MAX_CONCURRENCY, swap_space=SWAP_SPACE, enforce_eager=ENFORCE_EAGER, trust_remote_code=True, disable_mm_preprocessor_cache=DISABLE_MM_PREPROCESSOR_CACHE)

    request = {"prompt": PROMPT, "multi_modal_data": {"image": image_features}} if image_features else {"prompt": PROMPT}
    outputs = llm.generate([request], SamplingParams(temperature=0.0, max_tokens=8192))

    if outputs and outputs[0].outputs:
        result = outputs[0].outputs[0].text
        with open(f"{OUTPUT_PATH}/result.mmd", "w") as f: f.write(result)
        print(f"[INFO] Saved to {OUTPUT_PATH}/result.mmd")
    else:
        print("[ERROR] No output")
        sys.exit(1)

if __name__ == "__main__": main()
