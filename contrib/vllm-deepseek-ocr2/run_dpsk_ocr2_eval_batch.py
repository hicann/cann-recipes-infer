#!/usr/bin/env python3
import sys
import os
import warnings
import re
import glob
warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
os.environ["VLLM_USE_V1"] = "0"

from PIL import Image, ExifTags
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

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
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from config import MODEL_PATH, INPUT_PATH, OUTPUT_PATH, PROMPT, CROP_MODE, MAX_CONCURRENCY, NUM_WORKERS, DEVICE, ENFORCE_EAGER, MAX_MODEL_LEN, TENSOR_PARALLEL_SIZE, GPU_MEMORY_UTILIZATION, SWAP_SPACE

ModelRegistry.register_model("DeepseekOCR2ForCausalLM", DeepseekOCR2ForCausalLM)
print(f"[INFO] Device: {DEVICE}, Model: {MODEL_PATH}")

def correct_image_orientation(image):
    try:
        exif = image._getexif()
        if exif is not None:
            for tag, value in ExifTags.TAGS.items():
                if value == 'Orientation':
                    orientation_key = tag
                    break
            orientation = exif.get(orientation_key, 1)
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
    except Exception as e:
        print(f"[WARN] EXIF error: {e}")
    return image

def clean_formula(text):
    formula_pattern = r'\\\[(.*?)\\\]'
    def process_formula(match):
        formula = match.group(1)
        formula = re.sub(r'\\quad\s*\([^)]*\)', '', formula)
        return r'\[' + formula.strip() + r'\]'
    return re.sub(formula_pattern, process_formula, text)

def re_match(text):
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)
    mathes_other = [a_match[0] for a_match in matches]
    return matches, mathes_other

def process_single_image(image):
    return {
        "prompt": PROMPT,
        "multi_modal_data": {"image": DeepseekOCR2Processor().tokenize_with_images(
            images=[image], bos=True, eos=True, cropping=CROP_MODE
        )}
    }

def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    print("[INFO] Loading images...")
    images_path = glob.glob(f'{INPUT_PATH}/*')
    images = []
    valid_image_paths = []  # Store successfully loaded image paths

    for image_path in tqdm(images_path, desc="Loading"):
        try:
            image = Image.open(image_path)
            try:
                image = correct_image_orientation(image)
                images.append(image.convert('RGB'))
                valid_image_paths.append(image_path)
            finally:
                image.close()
        except Exception as e:
            print(f"[WARN] Failed to load image {image_path}: {e}")
            continue

    images_path = valid_image_paths

    print(f"[INFO] Loaded {len(images)} images")

    print("[INFO] Pre-processing images...")
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        batch_inputs = list(tqdm(
            executor.map(process_single_image, images),
            total=len(images),
            desc="Pre-processing"
        ))

    print("[INFO] Initializing model...")
    llm = LLM(
        model=MODEL_PATH,
        hf_overrides={"architectures": ["DeepseekOCR2ForCausalLM"]},
        block_size=256,
        enforce_eager=ENFORCE_EAGER,
        trust_remote_code=True,
        max_model_len=MAX_MODEL_LEN,
        swap_space=SWAP_SPACE,
        max_num_seqs=MAX_CONCURRENCY,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=0.7,
    )

    logits_processors = [NoRepeatNGramLogitsProcessor(
        ngram_size=40, window_size=90, whitelist_token_ids={128821, 128822}
    )]

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        logits_processors=logits_processors,
        skip_special_tokens=False,
    )

    print("[INFO] Generating outputs...")
    outputs_list = llm.generate(batch_inputs, sampling_params=sampling_params)

    print("[INFO] Processing and saving results...")
    for output, image_path in tqdm(
        zip(outputs_list, images_path),
        total=len(images_path),
        desc="Saving"
    ):
        content = output.outputs[0].text
        content = clean_formula(content)
        matches_ref, mathes_other = re_match(content)

        for a_match_other in mathes_other:
            content = content.replace(a_match_other, '').replace(
                '\n\n\n\n', '\n\n'
            ).replace(
                '\n\n\n', '\n\n'
            )

        output_filename = os.path.basename(image_path).replace(
            '.jpg', '.md'
        ).replace(
            '.png', '.md'
        )
        output_filepath = os.path.join(OUTPUT_PATH, output_filename)

        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write(content)

    print(f"[INFO] Saved {len(images_path)} results to {OUTPUT_PATH}")

if __name__ == "__main__": main()
