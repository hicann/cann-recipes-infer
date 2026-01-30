#!/usr/bin/env python3
import sys
import os
import warnings
import re
import asyncio
import time
warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
os.environ["VLLM_USE_V1"] = "0"

from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np
from tqdm import tqdm

try:
    import torch_npu
    print("[INFO] torch_npu loaded")
except ImportError:
    print("[WARN] torch_npu not available, running on CPU")
except Exception as e:
    print(f"[WARN] Failed to import torch_npu: {e}")

import deepseek_ocr2_npu

from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.model_executor.models.registry import ModelRegistry
from deepseek_ocr2 import DeepseekOCR2ForCausalLM
from process.image_process import DeepseekOCR2Processor
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from config import MODEL_PATH, INPUT_PATH, OUTPUT_PATH, PROMPT, CROP_MODE, DEVICE, ENFORCE_EAGER, MAX_MODEL_LEN, TENSOR_PARALLEL_SIZE, GPU_MEMORY_UTILIZATION, MAX_CONCURRENCY, SWAP_SPACE

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

def re_match(text):
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)
    mathes_image, mathes_other = [], []
    for a_match in matches:
        if '<|ref|>image<|/ref|>' in a_match[0]:
            mathes_image.append(a_match[0])
        else:
            mathes_other.append(a_match[0])
    return matches, mathes_image, mathes_other

def extract_coordinates_and_label(ref_text, image_width, image_height):
    try:
        return (ref_text[1], eval(ref_text[2]))
    except ImportError:
        print("[WARN] eval not available, running on CPU")
    except Exception as e:
        print(f"[WARN] Failed to parse coordinates: {ref_text[2]}, error: {e}")

def draw_bounding_boxes(image, refs):
    image_width, image_height = image.size
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    img_idx = 0

    for i, ref in enumerate(refs):
        try:
            result = extract_coordinates_and_label(ref, image_width, image_height)
            if result:
                label_type, points_list = result
                color = (np.random.randint(0, 200), np.random.randint(0, 200), np.random.randint(0, 255))
                color_a = color + (20, )
                for points in points_list:
                    x1 = int(points[0] / 999 * image_width)
                    y1 = int(points[1] / 999 * image_height)
                    x2 = int(points[2] / 999 * image_width)
                    y2 = int(points[3] / 999 * image_height)

                    if label_type == 'image':
                        try:
                            cropped = image.crop((x1, y1, x2, y2))
                            cropped.save(f"{OUTPUT_PATH}/images/{img_idx}.jpg")
                        except Exception as e:
                            print(f"[WARN] Failed to save cropped image {img_idx}: {e}")
                        img_idx += 1

                    try:
                        width = 4 if label_type == 'title' else 2
                        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
                        draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                        text_x, text_y = x1, max(0, y1 - 15)
                        text_bbox = draw.textbbox((0, 0), label_type, font=font)
                        tw, th = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
                        draw.rectangle([text_x, text_y, text_x + tw, text_y + th], fill=(255, 255, 255, 30))
                        draw.text((text_x, text_y), label_type, font=font, fill=color)
                    except Exception as e:
                        print(f"[WARN] Failed to draw text {label_type}: {e}")
        except Exception as e:
            print(f"[WARN] Failed to process ref {ref}: {e}")
            continue
    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw

async def stream_generate(image=None, prompt=''):
    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        hf_overrides={"architectures": ["DeepseekOCR2ForCausalLM"]},
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=ENFORCE_EAGER,
        trust_remote_code=True,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        swap_space=SWAP_SPACE,
    )

    print("[INFO] Initializing async engine...")
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    logits_processors = [NoRepeatNGramLogitsProcessor(
        ngram_size=20, window_size=90, whitelist_token_ids={128821, 128822}
    )]

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        logits_processors=logits_processors,
        skip_special_tokens=False,
    )

    request_id = f"request-{int(time.time())}"
    printed_length = 0
    final_output = ''

    if image and '<image>' in prompt:
        request = {
            "prompt": prompt,
            "multi_modal_data": {"image": image}
        }
    else:
        request = {"prompt": prompt}

    print("[INFO] Generating (streaming output):")
    print("-" * 60)

    async for request_output in engine.generate(request, sampling_params, request_id):
        if request_output.outputs:
            full_text = request_output.outputs[0].text
            new_text = full_text[printed_length:]
            print(new_text, end='', flush=True)
            printed_length = len(full_text)
            final_output = full_text

    print('\n' + "-" * 60)
    return final_output

def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(f"{OUTPUT_PATH}/images", exist_ok=True)

    image = load_image(INPUT_PATH)
    if not image:
        print("[ERROR] Failed to load image")
        sys.exit(1)
    image = image.convert('RGB')

    if '<image>' in PROMPT:
        image_features = DeepseekOCR2Processor().tokenize_with_images(
            images=[image], bos=True, eos=True, cropping=CROP_MODE
        )
    else:
        image_features = None

    result = asyncio.run(stream_generate(image_features, PROMPT))

    if result:
        with open(f"{OUTPUT_PATH}/result_ori.mmd", 'w', encoding='utf-8') as f:
            f.write(result)

        if '<image>' in PROMPT:
            print("[INFO] Processing references...")
            matches_ref, matches_images, mathes_other = re_match(result)

            result_image = draw_bounding_boxes(image.copy(), matches_ref)
            result_image.save(f"{OUTPUT_PATH}/result_with_boxes.jpg")

            for idx, a_match_image in enumerate(tqdm(matches_images, desc="Images")):
                result = result.replace(a_match_image, f'![](images/{idx}.jpg)\n')

            for a_match_other in tqdm(mathes_other, desc="Others"):
                result = result.replace(a_match_other, '').replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')

        with open(f"{OUTPUT_PATH}/result.mmd", 'w', encoding='utf-8') as f:
            f.write(result)

        print(f"[INFO] Saved to {OUTPUT_PATH}/result.mmd")
        if '<image>' in PROMPT:
            print(f"[INFO] Saved visualization to {OUTPUT_PATH}/result_with_boxes.jpg")
    else:
        print("[ERROR] No output")
        sys.exit(1)

if __name__ == "__main__": main()
