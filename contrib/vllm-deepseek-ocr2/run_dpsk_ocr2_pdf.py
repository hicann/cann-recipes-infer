#!/usr/bin/env python3
import sys
import os
import warnings
import re
import io
warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
os.environ["VLLM_USE_V1"] = "0"

try:
    import fitz, img2pdf
except ImportError:
    print("[ERROR] Install: pip install PyMuPDF img2pdf")
    sys.exit(1)

from PIL import Image, ImageDraw, ImageFont
import numpy as np
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
from config import MODEL_PATH, INPUT_PATH, OUTPUT_PATH, PROMPT, CROP_MODE, SKIP_REPEAT, MAX_CONCURRENCY, NUM_WORKERS, DEVICE, ENFORCE_EAGER, MAX_MODEL_LEN, TENSOR_PARALLEL_SIZE, GPU_MEMORY_UTILIZATION, SWAP_SPACE, DISABLE_MM_PREPROCESSOR_CACHE

ModelRegistry.register_model("DeepseekOCR2ForCausalLM", DeepseekOCR2ForCausalLM)
print(f"[INFO] Device: {DEVICE}, Model: {MODEL_PATH}")

def pdf_to_images(pdf_path, dpi=144):
    images = []
    pdf_document = None
    try:
        pdf_document = fitz.open(pdf_path)
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            Image.MAX_IMAGE_PIXELS = None
            img_data = pixmap.tobytes("png")
            # Use with statement to ensure BytesIO object is properly closed, following the principle of paired resource acquisition and release
            with io.BytesIO(img_data) as bio:
                img = Image.open(bio)
                if img.mode in ('RGBA', 'LA'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                    img = background
                images.append(img)
    finally:
        if pdf_document is not None:
            pdf_document.close()
    return images

def pil_to_pdf(pil_images, output_path):
    if not pil_images: return
    image_bytes_list = []
    for img in pil_images:
        if img.mode != 'RGB': img = img.convert('RGB')
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='JPEG', quality=95)
        image_bytes_list.append(img_buffer.getvalue())
    try:
        with open(output_path, "wb") as f:
            f.write(img2pdf.convert(image_bytes_list))
    except Exception as e:
        print(f"[ERROR] PDF creation failed: {e}")

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
        return None
    except Exception as e:
        print(f"[WARN] Failed to parse coordinates: {ref_text[2]}, error: {e}")
        return None

def draw_bounding_boxes(image, refs, jdx):
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
                            cropped.save(f"{OUTPUT_PATH}/images/{jdx}_{img_idx}.jpg")
                        except Exception as e:
                            print(f"[WARN] Failed to save cropped image {jdx}_{img_idx}: {e}")
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

def process_single_image(image):
    return {
        "prompt": PROMPT,
        "multi_modal_data": {"image": DeepseekOCR2Processor().tokenize_with_images(
            images=[image], bos=True, eos=True, cropping=CROP_MODE
        )}
    }

def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(f'{OUTPUT_PATH}/images', exist_ok=True)

    print("[INFO] Loading PDF...")
    images = pdf_to_images(INPUT_PATH)
    print(f"[INFO] Loaded {len(images)} pages")

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
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        disable_mm_preprocessor_cache=DISABLE_MM_PREPROCESSOR_CACHE,
    )

    logits_processors = [NoRepeatNGramLogitsProcessor(
        ngram_size=20, window_size=50, whitelist_token_ids={128821, 128822}
    )]

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        logits_processors=logits_processors,
        skip_special_tokens=False,
        include_stop_str_in_output=True,
    )

    print("[INFO] Generating outputs...")
    outputs_list = llm.generate(batch_inputs, sampling_params=sampling_params)

    pdf_name = os.path.basename(INPUT_PATH).replace('.pdf', '')
    mmd_det_path = f'{OUTPUT_PATH}/{pdf_name}_det.mmd'
    mmd_path = f'{OUTPUT_PATH}/{pdf_name}.mmd'
    pdf_out_path = f'{OUTPUT_PATH}/{pdf_name}_layouts.pdf'

    contents_det, contents = '', ''
    draw_images = []
    jdx = 0

    print("[INFO] Processing results...")
    for output, img in tqdm(zip(outputs_list, images), total=len(images), desc="Processing"):
        content = output.outputs[0].text

        if '<｜end▁of▁sentence｜>' in content:
            content = content.replace('<｜end▁of▁sentence｜>', '')
        elif SKIP_REPEAT:
            continue

        page_split = '\n<--- Page Split --->\n'
        contents_det += content + page_split

        matches_ref, matches_images, mathes_other = re_match(content)
        result_image = draw_bounding_boxes(img.copy(), matches_ref, jdx)
        draw_images.append(result_image)

        for idx, a_match_image in enumerate(matches_images):
            content = content.replace(a_match_image, f'![](images/{jdx}_{idx}.jpg)\n')

        for a_match_other in mathes_other:
            content = content.replace(a_match_other, '').replace(
                '\\coloneqq', ':='
            ).replace(
                '\\eqqcolon', '=:'
            ).replace(
                '\n\n\n\n', '\n\n'
            ).replace(
                '\n\n\n', '\n\n'
            )

        contents += content + page_split
        jdx += 1

    with open(mmd_det_path, 'w', encoding='utf-8') as f:
        f.write(contents_det)
    with open(mmd_path, 'w', encoding='utf-8') as f:
        f.write(contents)
    pil_to_pdf(draw_images, pdf_out_path)

    print(f"[INFO] Saved to {mmd_path}")
    print(f"[INFO] Saved layouts PDF to {pdf_out_path}")

if __name__ == "__main__":
    main()
