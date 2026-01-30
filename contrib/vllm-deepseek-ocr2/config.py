# DeepSeek-OCR-2 Ascend NPU Configuration for vLLM
# Adapted for Huawei Ascend NPU environment - Using NPU Card 1
# ==================== Model Configuration ====================
BASE_SIZE = 1024
IMAGE_SIZE = 768
CROP_MODE = True
MIN_CROPS = 2
MAX_CROPS = 6  # max: 6

# ==================== vLLM Server Configuration ====================
MAX_CONCURRENCY = 50  # Reduced for NPU memory constraints
NUM_WORKERS = 16  # Image pre-process workers (reduced for NPU)
PRINT_NUM_VIS_TOKENS = False
SKIP_REPEAT = True

# ==================== Model Path ====================
MODEL_PATH = './models/DeepSeek-OCR-2'
INPUT_PATH = './data/ocrtest.jpg'
OUTPUT_PATH = './output'

# ==================== Prompts ====================
PROMPT = '<image>\n<|grounding|>Convert the document to markdown.'

# ==================== Device Configuration ====================
DEVICE = 'npu'
ENFORCE_EAGER = True
MAX_MODEL_LEN = 8192
SWAP_SPACE = 0
TENSOR_PARALLEL_SIZE = 1
GPU_MEMORY_UTILIZATION = 0.85
DISABLE_MM_PREPROCESSOR_CACHE = True

# ==================== Tokenizer ====================
try:
    from transformers import AutoTokenizer
    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
except Exception as e:
    print(f"Warning: Failed to load tokenizer: {e}")
    TOKENIZER = None
