import json
import requests
from transformers import AutoTokenizer

TXT_PATH = "xiaoshuo.txt"
MODEL_PATH = "/data/models/Qwen3-Next-80B-A3B-Instruct-A8W8"
URL = "http://127.0.0.1:30002/generate"
PROMPT_LENGTH = 8192


def send_request():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    with open(TXT_PATH) as f:
        text = '\n'.join(f.readlines())
        chunk_token_ids = tokenizer.encode(text, add_special_tokens=False)[:PROMPT_LENGTH]
        text = tokenizer.decode(chunk_token_ids)
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "text": text,
        "sampling_params": {
            "max_new_tokens": 32,
            "temperature": 0
        }
    }
    reponse = requests.post(URL, headers=headers, json=payload, timeout=6000)
    result = reponse.json()
    print(json.dumps(result, indent=2, ensure_ascii=False))

send_request()