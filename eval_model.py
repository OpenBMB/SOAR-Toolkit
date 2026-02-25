import sys
import os
import json
import argparse
import re
import time
import random
import logging
from typing import Dict, List, Optional, Union

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
def get_logger():
    return logging.getLogger("SGLANG_INFERENCE")

# os.environ["SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK"] = "1"

import sglang as sgl
from sglang import Engine

# ==========================================
# Helper Functions
# ==========================================

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

def _convert_chat_messages(inputs):
    return [[{'role': 'user', 'content': s}] if isinstance(s, str) else s for s in inputs]

def call_sglang_api(api_base: str, model: str, prompt: str, sampling_kwargs: dict, timeout: int = 3000):
    url = f"{api_base}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    payload.update(sampling_kwargs)

    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        result = resp.json()
        content = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})
        return content, usage
    except Exception as e:
        get_logger().error(f"Request failed: {e}")
        return None, {}

# ==========================================
# SGLANGwithChatTemplate Class
# ==========================================

class SGLANGwithChatTemplate:
    """SGLang model wrapper with chat template support."""

    def __init__(
        self,
        path: str,
        api_base: str,
        model_name: str,
        generation_kwargs: dict = dict(),
        max_seq_len: int = None,
        chat_template_kwargs: Optional[dict] = None,
        mode: str = 'none',
        concurrency: int = 8,
    ):
        assert mode in ['none', 'mid'], 'mode must be one of none, mid'
        self.mode = mode
        self.logger = get_logger()
        self.path = path
        self.api_base = api_base
        self.model_name = model_name
        self.max_seq_len = max_seq_len
        self.concurrency = concurrency

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        # self._load_model(path, model_kwargs, self.max_seq_len) # No longer load engine

        self.generation_kwargs = generation_kwargs
        self.generation_kwargs.pop('do_sample', None)
        self.stop_words = self._get_potential_stop_words(path)
        self.chat_template_kwargs = chat_template_kwargs or {}

    def _get_potential_stop_words(self, path):
        from transformers import GenerationConfig
        potential_stop_words = []
        generation_config = None
        generation_config = GenerationConfig.from_pretrained(path)
        if generation_config and hasattr(generation_config, 'eos_token_id'):
            eos = generation_config.eos_token_id
            ids = [eos] if isinstance(eos, int) else (eos or [])
            for tid in ids:
                w = self.tokenizer.decode(tid)
                if w:
                    potential_stop_words.append(w)
        if self.tokenizer.eos_token:
            potential_stop_words.append(self.tokenizer.eos_token)
        return list(set(s for s in potential_stop_words if s))

    def mid_truncated(self, message, max_prompt_len):
        """Truncate message from the middle if it exceeds max_prompt_len."""
        truncated_message = message
        half_max_prompt_len = max_prompt_len // 2
        tokens = self.tokenizer.encode(message)
        if len(tokens) > max_prompt_len:
            self.logger.warning('=' * 100)
            self.logger.warning(
                "This prompt exceed the model's predefined maximum length.")
            self.logger.warning('=' * 100)
            front = tokens[:half_max_prompt_len - 1]
            back = tokens[-(half_max_prompt_len + 1):]
            truncated_tokens = front + back
            truncated_message = self.tokenizer.decode(truncated_tokens)
        return truncated_message

    def generate(self, inputs: List[str], max_out_len: int, stopping_criteria: List[str] = [], **kwargs) -> List[str]:
        """Generate results given a list of inputs."""
        messages = _convert_chat_messages(inputs)
        messages = [self.tokenizer.apply_chat_template(
            m, add_generation_prompt=True, tokenize=False, **self.chat_template_kwargs) for m in messages]
        if self.tokenizer.bos_token:
            bos_token = self.tokenizer.bos_token
            messages = [msg.removeprefix(bos_token) if msg.startswith(bos_token) else msg for msg in messages]

        if self.mode == 'mid':
            max_prompt_len = int(os.environ.get('MAX_PROMPT_LEN', 0)) or min(self.max_seq_len - max_out_len - 300, 128000)
            self.logger.info(f'mid truncation: max_out_len={max_out_len}, max_seq_len={self.max_seq_len}, max_prompt_len={max_prompt_len}')
            messages = [self.mid_truncated(m, max_prompt_len) for m in messages]

        sampling_kwargs = {
            'temperature': 0,
            'max_tokens': max_out_len,
            'stop': list(set(self.stop_words + stopping_criteria)),
        }
        sampling_kwargs.update(self.generation_kwargs)
        sampling_kwargs.update(kwargs)
        self.logger.info(f'SGLang sampling kwargs: {sampling_kwargs}')

        time_start = time.time()
        print(f"  Sending {len(messages)} requests to SGLang API (concurrency={self.concurrency})...")

        import tqdm
        outputs = [None] * len(messages)
        completed = 0
        
        # Use full prompt as user message content since the template is already applied and we want raw prompt testing
        # However, SGLang chat/completions expects roles. If we send pre-templated text as 'user' role, 
        # the server might apply template again. To avoid double template, we should send raw prompt 
        # and let server apply template, OR use /v1/completions for raw text.
        # But for simplicity and matching old gpqa_eval logic, let's just send the raw text in 'user' role.
        # Wait, the best way is to NOT apply template here if using /v1/chat/completions, 
        # OR use /v1/completions with the templated messages.
        # Let's use /v1/chat/completions but without apply_chat_template here, just use the raw input.
        # But wait, `mid_truncated` might be needed on raw string or templated?
        # Let's stick to the current logic: we apply template, but if we send it as "user" content to chat API,
        # it might get double-templated.
        # Let's just use raw inputs and rely on API's chat template, OR change call_sglang_api to use /v1/completions.
        # Since we want to use 'enable_thinking', we should use the API's template or format.
        # Let's send the raw prompt to API, but how to handle `mid_truncated`?
        # Let's just send raw inputs and do `mid_truncated` on raw inputs.
        
        # ACTUALLY, let's keep it simple. Let's send the raw `inputs` directly to the `call_sglang_api`.
        raw_inputs = inputs
        # if self.mode == 'mid':
        #     max_prompt_len = int(os.environ.get('MAX_PROMPT_LEN', 0)) or min(self.max_seq_len - max_out_len - 300, 128000)
        #     raw_inputs = [self.mid_truncated(m, max_prompt_len) for m in raw_inputs]

        def _infer(idx, prompt):
            content, usage = call_sglang_api(self.api_base, self.model_name, prompt, sampling_kwargs)
            return idx, content

        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            futures = {executor.submit(_infer, i, raw_inputs[i]): i for i in range(len(raw_inputs))}
            for future in tqdm.tqdm(as_completed(futures), total=len(raw_inputs), desc="Generating"):
                idx, content = future.result()
                outputs[idx] = content if content is not None else ""

        time_end = time.time()
        processing_time = time_end - time_start
        self.logger.info(f'Processing time: {processing_time:.2f}s')

        return outputs

    def get_token_len(self, prompt: str) -> int:
        m = _convert_chat_messages([prompt])[0]
        t = self.tokenizer.apply_chat_template(
            m, add_generation_prompt=True, return_dict=True)
        return len(t['input_ids'])

# ==========================================
# Main Test/Inference Script
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='openbmb/MiniCPM-SALA', help="Model Path")
    parser.add_argument('--api_base', type=str, default='http://127.0.0.1:30000', help="SGLang API base URL")
    parser.add_argument('--model_name', type=str, default=None, help="Model name for API requests. Auto-detected if not set.")
    parser.add_argument('--data_path', type=str, default='data/public_set.jsonl')
    parser.add_argument('--max_seq_len', type=int, default=262144)
    parser.add_argument('--concurrency', type=int, default=8, help="Number of concurrent API requests")
    parser.add_argument('--num_samples', type=int, default=None, help="Number of samples to test")
    parser.add_argument('--verbose', action='store_true', help="Print per-sample details")
    return parser.parse_args()

def extract_final_answer(pred):
    """Extract content after </think> tag, falling back to full prediction."""
    parts = pred.split('</think>')
    return parts[-1].strip() if len(parts) > 1 else pred

def extract_mcq_answer(pred):
    """Extract MCQ answer letter from prediction, supporting multiple formats."""
    # Format 1: ANSWER: X (standard)
    match = re.search(r'(?i)ANSWER\s*:\s*([A-D])', pred)
    if match:
        return match.group(1).upper()
    # Format 2: \boxed{\text{X}} or \boxed{X} (LaTeX)
    match = re.search(r'\\boxed\{\\text\{([A-D])\}\}', pred)
    if match:
        return match.group(1).upper()
    match = re.search(r'\\boxed\{([A-D])\}', pred)
    if match:
        return match.group(1).upper()
    return None

def score_mcq(pred, gold):
    if not pred or not gold: return 0, None
    final = extract_final_answer(pred)
    extracted = extract_mcq_answer(final)
    if extracted and extracted.upper() == gold.upper():
        return 1, extracted
    return 0, extracted

def score_exact_match(pred, gold, task="unknown"):
    if not pred or not gold: return 0
    final = extract_final_answer(pred)
    if not isinstance(gold, list): gold = [gold]
    
    # 针对长文本任务评分的瑕疵修复：
    # 如果是 QA 类型的任务，gold 列表通常是同一答案的不同表述（同义词），只要命中任意一个就算满分 1。
    # 如果是 CWE/FWE 类型的任务，gold 列表是必须全部提取出来的多个关键词，则算覆盖率。
    if task in ['qa', 'niah', 'lcx']:
        # 只要包含任意一个候选答案即为完全正确
        hits = any(str(r).lower() in final.lower() for r in gold)
        return 1.0 if hits else 0.0
    else:
        # cwe, fwe 等需要提取所有目标词汇的任务
        hits = sum([1.0 if str(r).lower() in final.lower() else 0.0 for r in gold])
        return hits / len(gold) if gold else 0

def print_json_result(record_id, user_id, task_id, state, error_msg="", acc=0.0, duration=0.0, total_tokens=0):
    result = {
        "record_id": record_id,
        "user_id": user_id,
        "task_id": task_id,
        "state": state,
        "result": {
            "error_msg": error_msg,
            "score": {
                "acc": acc,
                "duration": duration, # Current run duration
                "total_tokens": total_tokens
            },
            "sort_by": "acc"
        }
    }
    # Print a separator to help backend parsing if needed, though split by '{' logic usually handles it
    print("\n--- JSON RESULT START ---")
    print(json.dumps(result, ensure_ascii=False))
    print("--- JSON RESULT END ---")

def main():
    record_id = os.environ.get("RECORD_ID", "test_record")
    user_id = os.environ.get("USER_ID", "test_user")
    task_id = os.environ.get("TASK_ID", "test_task")
    
    args = parse_args()
    # Auto-detect model name if not set
    if not args.model_name:
        try:
            resp = requests.get(f"{args.api_base}/v1/models", timeout=10)
            resp.raise_for_status()
            models = resp.json()["data"]
            args.model_name = models[0]["id"]
            print(f"Auto-detected model name: {args.model_name}")
        except Exception as e:
            print(f"[ERROR] Could not auto-detect model name: {e}")
            print("Please specify using --model_name")
            sys.exit(1)

    print(f"API Base: {args.api_base}")
    print(f"Model Name: {args.model_name}")
    if os.environ.get("DATA_PATH"):
        args.data_path = os.environ.get("DATA_PATH")
    
    print(f"Model Path: {args.model_path}")
    print(f"Data Path: {args.data_path}")

    # Setup output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("outputs", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to {output_dir}")

    # 1. Load Data
    dataset = []
    if os.path.exists(args.data_path):
        with open(args.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    dataset.append(json.loads(line))
                    if args.num_samples and len(dataset) >= args.num_samples:
                        break
    else:
        raise FileNotFoundError(f"Data file not found: {args.data_path}")

    print(f"Testing with {len(dataset)} samples.")

    # 2. Initialize Model Client
    print("Initializing model client...")
    model = SGLANGwithChatTemplate(
        path=args.model_path,
        api_base=args.api_base,
        model_name=args.model_name,
        max_seq_len=args.max_seq_len,
        concurrency=args.concurrency,
        generation_kwargs={
            "temperature": 0.0,
        },
        chat_template_kwargs={"enable_thinking": True},
        mode='mid',
    )

    # 3. Generate
    inputs = [item['question'] for item in dataset]
    print("Generating responses...")
    start_time = time.time()
    outputs = model.generate(inputs, max_out_len=65536)
    end_time = time.time()
    print(f"\nGeneration completed in {end_time - start_time:.2f} seconds")

    # 4. Score & Save
    print("\n--- Evaluation Results ---")
    correct_count = 0
    results_to_save = []
    tmp_output_file = os.path.join(output_dir, "_tmp_prediction.jsonl")
    total_input_tokens = 0
    total_output_tokens = 0

    mcq_tasks = ['mcq']
    long_context_tasks = [
        'niah', 'cwe', 'fwe', 'qa', 'lcx'
    ]

    for i, item in enumerate(dataset):
        task = item.get('task', 'unknown')
        pred = outputs[i]
        gold = item.get('gold')

        in_len = model.get_token_len(inputs[i])
        out_len = model.get_token_len(pred)
        total_input_tokens += in_len
        total_output_tokens += out_len

        score = 0
        extracted = None

        if task in mcq_tasks:
            score, extracted = score_mcq(pred, gold)
        elif task in long_context_tasks:
            score = score_exact_match(pred, gold, task)
        else:
            if isinstance(gold, str) and gold.lower() in pred.lower():
                score = 1

        correct_count += score

        results_to_save.append({
            "index": i,
            "task": task,
            "question": item['question'],
            "gold": gold,
            "prediction": pred,
            "score": score,
            "extracted": extracted,
            "input_tokens": in_len,
            "output_tokens": out_len,
        })

        if args.verbose:
            print(f"\n[Sample {i+1}] Task: {task}")
            print(f"Gold: {gold}, Extracted: {extracted}, Score: {score}")
            print(f"Tokens: In={in_len}, Out={out_len}")

    avg_score = (correct_count / len(dataset)) * 100 if dataset else 0
    duration = end_time - start_time
    tps = total_output_tokens / duration if duration > 0 else 0

    print(f"\nAverage Score: {avg_score:.2f}%")
    print(f"Total Duration: {duration:.2f} s")
    print(f"Total Tokens: In={total_input_tokens}, Out={total_output_tokens}")
    if len(dataset) > 0:
        print(f"Average Tokens/Sample: In={total_input_tokens/len(dataset):.1f}, Out={total_output_tokens/len(dataset):.1f}")
    print(f"Overall TPS (Output): {tps:.2f} tokens/s")

    with open(tmp_output_file, "w", encoding="utf-8") as f:
        for res in results_to_save:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

    output_file = os.path.join(output_dir, "predictions.jsonl")
    os.rename(tmp_output_file, output_file)

    with open(os.path.join(output_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Data: {args.data_path}\n")
        f.write(f"Original Accuracy: {avg_score:.2f}%\n")
        f.write(f"Normalized Accuracy: {round(avg_score / 80 * 100, 2)}%\n")
        f.write(f"Num Samples: {len(dataset)}\n")
        f.write(f"Total Duration: {duration:.2f} s\n")
        f.write(f"Total Output Tokens: {total_output_tokens}\n")
        if len(dataset) > 0:
            f.write(f"Average Input Tokens: {total_input_tokens/len(dataset):.1f}\n")
            f.write(f"Average Output Tokens: {total_output_tokens/len(dataset):.1f}\n")
        f.write(f"TPS: {tps:.2f}\n")

    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "task_id": task_id,
            "record_id": record_id,
            "user_id": user_id,
            "ori_accuracy": round(avg_score, 2),
            "overall_accuracy": round(avg_score / 80 * 100, 2) ,
            "duration": duration,
            "total_tokens": total_output_tokens
        }, f, ensure_ascii=False, indent=2)

    print(f"Detailed results saved to {output_file}")
    # print_json_result(record_id, user_id, task_id, "1", "", acc=avg_score, duration=duration, total_tokens=total_output_tokens)

if __name__ == "__main__":
    main()
