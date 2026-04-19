import sys
import os
import json
import argparse
import re
import time
import logging
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def get_logger():
    return logging.getLogger("SGLANG_INFERENCE")


import sglang as sgl
from sglang import Engine

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

        self.generation_kwargs = generation_kwargs
        self.generation_kwargs.pop('do_sample', None)
        self.stop_words = self._get_potential_stop_words(path)
        self.chat_template_kwargs = chat_template_kwargs or {}

    def _get_potential_stop_words(self, path):
        from transformers import GenerationConfig
        potential_stop_words = []
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
        raw_inputs = inputs

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='openbmb/MiniCPM-SALA', help="Model Path")
    parser.add_argument('--api_base', type=str, default='http://127.0.0.1:30000', help="SGLang API base URL")
    parser.add_argument('--model_name', type=str, default=None, help="Model name for API requests. Auto-detected if not set.")
    parser.add_argument('--data_path', type=str, default='data/public_set.jsonl')
    parser.add_argument('--max_seq_len', type=int, default=262144)
    parser.add_argument('--concurrency', type=int, default=8, help="Number of concurrent API requests")
    parser.add_argument('--num_samples', type=int, default=None, help="Number of samples to test")
    parser.add_argument('--task_filter', type=str, default=None, help="Comma-separated task types to include (e.g. mcq,qa,cwe)")
    parser.add_argument('--num_samples_per_task', type=int, default=None, help="Max samples per task type (stratified sampling)")
    parser.add_argument('--verbose', action='store_true', help="Print per-sample details")
    return parser.parse_args()


def extract_final_answer(pred):
    parts = pred.split('</think>')
    return parts[-1].strip() if len(parts) > 1 else pred


def extract_mcq_answer(pred):
    match = re.search(r'(?i)ANSWER\s*:\s*([A-D])', pred)
    if match:
        return match.group(1).upper()
    match = re.search(r'\\boxed\{\\text\{([A-D])\}\}', pred)
    if match:
        return match.group(1).upper()
    match = re.search(r'\\boxed\{([A-D])\}', pred)
    if match:
        return match.group(1).upper()
    return None


def score_mcq(pred, gold):
    if not pred or not gold:
        return 0, None
    final = extract_final_answer(pred)
    extracted = extract_mcq_answer(final)
    if extracted and extracted.upper() == gold.upper():
        return 1, extracted
    return 0, extracted


def score_exact_match(pred, gold, task="unknown"):
    if not pred or not gold:
        return 0
    final = extract_final_answer(pred)
    if not isinstance(gold, list):
        gold = [gold]

    if task in ['qa', 'niah', 'lcx']:
        hits = any(str(r).lower() in final.lower() for r in gold)
        return 1.0 if hits else 0.0

    hits = sum([1.0 if str(r).lower() in final.lower() else 0.0 for r in gold])
    return hits / len(gold) if gold else 0


def _length_bucket(token_len: int) -> str:
    if token_len <= 4096:
        return 'len_0_4k'
    if token_len <= 32768:
        return 'len_4k_32k'
    if token_len <= 131072:
        return 'len_32k_128k'
    return 'len_128k_plus'


def _update_bucket(stats: Dict[str, dict], key: str, score: float, input_tokens: int, output_tokens: int) -> None:
    bucket = stats.setdefault(
        key,
        {
            'count': 0,
            'correct': 0.0,
            'accuracy': 0.0,
            'avg_input_tokens': 0.0,
            'avg_output_tokens': 0.0,
            'total_input_tokens': 0,
            'total_output_tokens': 0,
        },
    )
    bucket['count'] += 1
    bucket['correct'] += score
    bucket['total_input_tokens'] += input_tokens
    bucket['total_output_tokens'] += output_tokens


def _finalize_bucket_stats(stats: Dict[str, dict]) -> Dict[str, dict]:
    finalized = {}
    for key, bucket in sorted(stats.items()):
        count = bucket['count']
        correct = bucket['correct']
        avg_input_tokens = bucket['total_input_tokens'] / count if count else 0.0
        avg_output_tokens = bucket['total_output_tokens'] / count if count else 0.0
        finalized[key] = {
            'count': count,
            'correct': round(correct, 4),
            'accuracy': round((correct / count) * 100, 2) if count else 0.0,
            'avg_input_tokens': round(avg_input_tokens, 1),
            'avg_output_tokens': round(avg_output_tokens, 1),
            'total_input_tokens': bucket['total_input_tokens'],
            'total_output_tokens': bucket['total_output_tokens'],
        }
    return finalized


def _print_bucket_table(title: str, stats: Dict[str, dict]) -> None:
    print(f"\n{title}")
    if not stats:
        print("  <empty>")
        return
    for key, value in stats.items():
        print(
            f"  {key}: count={value['count']} correct={value['correct']} "
            f"accuracy={value['accuracy']:.2f}% avg_in={value['avg_input_tokens']} avg_out={value['avg_output_tokens']}"
        )


def main():
    record_id = os.environ.get("RECORD_ID", "test_record")
    user_id = os.environ.get("USER_ID", "test_user")
    task_id = os.environ.get("TASK_ID", "test_task")

    args = parse_args()
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

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("outputs", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to {output_dir}")

    dataset = []
    if os.path.exists(args.data_path):
        with open(args.data_path, 'r', encoding='utf-8') as f:
            all_items = [json.loads(line) for line in f if line.strip()]

        # Apply task filter
        task_filter = None
        if args.task_filter:
            task_filter = set(t.strip().lower() for t in args.task_filter.split(','))
            all_items = [item for item in all_items if item.get('task', '').lower() in task_filter]

        # Apply per-task sample limit (stratified)
        if args.num_samples_per_task:
            from collections import defaultdict
            by_task = defaultdict(list)
            for item in all_items:
                by_task[item.get('task', 'unknown')].append(item)
            all_items = []
            for task_items in by_task.values():
                all_items.extend(task_items[:args.num_samples_per_task])

        # Apply global sample limit
        if args.num_samples:
            all_items = all_items[:args.num_samples]

        dataset = all_items
    else:
        raise FileNotFoundError(f"Data file not found: {args.data_path}")

    print(f"Testing with {len(dataset)} samples.")

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

    inputs = [item['question'] for item in dataset]
    print("Generating responses...")
    start_time = time.time()
    outputs = model.generate(inputs, max_out_len=65536)
    end_time = time.time()
    print(f"\nGeneration completed in {end_time - start_time:.2f} seconds")

    print("\n--- Evaluation Results ---")
    correct_count = 0
    results_to_save = []
    tmp_output_file = os.path.join(output_dir, "_tmp_prediction.jsonl")
    total_input_tokens = 0
    total_output_tokens = 0
    task_stats: Dict[str, dict] = {}
    length_bucket_stats: Dict[str, dict] = {}
    task_length_bucket_stats: Dict[str, dict] = {}

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
        length_bucket = _length_bucket(in_len)
        task_length_bucket = f"task={task}|{length_bucket}"

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
        _update_bucket(task_stats, task, score, in_len, out_len)
        _update_bucket(length_bucket_stats, length_bucket, score, in_len, out_len)
        _update_bucket(task_length_bucket_stats, task_length_bucket, score, in_len, out_len)

        results_to_save.append({
            "index": i,
            "task": task,
            "length_bucket": length_bucket,
            "task_length_bucket": task_length_bucket,
            "question": item['question'],
            "gold": gold,
            "prediction": pred,
            "score": score,
            "extracted": extracted,
            "input_tokens": in_len,
            "output_tokens": out_len,
        })

        if args.verbose:
            print(f"\n[Sample {i+1}] Task: {task}, Length Bucket: {length_bucket}")
            print(f"Gold: {gold}, Extracted: {extracted}, Score: {score}")
            print(f"Tokens: In={in_len}, Out={out_len}")

    avg_score = (correct_count / len(dataset)) * 100 if dataset else 0
    duration = end_time - start_time
    tps = total_output_tokens / duration if duration > 0 else 0
    normalized_accuracy = min(round(avg_score / 80 * 100, 2), 100)

    finalized_task_stats = _finalize_bucket_stats(task_stats)
    finalized_length_bucket_stats = _finalize_bucket_stats(length_bucket_stats)
    finalized_task_length_bucket_stats = _finalize_bucket_stats(task_length_bucket_stats)
    bucket_accuracy = {
        'task': finalized_task_stats,
        'length_bucket': finalized_length_bucket_stats,
        'task_length_bucket': finalized_task_length_bucket_stats,
    }

    print(f"\nAverage Score: {avg_score:.2f}%")
    print(f"Total Duration: {duration:.2f} s")
    print(f"Total Tokens: In={total_input_tokens}, Out={total_output_tokens}")
    if len(dataset) > 0:
        print(f"Average Tokens/Sample: In={total_input_tokens/len(dataset):.1f}, Out={total_output_tokens/len(dataset):.1f}")
    print(f"Overall TPS (Output): {tps:.2f} tokens/s")
    _print_bucket_table("Per-task Accuracy", finalized_task_stats)
    _print_bucket_table("Per-length-bucket Accuracy", finalized_length_bucket_stats)
    _print_bucket_table("Per-task-length-bucket Accuracy", finalized_task_length_bucket_stats)
    print(f"Bucket Accuracy JSON: {json.dumps(bucket_accuracy, ensure_ascii=False, sort_keys=True)}")

    with open(tmp_output_file, "w", encoding="utf-8") as f:
        for res in results_to_save:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

    output_file = os.path.join(output_dir, "predictions.jsonl")
    os.rename(tmp_output_file, output_file)

    with open(os.path.join(output_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Data: {args.data_path}\n")
        f.write(f"Original Accuracy: {avg_score:.2f}%\n")
        f.write(f"Normalized Accuracy: {normalized_accuracy}%\n")
        f.write(f"Num Samples: {len(dataset)}\n")
        f.write(f"Total Duration: {duration:.2f} s\n")
        f.write(f"Total Output Tokens: {total_output_tokens}\n")
        if len(dataset) > 0:
            f.write(f"Average Input Tokens: {total_input_tokens/len(dataset):.1f}\n")
            f.write(f"Average Output Tokens: {total_output_tokens/len(dataset):.1f}\n")
        f.write(f"TPS: {tps:.2f}\n")
        f.write(f"Bucket Accuracy JSON: {json.dumps(bucket_accuracy, ensure_ascii=False, sort_keys=True)}\n")

    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "task_id": task_id,
            "record_id": record_id,
            "user_id": user_id,
            "ori_accuracy": round(avg_score, 2),
            "overall_accuracy": normalized_accuracy,
            "duration": duration,
            "total_tokens": total_output_tokens,
            "bucket_accuracy": bucket_accuracy,
        }, f, ensure_ascii=False, indent=2)

    print(f"Detailed results saved to {output_file}")


if __name__ == "__main__":
    main()