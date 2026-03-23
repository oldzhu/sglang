from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import requests


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def get_logger():
    return logging.getLogger("SGLANG_INFERENCE")


def _convert_chat_messages(inputs):
    return [[{"role": "user", "content": sample}] if isinstance(sample, str) else sample for sample in inputs]


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
    except Exception as exc:
        get_logger().error(f"Request failed: {exc}")
        return None, {}


class SGLANGwithChatTemplate:
    def __init__(
        self,
        path: str,
        api_base: str,
        model_name: str,
        generation_kwargs: dict = dict(),
        max_seq_len: int = None,
        chat_template_kwargs: Optional[dict] = None,
        mode: str = "none",
        concurrency: int = 8,
    ):
        assert mode in ["none", "mid"], "mode must be one of none, mid"
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
        self.generation_kwargs.pop("do_sample", None)
        self.stop_words = self._get_potential_stop_words(path)
        self.chat_template_kwargs = chat_template_kwargs or {}

    def _get_potential_stop_words(self, path):
        from transformers import GenerationConfig

        potential_stop_words = []
        generation_config = GenerationConfig.from_pretrained(path)
        if generation_config and hasattr(generation_config, "eos_token_id"):
            eos = generation_config.eos_token_id
            ids = [eos] if isinstance(eos, int) else (eos or [])
            for token_id in ids:
                word = self.tokenizer.decode(token_id)
                if word:
                    potential_stop_words.append(word)
        if self.tokenizer.eos_token:
            potential_stop_words.append(self.tokenizer.eos_token)
        return list(set(item for item in potential_stop_words if item))

    def mid_truncated(self, message, max_prompt_len):
        truncated_message = message
        half_max_prompt_len = max_prompt_len // 2
        tokens = self.tokenizer.encode(message)
        if len(tokens) > max_prompt_len:
            self.logger.warning("=" * 100)
            self.logger.warning("This prompt exceed the model's predefined maximum length.")
            self.logger.warning("=" * 100)
            front = tokens[:half_max_prompt_len - 1]
            back = tokens[-(half_max_prompt_len + 1):]
            truncated_message = self.tokenizer.decode(front + back)
        return truncated_message

    def generate(self, inputs: List[str], max_out_len: int, stopping_criteria: List[str] = [], **kwargs) -> List[str]:
        messages = _convert_chat_messages(inputs)
        messages = [
            self.tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                tokenize=False,
                **self.chat_template_kwargs,
            )
            for message in messages
        ]
        if self.tokenizer.bos_token:
            bos_token = self.tokenizer.bos_token
            messages = [message.removeprefix(bos_token) if message.startswith(bos_token) else message for message in messages]

        if self.mode == "mid":
            max_prompt_len = int(os.environ.get("MAX_PROMPT_LEN", 0)) or min(self.max_seq_len - max_out_len - 300, 128000)
            self.logger.info(
                f"mid truncation: max_out_len={max_out_len}, max_seq_len={self.max_seq_len}, max_prompt_len={max_prompt_len}"
            )
            messages = [self.mid_truncated(message, max_prompt_len) for message in messages]

        sampling_kwargs = {
            "temperature": 0,
            "max_tokens": max_out_len,
            "stop": list(set(self.stop_words + stopping_criteria)),
        }
        sampling_kwargs.update(self.generation_kwargs)
        sampling_kwargs.update(kwargs)
        self.logger.info(f"SGLang sampling kwargs: {sampling_kwargs}")

        import tqdm

        outputs = [None] * len(messages)

        def _infer(idx, prompt):
            content, _ = call_sglang_api(self.api_base, self.model_name, prompt, sampling_kwargs)
            return idx, content

        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            futures = {executor.submit(_infer, idx, inputs[idx]): idx for idx in range(len(inputs))}
            for future in tqdm.tqdm(as_completed(futures), total=len(inputs), desc="Generating"):
                idx, content = future.result()
                outputs[idx] = content if content is not None else ""

        return outputs

    def get_token_len(self, prompt: str) -> int:
        message = _convert_chat_messages([prompt])[0]
        tokenized = self.tokenizer.apply_chat_template(message, add_generation_prompt=True, return_dict=True)
        return len(tokenized["input_ids"])


def extract_final_answer(pred):
    parts = pred.split("</think>")
    return parts[-1].strip() if len(parts) > 1 else pred


def extract_mcq_answer(pred):
    match = re.search(r"(?i)ANSWER\s*:\s*([A-D])", pred)
    if match:
        return match.group(1).upper()
    match = re.search(r"\\boxed\{\\text\{([A-D])\}\}", pred)
    if match:
        return match.group(1).upper()
    match = re.search(r"\\boxed\{([A-D])\}", pred)
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

    if task in ["qa", "niah", "lcx"]:
        hits = any(str(reference).lower() in final.lower() for reference in gold)
        return 1.0 if hits else 0.0

    hits = sum(1.0 if str(reference).lower() in final.lower() else 0.0 for reference in gold)
    return hits / len(gold) if gold else 0


def _length_bucket(token_len: int) -> str:
    if token_len <= 4096:
        return "len_0_4k"
    if token_len <= 32768:
        return "len_4k_32k"
    if token_len <= 131072:
        return "len_32k_128k"
    return "len_128k_plus"


def _update_bucket(stats: Dict[str, dict], key: str, score: float, input_tokens: int, output_tokens: int) -> None:
    bucket = stats.setdefault(
        key,
        {
            "count": 0,
            "correct": 0.0,
            "accuracy": 0.0,
            "avg_input_tokens": 0.0,
            "avg_output_tokens": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
        },
    )
    bucket["count"] += 1
    bucket["correct"] += score
    bucket["total_input_tokens"] += input_tokens
    bucket["total_output_tokens"] += output_tokens


def _finalize_bucket_stats(stats: Dict[str, dict]) -> Dict[str, dict]:
    finalized = {}
    for key, bucket in sorted(stats.items()):
        count = bucket["count"]
        correct = bucket["correct"]
        avg_input_tokens = bucket["total_input_tokens"] / count if count else 0.0
        avg_output_tokens = bucket["total_output_tokens"] / count if count else 0.0
        finalized[key] = {
            "count": count,
            "correct": round(correct, 4),
            "accuracy": round((correct / count) * 100, 2) if count else 0.0,
            "avg_input_tokens": round(avg_input_tokens, 1),
            "avg_output_tokens": round(avg_output_tokens, 1),
            "total_input_tokens": bucket["total_input_tokens"],
            "total_output_tokens": bucket["total_output_tokens"],
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


ROOT = Path(__file__).resolve().parent
DEFAULT_SUBSET_CONFIG = ROOT / "quick_screen_public_subset.json"
MCQ_TASKS = {"mcq"}
LONG_CONTEXT_TASKS = {"niah", "cwe", "fwe", "qa", "lcx"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="openbmb/MiniCPM-SALA",
        help="Model path used for tokenizer loading and chat templating.",
    )
    parser.add_argument(
        "--api_base",
        type=str,
        default="http://127.0.0.1:30000",
        help="SGLang API base URL.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name for API requests. Auto-detected if not set.",
    )
    parser.add_argument(
        "--subset_config",
        type=str,
        default=str(DEFAULT_SUBSET_CONFIG),
        help="Subset config JSON file with fixed public indices.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Override the source dataset path from the subset config.",
    )
    parser.add_argument("--max_seq_len", type=int, default=262144)
    parser.add_argument(
        "--max_out_len",
        type=int,
        default=8192,
        help="Generation cap for the quick screen. Lower than full eval to reduce turnaround time.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Number of concurrent API requests.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate subset config against the source file without calling the model.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-sample scoring details.",
    )
    return parser.parse_args()


def load_subset_config(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    name = str(raw.get("name", "")).strip()
    description = str(raw.get("description", "")).strip()
    source_file = str(raw.get("source_file", "")).strip()
    indices = raw.get("indices")
    target_task_length_buckets = raw.get("target_task_length_buckets") or []
    expected_task_counts = raw.get("expected_task_counts") or {}

    if not name:
        raise ValueError(f"Invalid subset config {path}: missing name")
    if not source_file:
        raise ValueError(f"Invalid subset config {path}: missing source_file")
    if not isinstance(indices, list) or not indices:
        raise ValueError(f"Invalid subset config {path}: missing non-empty indices list")
    if len(indices) != len(set(indices)):
        raise ValueError(f"Invalid subset config {path}: indices must be unique")

    return {
        "name": name,
        "description": description,
        "source_file": source_file,
        "indices": [int(index) for index in indices],
        "target_task_length_buckets": [str(item) for item in target_task_length_buckets],
        "expected_task_counts": {str(key): int(value) for key, value in expected_task_counts.items()},
    }


def resolve_data_path(config: dict, override_path: str | None) -> Path:
    if override_path:
        return Path(override_path).resolve()
    return (ROOT / config["source_file"]).resolve()


def load_rows(path: Path) -> Dict[int, dict]:
    rows: Dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows[int(obj["index"])] = obj
    return rows


def build_subset(rows_by_index: Dict[int, dict], indices: List[int]) -> List[dict]:
    missing = [index for index in indices if index not in rows_by_index]
    if missing:
        raise KeyError(f"Missing subset indices in source file: {missing}")
    return [rows_by_index[index] for index in indices]


def validate_subset(config: dict, dataset: List[dict]) -> Dict[str, int]:
    task_counts: Dict[str, int] = {}
    for item in dataset:
        task = str(item.get("task", "unknown"))
        task_counts[task] = task_counts.get(task, 0) + 1

    expected_task_counts = config["expected_task_counts"]
    for task, expected_count in expected_task_counts.items():
        actual_count = task_counts.get(task, 0)
        if actual_count != expected_count:
            raise AssertionError(
                f"Subset task count mismatch for {task}: expected {expected_count}, got {actual_count}"
            )
    return dict(sorted(task_counts.items()))


def auto_detect_model_name(api_base: str) -> str:
    resp = requests.get(f"{api_base}/v1/models", timeout=10)
    resp.raise_for_status()
    models = resp.json()["data"]
    if not models:
        raise RuntimeError("No models exposed by SGLang /v1/models")
    return models[0]["id"]


def empty_bucket() -> dict:
    return {
        "count": 0,
        "correct": 0.0,
        "accuracy": 0.0,
        "avg_input_tokens": 0.0,
        "avg_output_tokens": 0.0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
    }


def main() -> None:
    record_id = os.environ.get("RECORD_ID", "quick_screen_record")
    user_id = os.environ.get("USER_ID", "quick_screen_user")
    task_id = os.environ.get("TASK_ID", "quick_screen_task")

    args = parse_args()
    subset_config_path = Path(args.subset_config).resolve()
    subset_config = load_subset_config(subset_config_path)
    data_path = resolve_data_path(subset_config, args.data_path)
    rows_by_index = load_rows(data_path)
    dataset = build_subset(rows_by_index, subset_config["indices"])
    task_counts = validate_subset(subset_config, dataset)

    print(f"Subset Name: {subset_config['name']}")
    print(f"Subset Config: {subset_config_path}")
    print(f"Source Data: {data_path}")
    print(f"Subset Size: {len(dataset)}")
    print(f"Subset Task Counts: {json.dumps(task_counts, ensure_ascii=False, sort_keys=True)}")
    print(f"Target Task-Length Buckets: {json.dumps(subset_config['target_task_length_buckets'], ensure_ascii=False)}")
    if subset_config["description"]:
        print(f"Description: {subset_config['description']}")

    if args.check:
        print("Subset config validation passed.")
        return

    if not args.model_name:
        try:
            args.model_name = auto_detect_model_name(args.api_base)
            print(f"Auto-detected model name: {args.model_name}")
        except Exception as exc:
            print(f"[ERROR] Could not auto-detect model name: {exc}")
            print("Please specify using --model_name")
            sys.exit(1)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs") / f"quick_screen_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"API Base: {args.api_base}")
    print(f"Model Name: {args.model_name}")
    print(f"Model Path: {args.model_path}")
    print(f"Saving results to {output_dir}")

    model = SGLANGwithChatTemplate(
        path=args.model_path,
        api_base=args.api_base,
        model_name=args.model_name,
        max_seq_len=args.max_seq_len,
        concurrency=args.concurrency,
        generation_kwargs={"temperature": 0.0},
        chat_template_kwargs={"enable_thinking": True},
        mode="mid",
    )

    inputs = [item["question"] for item in dataset]
    print("Generating responses for quick screen...")
    start_time = time.time()
    outputs = model.generate(inputs, max_out_len=args.max_out_len)
    end_time = time.time()
    duration = end_time - start_time
    print(f"Generation completed in {duration:.2f} seconds")

    correct_count = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    task_stats: Dict[str, dict] = {}
    length_bucket_stats: Dict[str, dict] = {}
    task_length_bucket_stats: Dict[str, dict] = {}
    results_to_save: List[dict] = []

    for item, pred, prompt in zip(dataset, outputs, inputs):
        source_index = int(item["index"])
        task = str(item.get("task", "unknown"))
        gold = item.get("gold")

        in_len = model.get_token_len(prompt)
        out_len = model.get_token_len(pred)
        total_input_tokens += in_len
        total_output_tokens += out_len

        length_bucket = _length_bucket(in_len)
        task_length_bucket = f"task={task}|{length_bucket}"

        score = 0.0
        extracted = None
        if task in MCQ_TASKS:
            score, extracted = score_mcq(pred, gold)
        elif task in LONG_CONTEXT_TASKS:
            score = score_exact_match(pred, gold, task)
        elif isinstance(gold, str) and gold.lower() in pred.lower():
            score = 1.0

        correct_count += score
        _update_bucket(task_stats, task, score, in_len, out_len)
        _update_bucket(length_bucket_stats, length_bucket, score, in_len, out_len)
        _update_bucket(task_length_bucket_stats, task_length_bucket, score, in_len, out_len)

        results_to_save.append(
            {
                "source_index": source_index,
                "task": task,
                "length_bucket": length_bucket,
                "task_length_bucket": task_length_bucket,
                "gold": gold,
                "prediction": pred,
                "score": score,
                "extracted": extracted,
                "input_tokens": in_len,
                "output_tokens": out_len,
            }
        )

        if args.verbose:
            print(
                f"[Index {source_index}] task={task} bucket={task_length_bucket} "
                f"score={score} extracted={extracted}"
            )

    quick_screen_accuracy = round((correct_count / len(dataset)) * 100, 2) if dataset else 0.0
    tps = total_output_tokens / duration if duration > 0 else 0.0

    finalized_task_stats = _finalize_bucket_stats(task_stats)
    finalized_length_bucket_stats = _finalize_bucket_stats(length_bucket_stats)
    finalized_task_length_bucket_stats = _finalize_bucket_stats(task_length_bucket_stats)

    focus_bucket_summary = {
        key: finalized_task_length_bucket_stats.get(key, empty_bucket())
        for key in subset_config["target_task_length_buckets"]
    }
    focus_bucket_accuracies = [bucket["accuracy"] for bucket in focus_bucket_summary.values()]
    focus_bucket_average = round(sum(focus_bucket_accuracies) / len(focus_bucket_accuracies), 2) if focus_bucket_accuracies else 0.0
    focus_bucket_min = round(min(focus_bucket_accuracies), 2) if focus_bucket_accuracies else 0.0

    print(f"\nQuick Screen Accuracy: {quick_screen_accuracy:.2f}%")
    print(f"Focus Bucket Average: {focus_bucket_average:.2f}%")
    print(f"Focus Bucket Minimum: {focus_bucket_min:.2f}%")
    print(f"Total Duration: {duration:.2f} s")
    print(f"Total Tokens: In={total_input_tokens}, Out={total_output_tokens}")
    if dataset:
        print(
            f"Average Tokens/Sample: In={total_input_tokens / len(dataset):.1f}, "
            f"Out={total_output_tokens / len(dataset):.1f}"
        )
    print(f"Overall TPS (Output): {tps:.2f} tokens/s")
    _print_bucket_table("Per-task Accuracy", finalized_task_stats)
    _print_bucket_table("Per-length-bucket Accuracy", finalized_length_bucket_stats)
    _print_bucket_table("Per-task-length-bucket Accuracy", finalized_task_length_bucket_stats)
    _print_bucket_table("Focus Task-Length Buckets", focus_bucket_summary)

    predictions_path = output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as file:
        for result in results_to_save:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")

    summary = {
        "task_id": task_id,
        "record_id": record_id,
        "user_id": user_id,
        "subset_name": subset_config["name"],
        "subset_config": str(subset_config_path),
        "data_path": str(data_path),
        "indices": subset_config["indices"],
        "target_task_length_buckets": subset_config["target_task_length_buckets"],
        "ori_accuracy": quick_screen_accuracy,
        "overall_accuracy": quick_screen_accuracy,
        "quick_screen_accuracy": quick_screen_accuracy,
        "focus_bucket_average": focus_bucket_average,
        "focus_bucket_min": focus_bucket_min,
        "duration": duration,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "tps": round(tps, 2),
        "bucket_accuracy": {
            "task": finalized_task_stats,
            "length_bucket": finalized_length_bucket_stats,
            "task_length_bucket": finalized_task_length_bucket_stats,
            "focus_task_length_bucket": focus_bucket_summary,
        },
    }

    summary_txt_path = output_dir / "summary.txt"
    summary_json_path = output_dir / "summary.json"
    summary_txt_path.write_text(
        "\n".join(
            [
                f"Subset Name: {subset_config['name']}",
                f"Subset Config: {subset_config_path}",
                f"Source Data: {data_path}",
                f"Quick Screen Accuracy: {quick_screen_accuracy:.2f}%",
                f"Focus Bucket Average: {focus_bucket_average:.2f}%",
                f"Focus Bucket Minimum: {focus_bucket_min:.2f}%",
                f"Total Duration: {duration:.2f} s",
                f"Total Input Tokens: {total_input_tokens}",
                f"Total Output Tokens: {total_output_tokens}",
                f"TPS: {tps:.2f}",
                f"Focus Buckets: {json.dumps(focus_bucket_summary, ensure_ascii=False, sort_keys=True)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Detailed results saved to {predictions_path}")
    print(f"Summary saved to {summary_json_path}")


if __name__ == "__main__":
    main()