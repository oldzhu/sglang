#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple


def run_and_capture(
    cmd: list[str], log_path: Path, label: str
) -> subprocess.CompletedProcess:
    cmd_str = " ".join(shlex.quote(x) for x in cmd)
    header = f"[{label}] Running: {cmd_str}\n"
    print(header, end="")

    output_lines: list[str] = []
    with log_path.open("w", encoding="utf-8") as logf:
        logf.write(header)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            output_lines.append(line)
            print(line, end="")
            logf.write(line)
            logf.flush()

        ret = proc.wait()

    return subprocess.CompletedProcess(cmd, ret, stdout="".join(output_lines))


def parse_accuracy_from_text(text: str) -> Dict[str, Optional[float]]:
    patterns = {
        "ori_accuracy": r"ori_accuracy[^0-9]*([0-9]+(?:\.[0-9]+)?)",
        "overall_accuracy": r"overall_accuracy[^0-9]*([0-9]+(?:\.[0-9]+)?)",
    }
    result: Dict[str, Optional[float]] = {"ori_accuracy": None, "overall_accuracy": None}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            result[key] = float(match.group(1))
    return result


def count_jsonl_rows(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def convert_speed_jsonl_to_custom_jsonl(src: Path, dst: Path) -> Tuple[Path, int]:
    first_obj = None
    lines = []
    with src.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            if first_obj is None:
                first_obj = obj
            lines.append(obj)

    if first_obj is None:
        raise ValueError(f"Empty dataset: {src}")

    if "conversations" in first_obj or "conversation" in first_obj:
        return src, len(lines)

    if "question" in first_obj and ("model_response" in first_obj or "gold" in first_obj):
        with dst.open("w", encoding="utf-8") as out:
            for obj in lines:
                prompt = str(obj.get("question", ""))
                response = str(obj.get("model_response", obj.get("gold", "")))
                rec = {
                    "conversations": [
                        {"content": prompt},
                        {"content": response},
                    ]
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return dst, len(lines)

    raise ValueError(
        f"Unsupported dataset format in {src}. Expected custom conversations JSONL or SOAR-style question/model_response JSONL."
    )


def read_last_jsonl(path: Path) -> dict:
    last = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                last = json.loads(stripped)
    if last is None:
        raise ValueError(f"No JSON record found in {path}")
    return last


def run_single_speed_tier(
    tier_name: str,
    max_concurrency: Optional[int],
    dataset_path: Path,
    api_base: str,
    model_path: str,
    output_dir: Path,
    num_prompts_override: Optional[int],
    disable_tqdm: bool,
) -> dict:
    prepared_path, inferred_num = convert_speed_jsonl_to_custom_jsonl(
        dataset_path, output_dir / f"{tier_name}.converted.custom.jsonl"
    )
    num_prompts = num_prompts_override if num_prompts_override is not None else inferred_num
    output_jsonl = output_dir / f"{tier_name}.bench.jsonl"
    log_path = output_dir / f"{tier_name}.bench.log"

    cmd = [
        sys.executable,
        "-m",
        "sglang.bench_serving",
        "--backend",
        "sglang-oai-chat",
        "--base-url",
        api_base,
        "--model",
        model_path,
        "--dataset-name",
        "custom",
        "--dataset-path",
        str(prepared_path),
        "--num-prompts",
        str(num_prompts),
        "--request-rate",
        "inf",
        "--flush-cache",
        "--output-file",
        str(output_jsonl),
    ]

    if disable_tqdm:
        cmd.append("--disable-tqdm")

    if max_concurrency is not None:
        cmd.extend(["--max-concurrency", str(max_concurrency)])

    proc = run_and_capture(cmd, log_path, label=f"speed/{tier_name}")
    if proc.returncode != 0:
        raise RuntimeError(
            f"Speed tier {tier_name} failed (exit={proc.returncode}). See {log_path}"
        )

    result = read_last_jsonl(output_jsonl)
    duration = float(result.get("duration", 0.0))
    completed = int(result.get("completed", 0))
    return {
        "tier": tier_name,
        "dataset": str(dataset_path),
        "prepared_dataset": str(prepared_path),
        "num_prompts": num_prompts,
        "max_concurrency": max_concurrency,
        "duration": duration,
        "completed": completed,
        "output_jsonl": str(output_jsonl),
        "log": str(log_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SOAR-style local correctness + speed suite and emit summary JSON."
    )
    parser.add_argument("--api-base", required=True, help="e.g. http://127.0.0.1:30000")
    parser.add_argument("--model-path", required=True, help="Model id/path used by serving API")
    parser.add_argument(
        "--output-dir",
        default="benchmark/soar/results",
        help="Directory to store logs/results",
    )

    parser.add_argument(
        "--eval-script",
        default="",
        help="Optional path to SOAR Toolkit eval_model.py for correctness evaluation",
    )
    parser.add_argument(
        "--public-data",
        default="",
        help="Optional path to perf_public_set.jsonl (used with --eval-script)",
    )
    parser.add_argument("--eval-concurrency", type=int, default=32)
    parser.add_argument("--eval-num-samples", type=int, default=0)

    parser.add_argument("--speed-data-s1", default="", help="JSONL for S1")
    parser.add_argument("--speed-data-s8", default="", help="JSONL for S8")
    parser.add_argument("--speed-data-smax", default="", help="JSONL for S∞")
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=0,
        help="Override request count per speed tier; 0 means use dataset rows",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Disable tqdm output from bench_serving sub-processes.",
    )

    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {
        "timestamp": ts,
        "api_base": args.api_base,
        "model_path": args.model_path,
        "run_dir": str(run_dir),
        "correctness": {},
        "speed": {},
    }

    if args.eval_script and args.public_data:
        eval_cmd = [
            sys.executable,
            args.eval_script,
            "--api_base",
            args.api_base,
            "--model_path",
            args.model_path,
            "--data_path",
            args.public_data,
            "--concurrency",
            str(args.eval_concurrency),
        ]
        if args.eval_num_samples > 0:
            eval_cmd.extend(["--num_samples", str(args.eval_num_samples)])

        eval_log = run_dir / "correctness.eval.log"
        eval_proc = run_and_capture(eval_cmd, eval_log, label="correctness")
        acc = parse_accuracy_from_text(eval_proc.stdout)
        summary["correctness"] = {
            "enabled": True,
            "returncode": eval_proc.returncode,
            "ori_accuracy": acc["ori_accuracy"],
            "overall_accuracy": acc["overall_accuracy"],
            "log": str(eval_log),
        }
    else:
        summary["correctness"] = {
            "enabled": False,
            "reason": "Set both --eval-script and --public-data to enable correctness eval",
        }

    tiers = [
        ("s1", 1, args.speed_data_s1),
        ("s8", 8, args.speed_data_s8),
        ("smax", None, args.speed_data_smax),
    ]

    for tier_name, max_concurrency, path_str in tiers:
        if not path_str:
            summary["speed"][tier_name] = {"enabled": False, "reason": "dataset not provided"}
            continue

        dataset_path = Path(path_str)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Speed dataset not found: {dataset_path}")

        tier_result = run_single_speed_tier(
            tier_name=tier_name,
            max_concurrency=max_concurrency,
            dataset_path=dataset_path,
            api_base=args.api_base,
            model_path=args.model_path,
            output_dir=run_dir,
            num_prompts_override=(args.num_prompts if args.num_prompts > 0 else None),
            disable_tqdm=args.disable_tqdm,
        )
        summary["speed"][tier_name] = {"enabled": True, **tier_result}

    weighted_duration = 0.0
    weights = {"s1": 0.4, "s8": 0.3, "smax": 0.3}
    all_present = True
    for tier_name, weight in weights.items():
        tier = summary["speed"].get(tier_name, {})
        if tier.get("enabled") and "duration" in tier:
            weighted_duration += weight * float(tier["duration"])
        else:
            all_present = False
    summary["speed"]["weighted_duration_proxy"] = weighted_duration if all_present else None

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
