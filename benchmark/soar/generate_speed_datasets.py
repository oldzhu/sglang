#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


EN_PHRASES = [
    "Please summarize the following content in detail and preserve key numbers.",
    "Explain the trade-offs, assumptions, and edge cases for this scenario.",
    "Provide a step-by-step reasoning path and final concise conclusion.",
    "Compare alternatives and justify the recommendation with practical constraints.",
    "Rewrite the text to be clear for both technical and non-technical readers.",
]

ZH_PHRASES = [
    "请你详细总结以下内容，并保留关键数字与结论。",
    "请解释该方案的权衡、前提假设和边界条件。",
    "请给出分步骤推理过程，并在最后给出简洁结论。",
    "请比较可选方案，并结合工程约束说明推荐理由。",
    "请将文本改写为技术和非技术读者都容易理解的版本。",
]


def make_text(target_tokens: int, zh_ratio: float, seed: int) -> str:
    rnd = random.Random(seed)
    words = []
    while len(words) < target_tokens:
        if rnd.random() < zh_ratio:
            words.append(rnd.choice(ZH_PHRASES))
        else:
            words.append(rnd.choice(EN_PHRASES))
    return " ".join(words[:target_tokens])


def make_row(input_tokens: int, output_tokens: int, idx: int) -> dict:
    zh_ratio = 0.35 if idx % 2 == 0 else 0.55
    question = make_text(input_tokens, zh_ratio=zh_ratio, seed=idx * 17 + 11)
    model_response = make_text(output_tokens, zh_ratio=zh_ratio, seed=idx * 19 + 23)
    return {"question": question, "model_response": model_response}


def sample_by_bins(total: int, bins: list[tuple[float, tuple[int, int]]], rnd: random.Random) -> list[int]:
    counts = [int(total * ratio) for ratio, _ in bins]
    while sum(counts) < total:
        counts[rnd.randrange(len(counts))] += 1

    values: list[int] = []
    for count, (_, (low, high)) in zip(counts, bins):
        for _ in range(count):
            values.append(rnd.randint(low, high))
    rnd.shuffle(values)
    return values


def enforce_context_budget(
    input_tokens: int,
    output_tokens: int,
    max_context_tokens: int,
    safety_margin: int,
    min_output_tokens: int,
    max_output_cap: int,
):
    budget = max_context_tokens - safety_margin
    if budget <= min_output_tokens:
        raise ValueError(
            f"Invalid budget: max_context_tokens ({max_context_tokens}) - safety_margin ({safety_margin}) must be > {min_output_tokens}"
        )

    input_tokens = max(1, min(input_tokens, budget - min_output_tokens))
    allowed_output = max(min_output_tokens, budget - input_tokens)
    output_tokens = min(output_tokens, max_output_cap, allowed_output)
    return input_tokens, output_tokens


def write_dataset(
    path: Path,
    num_rows: int,
    in_bins,
    out_bins,
    seed: int,
    max_context_tokens: int,
    safety_margin: int,
    min_output_tokens: int,
    max_output_cap: int,
) -> dict:
    rnd = random.Random(seed)
    in_lens = sample_by_bins(num_rows, in_bins, rnd)
    out_lens = sample_by_bins(num_rows, out_bins, rnd)

    max_in, max_out, max_total = 0, 0, 0
    total_in, total_out = 0, 0

    with path.open("w", encoding="utf-8") as f:
        for idx, (i_len, o_len) in enumerate(zip(in_lens, out_lens), start=1):
            i_len, o_len = enforce_context_budget(
                i_len,
                o_len,
                max_context_tokens=max_context_tokens,
                safety_margin=safety_margin,
                min_output_tokens=min_output_tokens,
                max_output_cap=max_output_cap,
            )

            max_in = max(max_in, i_len)
            max_out = max(max_out, o_len)
            max_total = max(max_total, i_len + o_len)
            total_in += i_len
            total_out += o_len

            row = make_row(i_len, o_len, idx)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "max_input_tokens": max_in,
        "max_output_tokens": max_out,
        "max_total_tokens": max_total,
        "budget": max_context_tokens - safety_margin,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate local SOAR-style speed JSONL datasets")
    parser.add_argument("--output-dir", default="benchmark/soar/data")
    parser.add_argument(
        "--profile",
        choices=["quick10", "balanced", "heavy"],
        default="quick10",
        help="Dataset profile. quick10 is for fast iteration (~10 min per tier depending hardware).",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=262144,
        help="Context length limit used to constrain generated samples.",
    )
    parser.add_argument(
        "--safety-margin",
        type=int,
        default=4096,
        help="Reserved token margin under max context to avoid warmup/effective-length edge failures.",
    )
    parser.add_argument(
        "--min-output-tokens",
        type=int,
        default=64,
        help="Minimum output tokens per sample after context-budget clamping.",
    )
    parser.add_argument(
        "--max-output-cap",
        type=int,
        default=20000,
        help="Hard upper bound for output tokens per sample.",
    )
    parser.add_argument("--rows-s1", type=int, default=0, help="Override row count for S1.")
    parser.add_argument("--rows-s8", type=int, default=0, help="Override row count for S8.")
    parser.add_argument("--rows-smax", type=int, default=0, help="Override row count for Smax.")
    parser.add_argument(
        "--estimate-input-tps",
        type=float,
        default=8000.0,
        help="Estimated input token throughput for rough duration estimate.",
    )
    parser.add_argument(
        "--estimate-output-tps",
        type=float,
        default=250.0,
        help="Estimated output token throughput for rough duration estimate.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.profile == "quick10":
        input_bins = [
            (0.40, (32, 320)),
            (0.30, (320, 1000)),
            (0.20, (1000, 2500)),
            (0.10, (2500, 6000)),
        ]
        output_bins = [
            (0.55, (32, 180)),
            (0.30, (180, 600)),
            (0.12, (600, 1200)),
            (0.03, (1200, 2400)),
        ]
        default_rows = {"s1": 16, "s8": 24, "smax": 32}
    elif args.profile == "balanced":
        input_bins = [
            (0.25, (128, 1500)),
            (0.20, (1500, 6000)),
            (0.25, (6000, 18000)),
            (0.20, (18000, 48000)),
            (0.10, (48000, 90000)),
        ]
        output_bins = [
            (0.40, (64, 500)),
            (0.25, (500, 1800)),
            (0.15, (1800, 3500)),
            (0.12, (3500, 9000)),
            (0.08, (9000, 18000)),
        ]
        default_rows = {"s1": 96, "s8": 192, "smax": 320}
    else:  # heavy
        input_bins = [
            (0.20, (256, 2000)),
            (0.20, (2000, 8000)),
            (0.25, (8000, 24000)),
            (0.20, (24000, 64000)),
            (0.15, (64000, 120000)),
        ]
        output_bins = [
            (0.30, (128, 700)),
            (0.25, (700, 2200)),
            (0.20, (2200, 5000)),
            (0.15, (5000, 12000)),
            (0.10, (12000, 24000)),
        ]
        default_rows = {"s1": 128, "s8": 256, "smax": 384}

    rows_s1 = args.rows_s1 if args.rows_s1 > 0 else default_rows["s1"]
    rows_s8 = args.rows_s8 if args.rows_s8 > 0 else default_rows["s8"]
    rows_smax = args.rows_smax if args.rows_smax > 0 else default_rows["smax"]

    # Separate files allow independent scaling by tier.
    s1_stats = write_dataset(
        out_dir / "speed_s1.jsonl",
        num_rows=rows_s1,
        in_bins=input_bins,
        out_bins=output_bins,
        seed=202601,
        max_context_tokens=args.max_context_tokens,
        safety_margin=args.safety_margin,
        min_output_tokens=args.min_output_tokens,
        max_output_cap=args.max_output_cap,
    )
    s8_stats = write_dataset(
        out_dir / "speed_s8.jsonl",
        num_rows=rows_s8,
        in_bins=input_bins,
        out_bins=output_bins,
        seed=202602,
        max_context_tokens=args.max_context_tokens,
        safety_margin=args.safety_margin,
        min_output_tokens=args.min_output_tokens,
        max_output_cap=args.max_output_cap,
    )
    smax_stats = write_dataset(
        out_dir / "speed_smax.jsonl",
        num_rows=rows_smax,
        in_bins=input_bins,
        out_bins=output_bins,
        seed=202603,
        max_context_tokens=args.max_context_tokens,
        safety_margin=args.safety_margin,
        min_output_tokens=args.min_output_tokens,
        max_output_cap=args.max_output_cap,
    )

    def _estimate_secs(stats: dict) -> float:
        in_tps = max(args.estimate_input_tps, 1.0)
        out_tps = max(args.estimate_output_tps, 1.0)
        return stats["total_input_tokens"] / in_tps + stats["total_output_tokens"] / out_tps

    s1_est = _estimate_secs(s1_stats)
    s8_est = _estimate_secs(s8_stats)
    smax_est = _estimate_secs(smax_stats)

    print(f"Generated datasets in {out_dir}")
    print(f"Profile: {args.profile}")
    print(f"- {out_dir / 'speed_s1.jsonl'}")
    print(
        f"  stats: rows={rows_s1}, max_in={s1_stats['max_input_tokens']}, max_out={s1_stats['max_output_tokens']}, max_total={s1_stats['max_total_tokens']}, budget={s1_stats['budget']}, est_minutes={s1_est/60:.1f}"
    )
    print(f"- {out_dir / 'speed_s8.jsonl'}")
    print(
        f"  stats: rows={rows_s8}, max_in={s8_stats['max_input_tokens']}, max_out={s8_stats['max_output_tokens']}, max_total={s8_stats['max_total_tokens']}, budget={s8_stats['budget']}, est_minutes={s8_est/60:.1f}"
    )
    print(f"- {out_dir / 'speed_smax.jsonl'}")
    print(
        f"  stats: rows={rows_smax}, max_in={smax_stats['max_input_tokens']}, max_out={smax_stats['max_output_tokens']}, max_total={smax_stats['max_total_tokens']}, budget={smax_stats['budget']}, est_minutes={smax_est/60:.1f}"
    )
    print(
        f"Estimated total time for all tiers: {(s1_est+s8_est+smax_est)/60:.1f} minutes (using input_tps={args.estimate_input_tps}, output_tps={args.estimate_output_tps})"
    )


if __name__ == "__main__":
    main()
