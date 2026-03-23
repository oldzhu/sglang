"""Submission preprocess entry for MiniCPM-SALA.

This script follows SOAR toolkit submission contract:
    python preprocess_model.py --input <raw_model_dir> --output <processed_model_dir>

Modes:
- copy: copy raw model as-is (default)
- gptq: run GPTQ offline quantization with GPTQModel
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


def copy_model(src: Path, dst: Path) -> int:
    dst.mkdir(parents=True, exist_ok=True)

    count = 0
    for f in sorted(src.iterdir()):
        if f.name.startswith("."):
            continue
        target = dst / f.name
        if target.exists():
            continue
        if f.is_dir():
            shutil.copytree(f, target)
        else:
            shutil.copy2(f, target)
        count += 1
    return count


def has_command(cmd: str) -> bool:
    try:
        subprocess.run(["bash", "-lc", f"command -v {cmd}"], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False


def gptq_preflight(src: Path) -> None:
    # Validate minimum model artifacts and python dependency availability.
    if not has_command("python3"):
        raise RuntimeError("python3 not found in environment")

    required = ["config.json", "tokenizer_config.json"]
    missing = [name for name in required if not (src / name).exists()]
    if missing:
        raise RuntimeError(
            "Input model missing required files for quant-prep: "
            + ", ".join(missing)
        )

    import importlib.util

    spec = importlib.util.find_spec("gptqmodel")
    if spec is None:
        raise RuntimeError(
            "SOAR_QUANT_MODE=gptq but `gptqmodel` is not installed. "
            "Install it in prepare_env.sh (e.g., `uv pip install gptqmodel`) "
            "or use SOAR_QUANT_MODE=copy."
        )


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _parse_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got: {value}") from exc


def _calibration_length_bucket(record: dict) -> str:
    prompt_tokens = record.get("prompt_tokens")
    if not isinstance(prompt_tokens, int):
        return "len_unknown"
    if prompt_tokens <= 4096:
        return "len_0_4k"
    if prompt_tokens <= 32768:
        return "len_4k_32k"
    if prompt_tokens <= 131072:
        return "len_32k_128k"
    return "len_128k_plus"


def _calibration_bucket_key(
    record: dict,
    task_balance: bool,
    use_prompt_tokens: bool,
) -> str:
    parts: List[str] = []
    if task_balance:
        parts.append(f"task={record.get('task', 'unknown')}")
    if use_prompt_tokens:
        parts.append(_calibration_length_bucket(record))
    if not parts:
        return "all"
    return "|".join(parts)


def _largest_remainder_allocate(capacities: Dict[str, int], total: int) -> Dict[str, int]:
    allocation = {key: 0 for key in capacities}
    if total <= 0:
        return allocation

    total_capacity = sum(capacities.values())
    if total_capacity <= 0:
        return allocation

    fractional: List[Tuple[float, str]] = []
    assigned = 0
    for key, capacity in capacities.items():
        if capacity <= 0:
            continue
        raw = total * capacity / total_capacity
        whole = min(capacity, int(raw))
        allocation[key] = whole
        assigned += whole
        fractional.append((raw - whole, key))

    remaining = total - assigned
    if remaining <= 0:
        return allocation

    fractional.sort(key=lambda item: (-item[0], item[1]))
    for _, key in fractional:
        if remaining <= 0:
            break
        if allocation[key] >= capacities[key]:
            continue
        allocation[key] += 1
        remaining -= 1

    return allocation


def _select_calibration_records(records: List[dict], max_samples: int) -> Tuple[List[dict], dict]:
    mode = os.environ.get("SOAR_GPTQ_CALIBRATION_SAMPLING", "sequential").strip().lower()
    seed = _parse_int_env("SOAR_GPTQ_CALIBRATION_SEED", 20260320)
    task_balance = _env_truthy("SOAR_GPTQ_CALIBRATION_TASK_BALANCE", default=True)
    use_prompt_tokens = _env_truthy(
        "SOAR_GPTQ_CALIBRATION_USE_PROMPT_TOKENS", default=True
    )

    if mode not in {"sequential", "shuffled", "stratified"}:
        raise ValueError(
            "SOAR_GPTQ_CALIBRATION_SAMPLING must be one of sequential, shuffled, stratified"
        )

    available = len(records)
    if max_samples <= 0 or max_samples >= available:
        selected = list(records)
        summary = {
            "mode": mode,
            "seed": seed,
            "available": available,
            "selected": len(selected),
            "task_balance": task_balance,
            "use_prompt_tokens": use_prompt_tokens,
            "selected_buckets": {"all": len(selected)},
        }
        return selected, summary

    if mode == "sequential":
        selected = list(records[:max_samples])
        summary = {
            "mode": mode,
            "seed": seed,
            "available": available,
            "selected": len(selected),
            "task_balance": task_balance,
            "use_prompt_tokens": use_prompt_tokens,
            "selected_buckets": {"all": len(selected)},
        }
        return selected, summary

    rng = random.Random(seed)

    if mode == "shuffled":
        selected = list(records)
        rng.shuffle(selected)
        selected = selected[:max_samples]
        summary = {
            "mode": mode,
            "seed": seed,
            "available": available,
            "selected": len(selected),
            "task_balance": task_balance,
            "use_prompt_tokens": use_prompt_tokens,
            "selected_buckets": {"all": len(selected)},
        }
        return selected, summary

    buckets: Dict[str, List[dict]] = {}
    for record in records:
        key = _calibration_bucket_key(record, task_balance, use_prompt_tokens)
        buckets.setdefault(key, []).append(record)

    bucket_items = sorted(buckets.items())
    for _, bucket_records in bucket_items:
        rng.shuffle(bucket_records)

    base_counts = {key: 0 for key, _ in bucket_items}
    if max_samples >= len(bucket_items):
        for key, bucket_records in bucket_items:
            if bucket_records:
                base_counts[key] = 1
    else:
        ranked = sorted(bucket_items, key=lambda item: (-len(item[1]), item[0]))
        for key, _ in ranked[:max_samples]:
            base_counts[key] = 1

    remaining = max_samples - sum(base_counts.values())
    capacities = {
        key: max(0, len(bucket_records) - base_counts[key])
        for key, bucket_records in bucket_items
    }
    extra_counts = _largest_remainder_allocate(capacities, remaining)

    selected = []
    selected_buckets: Dict[str, int] = {}
    for key, bucket_records in bucket_items:
        take = min(len(bucket_records), base_counts[key] + extra_counts[key])
        if take <= 0:
            continue
        selected.extend(bucket_records[:take])
        selected_buckets[key] = take

    if len(selected) < max_samples:
        used_ids = {id(record) for record in selected}
        leftovers = [record for record in records if id(record) not in used_ids]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: max_samples - len(selected)])

    selected = selected[:max_samples]
    summary = {
        "mode": mode,
        "seed": seed,
        "available": available,
        "selected": len(selected),
        "task_balance": task_balance,
        "use_prompt_tokens": use_prompt_tokens,
        "selected_buckets": selected_buckets,
    }
    return selected, summary


def load_calibration_texts(
    path: Path,
    max_samples: int,
    text_field: str,
) -> Tuple[List[str], dict]:
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")

    records = list(_iter_jsonl(path))
    selected_records, summary = _select_calibration_records(records, max_samples)

    samples: List[str] = []
    for obj in selected_records:
        text = obj.get(text_field)
        if not text and text_field != "question":
            text = obj.get("question")
        if not text:
            continue
        samples.append(str(text))

    if not samples:
        raise RuntimeError(
            f"No calibration text found in {path}. Checked field='{text_field}' and fallback='question'."
        )
    summary = dict(summary)
    summary["selected"] = len(samples)
    return samples, summary


def _env_truthy(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_csv_env(name: str, default: Optional[List[str]] = None) -> List[str]:
    raw = os.environ.get(name, "")
    if not raw.strip():
        return list(default or [])

    seen = set()
    values: List[str] = []
    for item in raw.split(","):
        value = item.strip()
        if not value or value in seen:
            continue
        seen.add(value)
        values.append(value)
    return values


def _call_with_supported_kwargs(
    func: Callable[..., Any],
    args: List[Any],
    kwargs: dict,
    optional_keys: List[str],
) -> Any:
    filtered = dict(kwargs)
    try:
        sig = inspect.signature(func)
        accepted = set(sig.parameters.keys())
        has_var_keyword = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in sig.parameters.values()
        )
        if not has_var_keyword:
            for key in list(filtered.keys()):
                if key not in accepted:
                    filtered.pop(key, None)
    except Exception:
        pass

    try:
        return func(*args, **filtered)
    except TypeError:
        retry_kwargs = dict(filtered)
        for key in optional_keys:
            if key in retry_kwargs:
                retry_kwargs.pop(key, None)
                try:
                    return func(*args, **retry_kwargs)
                except TypeError:
                    continue
        raise


def _include_value_for_attr(attr_name: str, modules: List[str]) -> Any:
    if attr_name in {"inside_layer_modules", "modules_in_block_to_quantize"}:
        return [modules]
    return modules


def _build_dynamic_rules(include_modules: List[str], exclude_modules: List[str]) -> dict:
    del include_modules
    dynamic = {}

    for module in exclude_modules:
        escaped = re.escape(module)
        dynamic[rf"-:.*{escaped}.*"] = {}

    return dynamic


def _is_module_mismatch_error(exc: Exception) -> bool:
    text = str(exc).lower()
    patterns = [
        "layer module item",
        "not found in model",
        "module mismatch",
        "incompatible module",
    ]
    return any(pattern in text for pattern in patterns)


def run_gptq_quantization(
    src: Path,
    dst: Path,
    bits: int,
    group_size: int,
    calibration_file: Path,
    calibration_samples: int,
    calibration_field: str,
    batch_size: int,
) -> None:
    # Import lazily so copy mode does not require gptq dependencies.
    from gptqmodel import GPTQModel, QuantizeConfig

    from gptqmodel_minicpm_sala import register_minicpm_sala_gptq_model

    register_minicpm_sala_gptq_model()

    calibration_texts, calibration_summary = load_calibration_texts(
        calibration_file,
        max_samples=calibration_samples,
        text_field=calibration_field,
    )

    default_include = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ]
    default_exclude = ["self_attn.o_gate", "self_attn.z_proj"]

    layer_aware = _env_truthy("SOAR_GPTQ_LAYER_AWARE", default=True)
    include_modules = _parse_csv_env("SOAR_GPTQ_INCLUDE_MODULES", default=default_include)
    exclude_modules = _parse_csv_env("SOAR_GPTQ_EXCLUDE_MODULES", default=default_exclude)
    exclude_set = set(exclude_modules)
    include_modules = [module for module in include_modules if module not in exclude_set]

    dynamic_rules = _build_dynamic_rules(include_modules, exclude_modules) if layer_aware else None
    quant_config = QuantizeConfig(bits=bits, group_size=group_size, dynamic=dynamic_rules)

    trust_remote_code = _env_truthy("SOAR_TRUST_REMOTE_CODE", default=True)
    attn_impl = os.environ.get("SOAR_GPTQ_ATTN_IMPL", "flash_attention_2").strip()
    print(
        "[preprocess] GPTQ start "
        f"bits={bits} group_size={group_size} "
        f"calibration_samples={len(calibration_texts)} batch_size={batch_size} "
        f"calibration_sampling={json.dumps(calibration_summary, sort_keys=True)} "
        f"trust_remote_code={trust_remote_code} attn_impl={attn_impl} "
        f"layer_aware={layer_aware} include={include_modules} exclude={exclude_modules} "
        f"dynamic_rules={dynamic_rules}"
    )
    print("[preprocess] GPTQ custom model support enabled for model_type=minicpm_sala")

    load_kwargs = {
        "trust_remote_code": trust_remote_code,
        "attn_implementation": attn_impl,
    }
    model = _call_with_supported_kwargs(
        GPTQModel.load,
        [str(src), quant_config],
        load_kwargs,
        optional_keys=["attn_implementation"],
    )
    if layer_aware:
        try:
            print(
                "[preprocess] GPTQ resolved modules "
                f"simple_layer_modules={model.simple_layer_modules(model.model.config, model.quantize_config)}"
            )
        except Exception as debug_exc:
            print(f"[preprocess] GPTQ module debug unavailable: {debug_exc}")

    try:
        _call_with_supported_kwargs(
            model.quantize,
            [calibration_texts],
            {"batch_size": batch_size},
            optional_keys=["batch_size"],
        )
    except Exception as exc:
        if not (layer_aware and _is_module_mismatch_error(exc)):
            raise

        retry_include = [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ]
        retry_exclude = ["self_attn.o_gate", "self_attn.z_proj"]
        print(
            "[preprocess] GPTQ retry after module mismatch "
            f"error={exc} retry_include={retry_include} retry_exclude={retry_exclude}"
        )

        retry_dynamic = _build_dynamic_rules(retry_include, retry_exclude)
        retry_config = QuantizeConfig(
            bits=bits,
            group_size=group_size,
            dynamic=retry_dynamic,
        )
        print(
            "[preprocess] GPTQ retry dynamic "
            f"dynamic_rules={retry_dynamic}"
        )

        retry_model = _call_with_supported_kwargs(
            GPTQModel.load,
            [str(src), retry_config],
            load_kwargs,
            optional_keys=["attn_implementation"],
        )
        try:
            print(
                "[preprocess] GPTQ retry resolved modules "
                f"simple_layer_modules={retry_model.simple_layer_modules(retry_model.model.config, retry_model.quantize_config)}"
            )
        except Exception as debug_exc:
            print(f"[preprocess] GPTQ retry module debug unavailable: {debug_exc}")
        _call_with_supported_kwargs(
            retry_model.quantize,
            [calibration_texts],
            {"batch_size": batch_size},
            optional_keys=["batch_size"],
        )
        model = retry_model

    dst.mkdir(parents=True, exist_ok=True)
    model.save(str(dst))

    if not (dst / "quantize_config.json").exists():
        raise RuntimeError(
            "GPTQ output missing quantize_config.json, which is required by SGLang loader."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode",
        choices=["copy", "gptq"],
        default=None,
        help="Preprocess mode. If unset, reads SOAR_QUANT_MODE (default: copy).",
    )
    parser.add_argument(
        "--calibration-file",
        default=os.environ.get(
            "SOAR_GPTQ_CALIBRATION_FILE",
            str(Path(__file__).resolve().parent / "perf_public_set.jsonl"),
        ),
        help="JSONL calibration file path for GPTQ mode.",
    )
    parser.add_argument(
        "--calibration-field",
        default=os.environ.get("SOAR_GPTQ_CALIBRATION_FIELD", "question"),
        help="Text field name used from calibration JSONL records.",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=int(os.environ.get("SOAR_GPTQ_CALIBRATION_SAMPLES", "32")),
        help="Maximum number of calibration samples.",
    )
    parser.add_argument(
        "--gptq-bits",
        type=int,
        default=int(os.environ.get("SOAR_GPTQ_BITS", "4")),
        help="GPTQ quantization bits.",
    )
    parser.add_argument(
        "--gptq-group-size",
        type=int,
        default=int(os.environ.get("SOAR_GPTQ_GROUP_SIZE", "128")),
        help="GPTQ group size.",
    )
    parser.add_argument(
        "--gptq-batch-size",
        type=int,
        default=int(os.environ.get("SOAR_GPTQ_BATCH_SIZE", "2")),
        help="Calibration batch size for GPTQModel.quantize.",
    )
    args = parser.parse_args()

    src = Path(args.input).resolve()
    dst = Path(args.output).resolve()

    if not src.is_dir():
        raise FileNotFoundError(f"Input model dir not found: {src}")

    mode = args.mode or os.environ.get("SOAR_QUANT_MODE", "copy")
    mode = mode.strip().lower()
    if mode not in {"copy", "gptq"}:
        raise ValueError(f"Unsupported preprocess mode: {mode}")

    if mode == "gptq":
        gptq_preflight(src)
        if not args.calibration_file:
            raise RuntimeError(
                "GPTQ mode requires --calibration-file (or SOAR_GPTQ_CALIBRATION_FILE)."
            )
        run_gptq_quantization(
            src=src,
            dst=dst,
            bits=args.gptq_bits,
            group_size=args.gptq_group_size,
            calibration_file=Path(args.calibration_file).resolve(),
            calibration_samples=args.calibration_samples,
            calibration_field=args.calibration_field,
            batch_size=args.gptq_batch_size,
        )
        print(f"[preprocess] mode={mode} done - quantized model saved to {dst}")
        return

    count = copy_model(src, dst)

    print(f"[preprocess] mode={mode} done - copied {count} files from {src} to {dst}")


if __name__ == "__main__":
    main()
