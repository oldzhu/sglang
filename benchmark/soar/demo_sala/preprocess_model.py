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
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional


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


def load_calibration_texts(path: Path, max_samples: int, text_field: str) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")

    samples: List[str] = []
    for obj in _iter_jsonl(path):
        text = obj.get(text_field)
        if not text and text_field != "question":
            text = obj.get("question")
        if not text:
            continue
        samples.append(str(text))
        if len(samples) >= max_samples:
            break

    if not samples:
        raise RuntimeError(
            f"No calibration text found in {path}. Checked field='{text_field}' and fallback='question'."
        )
    return samples


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

    calibration_texts = load_calibration_texts(
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
