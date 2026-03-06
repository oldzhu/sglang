"""Submission preprocess entry for MiniCPM-SALA.

This script follows SOAR toolkit submission contract:
    python preprocess_model.py --input <raw_model_dir> --output <processed_model_dir>

Modes:
- copy: copy raw model as-is (default)
- gptq: placeholder quant-prep mode with strict preflight checks
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


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
    # Keep this strict and explicit. Real GPTQ implementation can be plugged in later.
    if not has_command("python3"):
        raise RuntimeError("python3 not found in environment")

    required = ["config.json", "tokenizer_config.json"]
    missing = [name for name in required if not (src / name).exists()]
    if missing:
        raise RuntimeError(
            "Input model missing required files for quant-prep: "
            + ", ".join(missing)
        )

    try:
        import importlib.util

        spec = importlib.util.find_spec("gptqmodel")
        if spec is None:
            raise RuntimeError(
                "SOAR_QUANT_MODE=gptq but `gptqmodel` is not installed. "
                "Install it in prepare_env.sh (e.g., `uv pip install gptqmodel`) "
                "or use SOAR_QUANT_MODE=copy."
            )
    except ModuleNotFoundError:
        raise RuntimeError(
            "Python import system unavailable while checking gptqmodel dependency."
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
        raise RuntimeError(
            "GPTQ quantization scaffold is enabled but concrete quantization logic "
            "is not implemented yet. Add project-specific GPTQ workflow here."
        )

    count = copy_model(src, dst)

    print(f"[preprocess] mode={mode} done — copied {count} files from {src} to {dst}")


if __name__ == "__main__":
    main()
