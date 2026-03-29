"""Diagnostic preprocess probe for official environment version inspection.

This script follows the SOAR toolkit submission contract:
    python preprocess_model_001.py --input <raw_model_dir> --output <processed_model_dir>

It intentionally fails after printing dependency versions so the official
platform logs expose the exact environment state.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict


def _module_probe(module_name: str) -> Dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return {
            "module": module_name,
            "import_ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    return {
        "module": module_name,
        "import_ok": True,
        "version": getattr(module, "__version__", "unknown"),
        "file": getattr(module, "__file__", None),
    }


def _print_probe_line(label: str, payload: Dict[str, Any]) -> None:
    print(
        f"[preprocess-probe] {label} "
        f"{json.dumps(payload, ensure_ascii=False, sort_keys=True)}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    src = Path(args.input).resolve()
    dst = Path(args.output).resolve()

    _print_probe_line(
        "invocation",
        {
            "argv": sys.argv,
            "cwd": os.getcwd(),
            "input": str(src),
            "output": str(dst),
            "input_exists": src.exists(),
            "input_is_dir": src.is_dir(),
            "output_exists": dst.exists(),
        },
    )
    _print_probe_line(
        "python",
        {
            "version": sys.version.split()[0],
            "executable": sys.executable,
            "prefix": sys.prefix,
            "base_prefix": getattr(sys, "base_prefix", sys.prefix),
        },
    )
    _print_probe_line("torch", _module_probe("torch"))
    _print_probe_line("gptqmodel", _module_probe("gptqmodel"))
    _print_probe_line("transformers", _module_probe("transformers"))

    raise RuntimeError(
        "Intentional diagnostic failure after environment version probe. "
        "Compare official log output against fcloud versions before resuming GPTQ changes."
    )


if __name__ == "__main__":
    main()