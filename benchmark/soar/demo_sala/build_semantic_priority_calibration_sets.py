from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parent
SOURCE_FILE = ROOT / "perf_public_set.jsonl"
CONFIG_FILE = ROOT / "calibration_candidates.json"


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


def load_candidates(path: Path) -> Dict[str, dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    candidates = raw.get("candidates")
    if not isinstance(candidates, dict) or not candidates:
        raise ValueError(f"Invalid candidate config file: {path}")

    normalized: Dict[str, dict] = {}
    for name, config in candidates.items():
        indices = config.get("indices")
        output_file = config.get("output_file")
        if not isinstance(indices, list) or not indices:
            raise ValueError(f"Candidate {name} missing non-empty indices list")
        if not isinstance(output_file, str) or not output_file.strip():
            raise ValueError(f"Candidate {name} missing output_file")
        normalized[name] = {
            "indices": [int(index) for index in indices],
            "path": ROOT / output_file,
            "description": str(config.get("description", "")).strip(),
        }
    return normalized


def render_jsonl(rows_by_index: Dict[int, dict], indices: List[int]) -> str:
    missing = [index for index in indices if index not in rows_by_index]
    if missing:
        raise KeyError(f"Missing calibration indices in source file: {missing}")
    lines = [json.dumps(rows_by_index[index], ensure_ascii=False) for index in indices]
    return "\n".join(lines) + "\n"


def write_outputs(rows_by_index: Dict[int, dict], outputs: Dict[str, dict]) -> None:
    for config in outputs.values():
        content = render_jsonl(rows_by_index, config["indices"])
        output_path = config["path"]
        output_path.write_text(content, encoding="utf-8")
        print(f"[build-calib] wrote {output_path} rows={len(config['indices'])}")


def check_outputs(rows_by_index: Dict[int, dict], outputs: Dict[str, dict]) -> None:
    for name, config in outputs.items():
        expected = render_jsonl(rows_by_index, config["indices"])
        output_path = config["path"]
        actual = output_path.read_text(encoding="utf-8")
        if actual != expected:
            raise AssertionError(
                f"{name} mismatch: existing file does not match source rows for indices {config['indices']}"
            )
        print(f"[build-calib] verified {output_path} rows={len(config['indices'])}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(CONFIG_FILE),
        help="JSON config file describing named calibration candidates.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify generated files match the source dataset and embedded index lists.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print candidate names, output files, and descriptions without generating files.",
    )
    args = parser.parse_args()

    rows_by_index = load_rows(SOURCE_FILE)
    outputs = load_candidates(Path(args.config).resolve())
    if args.list:
        for name, config in outputs.items():
            print(
                f"{name}: output={config['path'].name} rows={len(config['indices'])} "
                f"indices={config['indices']} description={config['description']}"
            )
        return
    if args.check:
        check_outputs(rows_by_index, outputs)
        return
    write_outputs(rows_by_index, outputs)


if __name__ == "__main__":
    main()