from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from openvino.tools.ovc.cli_parser import get_absolute_path

MODEL_CONFIGS: dict[str, dict[str, str]] = {
    "segmentation": {"onnx": "segmentation.onnx", "output_prefix": "segmentation"},
    "embedding": {"onnx": "embedding.onnx", "output_prefix": "embedding"},
}

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ONNX exports to OpenVINO IR using the OpenVINO Model Optimizer."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_CONFIGS.keys(),
        default=list(MODEL_CONFIGS.keys()),
        help="Subset of models to convert.",
    )
    parser.add_argument(
        "--onnx-dir",
        type=Path,
        default=Path("models/onnx"),
        help="Directory containing the ONNX files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/ov"),
        help="Directory to save the IR files (XML + BIN) per model prefix.",
    )
    parser.add_argument(
        "--weight-format",
        choices=["fp32", "fp16"],
        default="fp32",
        help="Precision of the converted IR weights.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the Model Optimizer commands without executing them.",
    )
    return parser.parse_args()


def build_command(
    onnx_path: Path, output_prefix: Path, compress_to_fp16: bool
) -> list[str]:
    command: list[str] = [
        sys.executable,
        "-m",
        "openvino.tools.ovc.main",
        "--output_model",
        str(get_absolute_path(str(output_prefix))),
        "--compress_to_fp16",
        "True" if compress_to_fp16 else "False",
        str(get_absolute_path(str(onnx_path))),
    ]
    return command


def run_conversion(model_name: str, args: argparse.Namespace) -> None:
    config = MODEL_CONFIGS[model_name]
    onnx_path = (args.onnx_dir / config["onnx"]).resolve()
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found at {onnx_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = (args.output_dir / config["output_prefix"]).resolve()

    compress_to_fp16 = args.weight_format == "fp16"
    command = build_command(onnx_path, output_prefix, compress_to_fp16)

    print(f"Converting {model_name} ({onnx_path.name}) -> {output_prefix}")
    print(" ".join(command))

    if args.dry_run:
        return

    subprocess.run(command, check=True, cwd=REPO_ROOT)


def main() -> None:
    args = parse_args()

    for model_name in args.models:
        run_conversion(model_name, args)


if __name__ == "__main__":
    main()
