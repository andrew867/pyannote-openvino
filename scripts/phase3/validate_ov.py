from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from openvino import Core


MODEL_SPECS = {
    "segmentation": {
        "xml_name": "segmentation.xml",
        "dummy_shape": lambda args: (1, 1, args.samples),
    },
    "embedding": {
        "xml_name": "embedding.xml",
        "dummy_shape": lambda args: (1, args.frames, 80),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a sanity check on the OpenVINO IR segmentation and embedding models."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_SPECS.keys(),
        default=list(MODEL_SPECS.keys()),
        help="List of models to validate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/ov"),
        help="Directory where the XML/BIN files live.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="CPU",
        help="OpenVINO target device for compilation.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=32000,
        help="Number of waveform samples to feed the segmentation model.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=128,
        help="Number of Mel frames to feed the ResNet embedding model.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to generate dummy inputs.",
    )
    return parser.parse_args()


def random_tensor(shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(shape, dtype=np.float32)


def validate_model(
    model_name: str, args: argparse.Namespace, rng: np.random.Generator, core: Core
) -> None:
    spec = MODEL_SPECS[model_name]
    xml_path = args.output_dir / spec["xml_name"]
    if not xml_path.exists():
        raise FileNotFoundError(f"{model_name} IR missing at {xml_path}")

    print(f"\nValidating {model_name} ({xml_path.name}) on {args.device}")
    model = core.read_model(xml_path)
    compiled = core.compile_model(model, args.device)
    shape = spec["dummy_shape"](args)
    tensor = random_tensor(shape, rng)
    ov_outputs = compiled([tensor])
    outputs = dict(ov_outputs)

    for output_name, tensor in outputs.items():
        print(f"  {output_name.get_any_name()}: shape {tensor.shape}")


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    core = Core()

    for model_name in args.models:
        validate_model(model_name, args, rng, core)


if __name__ == "__main__":
    main()
