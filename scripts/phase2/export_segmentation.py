from __future__ import annotations

import argparse
from pathlib import Path

import torch
from pyannote.audio import Pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export pyannote.segmentation.3.0 (PyanNet) to ONNX."
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="Duration of the dummy waveform used for export (seconds).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/onnx/segmentation.onnx"),
        help="Path to write the ONNX model.",
    )
    parser.add_argument(
        "--token", type=str, default=None, help="Hugging Face token (optional)."
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Optional cache directory for HF downloads.",
    )
    return parser.parse_args()


def get_segmentation_model(
    token: str | None = None, cache_dir: Path | None = None
) -> tuple[torch.nn.Module, int, int]:
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=token,
        cache_dir=cache_dir,
    )
    model = pipeline._segmentation.model
    sample_rate = getattr(model, "sample_rate", 16000)
    num_channels = getattr(model, "num_channels", 1)
    model.eval()
    model.cpu()
    return model, sample_rate, num_channels


def export_segmentation(
    model: torch.nn.Module,
    sample_rate: int,
    num_channels: int,
    duration: float,
    output: Path,
) -> None:
    num_samples = int(sample_rate * duration)
    dummy = torch.randn(1, num_channels, num_samples, dtype=torch.float32)
    output.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        output,
        input_names=["waveforms"],
        output_names=["scores"],
        dynamo=False,
        dynamic_axes={"waveforms": {2: "samples"}, "scores": {1: "frames"}},
        opset_version=18,
        do_constant_folding=True,
        verbose=False,
        export_params=True,
    )
    print(f"Segmentation ONNX written to {output} (dynamic samples).")


def main() -> None:
    args = parse_args()
    model, sample_rate, num_channels = get_segmentation_model(
        token=args.token, cache_dir=args.cache
    )
    export_segmentation(
        model,
        sample_rate,
        num_channels,
        args.duration,
        args.output,
    )


if __name__ == "__main__":
    main()
