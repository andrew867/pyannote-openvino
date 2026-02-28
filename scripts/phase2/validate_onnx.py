from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from pyannote.audio import Pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate ONNX exports against torch.")
    parser.add_argument(
        "--segmentation",
        type=Path,
        default=Path("models/onnx/segmentation.onnx"),
        help="Segmentation ONNX path.",
    )
    parser.add_argument(
        "--embedding",
        type=Path,
        default=Path("models/onnx/embedding.onnx"),
        help="Embedding ONNX path.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="Duration of the dummy waveform used for comparison.",
    )
    return parser.parse_args()


def create_dummy_waveform(sample_rate: int, duration: float) -> torch.Tensor:
    num_samples = int(sample_rate * duration)
    return torch.randn(1, 1, num_samples, dtype=torch.float32)


def run_pytorch(model: torch.nn.Module, waveforms: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        return model(waveforms).cpu()


def run_onnx(path: Path, waveforms: torch.Tensor | np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(str(path))
    array = waveforms.numpy() if isinstance(waveforms, torch.Tensor) else waveforms
    inputs = {session.get_inputs()[0].name: array}
    return session.run(None, inputs)[0]


def compare(name: str, torch_out: torch.Tensor, onnx_out: np.ndarray) -> None:
    diff = np.max(np.abs(torch_out.numpy() - onnx_out))
    print(f"{name}: torch shape {torch_out.shape}, onnx shape {onnx_out.shape}, max diff {diff:.6f}")


def main() -> None:
    args = parse_args()
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    seg_model = pipeline._segmentation.model
    emb_model = pipeline._embedding.model_
    sample_rate = getattr(seg_model, "sample_rate", 16000)

    dummy = create_dummy_waveform(sample_rate, args.duration)
    torch_seg_out = run_pytorch(seg_model, dummy)
    torch_emb_out = run_pytorch(emb_model, dummy)
    fbanks = emb_model.compute_fbank(dummy)
    fbanks_np = fbanks.numpy()

    if args.segmentation.exists():
        onnx_seg_out = run_onnx(args.segmentation, dummy)
        compare("Segmentation", torch_seg_out, onnx_seg_out)
    else:
        print(f"Segmentation ONNX not found at {args.segmentation}")

    if args.embedding.exists():
        onnx_emb_out = run_onnx(args.embedding, fbanks_np)
        compare("Embedding", torch_emb_out, onnx_emb_out)
    else:
        print(f"Embedding ONNX not found at {args.embedding}")


if __name__ == "__main__":
    main()
