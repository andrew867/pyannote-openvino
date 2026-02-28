from __future__ import annotations

import argparse
from pathlib import Path

import torch
from pyannote.audio import Pipeline


class ResNetEmbeddingWrapper(torch.nn.Module):
    """ONNX wrapper that exposes only the ResNet embedding head."""

    def __init__(self, resnet: torch.nn.Module) -> None:
        super().__init__()
        self.resnet = resnet

    def forward(self, fbanks: torch.Tensor) -> torch.Tensor:
        # The original forward returns (noise_embedding, embedding); keep only the speaker embedding.
        outputs = self.resnet(fbanks)
        return outputs[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the pyannote wespeaker ResNet head to ONNX using precomputed FBanks."
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="Duration of the dummy waveform (seconds) used to shape the FBanks.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Force the exported model to use this number of FBank frames (pads with zeros if longer).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/onnx/embedding.onnx"),
        help="Target ONNX path for the ResNet embedding head.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token (optional).",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Cache directory for Hugging Face downloads.",
    )
    return parser.parse_args()


def get_embedding_model(token: str | None = None, cache_dir: Path | None = None) -> tuple[torch.nn.Module, int, int]:
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=token,
        cache_dir=cache_dir,
    )
    model = pipeline._embedding.model_
    model.eval()
    model.cpu()
    return (
        model,
        pipeline._embedding.sample_rate,
        getattr(pipeline._embedding, "num_channels", 1),
    )


def prepare_dummy_fbanks(
    model: torch.nn.Module,
    sample_rate: int,
    num_channels: int,
    duration: float,
    frames: int | None,
) -> torch.Tensor:
    num_samples = int(sample_rate * duration)

    waveform = torch.randn(1, num_channels, num_samples, dtype=torch.float32)
    waveform = waveform * (1 << 15)
    fbanks = model.compute_fbank(waveform)

    if frames is not None:
        fbanks = adjust_frames(fbanks, frames)

    return fbanks


def adjust_frames(fbanks: torch.Tensor, target_frames: int) -> torch.Tensor:
    current_frames = fbanks.shape[1]
    if current_frames == target_frames:
        return fbanks

    if current_frames > target_frames:
        return fbanks[:, :target_frames, :]

    padding = target_frames - current_frames
    pad_tensor = torch.zeros(
        fbanks.size(0), padding, fbanks.size(2), dtype=fbanks.dtype
    )
    return torch.cat([fbanks, pad_tensor], dim=1)


def export_resnet(
    resnet: torch.nn.Module,
    fbanks: torch.Tensor,
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    wrapper = ResNetEmbeddingWrapper(resnet)
    wrapper.eval()

    torch.onnx.export(
        wrapper,
        fbanks,
        output,
        input_names=["fbanks"],
        output_names=["embeddings"],
        dynamic_axes={"fbanks": {1: "frames"}},
        opset_version=18,
        do_constant_folding=True,
        export_params=True,
        dynamo=False,
        verbose=False,
    )
    print(f"ResNet embedding ONNX written to {output} (frames dynamic).")


def main() -> None:
    args = parse_args()
    model, sample_rate, num_channels = get_embedding_model(
        token=args.token, cache_dir=args.cache
    )
    dummy_fbanks = prepare_dummy_fbanks(
        model, sample_rate, num_channels, args.duration, args.frames
    )
    export_resnet(model.resnet, dummy_fbanks, args.output)


if __name__ == "__main__":
    main()
