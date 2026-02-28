import json
import importlib.metadata as metadata
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from pyannote.audio import Pipeline

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

PACKAGE_NAMES = {
    "pyannote.audio": "pyannote.audio",
    "torch": "torch",
    "onnxruntime": "onnxruntime",
    "optimum": "optimum",
    "openvino": "openvino",
}


def log(message: str) -> None:
    print(message)


def save_summary(summary: dict) -> None:
    summary_path = ARTIFACTS_DIR / "phase1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    log("Loading pyannote speaker diarization pipeline...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

    if not hasattr(pipeline, "_segmentation"):
        raise AttributeError("Pipeline is missing the internal '_segmentation' attribute")
    if not hasattr(pipeline, "_embedding"):
        raise AttributeError("Pipeline is missing the internal '_embedding' attribute")

    seg_model = pipeline._segmentation.model
    embedding_sample_rate = pipeline._embedding.sample_rate
    emb_model = pipeline._embedding.model_
    sample_rate = getattr(seg_model, "sample_rate", 16000)
    num_channels = getattr(seg_model, "num_channels", 1)
    duration_s = 2.0
    num_samples = int(sample_rate * duration_s)

    dummy_waveforms = torch.randn(1, num_channels, num_samples, dtype=torch.float32)

    log(f"Segmentation call shapes: waveforms={tuple(dummy_waveforms.shape)} sample_rate={sample_rate}")
    with torch.no_grad():
        segmentation_scores = seg_model(dummy_waveforms)
    log(f"Segmentation output shape (batch, frames, classes): {tuple(segmentation_scores.shape)}")

    log(f"Embedding call shapes: waveforms={tuple(dummy_waveforms.shape)}")
    with torch.no_grad():
        embeddings = emb_model(dummy_waveforms)
    log(f"Embedding output shape (batch, frames, dim): {tuple(embeddings.shape)}")

    summary = {
        "environment": {pkg: "<missing>" for pkg in PACKAGE_NAMES},
        "models": {
            "segmentation": {
                "input_shape": tuple(dummy_waveforms.shape),
                "output_shape": tuple(segmentation_scores.shape),
                "sample_rate": sample_rate,
                "num_channels": num_channels,
                "num_classes": seg_model.dimension,
            },
            "embedding": {
                "input_shape": tuple(dummy_waveforms.shape),
                "output_shape": tuple(embeddings.shape),
                "sample_rate": embedding_sample_rate,
                "embedding_dim": pipeline._embedding.dimension,
            },
        },
    }

    for pkg_name, display in PACKAGE_NAMES.items():
        version = "<missing>"
        try:
            version = metadata.version(pkg_name)
        except metadata.PackageNotFoundError:
            try:
                module = __import__(pkg_name)
                version = getattr(module, "__version__", "<unknown>")
            except ImportError:
                log(f"Warning: could not import {pkg_name} for version")
        summary["environment"][pkg_name] = version

    seg_path = ARTIFACTS_DIR / "segmentation.onnx"
    emb_path = ARTIFACTS_DIR / "embedding.onnx"

    def export_model(model, path: Path, name: str, dynamic_shapes: dict) -> bool:
        try:
            log(f"Exporting {name} model to ONNX at {path} (dynamic_shapes={dynamic_shapes})...")
            torch.onnx.export(
                model,
                dummy_waveforms,
                path,
                input_names=["waveforms"],
                output_names=[name],
                dynamic_shapes=dynamic_shapes,
                opset_version=18,
            )
            log(f"{name.capitalize()} export succeeded.")
            summary["onnx"][name] = str(path)
            return True
        except Exception as exc:
            log(f"Failed to export {name} model: {exc}")
            summary["onnx"][f"{name}_error"] = str(exc)
            return False

    summary["onnx"] = {
        "segmentation": None,
        "embedding": None,
        "segmentation_error": None,
        "embedding_error": None,
    }

    seg_exported = export_model(
        seg_model,
        seg_path,
        "segmentation",
        {"waveforms": {2: None}, "scores": {1: None}},
    )

    emb_exported = export_model(
        emb_model,
        emb_path,
        "embedding",
        {"waveforms": {2: None}, "embeddings": {1: None}},
    )

    def run_onnx(path: Path, name: str, torch_output: torch.Tensor):
        session = ort.InferenceSession(str(path))
        inputs = {session.get_inputs()[0].name: dummy_waveforms.numpy()}
        onnx_output = session.run(None, inputs)[0]
        log(f"{name} ONNX output shape: {onnx_output.shape}")
        diff = np.max(np.abs(torch_output.numpy() - onnx_output))
        log(f"{name} ONNX max abs diff: {diff:.6f}")
        return onnx_output

    if seg_exported:
        run_onnx(seg_path, "segmentation", segmentation_scores)
    if emb_exported:
        run_onnx(emb_path, "embedding", embeddings)

    save_summary(summary)
    log("Audit completed; summary written to artifacts/phase1_summary.json")
