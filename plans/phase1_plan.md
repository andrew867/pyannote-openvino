# Phase 1 – Environment and Model Audit

## SPEC
- Inventory segmentation and embedding model architectures used by `pyannote/speaker-diarization-3.1`, including their expected input tensor shapes (samples, channels, frames) and output tensors (activity, change-point, embeddings).
- Record the runtime environment that reproduces the pipeline: Python, PyTorch, pyannote.audio, OpenVINO, Optimum-Intel versions.
- Verify ONNX exportability by capturing any dynamic axes, control-flow operators, or unsupported ops that require special handling.
- Document the commands used to load each model, run a forward pass with representative audio, and capture tensor shapes/values.
- Acceptance criteria: audit notes exist in the repo (e.g., markdown tables) showing input/output shapes, export observations, and environment spec with explicit versions.

## TESTS
- Run a Python script that imports the segmentation and embedding checkpoints, feeds random tensors of representative sizes, and prints tensor shapes for inputs/outputs to confirm runtime behavior. Expect no exceptions and matching shapes with documented specs.
- Confirm ONNX export commands succeed for both models using `torch.onnx.export` with sample inputs; verify the exported graph loads via `onnxruntime` or `OpenVINO` for shape inspection.
- Manually record the versions returned by `python -c "import torch, openvino, optimum_intel, pyannote"` to compare against target environment spec.

## PLAN
1. Load pyannote speaker diarization 3.1 pipeline from HF (or local stub) and inspect constituent models to understand data flow and tensor dimensions.
2. Write small scripts to run segmentation and embedding models individually with dummy data to capture input/output shapes, list dynamic axes, and note layers operating on variable-length sequences.
3. Execute ONNX export attempts for both models with sample inputs (segmentation: variable frames, embedding: sample segment). Note any warnings/errors, dynamic axis hints, and ops requiring special attention.
4. Capture environment details (Python, torch, pyannote, openvino, optimum-intel versions) using Python version queries.
5. Assemble notes and commands in this plan file or companion audit doc for easy reference by Phase 2.
6. Commit the audit notes and plan before proceeding to Phase 2.

## CODE
- No production code yet; Phase 1 only produces audit scripts/snippets. Reference scripts can live under `scripts/phase1/` when implemented.
