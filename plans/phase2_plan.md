# Phase 2 – ONNX Export

## SPEC
  - Export segmentation and embedding models from pyannote speaker diarization 3.1 to ONNX, handling dynamic sequence length for segmentation.
  - Use representative inputs (segmentation: variable-length spectrogram frames; embedding: precomputed FBank tensors) to produce ONNX files with dynamic axes documented and note that the embedding export expects mel-filterbank features instead of raw waveforms.
- Validate exported ONNX graphs by loading with ONNX Runtime and checking that input/output shapes match documentation.
- Acceptance criteria: scripts `export_segmentation.py` and `export_embedding.py` exist, produce ONNX files, and documentation lists export settings and any special ops.

## TESTS
  - Run `python export_segmentation.py` with sample audio metadata and verify `onnxruntime.InferenceSession` loads the resulting model without errors.
  - Run `python export_embedding.py` with FBank tensors generated via the embedding model's `compute_fbank` helper and verify load succeeds.
- Compare the numerical outputs of Torch and ONNX runs on a small batch to ensure max absolute difference is within tolerance (e.g., 1e-5 for embeddings, 1e-3 for segmentation).

## PLAN
1. Inspect pyannote.audio segmentation and embedding model definitions to understand expected input shapes and preprocessing.
2. Create export scripts that recreate model state dicts, prepare dummy inputs, and call `torch.onnx.export`, specifying dynamic axes for time/frequency where needed.
3. Document export parameters (opset, dynamic axes, input/output names).
4. Implement validation scripts that load exported ONNX models and run inference on same dummy inputs to verify shapes and simple output match.
5. Capture any export warnings/errors for reporting in documentation.

## CODE
  - `export_segmentation.py`: loads segmentation weights, builds a sample input, exports ONNX with dynamic `sequence` axis.
  - `export_embedding.py`: wraps the ResNet embedding head, precomputes FBanks from a dummy waveform, and exports a model that consumes mel features instead of performing FFTs internally.
- Optional helper for ONNX validation in `scripts/phase2/validate_onnx.py` to reuse between models.
