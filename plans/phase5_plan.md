# Phase 5 – Testing and Validation

## SPEC
- Build a test suite exercising OV pipeline on representative audio clips covering single speaker, multi-speaker, overlap, silence, and noise.
- Compute DER for each clip with OV pipeline vs reference pyannote pipeline and ensure relative degradation <5%.
- Include regression tests for very short and very long clips, and silence-only inputs.
- Provide CI-friendly test runner script (`tests/run_tests.py` or similar).

## TESTS
- Run test suite with actual or synthetic audio (wave files or NumPy signals) and log DER values for each clip.
- Ensure tests assert output diarization matches expected speaker counts within tolerance (maybe using `pyannote.metrics.diarization` utilities).
- Validate pipeline handles edge cases (empty audio) without raising and returns empty timeline.

## PLAN
1. Gather or generate short audio files representing target scenarios (single speaker, overlapping speech, silence). If real audio unavailable, synthesize using sine waves plus noise.
2. Write test scripts that run both reference pyannote pipeline and OV pipeline on each audio clip, compute DER using `pyannote.metrics.diarization.DiarizationErrorRate`.
3. Set thresholds for DER differences; fail tests if difference exceeds 5% relative.
4. Wrap tests in CLI script or pytest-style runner to meet CI needs.
5. Document in README how to run tests and interpret DER metrics.

## CODE
- `tests/test_baseline_clips.py`: runs both pipelines on multiple clips and asserts DER limits.
- `tests/data/`: stores small audio assets or instructions to generate them programmatically.
- `tests/run_tests.py`: orchestrates running metrics and prints per-clip breakdown (DER, speakers).
