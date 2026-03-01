from pathlib import Path

import pytest
import torch
from pyannote.core.annotation import Annotation
from pyannote.audio.pipelines.speaker_diarization import DiarizeOutput

from pyannote_openvino import OVSpeakerDiarization


def test_ov_speaker_diarization_runs() -> None:
    ov_dir = Path("models/ov")
    segmentation_xml = ov_dir / "segmentation.xml"
    embedding_xml = ov_dir / "embedding.xml"
    if not segmentation_xml.exists() or not embedding_xml.exists():
        pytest.skip("Skipping OV pipeline test: models/ov export artifacts are missing.")

    pipeline = OVSpeakerDiarization.from_pretrained(ov_dir=ov_dir, device="CPU")
    waveform = torch.randn(1, 16_000, dtype=torch.float32)
    file_input = {"uri": "test", "waveform": waveform, "sample_rate": 16_000}

    output = pipeline(file_input)
    assert isinstance(output, DiarizeOutput)
    annotation = output.speaker_diarization
    assert isinstance(annotation, Annotation)
    # Tiny random audio may produce zero segments but the annotation object must exist.
    assert annotation.uri == "test"
