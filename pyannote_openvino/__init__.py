"""High-level helpers for running pyannote.audio with OpenVINO."""

from .pipeline import OVEmbeddingConfig, OVSpeakerDiarization
from .ov_model import OVEmbeddingModel, OVSegmentationModel

__all__ = [
    "OVSpeakerDiarization",
    "OVEmbeddingConfig",
    "OVSegmentationModel",
    "OVEmbeddingModel",
]
