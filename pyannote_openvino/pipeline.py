from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.compliance.kaldi as kaldi
from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
from pyannote.audio.pipelines import (
    speaker_diarization as speaker_diarization_module,
    speaker_verification,
)
from pyannote.audio.pipelines.utils import PipelineModel
from pyannote.audio.utils.receptive_field import conv1d_num_frames

from .ov_model import OVEmbeddingModel, OVSegmentationModel


@dataclass(frozen=True)
class OVEmbeddingConfig:
    xml_path: Path
    num_mel_bins: int = 80
    frame_length: float = 25.0
    frame_shift: float = 10.0
    round_to_power_of_two: bool = True
    snip_edges: bool = True
    dither: float = 0.0
    window_type: str = "hamming"
    use_energy: bool = False
    fbank_centering_span: Optional[float] = None
    min_num_samples: Optional[int] = None


class OVEmbeddingInference:
    def __init__(self, config: OVEmbeddingConfig, device: torch.device):
        self.config = config
        self.device = device or torch.device("cpu")
        self.sample_rate = 16000
        self.dimension = 256
        self.metric = "cosine"
        self.min_num_samples = (
            config.min_num_samples or int(0.5 * self.sample_rate)
        )
        self._model = OVEmbeddingModel(
            config.xml_path, device=self.device, sample_rate=self.sample_rate
        )
        self._build_fbank()
        self.min_num_frames = self._compute_min_num_frames()

    def _build_fbank(self):
        self._fbank = partial(
            kaldi.fbank,
            num_mel_bins=self.config.num_mel_bins,
            frame_length=self.config.frame_length,
            frame_shift=self.config.frame_shift,
            dither=self.config.dither,
            sample_frequency=self.sample_rate,
            window_type=self.config.window_type,
            use_energy=self.config.use_energy,
        )

    def _compute_min_num_frames(self) -> int:
        dummy = torch.randn(1, 1, self.min_num_samples)
        features = self._compute_fbank(dummy)
        return int(features.shape[1])

    def _compute_fbank(self, waveforms: torch.Tensor) -> torch.Tensor:
        waveforms = waveforms * (1 << 15)
        device = waveforms.device
        fft_device = torch.device("cpu") if device.type == "mps" else device
        features = torch.vmap(self._fbank)(waveforms.to(fft_device)).to(device)

        if self.config.fbank_centering_span is None:
            return features - features.mean(dim=1, keepdim=True)

        window_size = int(self.sample_rate * self.config.frame_length * 0.001)
        step_size = int(self.sample_rate * self.config.frame_shift * 0.001)
        kernel_size = conv1d_num_frames(
            num_samples=int(self.config.fbank_centering_span * self.sample_rate),
            kernel_size=window_size,
            stride=step_size,
            padding=0,
            dilation=1,
        )
        centered = F.avg_pool1d(
            features.transpose(1, 2),
            kernel_size=2 * (kernel_size // 2) + 1,
            stride=1,
            padding=kernel_size // 2,
            count_include_pad=False,
        ).transpose(1, 2)
        return features - centered

    def to(self, device: torch.device) -> "OVEmbeddingInference":
        self.device = device
        self._model.to(device)
        return self

    def __call__(self, waveforms: torch.Tensor, masks: Optional[torch.Tensor] = None):
        features = self._compute_fbank(waveforms)
        _, num_frames, _ = features.shape

        if masks is None:
            embeddings = self._model(features)
            return embeddings.cpu().numpy()

        imasks = (
            F.interpolate(masks.unsqueeze(1), size=num_frames, mode="nearest")
            .squeeze(1)
            .to(torch.bool)
        )

        embeddings = np.nan * np.zeros((features.shape[0], self.dimension), dtype=np.float32)

        for idx, (feature, imask) in enumerate(zip(features, imasks)):
            masked = feature[imask]
            if masked.shape[0] < self.min_num_frames:
                continue
            embedding = self._model(masked[None])
            embeddings[idx] = embedding.squeeze(0).cpu().numpy()

        return embeddings


@contextmanager
def _patch_pretrained_embedding(config: OVEmbeddingConfig, device: torch.device):
    original_verification = speaker_verification.PretrainedSpeakerEmbedding
    original_diarization = speaker_diarization_module.PretrainedSpeakerEmbedding

    def patched(
        embedding: Union[PipelineModel, OVEmbeddingConfig],
        device: Optional[torch.device] = None,
        token: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        if isinstance(embedding, OVEmbeddingConfig):
            return OVEmbeddingInference(config, device or torch.device("cpu"))
        return original_verification(
            embedding, device=device, token=token, cache_dir=cache_dir
        )

    speaker_verification.PretrainedSpeakerEmbedding = patched
    speaker_diarization_module.PretrainedSpeakerEmbedding = patched
    try:
        yield
    finally:
        speaker_verification.PretrainedSpeakerEmbedding = original_verification
        speaker_diarization_module.PretrainedSpeakerEmbedding = original_diarization


def _to_torch_device(device: Union[str, torch.device]) -> torch.device:
    if isinstance(device, torch.device):
        return device

    name = str(device).lower()
    # OpenVINO GPU targets Intel iGPU and does not imply CUDA support.
    # pyannote internals (feature extraction/clustering glue) should stay on CPU.
    if name.startswith("gpu") or name.startswith("auto"):
        return torch.device("cpu")

    return torch.device("cpu")


class OVSpeakerDiarization(SpeakerDiarization):
    def __init__(
        self,
        segmentation_xml: Union[str, Path],
        embedding_xml: Union[str, Path],
        device: Union[str, torch.device] = "cpu",
        embedding_config: Optional[OVEmbeddingConfig] = None,
        **kwargs,
    ):
        segmentation_model = OVSegmentationModel(Path(segmentation_xml), device=device)
        config = embedding_config or OVEmbeddingConfig(xml_path=Path(embedding_xml))
        torch_device = _to_torch_device(device)
        if torch_device.type == "cuda":
            raise RuntimeError(
                "CUDA torch device selected for OVSpeakerDiarization. "
                "This pipeline uses OpenVINO devices (e.g. GPU for Intel iGPU) and "
                "must keep torch-side execution on CPU."
            )

        with _patch_pretrained_embedding(config, torch_device):
            super().__init__(segmentation=segmentation_model, embedding=config, **kwargs)

        self.to(torch_device)

    @classmethod
    def from_pretrained(
        cls,
        ov_dir: Union[str, Path] = Path("models/ov"),
        segmentation: str = "segmentation.xml",
        embedding: str = "embedding.xml",
        device: Union[str, torch.device] = "cpu",
        **kwargs,
    ) -> "OVSpeakerDiarization":
        ov_dir = Path(ov_dir)
        return cls(
            segmentation_xml=ov_dir / segmentation,
            embedding_xml=ov_dir / embedding,
            device=device,
            **kwargs,
        )
