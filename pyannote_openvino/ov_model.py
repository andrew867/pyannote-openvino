from __future__ import annotations

from pathlib import Path

import torch
from openvino import Core
from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Problem, Resolution, Specifications
from pyannote.audio.models.blocks.sincnet import SincNet
from pyannote.audio.models.embedding.wespeaker.resnet import ResNet34
from pyannote.audio.utils.receptive_field import (
    conv1d_num_frames,
    conv1d_receptive_field_center,
    conv1d_receptive_field_size,
)


SEGMENTATION_SPECIFICATIONS = Specifications(
    problem=Problem.MONO_LABEL_CLASSIFICATION,
    resolution=Resolution.FRAME,
    duration=10.0,
    min_duration=None,
    classes=["speaker#1", "speaker#2", "speaker#3"],
    powerset_max_classes=2,
    permutation_invariant=True,
)


EMBEDDING_SPECIFICATIONS = Specifications(
    problem=Problem.REPRESENTATION,
    resolution=Resolution.CHUNK,
    duration=5.0,
    min_duration=None,
    classes=None,
    powerset_max_classes=None,
    permutation_invariant=False,
)


def _openvino_device(device: torch.device | str | None) -> str:
    if device is None:
        return "CPU"

    if isinstance(device, str):
        return device.upper()

    if device.type == "cuda":
        if device.index is None:
            return "GPU"
        return f"GPU.{device.index}"

    if device.type == "cpu":
        return "CPU"

    return device.type.upper()


class OVBaseModel(Model):
    def __init__(
        self,
        xml_path: Path,
        device: torch.device | str | None = None,
        sample_rate: int = 16000,
        num_channels: int = 1,
    ):
        super().__init__(sample_rate=sample_rate, num_channels=num_channels)
        self.core = Core()
        self._xml_path = Path(xml_path)
        self._model = self.core.read_model(str(self._xml_path))
        self._compiled = None
        self._device_str = _openvino_device(device)
        self._input_name = self._model.inputs[0].get_any_name()
        self._output_names = [output.get_any_name() for output in self._model.outputs]
        self.to(self._device_str)

    def _compile(self, device: str):
        self._compiled = self.core.compile_model(self._model, device)

    def to(self, device: torch.device | str) -> "OVBaseModel":
        ov_device = _openvino_device(device)
        self._device_str = ov_device
        self._compile(ov_device)
        return self

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self._compiled is None:
            self._compile(self._device_str)

        array = inputs.detach().cpu().numpy()
        outputs = self._compiled([array])
        first_output = next(iter(outputs.values()))
        return torch.from_numpy(first_output)

    @property
    def input_name(self) -> str:
        return self._input_name

    @property
    def output_name(self) -> str:
        return self._output_names[0]


class OVSegmentationModel(OVBaseModel):
    def __init__(
        self,
        xml_path: Path,
        device: torch.device | str | None = None,
        sample_rate: int = 16000,
    ):
        super().__init__(xml_path=xml_path, device=device, sample_rate=sample_rate)
        self.specifications = SEGMENTATION_SPECIFICATIONS
        self._sincnet = SincNet(sample_rate=sample_rate, stride=10)

    def num_frames(self, num_samples: int) -> int:
        return self._sincnet.num_frames(num_samples)

    def receptive_field_size(self, num_frames: int = 1) -> int:
        return self._sincnet.receptive_field_size(num_frames=num_frames)

    def receptive_field_center(self, frame: int = 0) -> int:
        return self._sincnet.receptive_field_center(frame=frame)


class OVEmbeddingModel(OVBaseModel):
    NUM_MEL_BINS = 80
    FRAME_LENGTH_MS = 25.0
    FRAME_SHIFT_MS = 10.0

    def __init__(
        self,
        xml_path: Path,
        device: torch.device | str | None = None,
        sample_rate: int = 16000,
        num_channels: int = 1,
    ):
        super().__init__(xml_path=xml_path, device=device, sample_rate=sample_rate)
        self.specifications = EMBEDDING_SPECIFICATIONS
        self._resnet = ResNet34(
            feat_dim=self.NUM_MEL_BINS,
            embed_dim=256,
        )

    def num_frames(self, num_samples: int) -> int:
        window_size = int(self.hparams.sample_rate * self.FRAME_LENGTH_MS * 0.001)
        step_size = int(self.hparams.sample_rate * self.FRAME_SHIFT_MS * 0.001)
        num_frames = conv1d_num_frames(
            num_samples=num_samples,
            kernel_size=window_size,
            stride=step_size,
            padding=0,
            dilation=1,
        )
        return self._resnet.num_frames(num_frames)

    def receptive_field_size(self, num_frames: int = 1) -> int:
        receptive_field_size = num_frames
        receptive_field_size = self._resnet.receptive_field_size(receptive_field_size)
        return conv1d_receptive_field_size(
            num_frames=receptive_field_size,
            kernel_size=int(self.hparams.sample_rate * self.FRAME_LENGTH_MS * 0.001),
            stride=int(self.hparams.sample_rate * self.FRAME_SHIFT_MS * 0.001),
            padding=0,
            dilation=1,
        )

    def receptive_field_center(self, frame: int = 0) -> int:
        receptive_field_center = frame
        receptive_field_center = self._resnet.receptive_field_center(frame=receptive_field_center)
        return conv1d_receptive_field_center(
            frame=receptive_field_center,
            kernel_size=int(self.hparams.sample_rate * self.FRAME_LENGTH_MS * 0.001),
            stride=int(self.hparams.sample_rate * self.FRAME_SHIFT_MS * 0.001),
            padding=0,
            dilation=1,
        )
