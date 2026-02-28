"""Speaker-aware transcription using the OpenVINO diarization pipeline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchaudio
from pyannote_openvino import OVSpeakerDiarization
from scipy.io import wavfile

try:
    import whisper
except ImportError as exc:
    raise SystemExit(
        "`openai-whisper` is required for transcription. Run `pip install openai-whisper` or ``python -m pip install -e .[stt]``."
    ) from exc

STT_SAMPLE_RATE = 16_000


@dataclass
class TranscriptTurn:
    start: float
    end: float
    speaker: str
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OV speaker diarization + Whisper and print per-speaker transcripts."
    )
    parser.add_argument("--audio", type=Path, required=True, help="Path to the WAV file.")
    parser.add_argument(
        "--device",
        type=str,
        default="CPU",
        help="OpenVINO device string (CPU, GPU, GPU.0, etc.).",
    )
    parser.add_argument(
        "--ov-dir",
        type=Path,
        default=Path("models/ov"),
        help="Directory that contains the IR XML/BIN pairs.",
    )
    parser.add_argument(
        "--stt-model",
        type=str,
        default="tiny",
        help="Whisper model name (tiny, base, small, etc.).",
    )
    parser.add_argument(
        "--stt-device",
        type=str,
        default=None,
        help="Device for Whisper (e.g., cpu or cuda). Defaults to CUDA if available, otherwise CPU.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Force whisper to decode the specified language (ISO639-1).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature used for Whisper fallback decoding.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional TSV path to save the speaker/text summary.",
    )
    return parser.parse_args()


def _load_waveform(audio_path: Path) -> tuple[torch.Tensor, int]:
    sample_rate, data = wavfile.read(audio_path)
    if data.ndim > 1:
        data = data.mean(axis=1)

    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    else:
        data = data.astype(np.float32)

    waveform = torch.from_numpy(data)

    if sample_rate != STT_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sample_rate, STT_SAMPLE_RATE)
        waveform = resampler(waveform)
        sample_rate = STT_SAMPLE_RATE

    return waveform, sample_rate


def _crop_segment(waveform: torch.Tensor, sr: int, start: float, end: float) -> np.ndarray:
    start_sample = max(0, int(round(start * sr)))
    end_sample = min(waveform.shape[0], int(round(end * sr)))
    chunk = waveform[start_sample:end_sample]
    if chunk.shape[0] == 0:
        return np.empty(0, dtype=np.float32)
    return chunk.numpy().astype(np.float32, copy=False)


def _run_whisper(
    model: whisper.Whisper,
    audio: np.ndarray,
    language: str | None,
    temperature: float,
) -> str:
    if audio.size == 0:
        return ""

    result = model.transcribe(
        audio,
        language=language,
        temperature=temperature,
        verbose=False,
    )
    return result.get("text", "").strip()


def main() -> None:
    args = parse_args()
    audio_path = args.audio.expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    waveform, sr = _load_waveform(audio_path)
    whisper_device = args.stt_device or ("cuda" if torch.cuda.is_available() else "cpu")
    whisper_model = whisper.load_model(args.stt_model, device=whisper_device)

    print(f"Loading OpenVINO speaker diarization ({args.device})...")
    pipeline = OVSpeakerDiarization.from_pretrained(
        ov_dir=args.ov_dir,
        device=args.device,
    )

    file_input = {
        "uri": audio_path.stem,
        "waveform": waveform.unsqueeze(0),
        "sample_rate": sr,
    }
    print(f"Running diarization on {audio_path.name}...")
    annotation = pipeline(file_input)
    diarization = annotation.speaker_diarization
    turns: list[TranscriptTurn] = []

    for segment, track, speaker in diarization.itertracks(yield_label=True):
        chunk = _crop_segment(waveform, sr, segment.start, segment.end)
        if chunk.size == 0:
            continue
        text = _run_whisper(whisper_model, chunk, args.language, args.temperature)
        if not text:
            text = "<no speech detected>"
        turns.append(TranscriptTurn(start=segment.start, end=segment.end, speaker=speaker, text=text))

    if not turns:
        print("No segments produced by the pipeline.")
        return

    for turn in turns:
        print(f"[{turn.start:.2f}-{turn.end:.2f}] {turn.speaker}: {turn.text}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as out:
            out.write("start\tend\tspeaker\ttext\n")
            for turn in turns:
                out.write(f"{turn.start:.3f}\t{turn.end:.3f}\t{turn.speaker}\t{turn.text}\n")
        print(f"Transcript summary written to {args.output}")


if __name__ == "__main__":
    main()
