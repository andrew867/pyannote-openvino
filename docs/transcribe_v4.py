"""Single-step OpenVINO diarization + Whisper transcription CLI."""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch


def load_audio(path: Path) -> dict[str, Any]:
    try:
        from torchcodec.decoders import AudioDecoder

        decoder = AudioDecoder(str(path))
        samples = decoder.get_all_samples()
        waveform = samples.data.mean(0).numpy()
        sr = samples.sample_rate
    except Exception:
        waveform, sr = librosa.load(str(path), sr=16_000, mono=True)
    if sr != 16_000:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16_000)
        sr = 16_000
    return {"array": waveform.astype(np.float32), "sampling_rate": sr}


def run_whisper(
    audio_path: Path,
    segments_cache: Path,
    whisper_model: str,
    ov_model_dir: Path | str,
    device: str,
) -> list[dict[str, Any]]:
    from optimum.intel import OVModelForSpeechSeq2Seq
    from transformers import AutoProcessor, pipeline as hf_pipeline

    if segments_cache.exists():
        with segments_cache.open("r", encoding="utf-8") as f:
            segments = json.load(f)
        print(f"Loaded {len(segments)} cached Whisper segments.")
        return segments

    audio_input = load_audio(audio_path)
    print(f"Audio length: {len(audio_input['array']) / AUDIO_RATE:.1f}s")

    processor = AutoProcessor.from_pretrained(whisper_model)
    ov_source = Path(ov_model_dir)
    ov_source = ov_source.resolve() if ov_source.exists() else ov_source

    # Accept either a dedicated whisper export directory or a generic models root.
    if ov_source.exists() and not (ov_source / "openvino_model.xml").exists():
        inferred_dir = ov_source / f"{whisper_model.split('/')[-1]}-ov"
    else:
        inferred_dir = ov_source
    inferred_dir.mkdir(parents=True, exist_ok=True)

    if not (inferred_dir / "openvino_model.xml").exists():
        print(f"OpenVINO Whisper export not found in {inferred_dir}, exporting once...")
        exported = OVModelForSpeechSeq2Seq.from_pretrained(
            whisper_model,
            export=True,
            compile=False,
        )
        try:
            exported.save_pretrained(str(inferred_dir))
        except RuntimeError as exc:
            print(
                "Warning: could not persist Whisper OpenVINO model to disk; "
                "falling back to in-memory export for this run."
            )
            model = OVModelForSpeechSeq2Seq.from_pretrained(
                whisper_model,
                export=True,
                device=device,
            )
        else:
            model = OVModelForSpeechSeq2Seq.from_pretrained(str(inferred_dir), device=device)
    else:
        model = OVModelForSpeechSeq2Seq.from_pretrained(str(inferred_dir), device=device)
    print(f"Whisper OpenVINO device target: {device}")
    pipe = hf_pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        return_timestamps=True,
        generate_kwargs={"language": "english"},
    )

    print("Running Whisper via OpenVINO...")
    result = pipe(audio_input)
    segments = result["chunks"]

    segments_cache.parent.mkdir(parents=True, exist_ok=True)
    with segments_cache.open("w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2)
    print(f"Cached {len(segments)} segments to {segments_cache}")
    return segments


def run_diarization(
    audio_path: Path,
    cache_path: Path,
    ov_dir: Path,
    device: str,
) -> Annotation:
    from pyannote_openvino import OVSpeakerDiarization
    from pyannote.core import Annotation, Segment

    if cache_path.exists():
        annotation = Annotation()
        with cache_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) < 8:
                    continue
                start = float(parts[3])
                duration = float(parts[4])
                speaker = parts[7]
                annotation[Segment(start, start + duration)] = speaker
        print(f"Loaded diarization cache from {cache_path}")
        return annotation

    pipeline = OVSpeakerDiarization.from_pretrained(ov_dir, device=device)
    print("Running OpenVINO speaker diarization...")
    waveform = torch.tensor(load_audio(audio_path)["array"]).unsqueeze(0)
    file_input = {
        "uri": audio_path.stem,
        "waveform": waveform,
        "sample_rate": AUDIO_RATE,
    }
    raw = pipeline(file_input)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        for turn, speaker in raw.speaker_diarization:
            f.write(
                f"SPEAKER audio 1 {turn.start:.3f} {turn.duration:.3f} "
                f"<NA> <NA> {speaker} <NA> <NA>\n"
            )

    return raw.speaker_diarization


def merge_segments(
    segments: list[dict[str, Any]],
    diarization: Annotation,
) -> list[dict[str, Any]]:
    transcript: list[dict[str, Any]] = []
    turns = list(diarization.itertracks(yield_label=True))
    for seg in segments:
        start = seg["timestamp"][0] or 0.0
        end = seg["timestamp"][1] or 0.0
        text = seg["text"].strip()
        speaker_scores: dict[str, float] = {}
        for turn, _, speaker in turns:
            overlap = min(end, turn.end) - max(start, turn.start)
            if overlap > 0:
                speaker_scores[speaker] = speaker_scores.get(speaker, 0.0) + overlap
        speaker = (
            max(speaker_scores, key=speaker_scores.get)
            if speaker_scores
            else "UNKNOWN"
        )
        transcript.append(
            {"start": start, "end": end, "speaker": speaker, "text": text}
        )
    return transcript


def dump_transcript(transcript: list[dict[str, Any]], json_path: Path, txt_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(transcript, handle, indent=2)

    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with txt_path.open("w", encoding="utf-8") as handle:
        current = None
        for turn in transcript:
            speaker = turn["speaker"]
            if speaker != current:
                current = speaker
                handle.write(f"\n{speaker}:\n")
            handle.write(f"  [{turn['start']:.1f}s] {turn['text']}\n")
    print(f"Transcript written to {txt_path} and {json_path}")


AUDIO_RATE = 16_000


def assert_openvino_device_available(device: str) -> None:
    from openvino import Core

    requested = str(device).upper()
    available = Core().available_devices
    if requested.startswith("GPU"):
        if not any(d.startswith("GPU") for d in available):
            raise RuntimeError(
                f"Requested OpenVINO device '{requested}' but no GPU device is available. "
                f"Available OpenVINO devices: {available}"
            )
    elif requested not in {"AUTO", "CPU"}:
        # Allow explicit ids like GPU.0 and unknown custom plugins only if available.
        if requested not in available and not any(requested.startswith(f"{d}.") for d in available):
            raise RuntimeError(
                f"Requested OpenVINO device '{requested}' is not available. "
                f"Available OpenVINO devices: {available}"
            )
    print(f"OpenVINO available devices: {available}. Requested: {requested}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Intel iGPU diarization + transcription.")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--segments-cache", type=Path, default=Path("artifacts/segments.json"))
    parser.add_argument("--rttm-cache", type=Path, default=Path("artifacts/diarization.rttm"))
    parser.add_argument("--output-json", type=Path, default=Path("artifacts/transcribed.json"))
    parser.add_argument("--output-txt", type=Path, default=Path("artifacts/transcribed.txt"))
    parser.add_argument("--whisper-model", type=str, default="openai/whisper-large-v3")
    parser.add_argument(
        "--whisper-ov",
        type=Path,
        default=Path("models/ov"),
        help="Folder containing the OpenVINO IR for Whisper (xml+bin)",
    )
    parser.add_argument("--ov-dir", type=Path, default=Path("models/ov"))
    parser.add_argument("--device", type=str, default="GPU")
    parser.add_argument("--ffmpeg", type=Path, default=Path("ffmpeg/bin"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ffmpeg_path = args.ffmpeg.expanduser()
    if ffmpeg_path.is_dir():
        ffmpeg_abs = ffmpeg_path.resolve()
        try:
            os.add_dll_directory(str(ffmpeg_abs))
        except (AttributeError, OSError) as exc:
            print(f"Warning: could not register FFmpeg DLL path {ffmpeg_abs}: {exc}")
        os.environ["PATH"] = str(ffmpeg_abs) + os.pathsep + os.environ.get("PATH", "")

    # pyannote warns loudly when torchcodec is unavailable; this pipeline does not rely on it.
    warnings.filterwarnings(
        "ignore",
        message="torchcodec is not installed correctly*",
        module="pyannote.audio.core.io",
    )
    assert_openvino_device_available(args.device)
    segments = run_whisper(
        args.audio,
        args.segments_cache,
        args.whisper_model,
        args.whisper_ov,
        args.device,
    )
    diarization = run_diarization(args.audio, args.rttm_cache, args.ov_dir, args.device)
    transcript = merge_segments(segments, diarization)
    dump_transcript(transcript, args.output_json, args.output_txt)


if __name__ == "__main__":
    main()
