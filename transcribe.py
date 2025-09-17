#!/usr/bin/env python3
"""Command-line tool to transcribe long Russian audio files into SRT subtitles using GigaAM.

This script relies on GigaAM's ``transcribe_longform`` method which performs voice
activity detection and chunking internally. It supports hours long audio files and
produces a standard ``.srt`` subtitle file.

Example:
    python transcribe.py input1.mp3 input2.mp3 --hf-token YOUR_HF_TOKEN
    python transcribe.py /path/to/folder --recursive

Before running install dependencies:
    pip install gigaam[longform]
"""
import argparse
import os
import shutil
import subprocess
import tempfile
from typing import Dict, Iterable, Iterator, List, Tuple

import gigaam


def format_srt_timestamp(seconds: float) -> str:
    """Format seconds to ``HH:MM:SS,mmm`` required by SRT."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"


def write_srt(
    segments: List[Dict[str, Tuple[float, float]]], output_path: str
) -> None:
    """Write transcription segments into an SRT file."""
    with open(output_path, "w", encoding="utf-8") as srt_file:
        for idx, seg in enumerate(segments, start=1):
            start, end = seg["boundaries"]
            transcription = seg["transcription"]
            srt_file.write(f"{idx}\n")
            srt_file.write(
                f"{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}\n"
            )
            srt_file.write(f"{transcription}\n\n")


def ensure_wav(path: str) -> Tuple[str, bool]:
    """Return a WAV path for ``path`` converting with ffmpeg if needed.

    Returns the path to a WAV file and a flag indicating whether the file is
    temporary and should be deleted afterwards.
    """
    if path.lower().endswith(".wav"):
        return path, False
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required to convert audio formats to WAV")
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()
    subprocess.run(
        ["ffmpeg", "-y", "-i", path, tmp_path],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return tmp_path, True


MEDIA_EXTENSIONS = {
    ".aac",
    ".aiff",
    ".flac",
    ".m4a",
    ".mka",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
}


def is_media_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in MEDIA_EXTENSIONS


def has_adjacent_srt(path: str) -> bool:
    return os.path.exists(os.path.splitext(path)[0] + ".srt")


def _iter_directory_files(directory: str, recursive: bool) -> Iterator[str]:
    if recursive:
        for root, _, files in os.walk(directory):
            for name in files:
                yield os.path.join(root, name)
    else:
        for entry in os.scandir(directory):
            if entry.is_file():
                yield entry.path


def collect_media_paths(inputs: Iterable[str], recursive: bool) -> List[str]:
    discovered: List[str] = []
    seen = set()
    for original in inputs:
        if not os.path.exists(original):
            raise FileNotFoundError(f"Input path does not exist: {original}")
        if os.path.isdir(original):
            for candidate in _iter_directory_files(original, recursive):
                if not is_media_file(candidate) or has_adjacent_srt(candidate):
                    continue
                if candidate not in seen:
                    seen.add(candidate)
                    discovered.append(candidate)
            continue
        if not is_media_file(original):
            raise ValueError(f"Unsupported media file: {original}")
        if has_adjacent_srt(original):
            continue
        if original not in seen:
            seen.add(original)
            discovered.append(original)
    return discovered


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe audio into Russian SRT subtitles using GigaAM"
    )
    parser.add_argument(
        "audio", nargs="+", help="Path(s) to input media file(s) or directories"
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to save the resulting SRT file (default: audio path with .srt)",
    )
    parser.add_argument(
        "--model",
        default="ctc",
        choices=["ctc", "rnnt"],
        help="ASR model to use (default: ctc)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device for inference (e.g. cuda or cpu)",
    )
    parser.add_argument(
        "--hf-token",
        help="Hugging Face token for pyannote VAD used in long-form transcription",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=22.0,
        help="Maximum chunk duration in seconds",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=15.0,
        help="Minimum chunk duration before splitting",
    )
    parser.add_argument(
        "--new-chunk-threshold",
        type=float,
        default=0.2,
        help="Pause threshold (seconds) to start a new chunk",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recursively search directories for media files",
    )

    args = parser.parse_args()

    try:
        audio_inputs = collect_media_paths(args.audio, args.recursive)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    if not audio_inputs:
        parser.error(
            "No audio or video files without adjacent SRT subtitles were found."
        )

    if args.output and len(audio_inputs) > 1:
        parser.error("--output can only be used with a single input media file")

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    model = gigaam.load_model(args.model, device=args.device)
    for audio_path in audio_inputs:
        wav_path, is_temp = ensure_wav(audio_path)
        try:
            segments = model.transcribe_longform(
                wav_path,
                max_duration=args.max_duration,
                min_duration=args.min_duration,
                new_chunk_threshold=args.new_chunk_threshold,
            )
        finally:
            if is_temp:
                os.remove(wav_path)

        output_path = args.output or os.path.splitext(audio_path)[0] + ".srt"
        write_srt(segments, output_path)


if __name__ == "__main__":
    main()
