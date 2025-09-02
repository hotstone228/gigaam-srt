#!/usr/bin/env python3
"""Command-line tool to transcribe long Russian audio files into SRT subtitles using GigaAM.

This script relies on GigaAM's ``transcribe_longform`` method which performs voice
activity detection and chunking internally. It supports hours long audio files and
produces a standard ``.srt`` subtitle file.

Example:
    python transcribe.py input.wav --hf-token YOUR_HF_TOKEN --output subtitles.srt

Before running install dependencies:
    pip install gigaam[longform]
"""
import argparse
import os
from typing import List, Dict, Tuple

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe audio into Russian SRT subtitles using GigaAM"
    )
    parser.add_argument("audio", help="Path to the input audio file")
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

    args = parser.parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    model = gigaam.load_model(args.model, device=args.device)
    segments = model.transcribe_longform(
        args.audio,
        max_duration=args.max_duration,
        min_duration=args.min_duration,
        new_chunk_threshold=args.new_chunk_threshold,
    )

    output_path = args.output or os.path.splitext(args.audio)[0] + ".srt"
    write_srt(segments, output_path)


if __name__ == "__main__":
    main()
