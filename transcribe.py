#!/usr/bin/env python3
"""Command-line tool to transcribe long Russian audio files into SRT subtitles using GigaAM.

This script relies on GigaAM's ``transcribe_longform`` method which performs voice
activity detection and chunking internally. It supports hours long audio files and
produces a standard ``.srt`` subtitle file.

Example:
    python transcribe.py input1.mp3 input2.mp3 --hf-token YOUR_HF_TOKEN
    python transcribe.py /path/to/folder --recursive

Before running install dependencies:
    pip install gigaam[longform] sberam

For drag-and-drop GUI mode:
    pip install tkinterdnd2
"""
import argparse
import logging
import os
import queue
import shutil
import subprocess
import tempfile
import threading
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from gigaam import GigaAMModel  # type: ignore
else:
    GigaAMModel = Any  # type: ignore


LOGGER = logging.getLogger(__name__)


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
            LOGGER.info(
                "Scanning directory %s for media files%s",
                original,
                " recursively" if recursive else "",
            )
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


def transcribe_audio_file(
    model: "GigaAMModel",
    audio_path: str,
    args: argparse.Namespace,
    output_override: Optional[str] = None,
) -> Optional[str]:
    """Transcribe a single ``audio_path`` and return the output SRT path."""

    LOGGER.info("Processing %s", audio_path)
    wav_path, is_temp = ensure_wav(audio_path)
    try:
        segments = model.transcribe_longform(
            wav_path,
            max_duration=args.max_duration,
            min_duration=args.min_duration,
            new_chunk_threshold=args.new_chunk_threshold,
        )
    except Exception as exc:  # pylint: disable=broad-except
        if args.ignore_errors:
            LOGGER.error("Transcription failed for %s: %s", audio_path, exc)
            return None
        raise
    finally:
        if is_temp:
            os.remove(wav_path)

    output_path = output_override or os.path.splitext(audio_path)[0] + ".srt"
    write_srt(segments, output_path)
    LOGGER.info("Saved subtitles to %s", output_path)
    return output_path


def launch_drag_and_drop_gui(
    model: "GigaAMModel",
    args: argparse.Namespace,
    initial_inputs: Optional[List[str]] = None,
) -> None:
    """Launch a drag-and-drop GUI for sequential transcription."""

    try:
        import tkinter as tk
        from tkinter import messagebox
    except ImportError as exc:  # pragma: no cover - platform dependent
        raise RuntimeError(
            "Tkinter is required for --gui mode but is not available on this system"
        ) from exc

    try:
        from tkinterdnd2 import DND_FILES, TkinterDnD  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "The 'tkinterdnd2' package is required for drag-and-drop support. "
            "Install it with 'pip install tkinterdnd2'."
        ) from exc

    task_queue: "queue.Queue[List[str]]" = queue.Queue()
    stop_event = threading.Event()

    root = TkinterDnD.Tk()
    root.title("GigaAM SRT Transcriber")

    status_var = tk.StringVar(value="Перетащите аудио или видео файлы в окно")

    frame = tk.Frame(root, padx=20, pady=20)
    frame.pack(fill="both", expand=True)

    drop_label = tk.Label(
        frame,
        textvariable=status_var,
        relief="groove",
        borderwidth=2,
        width=60,
        height=10,
        wraplength=400,
        justify="center",
    )
    drop_label.pack(fill="both", expand=True)
    drop_label.drop_target_register(DND_FILES)

    log_text = tk.Text(frame, height=10, state="disabled")
    log_text.pack(fill="both", expand=True, pady=(10, 0))

    def append_log(message: str) -> None:
        def _append() -> None:
            log_text.configure(state="normal")
            log_text.insert("end", message + "\n")
            log_text.see("end")
            log_text.configure(state="disabled")

        root.after(0, _append)

    def set_status(message: str) -> None:
        root.after(0, status_var.set, message)

    def process_inputs(inputs: List[str]) -> None:
        try:
            collected = collect_media_paths(inputs, args.recursive)
        except (FileNotFoundError, ValueError) as exc:
            append_log(f"Ошибка: {exc}")
            root.after(0, lambda: messagebox.showerror("Ошибка", str(exc)))
            return

        if not collected:
            set_status("Подходящие файлы не найдены или SRT уже существует")
            return

        for audio_path in collected:
            set_status(f"Обработка {audio_path}")
            try:
                output_path = transcribe_audio_file(model, audio_path, args)
            except Exception as exc:  # pylint: disable=broad-except
                append_log(f"Ошибка при обработке {audio_path}: {exc}")
                root.after(
                    0,
                    lambda p=audio_path, e=exc: messagebox.showerror(
                        "Ошибка транскрибации", f"{p}: {e}"
                    ),
                )
                if not args.ignore_errors:
                    break
                continue

            if output_path:
                append_log(f"Готово: {output_path}")
                set_status(f"Субтитры сохранены в {output_path}")

    def worker() -> None:
        while not stop_event.is_set():
            try:
                inputs = task_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if inputs is None:
                task_queue.task_done()
                break
            process_inputs(inputs)
            task_queue.task_done()

    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()

    def on_drop(event: "tk.Event[tk.Misc]") -> None:  # type: ignore[name-defined]
        files = list(root.tk.splitlist(event.data))
        if files:
            append_log("Добавлены файлы: " + ", ".join(files))
            task_queue.put(files)

    drop_label.dnd_bind("<<Drop>>", on_drop)

    def on_close() -> None:
        stop_event.set()
        task_queue.put(None)  # type: ignore[arg-type]
        root.after(100, root.destroy)

    root.protocol("WM_DELETE_WINDOW", on_close)

    if initial_inputs:
        task_queue.put(initial_inputs)

    root.mainloop()
    worker_thread.join(timeout=1)


def load_asr_model(model_name: str, device: Optional[str]) -> "GigaAMModel":
    """Import required dependencies and return an initialized ASR model."""

    try:
        import sberam  # type: ignore  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The 'sberam' package is required. Install it with 'pip install sberam'."
        ) from exc

    try:
        import gigaam  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The 'gigaam' package is required. Install it with "
            "'pip install gigaam[longform]'."
        ) from exc

    return gigaam.load_model(model_name, device=device)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe audio into Russian SRT subtitles using GigaAM"
    )
    parser.add_argument(
        "inputs", nargs="*", help="Path(s) to input media file(s) or directories"
    )
    parser.add_argument(
        "-d",
        "--directory",
        dest="directories",
        action="append",
        default=[],
        help="Directory to scan for media files (can be specified multiple times)",
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
    parser.add_argument(
        "--ignore-errors",
        dest="ignore_errors",
        action="store_true",
        help="Continue processing other files when transcription fails",
    )
    parser.add_argument(
        "--raise-errors",
        dest="ignore_errors",
        action="store_false",
        help="Stop immediately and show full tracebacks on errors",
    )
    parser.add_argument(
        "--logging",
        dest="logging_enabled",
        action="store_true",
        help="Show informational logging during processing",
    )
    parser.add_argument(
        "--no-logging",
        dest="logging_enabled",
        action="store_false",
        help="Disable logging output",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help=(
            "Launch a simple drag-and-drop window that keeps the model loaded "
            "while the window remains open"
        ),
    )
    parser.set_defaults(ignore_errors=True, logging_enabled=True)

    args = parser.parse_args()

    if args.logging_enabled:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.CRITICAL + 1)

    combined_inputs: List[str] = []
    if args.inputs:
        combined_inputs.extend(args.inputs)
    if args.directories:
        combined_inputs.extend(args.directories)

    if not combined_inputs and not args.gui:
        parser.error("Please provide at least one media file or directory to process.")

    try:
        audio_inputs = collect_media_paths(combined_inputs, args.recursive)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    if not audio_inputs and not args.gui:
        parser.error(
            "No audio or video files without adjacent SRT subtitles were found."
        )

    if audio_inputs:
        LOGGER.info("Discovered %d media file(s) for transcription", len(audio_inputs))

    if args.gui and args.output:
        parser.error("--output cannot be used together with --gui mode")

    if args.output and len(audio_inputs) > 1:
        parser.error("--output can only be used with a single input media file")

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    try:
        model = load_asr_model(args.model, args.device)
    except RuntimeError as exc:
        parser.error(str(exc))

    if args.gui:
        try:
            launch_drag_and_drop_gui(model, args, audio_inputs or None)
        except RuntimeError as exc:
            parser.error(str(exc))
        return

    for audio_path in audio_inputs:
        output_path = transcribe_audio_file(
            model, audio_path, args, output_override=args.output
        )
        if output_path is None and not args.ignore_errors:
            # The helper only returns ``None`` when ``ignore_errors`` is enabled,
            # so this branch should normally never be reached.
            break


if __name__ == "__main__":
    main()
