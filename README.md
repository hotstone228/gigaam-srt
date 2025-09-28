# GigaAM SRT Transcriber

A small command-line utility that uses [GigaAM](https://github.com/salute-developers/GigaAM)
to convert long-form Russian audio (e.g. lectures or podcasts) into subtitle files in the
SRT format.

## Requirements

```bash
pip install gigaam[longform] sberam

# Optional: enable drag-and-drop GUI mode
pip install tkinterdnd2
```

For non-WAV inputs the script relies on `ffmpeg` being available in your `PATH` to
convert other formats (mp3, m4a, flac, ...).

You also need a [HuggingFace token](https://huggingface.co/docs/hub/security-tokens) in
order to download the VAD models used for splitting long audio files.

## Usage

```bash
python transcribe.py /path/to/audio1.mp3 /path/to/audio2.mp3 \
    --hf-token YOUR_TOKEN

# Transcribe every media file without an existing .srt next to it
python transcribe.py /path/to/folder --recursive

# Mix and match files and directories, or list directories explicitly
python transcribe.py video.mp4 -d recordings/lectures -d recordings/meetups
```

For a single file you may also specify `--output subtitles.srt` to set the
destination path. When multiple inputs (or directories) are provided, `.srt`
files are written next to each media file without an existing subtitle.

Additional useful options:

* `--model` – choose `ctc` (default) or `rnnt` model.
* `--device` – specify inference device, e.g. `cuda` or `cpu`.
* `--recursive` – look through folders recursively for audio/video files without
  an `.srt` subtitle.
* `--ignore-errors` / `--raise-errors` – keep processing other files after a failure
  (default) or stop immediately to view the traceback.
* `-d/--directory` – add one or more directories to scan in addition to positional
  inputs.
* `--logging` / `--no-logging` – turn informational logging on or off.
* `--max-duration`, `--min-duration`, `--new-chunk-threshold` – control how the audio is
  segmented before transcription.

The script accepts any audio format supported by `ffmpeg` and writes a UTF‑8 encoded
`.srt` file with time-coded Russian subtitles.

### Drag-and-drop GUI mode

To keep the model loaded in memory while processing multiple files, run the script with
`--gui`. A window will open where you can drop audio or video files; each file is
transcribed sequentially without reloading the ASR model. The same validation rules for
supported media formats and existing `.srt` files apply. You may optionally pass initial
inputs on the command line – they will be queued automatically when the window opens.
