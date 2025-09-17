# GigaAM SRT Transcriber

A small command-line utility that uses [GigaAM](https://github.com/salute-developers/GigaAM)
to convert long-form Russian audio (e.g. lectures or podcasts) into subtitle files in the
SRT format.

## Requirements

```bash
pip install gigaam[longform]
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
```

For a single file you may also specify `--output subtitles.srt` to set the
destination path. When multiple inputs (or directories) are provided, `.srt`
files are written next to each media file without an existing subtitle.

Additional useful options:

* `--model` – choose `ctc` (default) or `rnnt` model.
* `--device` – specify inference device, e.g. `cuda` or `cpu`.
* `--recursive` – look through folders recursively for audio/video files without
  an `.srt` subtitle.
* `--max-duration`, `--min-duration`, `--new-chunk-threshold` – control how the audio is
  segmented before transcription.

The script accepts any audio format supported by `ffmpeg` and writes a UTF‑8 encoded
`.srt` file with time-coded Russian subtitles.
