# WhisperX Transcription API · v1.8.7

Open‑source, **OpenAI‑compatible** HTTP service built on top of [WhisperX](https://github.com/m-bain/whisperX) with optional alignment & diarisation.  
Runs GPU‑only, supports every Faster‑Whisper variant, and can operate fully offline.

---

## What’s new in 1.8.7  (2025‑08‑23)

* Added `DIARIZATION_MODEL` environment variable to overwrite default model for diarization (pyannote/speaker-diarization-3.1)

## What’s new in 1.8.5 (2025‑08‑04)

* `/v1/models` (offline mode) now scans **all** HF cache roots
  (`HF_HOME`, `XDG_CACHE_HOME`, `~/.cache`, `/root/.cache`, `/.cache`), so
  every locally‑downloaded model is listed.
* Everything else unchanged (TF32 off, TTL eviction, detailed logging).

---

## Quick start

### One‑liner Docker

```bash
docker run -it --gpus all \
  -p 8000:8000 \
  -v whisper-cache:/root/.cache \
  -e MODEL_TTL_SEC=600 \
  -e HF_TOKEN=<your-hf-token> \
  ghcr.io/your-org/whisperx-api:latest
```

### docker‑compose.yml

```yaml
services:
  whisperx:
    image: ghcr.io/your-org/whisperx-api:latest
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
    ports: ["8000:8000"]
    volumes:
      - whisper-cache:/root/.cache
    environment:
      MODEL_TTL_SEC:          600
      MAX_THREADS:            4
      FASTER_WHISPER_THREADS: 0
      HF_TOKEN:               "${HF_TOKEN}"
      LOCAL_ONLY_MODELS:      0
volumes:
  whisper-cache:
```

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_TTL_SEC` | `600` | Seconds of inactivity after which a model is evicted from VRAM. |
| `MAX_THREADS` | `4` | Size of the ThreadPoolExecutor for blocking work. |
| `FASTER_WHISPER_THREADS` | `0` | Value forwarded to Faster‑Whisper `threads` (0 = not passed). |
| `HF_TOKEN` | — | HF access token for private diarisation models. |
| `LOCAL_ONLY_MODELS` | `0` | `1` → forbid downloads, fail if model not cached. |
| `WARMUP_MODEL` | `large-v3` | Whisper model ID to preload on startup. |
| `WARMUP_ALIGN_LANGS` | `en` | Comma-separated list of language codes to preload alignment models for. |
| `WARMUP_DIARIZE` | `0` | `1` → preload the diarization model. |
| `ASR_CONFIG_JSON` | — | JSON string to configure ASR options per model. See code for default. |
| `DIARIZATION_MODEL` | - | Override default diarization model. |
| `HF_HOME`, `XDG_CACHE_HOME` | — | Override HuggingFace cache location. |

⚠️ **TF32 is disabled globally** for reproducibility.

---

## Endpoints

### `POST /v1/audio/transcriptions`

Upload audio, receive transcription (optionally aligned & diarised).

| Field | Default | Notes |
|-------|---------|-------|
| `file` | — | Binary audio (any FFmpeg‑decodable format). |
| `model` | `large-v3` | Faster‑Whisper model id (see `/v1/models`). |
| `language` | _(auto)_ | Force language code; autodetect when omitted. |
| `align` | `false` | Word‑level alignment via Wav2Vec2. |
| `diarize` | `false` | Speaker diarisation with `[SPK_n]` tags. |
| `response_format` | `json` | `json`, `text`, `srt`, `vtt`, `verbose_json`. |
| `batch_size` | `16` | Whisper batch size. |
| `word_timestamps` | `false` | Include word timestamps (needs new FW build). |
| `vad_filter` | `false` | Apply VAD before transcription. |
| `vad_threshold` | `0.5` | VAD probability threshold. |
| `min_speakers` | `0` | Lower bound for diarisation clustering. |
| `max_speakers` | `0` | Upper bound for diarisation clustering. |

Example:

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@sample.mp3 \
  -F model=medium \
  -F align=true \
  -F diarize=true \
  -F response_format=srt > out.srt
```

### `POST /v1/audio/translations`

Same contract as `/transcriptions`, but forces English output
(WhisperX “translate” task).

### `GET /v1/models`

* **Online** → every Faster‑Whisper variant, `"downloaded": true/false`  
* **Offline** → only variants physically present in cache.

```json
{
  "data": [
    { "id": "large-v3", "downloaded": true },
    { "id": "small",    "downloaded": false }
  ]
}
```

---

## Logging example

```
[transcribe_start] meeting.wav  freeVRAM=22546 MB model=large-v3
[whisper_model_load_end]    model=large-v3  used=+4096 MB  freeVRAM=18450 MB
[align_end]   meeting.wav  freeVRAM=17320 MB Δ=4.35s
[summary]     meeting.wav Δ=12.3s audio=180.0s speed=14.6x
```

---

## Offline mode

Set `LOCAL_ONLY_MODELS=1` to disable downloads completely.

```bash
python - <<'PY'
import whisperx
whisperx.load_model("large-v3", device="cuda", local_files_only=False)
PY
```

---

## License

API code © 2025, MIT license.  
Whisper / WhisperX / Faster‑Whisper remain under their respective OSS licenses.
