# WhisperX Transcription API · v1.8.3

Open-source, **OpenAI-compatible** HTTP service built on top of [WhisperX](https://github.com/m-bain/whisperX) with optional alignment & diarisation.  
Runs GPU-only, supports every Faster-Whisper variant, and can operate fully offline.

---

## Quick start

### One-liner Docker

```bash
docker run -it --gpus all   -p 8000:8000   -v whisper-cache:/root/.cache   -e MODEL_TTL_SEC=600   -e HF_TOKEN=<your-hf-token>   ghcr.io/your-org/whisperx-api:latest
```

### docker-compose.yml

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
      - whisper-cache:/root/.cache        # persistent HF cache
    environment:
      MODEL_TTL_SEC:          600         # evict idle models after 10 min
      MAX_THREADS:            4           # CPU worker threads
      FASTER_WHISPER_THREADS: 0           # 0 → default
      HF_TOKEN:               "${HF_TOKEN}"
      LOCAL_ONLY_MODELS:      0           # 1 → full offline mode
volumes:
  whisper-cache:
```

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_TTL_SEC` | `600` | Seconds of inactivity after which a model is evicted from VRAM. |
| `MAX_THREADS` | `4` | Size of the `ThreadPoolExecutor` used for blocking work. |
| `FASTER_WHISPER_THREADS` | `0` | Value forwarded to Faster-Whisper `threads` (0 = not passed). |
| `HF_TOKEN` | — | HF access token for private diarisation models. |
| `LOCAL_ONLY_MODELS` | `0` | `1` → forbid network downloads, fail if model not cached. |
| `HF_HOME`, `XDG_CACHE_HOME` | — | Override HuggingFace cache location. |

⚠️ **TF32 is disabled globally** for reproducibility.

---

## Endpoints

### `POST /v1/audio/transcriptions`

Upload an audio file, receive a transcription (optionally aligned & diarised).

| Field | Default | Notes |
|-------|---------|-------|
| `file` | — | Binary audio (any FFmpeg-decodable format). |
| `model` | `large-v3` | Faster-Whisper model id. See `/v1/models`. |
| `language` | _(auto)_ | Force language code; autodetect when omitted. |
| `align` | `false` | Word-level alignment via Wav2Vec2. |
| `diarize` | `false` | Speaker diarisation with `[SPK_n]` tags. |
| `response_format` | `json` | `json`, `text`, `srt`, `vtt`, `verbose_json`. |
| `batch_size` | `16` | Whisper batch size. |
| `beam_size` | `0` | Beam width (`0` → greedy). |
| `best_of` | `0` | n-best for greedy. |
| `patience` | `0.0` | Beam search patience. |
| `length_penalty` | `0.0` | Beam length penalty. |
| `word_timestamps` | `false` | Include word timestamps (needs new FW build). |
| `vad_filter` | `false` | Apply VAD before transcription. |
| `vad_threshold` | `0.5` | VAD probability threshold. |
| `min_speakers` | `0` | Lower bound for diarisation clustering. |
| `max_speakers` | `0` | Upper bound for diarisation clustering. |

Example:

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions   -F file=@sample.mp3   -F model=medium   -F align=true   -F diarize=true   -F response_format=srt > out.srt
```

### `POST /v1/audio/translations`

Same contract as `/transcriptions`, but forces English output (WhisperX “translate” task).

### `GET /v1/models`

Returns the model catalogue:

* **Online mode** – every Faster-Whisper variant is listed.  
* **Offline mode** – only models already cached on disk / in VRAM.

```json
{
  "data": [
    { "id": "large-v3", "object": "model", "created": 0, "owned_by": "you", "downloaded": true },
    { "id": "small",    "object": "model", "created": 0, "owned_by": "you", "downloaded": false }
  ]
}
```

---

## Logging

```
[transcribe_start] meeting.wav  freeVRAM=22546 MB model=large-v3
[whisper_model_load_end]    model=large-v3  used=+4096 MB  freeVRAM=18450 MB
[align_end]   meeting.wav  freeVRAM=17320 MB Δ=4.35s
[summary]     meeting.wav Δ=12.3s audio=180.0s speed=14.6x
```
* **freeVRAM** – globally free GPU memory *before* the log event.  
* **used=±MB** – delta allocated or released during load/unload.

---

## Offline mode

Set `LOCAL_ONLY_MODELS=1` to disable downloads.  Requests for uncached
models raise a clear Faster‑Whisper error.

Pre‑seed the cache:

```bash
python - <<'PY'
import whisperx
whisperx.load_model("large-v3", device="cuda", local_files_only=False)
PY
```

---

## License

API code © 2025, MIT license.  
Whisper / WhisperX / Faster‑Whisper remain under their respective open-source licenses.
