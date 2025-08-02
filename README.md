# WhisperX Transcription API

Open‑source, **OpenAI‑compatible** HTTP service that wraps [WhisperX](https://github.com/m-bain/whisperX) with optional alignment & diarisation. Runs GPU‑only, supports all Faster‑Whisper model variants and works fully offline when required.

---

## Quick start

### Docker (one‑liner)
```bash
# mounts a persistent HF cache volume and exposes :8000
 docker run -it --gpus all \
    -p 8000:8000 \
    -v whisper-cache:/root/.cache \
    -e MODEL_TTL_SEC=600 \
    -e HF_TOKEN=<your‑hf‑token> \
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
      MODEL_TTL_SEC: 600        # evict models after 10 min idle
      MAX_THREADS:   4          # CPU executors
      FASTER_WHISPER_THREADS: 0 # per‑model threads (0 = default)
      HF_TOKEN:      "${HF_TOKEN}"
      LOCAL_ONLY_MODELS: 0      # 1 = offline mode (no downloads)
volumes:
  whisper-cache:
```

---

## Runtime environment variables
| Variable | Default | Purpose |
|----------|---------|---------|
| `MODEL_TTL_SEC` | `600` | Seconds after which an *idle* model is removed from VRAM. |
| `MAX_THREADS` | `4` | Size of `ThreadPoolExecutor` used for blocking work. |
| `FASTER_WHISPER_THREADS` | `0` | Value forwarded to Faster‑Whisper\`threads` (0 → not passed). |
| `HF_TOKEN` | — | HF access token for private diarisation models. |
| `LOCAL_ONLY_MODELS` | `0` | `1` → do **not** download models; fails if absent in cache. |
| `HF_HOME`, `XDG_CACHE_HOME` | — | Customise HuggingFace cache location. |

ℹ️ TF32 is **disabled** application‑wide for reproducibility.

---

## Endpoints
### `POST /v1/audio/transcriptions`
Transcribes (and optionally aligns / diarises) an uploaded audio file.

| Form field | Type | Default | Description |
|------------|------|---------|-------------|
| `file` | *binary* | — | Audio file (any format decodable by FFmpeg). |
| `model` | str | `large-v3` | Faster‑Whisper model id. See `/v1/models`. |
| `language` | str | `None` | Force language code; if omitted WhisperX auto‑detects. |
| `align` | bool | `false` | Word‑level alignment via Wav2Vec2. |
| `diarize` | bool | `false` | Speaker diarisation. Adds `[SPK_n]` tags. |
| `response_format` | enum | `json` | `json`, `text`, `srt`, `vtt`, `verbose_json`. |
| `batch_size` | int | `16` | Whisper batch size. |
| `beam_size` | int | `0` | Beam width (`0` → greedy). |
| `best_of` | int | `0` | *n‑best* for greedy. |
| `patience` | float | `0.0` | Beam search patience. |
| `length_penalty` | float | `0.0` | Beam length penalty. |
| `word_timestamps` | bool | `false` | Include word‑level timestamps (requires new FW build). |
| `vad_filter` | bool | `false` | Apply VAD before transcription. |
| `vad_threshold` | float | `0.5` | VAD probability threshold. |
| `min_speakers` | int | `0` | Lower bound for diarisation. |
| `max_speakers` | int | `0` | Upper bound for diarisation. |

#### Example (cURL)
```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@sample.mp3 \
  -F model=medium \
  -F align=true \
  -F diarize=true \
  -F response_format=srt > out.srt
```

### `POST /v1/audio/translations`
Same contract as `/transcriptions` but forces English output (WhisperX task `translate`).

### `GET /v1/models`
Returns the union of models currently cached in VRAM *plus* models present on disk:
```json
{
  "data": [
    {"id":"large-v3","object":"model","created":0,"owned_by":"you"},
    {"id":"small","object":"model","created":0,"owned_by":"you"}
  ]
}
```

---

## Logging
All messages are timestamped and go to stdout:
```
[transcribe_start] sample.wav  freeVRAM=22546 MB model=large-v3
[whisper_model_load_start]  model=large-v3  freeVRAM=22546 MB
[whisper_model_load_end]    model=large-v3  used=+4096 MB  freeVRAM=18450 MB
[align_start]  sample.wav  freeVRAM=18450 MB lang=en
...
[summary] sample.wav Δ=12.3s audio=180.0s speed=14.6x
```
* `freeVRAM` – global free GPU memory before the log event.
* `used=±MB` – delta allocated or freed by that step.

---

## Offline mode
Set `LOCAL_ONLY_MODELS=1` to forbid network downloads.  If a requested
model is missing from the local HF cache, the request raises a clear
error from Faster‑Whisper.

You can pre‑seed the cache by running:
```bash
python - <<'PY'
import whisperx, torch
whisperx.load_model("large-v3", device="cuda", local_files_only=False)
PY
```
and persisting the cache volume.

---

## License
Code is MIT‑licensed.  Whisper, WhisperX and Faster‑Whisper remain under
their respective open‑source licenses.

