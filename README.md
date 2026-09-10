# WhisperX Transcription API · v1.13.0

Open-source, **OpenAI-compatible** HTTP service built on top of [WhisperX](https://github.com/m-bain/whisperX) with optional alignment & diarisation.
Runs GPU-only, supports every Faster-Whisper variant, and can operate fully offline.

---

## What’s new in 1.13.0  (2026-09-10)

* **More Prometheus metrics** for cold-start visibility, capacity planning, and audio/request
  shape: `whisperx_model_load_seconds{kind}`, `whisperx_model_load_events_total{kind}`,
  `whisperx_model_vram_usage_mb{kind,key}`, `whisperx_model_evictions_total{kind}`,
  `whisperx_pool_wait_seconds{model}`, `whisperx_audio_duration_seconds`,
  `whisperx_upload_size_bytes`, `whisperx_num_speakers_detected`,
  `whisperx_language_detected_total{language}`, and a `model` label added to
  `whisperx_requests_total`. See "Metrics" below.

## What’s new in 1.12.1  (2026-09-09)

* **Fixed Docker build.** The upstream `ghcr.io/jim60105/whisperx:no_model`
  base image bumped Python 3.11 → 3.13; the Dockerfile's wheel-builder stage
  and hardcoded `site-packages` paths now target Python 3.13 to match.

## What’s new in 1.12.0  (2026-09-09)

* **`GET /metrics` Prometheus endpoint.** Request counts/latency per
  endpoint, per-thread transcription speed ratio
  (`whisperx_transcribe_speed_ratio`, `audio_s/wall_s` labeled by `model`
  and executor thread), active transcriptions, whisper pool sizes, free
  GPU memory, and error counters. See "Metrics" below.

## What’s new in 1.11.0  (2026-04-22)

* **OpenAI-compatible `usage` field.** JSON responses (`json` and
  `verbose_json`) now include a `usage` object matching OpenAI's
  `gpt-4o-transcribe` *duration* variant:

  ```json
  {
    "text": "...",
    "usage": { "type": "duration", "seconds": 12.34 }
  }
  ```

  `verbose_json` additionally exposes `duration` at the top level (matching
  OpenAI `whisper-1`). `seconds` is the audio duration, not wall-clock
  processing time. Plain-text formats (`text`, `srt`, `vtt`) are unchanged.

## What’s new in 1.10.0  (2026-04-22)

* **Concurrent transcriptions on a single GPU.** Each `(model, asr_options)`
  pair now backs a *pool* of N WhisperX instances (`TRANSCRIBE_CONCURRENCY`,
  default `1`). Several requests for the same model now run truly in parallel
  on big GPUs (e.g. L40s with 48 GB VRAM can comfortably hold 3–5 large-v3
  instances).
* **Event-loop never blocks.** All model loading (whisper / align / diarize)
  is offloaded to the thread pool and guarded by `asyncio.Lock` instead of
  `threading.Lock`. The previous design synchronously acquired a thread lock
  inside the async handler, which froze the whole server while another
  request was transcribing. Health-checks, `/v1/models` and other endpoints
  now stay responsive under load.
* `whisperx.load_audio` is also offloaded to the worker pool.
* Pool TTL eviction: when every instance of a pool is idle longer than
  `MODEL_TTL_SEC`, the whole pool is unloaded and VRAM freed.

## What’s new in 1.9.0  (2025-09-08)

* Minor release: formalizes recent improvements (1.8.9) under 1.9.0.
* No additional functional changes beyond 1.8.9.

## What’s new in 1.8.9  (2025-09-08)

* Respects explicit `language` parameter by forwarding it to
  `whisper.transcribe` when provided.
* Serialized model loading to avoid concurrent initializations.
* Fixed TTL cache semantics and zero VAD threshold handling.
* Fixed response format detection.
* Minor docs clarifications around diarisation defaults.

## What’s new in 1.8.8  (2025-08-24)

* New **`diarization_model`** request field:
  allows clients to override the diarisation backend per request
  (e.g. `-F diarization_model=pyannote/speaker-diarization-3.1`).
* Diarisation pipelines are cached **per model name** with TTL eviction.
* Logging clarified – diarisation unloads now logged as
  `model=<name>` instead of the generic `pipeline`.

## What’s new in 1.8.7  (2025-08-23)

* Added `DIARIZATION_MODEL` environment variable to overwrite default model for diarisation (pyannote/speaker-diarization-3.1)

## What’s new in 1.8.5 (2025-08-04)

* `/v1/models` (offline mode) now scans **all** HF cache roots
  (`HF_HOME`, `XDG_CACHE_HOME`, `~/.cache`, `/root/.cache`, `/.cache`), so
  every locally-downloaded model is listed.
* Everything else unchanged (TF32 off, TTL eviction, detailed logging).

---

## Quick start

### One-liner Docker

```bash
docker run -it --gpus all \
  -p 8000:8000 \
  -v whisper-cache:/root/.cache \
  -e MODEL_TTL_SEC=600 \
  -e HF_TOKEN=<your-hf-token> \
  ghcr.io/your-org/whisperx-api:latest
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
      - whisper-cache:/root/.cache
    environment:
      MODEL_TTL_SEC:          600
      MAX_THREADS:            4
      FASTER_WHISPER_THREADS: 0
      HF_TOKEN:               "${HF_TOKEN}"
      LOCAL_ONLY_MODELS:      0
      DIARIZATION_MODEL:      "pyannote/speaker-diarization-3.1"
volumes:
  whisper-cache:
```

---

## Environment variables

| Variable                    | Default    | Description                                                                                  |
| --------------------------- | ---------- | -------------------------------------------------------------------------------------------- |
| `MODEL_TTL_SEC`             | `600`      | Seconds of inactivity after which a model is evicted from VRAM.                              |
| `MAX_THREADS`               | `4`        | Size of the ThreadPoolExecutor for blocking work.                                            |
| `TRANSCRIBE_CONCURRENCY`    | `1`        | Number of warm Whisper instances per `(model, asr_options)` pool. Increase on big GPUs (L40s: 3-5 for large-v3). |
| `GPU_HOURLY_COST_USD`       | `0`        | Hourly GPU rate used to compute `whisperx_estimated_cost_usd_total` (0 = cost tracking disabled). |
| `FASTER_WHISPER_THREADS`    | `0`        | Value forwarded to Faster-Whisper `threads` (0 = not passed).                                |
| `HF_TOKEN`                  | —          | HF access token for private diarization models.                                              |
| `LOCAL_ONLY_MODELS`         | `0`        | `1` → forbid downloads, fail if model not cached.                                            |
| `WARMUP_MODEL`              | `large-v3` | Whisper model ID to preload on startup.                                                      |
| `WARMUP_ALIGN_LANGS`        | `en`       | Comma-separated list of language codes to preload alignment models for.                      |
| `WARMUP_DIARIZE`            | `0`        | `1` → preload the diarization model.                                                         |
| `ASR_CONFIG_JSON`           | —          | JSON string to configure ASR options per model. See code for default.                        |
| `DIARIZATION_MODEL`         | `pyannote/speaker-diarization-3.1` | Override default diarization model (used when request does not specify `diarization_model`). |
| `HF_HOME`, `XDG_CACHE_HOME` | —          | Override HuggingFace cache location.                                                         |

⚠️ **TF32 is disabled globally** for reproducibility.

---

## Metrics

`GET /metrics` exposes Prometheus metrics, including:

* `whisperx_requests_total`, `whisperx_request_duration_seconds` — per-endpoint request counts/latency.
* `whisperx_transcribe_speed_ratio` — realtime factor (`audio_seconds / wall_seconds`) of the
  `whisper.transcribe` call, labeled by `model` and by the **executor thread** that ran it. Use
  this to see whether raising `TRANSCRIBE_CONCURRENCY` / `MAX_THREADS` is actually improving
  per-thread throughput on your GPU, rather than just overall request latency.
* `whisperx_transcribe_thread_seconds_total`, `whisperx_audio_seconds_total` — cumulative time/audio
  processed per model. Wrap in `increase(whisperx_audio_seconds_total[$range])/60` for "minutes of
  audio processed in a period".
* `whisperx_process_stage_seconds{stage,model}` — per-request wall time of each pipeline stage
  (`transcribe`/`align`/`diarize`), so you can see each stage's **% share** of total processing
  time (e.g. `sum(rate(whisperx_process_stage_seconds_sum[5m])) by (stage)`).
* `whisperx_estimated_cost_usd_total{stage}` — estimated GPU cost, computed as
  `stage_wall_seconds * GPU_HOURLY_COST_USD / 3600` (set the `GPU_HOURLY_COST_USD` env var to your
  hourly GPU rate; defaults to `0`, i.e. disabled). Divide by processed audio-minutes for a
  $/audio-minute figure comparable to hosted transcription APIs.
* `whisperx_active_transcriptions`, `whisperx_model_pool_instances`, `whisperx_model_pool_available`,
  `whisperx_model_pool_target_size` — in-flight requests, current vs. idle vs. configured-max whisper
  pool sizes (`% capacity used = (instances-available)/target_size`).
* `whisperx_executor_active_tasks` / `whisperx_executor_max_workers` — saturation of the shared
  thread pool that backs **all** blocking work (model loads, audio decode, transcribe, align,
  diarize), i.e. the real throughput ceiling, not just the whisper pools.
* `whisperx_gpu_free_memory_mb` — free CUDA memory as of the last scrape.
* `whisperx_errors_total{stage}` — errors during audio loading vs. transcription/align/diarize.
* `whisperx_model_load_seconds{kind}`, `whisperx_model_load_events_total{kind}` — cold-start
  latency and frequency for `whisper`/`align`/`diarize` model loads (not counted when reused
  from a warm pool/cache).
* `whisperx_model_vram_usage_mb{kind,key}` — VRAM delta of the most recent load of each
  model/language/diarization-model key.
* `whisperx_model_evictions_total{kind}` — TTL-based unload events per model kind.
* `whisperx_pool_wait_seconds{model}` — time a request spent waiting to acquire a whisper
  instance from its pool (includes cold-load time when applicable) — the real "did I have to
  queue" latency, distinct from `whisperx_executor_active_tasks`.
* `whisperx_audio_duration_seconds`, `whisperx_upload_size_bytes` — distribution of input audio
  length and uploaded file size, useful for sizing `batch_size`/timeouts.
* `whisperx_num_speakers_detected` — distribution of distinct speakers found by diarisation.
* `whisperx_language_detected_total{language}` — count of requests by (autodetected or forced)
  transcription language.
* `whisperx_requests_total` now also carries a `model` label, so you can break down request
  volume/errors per Faster-Whisper model.

A ready-to-import Grafana dashboard covering all of these metrics is available in
[`grafana/whisperx-dashboard.json`](grafana/whisperx-dashboard.json) (see `grafana/README.md` for import steps).

---

## Endpoints

### `POST /v1/audio/transcriptions`

Upload audio, receive transcription (optionally aligned & diarised).

| Field               | Default         | Notes                                           |
| ------------------- | --------------- | ----------------------------------------------- |
| `file`              | —               | Binary audio (any FFmpeg-decodable format).     |
| `model`             | `large-v3`      | Faster-Whisper model id (see `/v1/models`).     |
| `language`          | *(auto)*        | Force language code; autodetect when omitted.   |
| `align`             | `false`         | Word-level alignment via Wav2Vec2.              |
| `diarize`           | `false`         | Speaker diarisation with `[SPK_n]` tags.        |
| `diarization_model` | *(env/default)* | Override diarisation backend model per request. |
| `response_format`   | `json`          | `json`, `text`, `srt`, `vtt`, `verbose_json`.   |
| `batch_size`        | `16`            | Whisper batch size.                             |
| `word_timestamps`   | `false`         | Include word timestamps (needs new FW build).   |
| `vad_filter`        | `false`         | Apply VAD before transcription.                 |
| `vad_threshold`     | `0.5`           | VAD probability threshold.                      |
| `min_speakers`      | `0`             | Lower bound for diarisation clustering.         |
| `max_speakers`      | `0`             | Upper bound for diarisation clustering.         |

Example:

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@sample.mp3 \
  -F model=medium \
  -F align=true \
  -F diarize=true \
  -F diarization_model=pyannote/speaker-diarization-3.1 \
  -F response_format=srt > out.srt
```

### `POST /v1/audio/translations`

Same contract as `/transcriptions`, but forces English output
(WhisperX “translate” task).

### `GET /v1/models`

* **Online** → every Faster-Whisper variant, `"downloaded": true/false`
* **Offline** → only variants physically present in cache.

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
[diarize_model_load_end]    model=pyannote/speaker-diarization-3.1  used=+512 MB freeVRAM=17938 MB
[align_end]   meeting.wav  freeVRAM=17320 MB Δ=4.35s
[summary]     meeting.wav Δ=12.3s audio=180.0s speed=14.6x
```

---

## Offline mode

Set `LOCAL_ONLY_MODELS=1` to disable downloads completely.

```bash
python - <<'PY'
import whisperx
whisperx.load_model("large-v3", device="cuda", local_files_only=False)
PY
```

---

## License

API code © 2025, MIT license.
Whisper / WhisperX / Faster-Whisper remain under their respective OSS licenses.
