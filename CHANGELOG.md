# Changelog

All notable changes to **WhisperX Transcription API** are documented in this file.

---

## [1.14.0] – 2026-09-10
### Added
- Response/segment shape metrics:
  - `whisperx_response_size_bytes{response_format}` – size of the formatted
    response body.
  - `whisperx_segments_count`, `whisperx_words_count` – size of the final
    transcription result per request.
  - `whisperx_upload_seconds` – wall time spent reading the uploaded file
    (separates client-upload time from processing time in overall latency).
- Alignment/diarization quality metrics:
  - `whisperx_align_word_coverage_ratio` – fraction of words that received a
    word-level timestamp from alignment.
  - `whisperx_unassigned_speaker_ratio` – fraction of segments diarization
    could not assign a speaker to.
- GPU/process utilization metrics (soft dependencies, degrade gracefully if
  unavailable):
  - `whisperx_gpu_utilization_percent`, `whisperx_gpu_temperature_celsius`,
    `whisperx_gpu_power_watts` via NVML (`nvidia-ml-py`).
  - `whisperx_process_cpu_percent`, `whisperx_process_rss_mb` via `psutil`.
- New optional dependencies `nvidia-ml-py` and `psutil` added to
  `requirements.txt` and the Dockerfile wheel-builder stage.

---

## [1.13.0] – 2026-09-10
### Added
- Capacity, cost & stage-breakdown metrics:
  - `whisperx_process_stage_seconds{stage,model}` – per-stage
    (`transcribe`/`align`/`diarize`) wall time, for time-share breakdown.
  - `whisperx_estimated_cost_usd_total{stage}` and
    `whisperx_gpu_hourly_cost_usd` – estimated GPU cost from
    `stage_wall_seconds * GPU_HOURLY_COST_USD / 3600` (new `GPU_HOURLY_COST_USD`
    env var, defaults to `0` = disabled).
  - `whisperx_executor_active_tasks` / `whisperx_executor_max_workers` –
    shared-executor saturation (the real throughput ceiling across all
    blocking work).
  - `whisperx_model_pool_target_size` – configured max capacity per whisper
    pool, for `% capacity used` calculations.
- Cold-start & queueing visibility:
  - `whisperx_model_load_seconds{kind}` / `whisperx_model_load_events_total{kind}` –
    duration and count of actual model (re)loads per kind.
  - `whisperx_model_vram_usage_mb{kind,key}` – VRAM delta of the most recent
    load per model/language/diarization key.
  - `whisperx_model_evictions_total{kind}` – TTL-based unload events.
  - `whisperx_pool_wait_seconds{model}` – time spent waiting to acquire a
    whisper instance from its pool.
- Request/audio shape & quality signals:
  - `whisperx_audio_duration_seconds`, `whisperx_upload_size_bytes` –
    distribution of input duration and upload size.
  - `whisperx_num_speakers_detected` – distribution of distinct diarized
    speakers per request.
  - `whisperx_language_detected_total{language}` – count by transcription
    language.
  - `model` label added to `whisperx_requests_total`.
- Grafana dashboard (`grafana/whisperx-dashboard.json`) extended with a new
  "Capacity, cost & stage breakdown" row (audio minutes/period, estimated
  cost, cost per audio-minute, executor saturation %, stage time-share,
  whisper pool utilization %).

---

## [1.12.1] – 2026-09-09
### Fixed
- Docker build was broken because the upstream base image
  `ghcr.io/jim60105/whisperx:no_model` bumped its Python version from 3.11
  to 3.13 (its venv now lives at `/venv/lib/python3.13/site-packages`), so
  our `cp -r /wheels/* /venv/lib/python3.11/site-packages/` step failed with
  "No such file or directory". Updated the wheel-builder stage to
  `python:3.13-slim` and all `python3.11` paths/env vars to `python3.13` to
  match the new base image.

---

## [1.12.0] – 2026-09-09
### Added
- `GET /metrics` Prometheus endpoint (`prometheus-client`):
  - `whisperx_requests_total` / `whisperx_request_duration_seconds` per endpoint.
  - `whisperx_transcribe_speed_ratio` – realtime factor (`audio_s/wall_s`) of
    `whisper.transcribe`, labeled by `model` and by the executor thread that
    ran it, to observe per-thread throughput under `TRANSCRIBE_CONCURRENCY`.
  - `whisperx_transcribe_thread_seconds_total`, `whisperx_audio_seconds_total`.
  - `whisperx_active_transcriptions`, `whisperx_model_pool_instances`,
    `whisperx_model_pool_available`, `whisperx_gpu_free_memory_mb`.
  - `whisperx_errors_total{stage}`.
- `.github/copilot-instructions.md` documenting architecture/conventions.

---

## [1.11.0] – 2026-04-22
### Added
- OpenAI-compatible `usage` object on JSON responses
  (`{"type": "duration", "seconds": <audio_seconds>}`), matching the
  *duration* variant returned by OpenAI's `gpt-4o-transcribe` endpoint.
- `verbose_json` now also exposes top-level `duration` (audio seconds),
  matching OpenAI `whisper-1`.

---

## [1.10.0] – 2026-04-22
### Added
- New `TRANSCRIBE_CONCURRENCY` environment variable (default `1`). Each
  `(model, asr_options)` key is now backed by a pool of N warm WhisperX
  instances, enabling true parallel transcriptions on a single large GPU.
- Pool TTL eviction: a whole pool is unloaded once every instance has been
  idle for `MODEL_TTL_SEC`.

### Fixed
- **Server hang under concurrent requests.** The previous implementation
  acquired a `threading.Lock` synchronously from inside the async handler.
  When one request was transcribing (holding the lock in a worker thread),
  any second request to the same model blocked the asyncio event loop for
  the entire transcription, freezing the whole server (including health
  checks and `/v1/models`).
- All model loading (whisper / align / diarize) and `whisperx.load_audio`
  are now offloaded to the worker thread pool. Loading is serialised per
  key with `asyncio.Lock`, never blocking the event loop.

### Changed
- `load_align` and `load_diar` are now `async`.
- Removed the per-instance `threading.Lock` (`LOCKS` dict) and the global
  `W_CACHE` (replaced by `WHISPER_POOLS`).

---

## [1.9.0] – 2025-09-08
### Changed
- Minor release: promote 1.8.9 changes under 1.9.0. No additional changes.

---

## [1.8.9] – 2025-09-08
### Added
- Forward explicit `language` argument into `whisper.transcribe` so client-forced
  language is honored.

### Changed
- Serialized Whisper model loading to avoid concurrent initializations.

### Fixed
- Corrected TTL cache `get` semantics.
- Handled zero VAD threshold correctly.
- Fixed response format detection.

---

## [1.8.8] - 2025-08-24
### Added
- **Per-request diarisation model selection** – new form field  
  `diarization_model` allows clients to override the diarisation backend
  (e.g. `pyannote/speaker-diarization-3.1`) on each request.  
  Falls back to `$DIARIZATION_MODEL` or the default *pyannote/3.1* when omitted.
- **Cache-per-model diarisation** – diarisation pipelines are now cached
  separately per model name with TTL-based eviction.

### Changed
- Diarisation cache & unload logging now use `model=<name>` instead of the
  generic `pipeline` label, improving clarity in logs.
- `/v1/audio/transcriptions` and `/v1/audio/translations` endpoints updated to
  forward the requested diarisation model through the processing pipeline.

---

## [1.8.5] – 2025-08-04
### Fixed
- **Offline catalogue** – `/v1/models` now scans *all* plausible HuggingFace
  cache roots (`HF_HOME`, `XDG_CACHE_HOME`, `~/.cache`, `/root/.cache`,
  `/.cache`) so every locally-downloaded model is listed.
- Graceful 400 response preserved for requests that ask for a model not
  present in the cache when `LOCAL_ONLY_MODELS=1`.

## [1.8.3] – 2025-08-03
### Added
- **Model catalogue overhaul** – `GET /v1/models` now  
  * lists every Faster-Whisper variant in online mode  
  * lists only locally-cached variants when `LOCAL_ONLY_MODELS=1`  
  * adds `"downloaded": true / false` flag to each entry.
### Removed
- Dropped the “-en” suffix from the header comment (no functional impact).

## [1.8.2] – 2025-08-02
### Changed
- TF32 permanently disabled (`torch.backends.cuda.*.allow_tf32 = False`).

## [1.8.1] – 2025-08-02
### Added
- Switched to free-VRAM logging with `used = ±MB` deltas.
### Removed
- Manual cache check (`ensure_local`) – offline errors now handled by Faster-Whisper itself.

## [1.8.0] – 2025-08-02
### Added
- Accurate VRAM usage logging at model load / unload.
- Correct diarisation unload label (`pipeline`).
- Re-enable TF32 after Pyannote disables it.

## [1.7.4] – 2025-08-02
### Added
- Conditional kwarg builder to avoid *TypeError* on older Faster-Whisper builds.

## [1.7.3] – 2025-08-02
### Added
- Full parameter set exposed (VAD, word-timestamps, beam search, etc.).

## [1.7.2] – 2025-08-02
### Fixed
- `threads` argument forwarded only when `FASTER_WHISPER_THREADS > 0`.

## [1.7.1] – 2025-08-02
### Added
- Complete Faster-Whisper model map (including *distil-\** and *turbo* lines).
- TTL-based cache sweeping with unload logging.

## [1.5.3] – 2025-08-02
### Added
- Concurrency via `ThreadPoolExecutor` + per-model locks.
- Detailed step logs & speaker tags `[SPK_n]`.

## [1.3.2] – 2025-08-02
### Added
- First public GPU-only API wrapper.
- OpenAI-compatible routes and multi-format outputs.

## [1.3.1] – 2025-08-02
### Changed
- Unified log labels (`[transcribe_start]`, `[transcribe_end]`, …).

## [1.3.0] – 2025-08-02
### Added
- Alignment & diarisation toggles with model selection.
### Changed
- Default model fallback to **large-v3** when the client omits `model`.

## [1.0.0] – 2025-08-02
### Added
- Initial FastAPI proof-of-concept wrapping WhisperX `transcribe`.
