# Copilot instructions for whisperx-api

Single-file FastAPI service (`app.py`, ~950 lines) wrapping WhisperX for an
OpenAI-compatible transcription/translation HTTP API. GPU-only (asserts
`torch.cuda.is_available()` at import time), designed to run inside the
Docker image built from `Dockerfile`.

## Running / testing

- No local dev server target — the app requires a CUDA GPU and WhisperX to
  actually run (`torch.cuda.is_available()` assertion at import time), so it
  is normally only run via the Docker image (`docker build . && docker run
  --gpus all -p 8000:8000 ...`, see README "Quick start").
- **Unit tests** (`tests/test_ttlcache.py`) run on plain CPU/no-GPU machines
  by stubbing out all heavy/CUDA-only imports (`whisperx`, `srt`, `webvtt`,
  `fastapi`, `huggingface_hub`, `torch`) with fake modules *before*
  `import app`, so `app.py` module-level code (including the CUDA assertion)
  never actually executes against real libraries. When adding tests, follow
  this same stubbing pattern at the top of the test file rather than trying
  to install real GPU dependencies.
- Run all tests: `python -m pytest tests/ -q`
- Run a single test: `python -m pytest tests/test_ttlcache.py::test_sweep_removes_expired -q`
- Requires Python **3.13** (matches the `Dockerfile` builder stage and the
  `ghcr.io/jim60105/whisperx:no_model` base image's venv); the
  `int | None` style annotations used throughout `app.py` fail to import on
  Python 3.9/3.10.
- If a test run needs packages not present locally (`pytest`,
  `prometheus_client`, etc.), install them with `pip install <pkg>` — these
  are lightweight, CPU-only packages, unlike `torch`/`whisperx` which are
  stubbed instead of installed.

## Architecture

- **Everything lives in `app.py`.** There's no package structure — routes,
  model loading, caching, and formatting are all in one module. Read the
  whole file (or use `grep -n "^def \|^async def \|^class "`) before making
  structural changes; there's no separate "service layer" to find.
- **Two-level caching by design, don't conflate them:**
  - `WhisperPool` (per `(model_id, ASROptions)` key, stored in
    `WHISPER_POOLS`) holds a *pool* of `TRANSCRIBE_CONCURRENCY` warm
    Faster-Whisper instances so multiple requests for the same model can
    transcribe truly in parallel on one GPU. Acquire/release via
    `pool.acquire()` (async context manager over an `asyncio.Queue`).
  - `TTLCache` (`A_CACHE` for alignment models, `D_CACHE` for diarization
    pipelines) is a plain dict keyed by language/model-name; only ever holds
    one instance per key.
  - Both are swept every 60s by the daemon thread started via
    `threading.Thread(target=_sweep, daemon=True).start()` at import time
    (`_sweep_pools()` for pools, `TTLCache.sweep()` for align/diarize), using
    `MODEL_TTL_SEC`. Whisper pools are only evicted when **fully idle**
    (`pool.is_idle()`), never mid-use.
- **All blocking/CPU work goes through `run_sync()`**, which offloads to the
  shared `EXECUTOR` (`ThreadPoolExecutor(max_workers=MAX_THREADS)`) so the
  asyncio event loop never blocks — model loads, `whisperx.load_audio`,
  `whisper.transcribe`, alignment, and diarization all use this. Locking
  uses `asyncio.Lock` (not `threading.Lock`) for first-load coordination
  across concurrent requests, precisely so the event loop stays responsive
  while a lock is held.
- **Request pipeline** (`process()` in `app.py`) is linear:
  load audio → acquire a whisper instance from its pool → transcribe →
  optionally align (`load_align` + `whisperx.align`) → optionally diarize
  (`load_diar` + `whisperx.assign_word_speakers`) → `standardize()` the
  result → format via `_fmt()` for the requested `response_format`
  (`json`/`text`/`srt`/`vtt`/`verbose_json`).
- **Model IDs are indirected**: the `_MODELS` dict maps short public IDs
  (e.g. `"large-v3"`, `"turbo"`) to actual HF repo IDs (e.g.
  `Systran/faster-whisper-large-v3`). The `ModelId` enum (used for FastAPI
  request validation) must stay in sync with `_MODELS`'s keys.
- **Offline mode** (`LOCAL_ONLY_MODELS=1` → `OFFLINE=True`) changes behavior
  in several places at once: sets `HF_HUB_OFFLINE=1`, makes `/v1/models`
  filter to only cached models (`is_cached()`/`local_sizes()` scan HF cache
  dirs on disk), and makes model loads raise HTTP 400 instead of downloading
  (`LocalEntryNotFoundError` → `HTTPException(400, ...)`). Preserve all three
  when touching this logic.
- **Prometheus metrics** (`GET /metrics`) are recorded inline in the request
  path, not via middleware — `_track_request()` (async context manager)
  wraps each endpoint body for request counts/latency, `process()` tracks
  `ACTIVE_TRANSCRIPTIONS`/`ERRORS_TOTAL`, and `_transcribe_with_metrics()`
  runs *inside* the executor thread (via `run_sync`) so
  `threading.current_thread().name` correctly labels which worker thread
  did the transcription — this is intentional per-thread throughput
  visibility, not per-request latency. Keep new metrics consistent with this
  pattern (record where the work happens, not from a wrapper afterward).
  `_record_stage(stage, model, elapsed)` is the single place that records
  both `whisperx_process_stage_seconds` (transcribe/align/diarize share) and
  `whisperx_estimated_cost_usd_total` (elapsed × `GPU_HOURLY_COST_USD`/3600,
  a no-op when that env var is `0`) — call it from `process()` rather than
  duplicating the cost math elsewhere. `run_sync()` increments/decrements
  `EXECUTOR_ACTIVE_TASKS` around every blocking call (model loads, audio
  decode, transcribe, align, diarize all share one `ThreadPoolExecutor`), so
  `whisperx_executor_active_tasks / whisperx_executor_max_workers` reflects
  *overall* saturation, not just transcription. `_load_start()`/`_load_end()`
  (used by `WhisperPool.ensure_loaded()`, `load_align()`, `load_diar()`) also
  record `whisperx_model_load_seconds`/`whisperx_model_load_events_total`
  (cold-start duration/count) and `whisperx_model_vram_usage_mb` per kind —
  `_load_start()` returns a `time.perf_counter()` start value that must be
  threaded through to the matching `_load_end()` call. `TTLCache.sweep()` and
  the whisper-pool idle sweeper (`_sweep_pools()`) both increment
  `whisperx_model_evictions_total{kind}`. `WhisperPool.acquire()` records
  `whisperx_pool_wait_seconds` for the time spent waiting for an instance
  (ensure_loaded + queue wait combined — the real "did this request have to
  queue" latency). `process()` also observes `whisperx_audio_duration_seconds`,
  `whisperx_num_speakers_detected`, and increments
  `whisperx_language_detected_total`; the endpoints observe
  `whisperx_upload_size_bytes` right after reading the uploaded file.

## Conventions

- Config is env-var driven (see README "Environment variables" table) and
  read once at module import time into module-level constants (`OFFLINE`,
  `TTL_SEC`, `TRANSCRIBE_CONCURRENCY`, `ASR_CONFIG`, etc.) — don't re-read
  `os.getenv` deep inside request-handling code.
- `ASR_CONFIG` (per-model beam size/patience/etc.) comes from
  `ASR_CONFIG_JSON` env var if set, else `DEFAULT_ASR_CONFIG`; it's looked up
  per-request by model name in `process()` and turned into an `ASROptions`
  dataclass, which doubles as (part of) the whisper pool cache key — so
  changing `ASROptions` fields changes pool identity/cache-key semantics.
- Logging is structured via small helpers (`_log`, `_load_start`,
  `_load_end`) that always include `freeVRAM=%d MB`, follow the
  `[tag] filename ...` / `[label_model_load_start|end]` conventions already
  used — match this format for new log lines rather than ad hoc
  `logging.info` calls, since these lines are relied on for operational
  debugging (see README "Logging example").
- Numeric form fields that can legitimately be `0` (e.g. `vad_threshold`)
  must be checked with `is not None`, not truthiness — see the comment in
  `build_transcribe_kwargs()` for a past bug from this exact mistake.
- New response formats/fields should stay OpenAI-compatible where possible
  (see `_usage()`, `_fmt()`) — the `usage`/`duration` fields intentionally
  mirror OpenAI's `gpt-4o-transcribe`/`whisper-1` shapes.
- Update `README.md`'s environment-variable table, endpoint field table,
  and/or "What's new" changelog section (and `CHANGELOG.md`) when adding
  env vars, request fields, or behavior changes — they're the source of
  truth for operators, not code comments.
