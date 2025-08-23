# Changelog

All notable changes to **WhisperX Transcription API** are documented in this file.

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
