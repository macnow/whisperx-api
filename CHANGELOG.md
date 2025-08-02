# Changelog

All notable changes to **WhisperX Transcription API** are documented in this file.

---

## [1.8.2] – 2025-08-02
### Changed
- **TF32 permanently disabled** (`torch.backends.cuda.matmul.allow_tf32 = False`, `torch.backends.cudnn.allow_tf32 = False`).

## [1.8.1] – 2025-08-02
### Added
- Switched to free VRAM logging (`freeVRAM`) with `used=±MB` deltas.
### Removed
- Manual cache verification (`ensure_local`) – offline errors now handled natively by Faster‑Whisper.

## [1.8.0] – 2025-08-02
### Added
- Accurate VRAM usage reporting at model load/unload.
- Correct unload label for diarization (`pipeline`).
- Re‑enable TF32 after Pyannote disables it.

## [1.7.4] – 2025-08-02
### Added
- Conditional kwargs builder to avoid `TypeError` on older Faster‑Whisper builds.

## [1.7.3] – 2025-08-02
### Added
- Full parameter set exposed (VAD, word timestamps, beam search, etc.).

## [1.7.2] – 2025-08-02
### Fixed
- `threads` argument forwarded only when `FASTER_WHISPER_THREADS > 0`.

## [1.7.1] – 2025-08-02
### Added
- Complete Faster‑Whisper model map (including `distil-*`, `turbo`).
- TTL‑based cache sweeping with unload logs.

## [1.5.3] – 2025-08-02
### Added
- Concurrency via `ThreadPoolExecutor` + per‑model locks.
- Detailed step logging & speaker tags `[SPK_n]`.

## [1.3.2] – 2025-08-02
### Added
- First public **GPU‑only** API wrapper.
- OpenAI‑compatible routes and multiple output formats.

## [1.3.1] – 2025-08-02
### Changed
- Unified step log labels (`[transcribe_start]`, `[transcribe_end]`, etc.).
- Renamed load/unload labels to `*_model_load_start/end`.

## [1.3.0] – 2025-08-02
### Added
- Alignment & diarization toggles with model selection.
### Changed
- Default model fallback to `large-v3` when `model` not provided.
### Removed
- Deprecated Whisper arguments (`temperature`, `initial_prompt`).

## [1.0.0] – 2025-08-02
### Added
- Initial FastAPI proof‑of‑concept wrapping WhisperX `transcribe`.
- Basic endpoints with JSON and plain‑text responses.

---

> **Note** Version numbers correspond to the `app.py` header comment in each iteration.

