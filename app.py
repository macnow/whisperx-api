"""
WhisperX Transcription API · v1.8.8
(OpenAI-compatible)

•  GPU-only WhisperX wrapper with optional alignment & diarisation
•  One GPU instance per model, thread-safe, TTL-based eviction
•  Detailed logging (freeVRAM, used=±MB, step timings)
•  `/v1/models`
      – online:  every Faster-Whisper variant + "downloaded" flag
      – offline: only variants that truly exist on disk
•  Offline mode now returns HTTP 400 when the requested model isn’t cached
•  TF32 permanently disabled (reproducibility)
"""

# ───────── CUDA / TF32 OFF ─────────
import json
import os, time, logging, threading, tempfile, gc, torch, asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, Any

import whisperx, srt, webvtt
from fastapi import Depends, FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from huggingface_hub.errors import LocalEntryNotFoundError
from urllib.parse import quote_plus

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)8s  %(message)s")

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert torch.cuda.is_available(), "CUDA GPU required"

DEVICE, COMPUTE_TYPE, BATCH_SIZE = "cuda", "float16", 16
EXECUTOR   = ThreadPoolExecutor(max_workers=int(os.getenv("MAX_THREADS", "4")))
FW_THREADS = int(os.getenv("FASTER_WHISPER_THREADS", "0"))  # 0 ⇒ not forwarded

_MB = 1024 * 1024
def free_mb() -> int:
    return torch.cuda.mem_get_info()[0] // _MB

async def run_sync(func, *a, **kw):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(EXECUTOR, lambda: func(*a, **kw))

# ───────── Faster-Whisper catalogue ─────────
_MODELS = {
    "tiny.en":          "Systran/faster-whisper-tiny.en",
    "tiny":             "Systran/faster-whisper-tiny",
    "base.en":          "Systran/faster-whisper-base.en",
    "base":             "Systran/faster-whisper-base",
    "small.en":         "Systran/faster-whisper-small.en",
    "small":            "Systran/faster-whisper-small",
    "medium.en":        "Systran/faster-whisper-medium.en",
    "medium":           "Systran/faster-whisper-medium",
    "large-v1":         "Systran/faster-whisper-large-v1",
    "large-v2":         "Systran/faster-whisper-large-v2",
    "large-v3":         "Systran/faster-whisper-large-v3",
    "large":            "Systran/faster-whisper-large-v3",
    "distil-large-v2":  "Systran/faster-distil-whisper-large-v2",
    "distil-medium.en": "Systran/faster-distil-whisper-medium.en",
    "distil-small.en":  "Systran/faster-distil-whisper-small.en",
    "distil-large-v3":  "Systran/faster-distil-whisper-large-v3",
    "large-v3-turbo":   "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "turbo":            "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}

class ModelId(str, Enum):
    TINY_EN = "tiny.en"
    TINY = "tiny"
    BASE_EN = "base.en"
    BASE = "base"
    SMALL_EN = "small.en"
    SMALL = "small"
    MEDIUM_EN = "medium.en"
    MEDIUM = "medium"
    LARGE_V1 = "large-v1"
    LARGE_V2 = "large-v2"
    LARGE_V3 = "large-v3"
    LARGE = "large"
    DISTIL_LARGE_V2 = "distil-large-v2"
    DISTIL_MEDIUM_EN = "distil-medium.en"
    DISTIL_SMALL_EN = "distil-small.en"
    DISTIL_LARGE_V3 = "distil-large-v3"
    LARGE_V3_TURBO = "large-v3-turbo"
    TURBO = "turbo"

class ResponseFormat(str, Enum):
    JSON = "json"
    TEXT = "text"
    SRT = "srt"
    VTT = "vtt"
    VERBOSE_JSON = "verbose_json"

@dataclass(frozen=True)
class ASROptions:
    beam_size: int | None
    patience: float | None
    length_penalty: float | None
    best_of: int | None

# ───────── Cache-scan helpers ─────────
def _cache_roots() -> list[Path]:
    """Return every plausible HF cache directory that exists."""
    roots: set[Path] = set()
    if "HF_HOME" in os.environ:
        roots.add(Path(os.environ["HF_HOME"]))
    if "XDG_CACHE_HOME" in os.environ:
        roots.add(Path(os.environ["XDG_CACHE_HOME"]) / "huggingface" / "hub")
    roots.update({
        Path.home() / ".cache/huggingface/hub",
        Path("/root/.cache/huggingface/hub"),
        Path("/.cache/huggingface/hub"),
    })
    return [r for r in roots if r.exists()]

def local_sizes() -> list[str]:
    """Return model IDs whose snapshot exists under any cache root."""
    cached: list[str] = []
    for mid, repo in _MODELS.items():
        org, name = repo.split("/", 1)
        fragment = f"models--{org}--{quote_plus(name)}"
        if any((root / fragment).exists() for root in _cache_roots()):
            cached.append(mid)
    return cached

# ───────── Runtime flags ─────────
OFFLINE  = os.getenv("LOCAL_ONLY_MODELS", "0") == "1"
TTL_SEC  = int(os.getenv("MODEL_TTL_SEC", "600"))
HF_TOKEN = os.getenv("HF_TOKEN", "").strip() or None
DIARIZATION_MODEL = os.getenv("DIARIZATION_MODEL", "").strip() or None

if OFFLINE:
    os.environ["HF_HUB_OFFLINE"] = "1"

# Warmup flags
WARMUP_MODEL = os.getenv("WARMUP_MODEL", "large-v3")
WARMUP_ALIGN_LANGS = [lang.strip() for lang in os.getenv("WARMUP_ALIGN_LANGS", "en").split(",")]
WARMUP_DIARIZE = os.getenv("WARMUP_DIARIZE", "0") == "1"

# ASR configuration
DEFAULT_ASR_CONFIG = {
    "large-v3": {
        "beam_size": 5,
        "patience": 1.0,
        "length_penalty": 1.0,
        "best_of": 5
    }
}
ASR_CONFIG_JSON = os.getenv("ASR_CONFIG_JSON")
ASR_CONFIG = json.loads(ASR_CONFIG_JSON) if ASR_CONFIG_JSON else DEFAULT_ASR_CONFIG

app = FastAPI(title="WhisperX Transcription API", version="1.8.8")

@app.on_event("startup")
async def on_startup():
    warmup()

# ───────── TTL caches ─────────
class TTLCache(dict):
    """Dict with TTL eviction & VRAM-usage logging."""
    def __init__(self, label: str): super().__init__(); self.label = label
    def get(self, k):
        v = super().get(k)
        if v is None:
            return None
        super().__setitem__(k, (v[0], time.time()))
        return v[0]
    def put(self, k, obj): super().__setitem__(k, (obj, time.time()))
    def sweep(self, ttl: int):
        now = time.time()
        for k, (_, ts) in list(self.items()):
            if now - ts > ttl:
                del self[k]; torch.cuda.empty_cache()
                key = {"whisper": "model", "align": "lang", "diarize": "model"}[self.label]
                logging.info("[%s_model_unload]  %s=%s  freeVRAM=%d MB",
                             self.label, key, k, free_mb())

W_CACHE, A_CACHE, D_CACHE = TTLCache("whisper"), TTLCache("align"), TTLCache("diarize")
LOCKS: Dict[str, threading.Lock] = {}

# ───────── Logging helpers ─────────
def _log(tag: str, fname: str, msg: str = "", *a):
    logging.info("[%s] %s  freeVRAM=%d MB " + msg,
                 tag, Path(fname).name, free_mb(), *a)

def _load_start(lbl: str, key: str):
    logging.info("[%s_model_load_start]  %s=%s  freeVRAM=%d MB",
                 lbl, "model" if lbl in ("whisper", "diarize") else "lang", key, free_mb())

def _load_end(lbl: str, key: str, before: int):
    delta = before - free_mb()
    logging.info("[%s_model_load_end]    %s=%s  used=%+d MB  freeVRAM=%d MB",
                 lbl, "model" if lbl in ("whisper", "diarize") else "lang",
                 key, delta, free_mb())

# ───────── Loaders ─────────
def load_whisper(model_id: str, asr_options: ASROptions):
    """Return (pipeline, lock); raise 400 offline if model isn’t cached."""
    cache_key = (model_id, asr_options)
    lock_key = str(cache_key)
    lock = LOCKS.setdefault(lock_key, threading.Lock())
    with lock:
        existing = W_CACHE.get(cache_key)
        if existing:
            return existing, lock

        log_key = f"{model_id} asr_opts={asr_options}"
        before = free_mb(); _load_start("whisper", log_key)

        options_dict = {
            k: v for k, v in asr_options.__dict__.items()
            if v is not None and v != 0 and v != 0.0
        }

        load_kw = dict(
            compute_type=COMPUTE_TYPE,
            local_files_only=OFFLINE,
            device=DEVICE,
            asr_options=options_dict,
        )

        if FW_THREADS:
            load_kw["threads"] = FW_THREADS

        try:
            model = whisperx.load_model(model_id, **load_kw)
        except LocalEntryNotFoundError:
            raise HTTPException(
                status_code=400,
                detail=(f"Model '{model_id}' is not cached locally and "
                        "LOCAL_ONLY_MODELS=1 prevents downloading.")
            ) from None

        W_CACHE.put(cache_key, model); _load_end("whisper", log_key, before)
    return model, lock

def load_align(lang: str):
    key = lang or "default"
    pair = A_CACHE.get(key)
    if pair: return pair
    before = free_mb(); _load_start("align", key)
    model, meta = whisperx.load_align_model(language_code=lang or "en", device=DEVICE)
    A_CACHE.put(key, (model, meta)); _load_end("align", key, before)
    return model, meta

def load_diar(model_name: str | None = None):
    """Load diarization pipeline; cache per model name."""
    diar_model_name = model_name or DIARIZATION_MODEL or "pyannote/speaker-diarization-3.1"
    pip = D_CACHE.get(diar_model_name)
    if pip: return pip
    before = free_mb(); _load_start("diarize", diar_model_name)
    try:
        from whisperx.diarize import DiarizationPipeline as _DP
    except ImportError:
        from whisperx.diarization import DiarizationPipeline as _DP
    pip = _DP(model_name=diar_model_name, use_auth_token=HF_TOKEN, device=DEVICE)
    D_CACHE.put(diar_model_name, pip); _load_end("diarize", diar_model_name, before)
    return pip

# ───────── Warmup ─────────
def warmup():
    """Pre-loads default models for faster first-request processing."""
    logging.info("Warming up...")
    if WARMUP_MODEL not in _MODELS:
        logging.warning("WARMUP_MODEL '%s' not found in _MODELS, skipping.", WARMUP_MODEL)
        return

    asr_config = ASR_CONFIG.get(WARMUP_MODEL, {})
    asr_options = ASROptions(
        beam_size=asr_config.get("beam_size"),
        patience=asr_config.get("patience"),
        length_penalty=asr_config.get("length_penalty"),
        best_of=asr_config.get("best_of"),
    )
    load_whisper(WARMUP_MODEL, asr_options)

    for lang in WARMUP_ALIGN_LANGS:
        if lang: load_align(lang)
    if WARMUP_DIARIZE:
        load_diar()
    logging.info("Warmup complete.")

# ───────── /v1/models ─────────
def is_cached(mid: str) -> bool:
    # Note: quick heuristic; W_CACHE keys include ASR options so this isn't exhaustive.
    return mid in local_sizes() or any(isinstance(k, tuple) and k[0] == mid for k in W_CACHE.keys())

@app.get("/v1/models")
def models():
    ids = (m for m in _MODELS) if not OFFLINE else (m for m in _MODELS if is_cached(m))
    return {"data": [
        {"id": m, "object": "model", "created": 0, "owned_by": "you",
         "downloaded": is_cached(m)}
        for m in ids
    ]}

# ───────── Text helpers ─────────
def split_segments_by_speaker(segments: list[dict]) -> list[dict]:
    """Splits segments into smaller segments whenever the speaker changes at a sentence boundary."""
    if not segments or "words" not in segments[0] or not segments[0]["words"]:
        return segments

    new_segments = []
    for segment in segments:
        if "words" not in segment or not segment["words"]:
            new_segments.append(segment)
            continue

        current_speaker = segment["words"][0].get("speaker")
        current_words = []
        for i, word in enumerate(segment["words"]):
            speaker = word.get("speaker")

            is_new_sentence = False
            if i > 0:
                prev_word = segment["words"][i-1]["word"]
                if prev_word.endswith(('.', '?', '!')):
                    is_new_sentence = True
            else:
                is_new_sentence = True

            if speaker != current_speaker and is_new_sentence and current_words:
                new_segments.append({
                    "start": current_words[0]["start"],
                    "end": current_words[-1]["end"],
                    "text": " ".join(w["word"] for w in current_words),
                    "speaker": current_speaker,
                    "words": current_words,
                })
                current_words = []

            current_speaker = speaker
            current_words.append(word)

        if current_words:
            new_segments.append({
                "start": current_words[0]["start"],
                "end": current_words[-1]["end"],
                "text": " ".join(w["word"] for w in current_words),
                "speaker": current_speaker,
                "words": current_words,
            })

    return new_segments

def _tagged(seg):
    out, cur = [], None
    for s in seg:
        if s.get("speaker") != cur:
            out.append(f"[{s['speaker']}]" if s.get("speaker") else "")
            cur = s.get("speaker")
        out.append(s["text"].strip())
    return " ".join(out).strip()

def standardize(raw, spk=False):
    if isinstance(raw, dict) and "segments" in raw:
        seg = raw["segments"]
        if spk:
            seg = split_segments_by_speaker(seg)
        txt = _tagged(seg) if spk else raw.get("text") or " ".join(s["text"].strip() for s in seg)
        return {"text": txt, "segments": seg, "language": raw.get("language")}
    if isinstance(raw, list):
        if spk:
            raw = split_segments_by_speaker(raw)
        txt = _tagged(raw) if spk else " ".join(s["text"].strip() for s in raw)
        return {"text": txt, "segments": raw, "language": None}
    return {"text": "", "segments": [], "language": None}

def srt_from(seg):
    return srt.compose([srt.Subtitle(i+1,
                                     timedelta(seconds=s["start"]),
                                     timedelta(seconds=s["end"]),
                                     (f"[{s['speaker']}] " if s.get("speaker") else "") +
                                     s["text"].strip())
                        for i, s in enumerate(seg)])

def vtt_from(seg):
    v = webvtt.WebVTT()
    for s in seg:
        v.captions.append(webvtt.Caption(
            start=webvtt.Caption.time_to_webvtt(s["start"]),
            end  = webvtt.Caption.time_to_webvtt(s["end"]),
            text=(f"[{s['speaker']}] " if s.get("speaker") else "") + s["text"].strip()))
    return v.content

# ───────── Sweeper thread ─────────
def _sweep():
    while True:
        time.sleep(60)
        W_CACHE.sweep(TTL_SEC); A_CACHE.sweep(TTL_SEC); D_CACHE.sweep(TTL_SEC); gc.collect()
threading.Thread(target=_sweep, daemon=True).start()

# ───────── Pipeline ─────────
async def process(path, model, lang, do_align, do_diar, trans_kw, diar_kw, diar_model_name: str | None):
    fname = Path(path).name
    try:
        wav = whisperx.load_audio(path)
    except Exception as e:
        logging.error("Error loading audio file %s: %s", fname, e, exc_info=True)
        raise HTTPException(status_code=400, detail=f"Error loading audio file: {e}")

    audio_sec = len(wav) / 16000
    t0 = time.perf_counter()

    try:
        # transcription
        _log("transcribe_start", fname, "model=%s", model)
        asr_config = ASR_CONFIG.get(model, {})
        asr_options = ASROptions(
            beam_size=asr_config.get("beam_size"),
            patience=asr_config.get("patience"),
            length_penalty=asr_config.get("length_penalty"),
            best_of=asr_config.get("best_of"),
        )
        whisper, lock = load_whisper(model, asr_options)
        await run_sync(lock.acquire)
        try:
            raw = await run_sync(whisper.transcribe, wav, **trans_kw)
        finally:
            lock.release()
        res = standardize(raw)
        _log("transcribe_end", fname, "Δ=%.2fs", time.perf_counter() - t0)

        # alignment
        if do_align:
            t = time.perf_counter(); lang_used = res.get("language") or lang
            _log("align_start", fname, "lang=%s", lang_used)
            model_a, meta = load_align(lang_used)
            res = standardize(await run_sync(
                whisperx.align, res["segments"], model_a, meta, wav, DEVICE))
            _log("align_end", fname, "Δ=%.2fs", time.perf_counter() - t)

        # diarisation
        if do_diar:
            t = time.perf_counter(); _log("diarize_start", fname)
            diar_pipe = load_diar(diar_model_name)
            spk = await run_sync(diar_pipe, wav, **diar_kw)
            res = standardize(await run_sync(
                whisperx.assign_word_speakers, spk, res), spk=True)
            _log("diarize_end", fname, "Δ=%.2fs", time.perf_counter() - t)

    except Exception as e:
        logging.error("Error during processing of %s: %s", fname, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during processing: {e}")

    wall = time.perf_counter() - t0
    logging.info("[summary] %s Δ=%.2fs audio=%.2fs speed=%.1fx",
                 fname, wall, audio_sec, audio_sec / wall if wall else 0)
    return res

# ───────── KW builders ─────────
def build_transcribe_kwargs(batch, word_ts, vad, vad_thr):
    kw = {"batch_size": batch or BATCH_SIZE}
    if vad:
        kw["vad_filter"] = True
        # `vad_threshold` may legitimately be ``0``.  The previous truthy check
        # skipped applying the user-provided threshold when it was ``0`` and
        # fell back to the default.  Explicitly guard against ``None`` instead
        # so zero is forwarded correctly.
        if vad_thr is not None:
            kw["vad_parameters"] = {"threshold": vad_thr}
    if word_ts: kw["word_timestamps"] = True
    return kw

def build_diar_kwargs(min_spk, max_spk):
    d = {}
    if min_spk: d["min_speakers"] = min_spk
    if max_spk: d["max_speakers"] = max_spk
    return d

def _fmt(res, fmt):
    """Format transcription results according to the requested response format."""
    if isinstance(fmt, ResponseFormat):
        fmt = fmt.value
    text, seg = res["text"], res["segments"]
    if fmt == "text":
        return PlainTextResponse(text)
    if fmt == "srt":
        return PlainTextResponse(srt_from(seg), media_type="text/srt")
    if fmt == "vtt":
        return PlainTextResponse(vtt_from(seg), media_type="text/vtt")
    if fmt == "verbose_json":
        return JSONResponse(res)
    return JSONResponse({"text": text})

# ───────── Dependencies ─────────
def common_form_params(
    model: ModelId = Form("large-v3", description="Faster-Whisper model ID (see `/v1/models`)."),
    align: bool = Form(False, description="Word-level alignment via Wav2Vec2."),
    diarize: bool = Form(False, description="Speaker diarisation with `[SPK_n]` tags."),
    response_format: ResponseFormat = Form("json", description="Response format (`json`, `text`, `srt`, `vtt`, `verbose_json`)."),
    batch_size: int | None = Form(BATCH_SIZE, description="Whisper batch size."),
    word_timestamps: bool = Form(False, description="Include word timestamps (needs new FW build)."),
    vad_filter: bool = Form(False, description="Apply VAD before transcription."),
    vad_threshold: float | None = Form(0.5, description="VAD probability threshold."),
    min_speakers: int | None = Form(0, description="Lower bound for diarisation clustering."),
    max_speakers: int | None = Form(0, description="Upper bound for diarisation clustering."),
    diarization_model: str | None = Form(
        None,
        description="Override diarization model (e.g. 'pyannote/speaker-diarization-3.1'). "
                    "Defaults to env DIARIZATION_MODEL or 'pyannote/speaker-diarization-3.1'"
    ),
):
    return dict(
        model=model, align=align, diarize=diarize, response_format=response_format,
        batch_size=batch_size, word_timestamps=word_timestamps,
        vad_filter=vad_filter, vad_threshold=vad_threshold,
        min_speakers=min_speakers, max_speakers=max_speakers,
        diarization_model=diarization_model,
    )

# ───────── Endpoints ─────────
@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(..., description="Binary audio (any FFmpeg-decodable format)."),
    language: str | None = Form(None, description="Force language code; autodetect when omitted."),
    params: dict = Depends(common_form_params),
):
    """Transcribes an audio file."""
    with tempfile.NamedTemporaryFile(suffix=".audio") as tmp:
        tmp.write(await file.read())
        tmp.flush()
        res = await process(
            tmp.name, params["model"], language, params["align"], params["diarize"],
            build_transcribe_kwargs(
                params["batch_size"], params["word_timestamps"],
                params["vad_filter"], params["vad_threshold"]),
            build_diar_kwargs(params["min_speakers"], params["max_speakers"]),
            params["diarization_model"],
        )
        return _fmt(res, params["response_format"])

@app.post("/v1/audio/translations")
async def translations(
    file: UploadFile = File(..., description="Binary audio (any FFmpeg-decodable format)."),
    params: dict = Depends(common_form_params),
):
    """Translates an audio file to English."""
    with tempfile.NamedTemporaryFile(suffix=".audio") as tmp:
        tmp.write(await file.read())
        tmp.flush()
        res = await process(
            tmp.name, params["model"], None, params["align"], params["diarize"],
            build_transcribe_kwargs(
                params["batch_size"], params["word_timestamps"],
                params["vad_filter"], params["vad_threshold"]),
            build_diar_kwargs(params["min_speakers"], params["max_speakers"]),
            params["diarization_model"],
        )
        return _fmt(res, params["response_format"])
