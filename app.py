"""
WhisperX Transcription API · v1.8.5
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
import os, time, logging, threading, tempfile, gc, torch, asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from typing import Dict, Any

import whisperx, srt, webvtt
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
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
if OFFLINE:
    os.environ["HF_HUB_OFFLINE"] = "1"

app = FastAPI(title="WhisperX Transcription API", version="1.8.5")

# ───────── TTL caches ─────────
class TTLCache(dict):
    """Dict with TTL eviction & VRAM-usage logging."""
    def __init__(self, label: str): super().__init__(); self.label = label
    def get(self, k):
        v = super().get(k)
        if v: super().__setitem__(k, (v[0], time.time())); return v[0]
    def put(self, k, obj): super().__setitem__(k, (obj, time.time()))
    def sweep(self, ttl: int):
        now = time.time()
        for k, (_, ts) in list(self.items()):
            if now - ts > ttl:
                del self[k]; torch.cuda.empty_cache()
                key = {"whisper": "model", "align": "lang", "diarize": "pipeline"}[self.label]
                logging.info("[%s_model_unload]  %s=%s  freeVRAM=%d MB",
                             self.label, key, k, free_mb())

W_CACHE, A_CACHE, D_CACHE = TTLCache("whisper"), TTLCache("align"), TTLCache("diarize")
LOCKS: Dict[str, threading.Lock] = {}

# ───────── Logging helpers ─────────
from pathlib import Path
def _log(tag: str, fname: str, msg: str = "", *a):
    logging.info("[%s] %s  freeVRAM=%d MB " + msg,
                 tag, Path(fname).name, free_mb(), *a)

def _load_start(lbl: str, key: str):
    logging.info("[%s_model_load_start]  %s=%s  freeVRAM=%d MB",
                 lbl, "model" if lbl == "whisper" else "lang", key, free_mb())

def _load_end(lbl: str, key: str, before: int):
    delta = before - free_mb()
    logging.info("[%s_model_load_end]    %s=%s  used=%+d MB  freeVRAM=%d MB",
                 lbl, "model" if lbl == "whisper" else "lang",
                 key, delta, free_mb())

# ───────── Loaders ─────────
def load_whisper(model_id: str):
    """Return (pipeline, lock); raise 400 offline if model isn’t cached."""
    existing = W_CACHE.get(model_id)
    if existing:
        if model_id not in LOCKS:
            LOCKS[model_id] = threading.Lock()
        return existing, LOCKS[model_id]

    before = free_mb(); _load_start("whisper", model_id)
    kw = dict(compute_type=COMPUTE_TYPE,
              local_files_only=OFFLINE,
              device=DEVICE)
    if FW_THREADS:
        kw["threads"] = FW_THREADS
    try:
        model = whisperx.load_model(model_id, **kw)
    except LocalEntryNotFoundError:
        raise HTTPException(
            status_code=400,
            detail=(f"Model '{model_id}' is not cached locally and "
                    "LOCAL_ONLY_MODELS=1 prevents downloading.")
        ) from None

    W_CACHE.put(model_id, model); _load_end("whisper", model_id, before)
    LOCKS[model_id] = threading.Lock()
    return model, LOCKS[model_id]

def load_align(lang: str):
    key = lang or "default"
    pair = A_CACHE.get(key)
    if pair: return pair
    before = free_mb(); _load_start("align", key)
    model, meta = whisperx.load_align_model(language_code=lang or "en", device=DEVICE)
    A_CACHE.put(key, (model, meta)); _load_end("align", key, before)
    return model, meta

def load_diar():
    pip = D_CACHE.get("pipeline")
    if pip: return pip
    before = free_mb(); _load_start("diarize", "pipeline")
    try: from whisperx.diarize import DiarizationPipeline as _DP
    except ImportError: from whisperx.diarization import DiarizationPipeline as _DP
    pip = _DP(use_auth_token=HF_TOKEN, device=DEVICE)
    D_CACHE.put("pipeline", pip); _load_end("diarize", "pipeline", before)
    return pip

# ───────── /v1/models ─────────
def is_cached(mid: str) -> bool:
    return mid in local_sizes() or mid in W_CACHE

@app.get("/v1/models")
def models():
    ids = (m for m in _MODELS) if not OFFLINE else (m for m in _MODELS if is_cached(m))
    return {"data": [
        {"id": m, "object": "model", "created": 0, "owned_by": "you",
         "downloaded": is_cached(m)}
        for m in ids
    ]}

# ───────── Text helpers ─────────
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
        txt = _tagged(seg) if spk else raw.get("text") or " ".join(s["text"].strip() for s in seg)
        return {"text": txt, "segments": seg, "language": raw.get("language")}
    if isinstance(raw, list):
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
            end  =webvtt.Caption.time_to_webvtt(s["end"]),
            text=(f"[{s['speaker']}] " if s.get("speaker") else "") + s["text"].strip()))
    return v.content

# ───────── Sweeper thread ─────────
def _sweep():
    while True:
        time.sleep(60)
        W_CACHE.sweep(TTL_SEC); A_CACHE.sweep(TTL_SEC); D_CACHE.sweep(TTL_SEC); gc.collect()
threading.Thread(target=_sweep, daemon=True).start()

# ───────── Pipeline ─────────
async def process(path, model, lang, do_align, do_diar, trans_kw, diar_kw):
    fname = Path(path).name
    wav = whisperx.load_audio(path)
    audio_sec = len(wav) / 16000
    t0 = time.perf_counter()

    # transcription
    _log("transcribe_start", fname, "model=%s", model)
    whisper, lock = load_whisper(model)
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
        spk = await run_sync(load_diar(), wav, **diar_kw)
        res = standardize(await run_sync(
            whisperx.assign_word_speakers, spk, res), spk=True)
        _log("diarize_end", fname, "Δ=%.2fs", time.perf_counter() - t)

    wall = time.perf_counter() - t0
    logging.info("[summary] %s Δ=%.2fs audio=%.2fs speed=%.1fx",
                 fname, wall, audio_sec, audio_sec / wall if wall else 0)
    return res

# ───────── KW builders ─────────
def build_transcribe_kwargs(batch, beam, best, patience, length_penalty,
                            word_ts, vad, vad_thr):
    kw = {"batch_size": batch or BATCH_SIZE}
    if beam: kw["beam_size"] = beam
    if best: kw["best_of"] = best
    if patience: kw["patience"] = patience
    if length_penalty: kw["length_penalty"] = length_penalty
    if vad:
        kw["vad_filter"] = True
        if vad_thr:
            kw["vad_parameters"] = {"threshold": vad_thr}
    if word_ts: kw["word_timestamps"] = True
    return kw

def build_diar_kwargs(min_spk, max_spk):
    d = {}
    if min_spk: d["min_speakers"] = min_spk
    if max_spk: d["max_speakers"] = max_spk
    return d

def _fmt(res, fmt):
    text, seg = res["text"], res["segments"]
    if fmt == "text": return PlainTextResponse(text)
    if fmt == "srt":  return PlainTextResponse(srt_from(seg), media_type="text/srt")
    if fmt == "vtt":  return PlainTextResponse(vtt_from(seg), media_type="text/vtt")
    if fmt == "verbose_json": return JSONResponse(res)
    return JSONResponse({"text": text})

# ───────── Endpoints ─────────
@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form("large-v3"),
    language: str | None = Form(None),
    align: bool = Form(False),
    diarize: bool = Form(False),
    response_format: str = Form("json"),
    batch_size: int | None = Form(BATCH_SIZE),
    beam_size: int | None = Form(0),
    best_of: int | None = Form(0),
    patience: float | None = Form(0.0),
    length_penalty: float | None = Form(0.0),
    word_timestamps: bool = Form(False),
    vad_filter: bool = Form(False),
    vad_threshold: float | None = Form(0.5),
    min_speakers: int | None = Form(0),
    max_speakers: int | None = Form(0),
):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".audio")
    tmp.write(await file.read()); tmp.close()
    try:
        res = await process(
            tmp.name, model, language, align, diarize,
            build_transcribe_kwargs(batch_size, beam_size, best_of, patience,
                                    length_penalty, word_timestamps,
                                    vad_filter, vad_threshold),
            build_diar_kwargs(min_speakers, max_speakers))
        return _fmt(res, response_format)
    finally:
        os.remove(tmp.name)

@app.post("/v1/audio/translations")
async def translations(
    file: UploadFile = File(...),
    model: str = Form("large-v3"),
    align: bool = Form(False),
    diarize: bool = Form(False),
    response_format: str = Form("json"),
    batch_size: int | None = Form(BATCH_SIZE),
    beam_size: int | None = Form(0),
    best_of: int | None = Form(0),
    patience: float | None = Form(0.0),
    length_penalty: float | None = Form(0.0),
    word_timestamps: bool = Form(False),
    vad_filter: bool = Form(False),
    vad_threshold: float | None = Form(0.5),
    min_speakers: int | None = Form(0),
    max_speakers: int | None = Form(0),
):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".audio")
    tmp.write(await file.read()); tmp.close()
    try:
        res = await process(
            tmp.name, model, None, align, diarize,
            build_transcribe_kwargs(batch_size, beam_size, best_of, patience,
                                    length_penalty, word_timestamps,
                                    vad_filter, vad_threshold),
            build_diar_kwargs(min_speakers, max_speakers))
        return _fmt(res, response_format)
    finally:
        os.remove(tmp.name)
