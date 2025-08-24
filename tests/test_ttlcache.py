import sys, types, time
from pathlib import Path

# Ensure root path for importing app module
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Stub heavy dependencies before importing app
stub_module = types.ModuleType("stub")
for name in ["whisperx", "srt", "webvtt"]:
    sys.modules[name] = stub_module

# fastapi stubs
class DummyApp:
    def __init__(self, *args, **kwargs):
        pass
    def on_event(self, *args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def get(self, *args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def post(self, *args, **kwargs):
        def decorator(func):
            return func
        return decorator

fastapi_stub = types.ModuleType("fastapi")
fastapi_stub.Depends = lambda *a, **k: None
fastapi_stub.FastAPI = DummyApp
fastapi_stub.UploadFile = object
fastapi_stub.File = lambda *a, **k: None
fastapi_stub.Form = lambda *a, **k: None
fastapi_stub.HTTPException = Exception
responses_stub = types.ModuleType("fastapi.responses")
responses_stub.JSONResponse = object
responses_stub.PlainTextResponse = object
fastapi_stub.responses = responses_stub
sys.modules['fastapi'] = fastapi_stub
sys.modules['fastapi.responses'] = responses_stub

# huggingface_hub errors stub
errors_mod = types.ModuleType('huggingface_hub.errors')
errors_mod.LocalEntryNotFoundError = Exception
huggingface_hub_mod = types.ModuleType('huggingface_hub')
sys.modules['huggingface_hub'] = huggingface_hub_mod
sys.modules['huggingface_hub.errors'] = errors_mod

# torch stub
class CudaStub:
    def is_available(self):
        return True
    def mem_get_info(self):
        return (0, 0)
    def empty_cache(self):
        pass

cuda_stub = CudaStub()
backends_stub = types.ModuleType('backends')
backends_stub.cuda = types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=False))
backends_stub.cudnn = types.SimpleNamespace(allow_tf32=False)
torch_stub = types.ModuleType('torch')
torch_stub.cuda = cuda_stub
torch_stub.backends = backends_stub
sys.modules['torch'] = torch_stub

import app
from app import TTLCache
import pytest


def test_get_missing_key_returns_none():
    cache = TTLCache("test")
    assert cache.get("missing") is None


def test_get_updates_timestamp():
    cache = TTLCache("test")
    cache.put("key", "value")
    old_ts = cache["key"][1]
    time.sleep(0.01)
    value = cache.get("key")
    new_ts = cache["key"][1]
    assert value == "value"
    assert new_ts > old_ts


def test_sweep_removes_expired(monkeypatch):
    cache = TTLCache("whisper")
    cache.put("old", 1)
    cache.put("fresh", 2)
    # Make "old" entry appear expired
    cache["old"] = (1, time.time() - 100)
    monkeypatch.setattr(app.torch.cuda, "empty_cache", lambda: None)
    cache.sweep(ttl=10)
    assert "old" not in cache
    assert "fresh" in cache
