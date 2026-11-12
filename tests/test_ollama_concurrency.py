"""Regression tests for bounded Ollama inference queue waits."""
import pytest


class _StubSemaphore:
    def __init__(self, acquired):
        self.acquired = acquired
        self.acquire_timeout = None
        self.release_count = 0

    def acquire(self, timeout=None):
        self.acquire_timeout = timeout
        return self.acquired

    def release(self):
        self.release_count += 1


def test_generation_fails_fast_when_inference_queue_is_busy(monkeypatch):
    import app.retrieval.ollama as ollama

    slot = _StubSemaphore(acquired=False)
    monkeypatch.setattr(ollama, "_MODEL_SLOT", slot)
    monkeypatch.setattr(ollama.settings, "inference_queue_timeout_seconds", 0.25)

    with pytest.raises(ollama.OllamaUnavailableError, match="inference queue"):
        ollama.generate("test prompt")

    assert slot.acquire_timeout == 0.25
    assert slot.release_count == 0


def test_embedding_fails_fast_when_inference_queue_is_busy(monkeypatch):
    import app.retrieval.ollama as ollama

    slot = _StubSemaphore(acquired=False)
    monkeypatch.setattr(ollama, "_MODEL_SLOT", slot)
    monkeypatch.setattr(ollama.settings, "inference_queue_timeout_seconds", 0.5)

    with pytest.raises(ollama.OllamaUnavailableError, match="inference queue"):
        ollama.embed(["test text"])

    assert slot.acquire_timeout == 0.5
    assert slot.release_count == 0


def test_generation_releases_inference_slot_after_request_error(monkeypatch):
    import app.retrieval.ollama as ollama

    slot = _StubSemaphore(acquired=True)
    monkeypatch.setattr(ollama, "_MODEL_SLOT", slot)
    monkeypatch.setattr(
        ollama, "_request_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(ollama.OllamaUnavailableError("offline")),
    )

    with pytest.raises(ollama.OllamaUnavailableError, match="offline"):
        ollama.generate("test prompt")

    assert slot.acquire_timeout == ollama.settings.inference_queue_timeout_seconds
    assert slot.release_count == 1
