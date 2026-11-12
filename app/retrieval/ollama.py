"""Ollama client with actionable diagnostics for local model failures."""
import logging
import math
import time
import requests
from threading import BoundedSemaphore
from contextlib import contextmanager
from app.core.config import settings

logger = logging.getLogger(__name__)

# Keep local inference bounded on low-memory machines. Ollama remains responsible
# for model lifetime; this semaphore prevents this API from flooding it with work.
# The configured inference_concurrency was previously ignored (always 1).
_MODEL_SLOT = BoundedSemaphore(value=max(1, int(settings.inference_concurrency)))


class OllamaUnavailableError(RuntimeError):
    """Raised when Ollama or a configured local model cannot be used."""

def _validate_embedding_vectors(vectors, expected_count):
    """Reject malformed vectors before they reach the persistent vector store."""
    if not isinstance(vectors, list) or len(vectors) != expected_count:
        raise OllamaUnavailableError(
            "Ollama returned an unexpected number of embeddings."
        )
    dimension = None
    for index, vector in enumerate(vectors):
        if not isinstance(vector, list) or not vector:
            raise OllamaUnavailableError(
                f"Ollama returned an empty or malformed embedding at batch item {index}."
            )
        if dimension is None:
            dimension = len(vector)
        elif len(vector) != dimension:
            raise OllamaUnavailableError(
                "Ollama returned embeddings with inconsistent dimensions."
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in vector
        ):
            raise OllamaUnavailableError(
                f"Ollama returned a non-numeric or non-finite embedding at batch item {index}."
            )
    return vectors



@contextmanager
def _model_slot(operation):
    """Acquire the shared local-inference slot with a bounded queue wait.

    The HTTP timeout does not cover time spent waiting for this semaphore. A
    separate deadline prevents chat requests from appearing hung behind long
    embedding or generation work.
    """
    timeout = max(0.0, float(getattr(settings, "inference_queue_timeout_seconds", 30.0)))
    wait_started = time.perf_counter()
    acquired = _MODEL_SLOT.acquire(timeout=timeout)
    wait_seconds = time.perf_counter() - wait_started
    if not acquired:
        logger.warning(
            "Ollama inference queue timeout: operation=%s wait_seconds=%.3f limit_seconds=%.3f",
            operation, wait_seconds, timeout,
        )
        raise OllamaUnavailableError(
            f"Timed out waiting for the local inference queue ({operation}) after "
            f"{timeout:g} seconds. Another Ollama request is still running. "
            "Wait for the current ingestion or generation task to finish, then retry."
        )
    logger.info(
        "Ollama inference slot acquired: operation=%s queue_wait_seconds=%.3f",
        operation, wait_seconds,
    )
    try:
        yield wait_seconds
    finally:
        _MODEL_SLOT.release()


def _model_names():
    try:
        response = requests.get(f"{settings.ollama_base_url.rstrip('/')}/api/tags", timeout=5)
        response.raise_for_status()
        return [item.get("name", "") for item in response.json().get("models", [])]
    except requests.RequestException as exc:
        raise OllamaUnavailableError(
            f"Cannot connect to Ollama at {settings.ollama_base_url}. "
            "Start Ollama and verify OLLAMA_BASE_URL. Original error: "
            f"{exc}"
        ) from exc
    except (ValueError, AttributeError) as exc:
        raise OllamaUnavailableError(
            "Ollama responded with an unexpected /api/tags response."
        ) from exc


def _has_model(names, configured):
    # Ollama's /api/tags normally returns names including the tag, e.g. gemma2:2b.
    return configured in names or f"{configured}:latest" in names


def status():
    try:
        names = _model_names()
        return {
            "available": True,
            "base_url": settings.ollama_base_url,
            "installed_models": names,
            "generation_model": settings.generation_model,
            "generation_model_available": _has_model(names, settings.generation_model),
            "verification_model": settings.verification_model,
            "verification_model_available": _has_model(names, settings.verification_model),
            "embedding_model": settings.embedding_model,
            "embedding_model_available": _has_model(names, settings.embedding_model),
        }
    except OllamaUnavailableError as exc:
        return {
            "available": False,
            "base_url": settings.ollama_base_url,
            "generation_model": settings.generation_model,
            "generation_model_available": False,
            "verification_model": settings.verification_model,
            "verification_model_available": False,
            "embedding_model": settings.embedding_model,
            "embedding_model_available": False,
            "error": str(exc),
        }


def _request_model(model, endpoint, payload, timeout):
    request_started = time.perf_counter()
    try:
        response = requests.post(
            f"{settings.ollama_base_url.rstrip('/')}{endpoint}",
            json=payload,
            timeout=timeout,
        )
        if response.status_code == 404:
            raise OllamaUnavailableError(
                f"Ollama could not find model '{model}'. Install it in a terminal with "
                f"`ollama pull {model}`, then verify with `ollama list`. "
                f"Endpoint response: {response.text[:500]}"
            )
        response.raise_for_status()
        return response.json()
    except OllamaUnavailableError:
        raise
    except requests.ConnectionError as exc:
        raise OllamaUnavailableError(
            f"Cannot connect to Ollama at {settings.ollama_base_url}. "
            "Make sure Ollama is running. Original error: " + str(exc)
        ) from exc
    except requests.Timeout as exc:
        raise OllamaUnavailableError(
            f"Ollama timed out while using model '{model}' after {timeout} seconds. "
            "Try a smaller model, increase TIMEOUT in .env, or check available RAM. "
            f"Original error: {exc}"
        ) from exc
    except requests.HTTPError as exc:
        detail = exc.response.text[:700] if exc.response is not None else str(exc)
        raise OllamaUnavailableError(
            f"Ollama returned an HTTP error for model '{model}': {detail}"
        ) from exc
    except (ValueError, KeyError) as exc:
        raise OllamaUnavailableError(
            f"Ollama returned an invalid response for model '{model}': {exc}"
        ) from exc
    finally:
        logger.info(
            "Ollama HTTP request finished: model=%s endpoint=%s elapsed_seconds=%.3f",
            model, endpoint, time.perf_counter() - request_started,
        )


def embed(texts):
    """Embed a batch in one Ollama request when supported, with legacy fallback.

    Ollama's /api/embed accepts an input list and avoids one HTTP/model call per
    chunk. Older Ollama versions fall back to /api/embeddings one text at a time.
    """
    if not texts:
        return []
    with _model_slot("embedding"):
        url = f"{settings.ollama_base_url.rstrip('/')}/api/embed"
        request_started = time.perf_counter()
        try:
            response = requests.post(
                url,
                json={"model": settings.embedding_model, "input": texts},
                timeout=settings.timeout,
            )
            if response.status_code not in (404, 405):
                response.raise_for_status()
                payload = response.json()
                vectors = payload.get("embeddings")
                return _validate_embedding_vectors(vectors, len(texts))
        except requests.ConnectionError as exc:
            raise OllamaUnavailableError(
                f"Cannot connect to Ollama at {settings.ollama_base_url}: {exc}"
            ) from exc
        except requests.Timeout as exc:
            raise OllamaUnavailableError(
                f"Ollama embedding timed out after {settings.timeout} seconds: {exc}"
            ) from exc
        except requests.HTTPError as exc:
            detail = exc.response.text[:500] if exc.response is not None else str(exc)
            raise OllamaUnavailableError(f"Ollama embedding failed: {detail}") from exc
        except (ValueError, AttributeError, TypeError) as exc:
            raise OllamaUnavailableError(
                f"Ollama /api/embed returned an invalid JSON or embedding payload: {exc}"
            ) from exc
        finally:
            logger.info(
                "Ollama embedding request finished: model=%s endpoint=/api/embed "
                "batch_size=%d elapsed_seconds=%.3f",
                settings.embedding_model, len(texts), time.perf_counter() - request_started,
            )

        # Compatibility path for Ollama versions without /api/embed.
        out = []
        for item in texts:
            result = _request_model(
                settings.embedding_model,
                "/api/embeddings",
                {"model": settings.embedding_model, "prompt": item},
                settings.timeout,
            )
            if "embedding" not in result:
                raise OllamaUnavailableError(
                    f"Ollama returned no embedding for '{settings.embedding_model}'."
                )
            out.append(result["embedding"])
        return _validate_embedding_vectors(out, len(texts))



def embed_query(text):
    """Return a cached query embedding, computing it with Ollama on a miss.

    Document/chunk embeddings still use ``embed`` directly so a large corpus
    cannot fill the query cache with every indexed chunk.
    """
    from app.cache.manager import get_embedding, put_embedding, normalize_query
    cache_text = normalize_query(text)
    cached = get_embedding(cache_text, settings.embedding_model)
    if cached is not None:
        return cached
    vector = embed([cache_text])[0]
    put_embedding(cache_text, settings.embedding_model, vector)
    return vector

# Keep the generation model resident between the closely spaced calls of a single
# chat request (generate -> optional retry -> verify). On CPU-only hosts Ollama
# otherwise unloads the model between calls, and each reload is paid again as
# latency. This is a residency hint only; Ollama still evicts the model when the
# window lapses, so it does not pin memory indefinitely.
_MODEL_KEEP_ALIVE = "30m"


def generate(prompt, num_predict=None, model=None):
    """Run one completion, capped so a small model cannot ramble on CPU.

    ``num_predict`` overrides the configured answer cap for callers that need a
    much shorter reply, such as the one-word verifier verdict. ``model`` selects
    a specific local model so the accuracy-critical verification call can use a
    stronger model than the one that drafted the answer.
    """
    target_model = model or settings.generation_model
    cap = settings.answer_max_tokens if num_predict is None else num_predict
    options = {"temperature": settings.temperature, "num_ctx": max(1024, int(settings.model_context_tokens))}
    if cap:
        options["num_predict"] = max(1, int(cap))
    with _model_slot("generation"):
        result = _request_model(
            target_model,
            "/api/generate",
            {
                "model": target_model,
                "prompt": prompt,
                "stream": False,
                "keep_alive": _MODEL_KEEP_ALIVE,
                "options": options,
            },
            settings.timeout,
        )
    if "response" not in result:
        raise OllamaUnavailableError(
            f"Ollama returned no generated response for '{target_model}'."
        )
    return result["response"].strip()
