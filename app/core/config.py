from pathlib import Path
from typing import Any
from pydantic_settings import BaseSettings, SettingsConfigDict
from urllib.parse import urlparse
import ipaddress


def validate_loopback_url(url: str) -> bool:
    """Return True only for an HTTP endpoint bound to a loopback host."""
    try:
        parsed = urlparse(url)
        if parsed.scheme != "http" or not parsed.hostname:
            return False

        host = parsed.hostname.lower()

        if host == "localhost":
            return True

        return ipaddress.ip_address(host).is_loopback

    except (ValueError, TypeError):
        return False


ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):

    # LOCAL LLM CONFIGURATION


    ollama_base_url: str = "http://127.0.0.1:11434"

    # Pinned supported generation model. This default used to read "qwen3:8b",
    # which is not the model the application is tuned for and is not installed
    # in the supported Ollama setup, so a run without a .env file silently
    # pointed at a missing model. It now matches GENERATION_MODEL in .env.
    generation_model: str = "qwen2.5:3b"

    # Keep the existing embedding model to maintain
    # compatibility with the existing ChromaDB vectors.
    embedding_model: str = "nomic-embed-text"

    temperature: float = 0.0
    timeout: int = 300

    # Generation limits
    answer_max_tokens: int = 350
    verify_max_tokens: int = 128

    # --------------------------------------------------
    # APPLICATION DATA PATHS
    # --------------------------------------------------

    data_path: Path = ROOT / "data"

    uploads_path: Path | None = None
    extracted_path: Path | None = None
    chroma_path: Path | None = None
    metadata_path: Path | None = None
    logs_path: Path | None = None
    jobs_path: Path | None = None
    sqlite_path: Path | None = None

    # --------------------------------------------------
    # CHROMADB CONFIGURATION
    # --------------------------------------------------

    collection_name: str = "document_chunks"

    # Chunk configuration (character-based if the
    # existing splitter consumes character lengths).
    chunk_size: int = 1200
    chunk_overlap: int = 180

    # --------------------------------------------------
    # SEMANTIC RETRIEVAL
    # --------------------------------------------------

    # Final number of relevant chunks sent to the LLM. A four-chunk window was
    # too narrow for long documents: the passage answering a factual question
    # (for example the synopsis line naming a character's profession) often
    # ranked just outside it. Six slots recovered most near-miss passages, but
    # a single-fact question can still be answered by a passage ranked seventh
    # (for example the line naming the ship whose log the novel includes), so
    # eight slots are used; the context-character budget below still bounds the
    # prompt size.
    top_k: int = 8

    # Retrieve additional candidates before ranking.
    retrieval_candidate_multiplier: int = 5

    # Similarity filtering.
    # Confirm whether your retrieval code uses similarity
    # scores or distance scores before changing this.
    min_similarity: float = 0.25

    # --------------------------------------------------
    # HYBRID RETRIEVAL
    # --------------------------------------------------

    # Balance semantic and lexical retrieval.
    hybrid_lexical_weight: float = 0.35

    # Number of lexical candidates.
    lexical_candidate_limit: int = 40

    # --------------------------------------------------
    # EMBEDDING CONFIGURATION
    # --------------------------------------------------

    embedding_batch_size: int = 32

    # --------------------------------------------------
    # DOCUMENT PROCESSING
    # --------------------------------------------------

    max_file_mb: int = 200
    max_batch_mb: int = 500
    max_upload_files: int = 50

    ingestion_workers: int = 4

    # Local LLM inference should be serialized to avoid
    # memory pressure on limited hardware.
    inference_concurrency: int = 1

    tesseract_cmd: str = "C:\Program Files\Tesseract-OCR\tesseract.exe"
    # Matches OCR_ENABLED in .env. Image-only pages are common in the supported
    # corpus, so OCR is on by default; set tesseract_cmd (or put Tesseract on
    # PATH) in environments where the binary is not installed at the .env path.
    ocr_enabled: bool = True

    max_pdf_pages: int = 5000
    max_image_pixels: int = 40_000_000
    ocr_timeout_seconds: int = 45

    # Archive and queue resource guardrails for local document processing.
    max_archive_uncompressed_mb: int = 1024
    max_archive_compression_ratio: float = 200.0
    max_archive_entries: int = 20000
    max_pending_jobs: int = 50
    max_pending_upload_mb: int = 1000
    max_data_storage_mb: int = 20480

    extraction_block_rows: int = 1000
    max_question_parts: int = 10

    # LLM CONTEXT CONFIGURATION

    max_context_chars: int = 16000
    model_context_tokens: int = 4096
    context_safety_tokens: int = 900
    context_chars_per_token: float = 3.5

    # Verify answers against retrieved evidence.
    verify_answers: bool = True

    # CACHE CONFIGURATION


    cache_enabled: bool = True
    cache_ttl_seconds: int = 604800
    cache_max_entries: int = 2000

    semantic_retrieval_threshold: float = 0.94

    # Keep semantic answer caching disabled because
    # similar questions may have different meanings.
    semantic_answer_cache_enabled: bool = False
    semantic_answer_threshold: float = 0.975

    # RECOVERY

    recovery_on_startup: bool = True
    # ENVIRONMENT CONFIGURATION


    model_config = SettingsConfigDict(
        env_file=ROOT / ".env",
        extra="ignore",
        case_sensitive=False
    )


    # INITIALIZATION AND PATH VALIDATION

    def model_post_init(self, __context: Any) -> None:

        if not validate_loopback_url(self.ollama_base_url):
            raise ValueError(
                "OLLAMA_BASE_URL must use plain HTTP on "
                "localhost/127.0.0.1/::1 for offline-only operation."
            )

        configured_data_path = Path(
            self.data_path
        ).expanduser()

        self.data_path = (
            configured_data_path
            if configured_data_path.is_absolute()
            else ROOT / configured_data_path
        ).resolve()

        defaults = {
            "uploads_path": self.data_path / "uploads",
            "extracted_path": self.data_path / "extracted",
            "chroma_path": self.data_path / "chroma_db",
            "metadata_path": self.data_path / "metadata",
            "logs_path": self.data_path / "logs",
            "jobs_path": self.data_path / "jobs",
            "sqlite_path": (
                self.data_path / "metadata" / "documents.sqlite3"
            ),
        }

        for key, default in defaults.items():

            value = getattr(self, key)

            if value is not None:
                path = Path(value).expanduser()

                setattr(
                    self,
                    key,
                    path if path.is_absolute()
                    else (ROOT / path).resolve()
                )

            else:
                setattr(self, key, default.resolve())


settings = Settings()
