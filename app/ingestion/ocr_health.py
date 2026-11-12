"""Local OCR capability checks; no model or binary downloads are performed."""
import importlib.util
import shutil
from pathlib import Path

from app.core.config import settings


def local_ocr_status() -> dict:
    """Return whether the configured local OCR stack is available."""
    if not settings.ocr_enabled:
        return {"enabled": False, "available": False, "reason": "OCR is disabled by configuration"}
    try:
        package_available = importlib.util.find_spec("pytesseract") is not None
    except (ImportError, ValueError):
        package_available = False
    if not package_available:
        return {
            "enabled": True, "available": False,
            "reason": "Python package pytesseract is not installed",
        }
    configured = Path(settings.tesseract_cmd).expanduser() if settings.tesseract_cmd else None
    binary = str(configured) if configured and configured.is_file() else shutil.which("tesseract")
    if not binary:
        return {
            "enabled": True, "available": False,
            "reason": "Tesseract executable was not found; configure TESSERACT_CMD or add it to PATH",
        }
    return {"enabled": True, "available": True, "executable": binary}
