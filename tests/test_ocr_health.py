import app.ingestion.ocr_health as ocr_health


def test_ocr_status_reports_disabled_capability(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "ocr_enabled", False)
    result = ocr_health.local_ocr_status()

    assert result["enabled"] is False
    assert result["available"] is False
    assert "disabled" in result["reason"]


def test_ocr_status_reports_missing_python_package(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "ocr_enabled", True)
    monkeypatch.setattr(ocr_health.importlib.util, "find_spec", lambda name: None)
    result = ocr_health.local_ocr_status()

    assert result["enabled"] is True
    assert result["available"] is False
    assert "pytesseract" in result["reason"]


def test_ocr_status_accepts_configured_local_executable(monkeypatch, tmp_path):
    from app.core.config import settings

    binary = tmp_path / "tesseract.exe"
    binary.write_bytes(b"local test executable")
    monkeypatch.setattr(settings, "ocr_enabled", True)
    monkeypatch.setattr(settings, "tesseract_cmd", str(binary))
    monkeypatch.setattr(ocr_health.importlib.util, "find_spec", lambda name: object())

    result = ocr_health.local_ocr_status()

    assert result["enabled"] is True
    assert result["available"] is True
    assert result["executable"] == str(binary)


def test_ocr_status_uses_path_lookup_when_configured_path_is_missing(monkeypatch, tmp_path):
    from app.core.config import settings

    monkeypatch.setattr(settings, "ocr_enabled", True)
    monkeypatch.setattr(settings, "tesseract_cmd", str(tmp_path / "missing.exe"))
    monkeypatch.setattr(ocr_health.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(ocr_health.shutil, "which", lambda name: "/usr/bin/tesseract")

    result = ocr_health.local_ocr_status()

    assert result["available"] is True
    assert result["executable"] == "/usr/bin/tesseract"
