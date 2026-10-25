"""Tests for language detection and filtering."""

from pathlib import Path
import pytest

from backend.ingestion.language import LanguageDetector


@pytest.fixture
def detector():
    return LanguageDetector()


def test_detect_language(detector: LanguageDetector):
    assert detector.detect_language(Path("main.py")) == "python"
    assert detector.detect_language(Path("script.pyi")) == "python"
    assert detector.detect_language(Path("index.js")) == "javascript"
    assert detector.detect_language(Path("component.jsx")) == "javascript"
    assert detector.detect_language(Path("server.ts")) == "typescript"
    assert detector.detect_language(Path("app.tsx")) == "tsx"
    assert detector.detect_language(Path("main.go")) == "go"
    assert detector.detect_language(Path("Main.java")) == "java"
    assert detector.detect_language(Path("lib.rs")) == "rust"
    assert detector.detect_language(Path("core.cpp")) == "cpp"
    assert detector.detect_language(Path("header.hpp")) == "cpp"
    assert detector.detect_language(Path("readme.md")) is None
    assert detector.detect_language(Path("config.yaml")) is None


def test_is_ignored(detector: LanguageDetector, tmp_path: Path):
    repo_root = tmp_path

    # Ignored directories
    assert detector.is_ignored(repo_root / ".git" / "config", repo_root) is True
    assert detector.is_ignored(repo_root / "node_modules" / "pkg" / "index.js", repo_root) is True
    assert detector.is_ignored(repo_root / ".venv" / "lib" / "site.py", repo_root) is True
    assert detector.is_ignored(repo_root / "__pycache__" / "mod.cpython-312.pyc", repo_root) is True

    # Ignored files
    assert detector.is_ignored(repo_root / "package-lock.json", repo_root) is True
    assert detector.is_ignored(repo_root / "bundle.min.js", repo_root) is True
    assert detector.is_ignored(repo_root / "image.png", repo_root) is True

    # Valid source file
    assert detector.is_ignored(repo_root / "src" / "app.py", repo_root) is False


def test_scan_repository(detector: LanguageDetector, tmp_path: Path):
    # Create sample folder structure
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hello')", encoding="utf-8")
    (tmp_path / "src" / "utils.js").write_text("console.log('hi')", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "index.md").write_text("# Docs", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/main", encoding="utf-8")

    supported, unsupported, stats = detector.scan_repository(tmp_path)

    assert len(supported) == 2
    assert len(unsupported) == 1
    assert "python" in stats
    assert "javascript" in stats
    assert stats["python"].file_count == 1
    assert stats["javascript"].file_count == 1
