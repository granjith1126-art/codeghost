"""Language detection, file classification, and repository statistics."""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from backend.models.schemas import LanguageStat

logger = logging.getLogger(__name__)

# Supported programming languages mapped to file extensions
EXTENSION_TO_LANGUAGE: Dict[str, str] = {
    # Python
    ".py": "python",
    ".pyi": "python",
    ".pyw": "python",
    # JavaScript
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    # TypeScript
    ".ts": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".tsx": "tsx",
    # Go
    ".go": "go",
    # Java
    ".java": "java",
    # Rust
    ".rs": "rust",
    # C / C++
    ".c": "cpp",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".h": "cpp",
    ".hh": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
}

# Directories to ignore during scanning
IGNORED_DIRECTORIES: Set[str] = {
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "venv",
    ".venv",
    "env",
    ".env",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".tox",
    ".nox",
    "build",
    "dist",
    "target",
    "vendor",
    ".idea",
    ".vscode",
    "coverage",
    "htmlcov",
    ".eggs",
    "site-packages",
    "docs/_build",
    ".next",
    ".nuxt",
}

# File patterns to exclude (binaries, lockfiles, generated assets)
IGNORED_EXTENSIONS: Set[str] = {
    # Binaries & compiled
    ".exe", ".dll", ".so", ".dylib", ".bin", ".pyc", ".pyo", ".pyd", ".class", ".o", ".a",
    # Archives
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar", ".whl", ".jar",
    # Media
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".webp", ".mp3", ".mp4", ".wav", ".avi",
    # Documents
    ".pdf", ".docx", ".xlsx", ".pptx",
    # Locks & data caches
    ".lock", ".sqlite", ".db",
}

IGNORED_FILE_NAMES: Set[str] = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "Pipfile.lock",
    "Cargo.lock",
    "go.sum",
    "composer.lock",
}


class LanguageDetector:
    """Classifies source files and detects repository language distributions."""

    def __init__(self, custom_extensions: Optional[Dict[str, str]] = None):
        self.extension_map = dict(EXTENSION_TO_LANGUAGE)
        if custom_extensions:
            self.extension_map.update(custom_extensions)

    def detect_language(self, file_path: Path) -> Optional[str]:
        """Detect the programming language for a given file path based on its extension."""
        ext = file_path.suffix.lower()
        return self.extension_map.get(ext)

    def is_ignored(self, path: Path, repo_root: Path) -> bool:
        """Check whether a path or any of its ancestors should be ignored."""
        try:
            rel_parts = path.relative_to(repo_root).parts
        except ValueError:
            rel_parts = path.parts

        # Check ignored folder names in relative path
        for part in rel_parts[:-1]:
            if part in IGNORED_DIRECTORIES or part.startswith("."):
                return True

        filename = path.name
        # Ignore hidden files, lock files, and minified bundles
        if filename.startswith(".") and not filename == ".":
            return True
        if filename in IGNORED_FILE_NAMES:
            return True
        if filename.endswith(".min.js") or filename.endswith(".min.css"):
            return True

        if path.suffix.lower() in IGNORED_EXTENSIONS:
            return True

        return False

    def scan_repository(
        self, repo_root: Path
    ) -> Tuple[List[Path], List[Path], Dict[str, LanguageStat]]:
        """
        Scan a repository and categorize all discovered files.

        Returns:
            Tuple of (supported_files, unsupported_files, language_stats_map)
        """
        repo_root = Path(repo_root).resolve()
        supported_files: List[Path] = []
        unsupported_files: List[Path] = []
        stats: Dict[str, Dict[str, int]] = {}

        total_bytes = 0

        for root, dirs, files in os.walk(repo_root):
            # Prune ignored directories in-place for performance
            dirs[:] = [d for d in dirs if d not in IGNORED_DIRECTORIES and not d.startswith(".")]

            for file in files:
                file_path = Path(root) / file

                if self.is_ignored(file_path, repo_root):
                    continue

                try:
                    file_size = file_path.stat().st_size
                except (OSError, FileNotFoundError):
                    file_size = 0

                language = self.detect_language(file_path)
                if language:
                    supported_files.append(file_path)
                    if language not in stats:
                        stats[language] = {"files": 0, "bytes": 0}
                    stats[language]["files"] += 1
                    stats[language]["bytes"] += file_size
                    total_bytes += file_size
                else:
                    unsupported_files.append(file_path)

        # Compute percentage stats
        language_stats: Dict[str, LanguageStat] = {}
        for lang, data in stats.items():
            pct = (data["bytes"] / total_bytes * 100.0) if total_bytes > 0 else 0.0
            language_stats[lang] = LanguageStat(
                language=lang,
                file_count=data["files"],
                total_bytes=data["bytes"],
                percentage=round(pct, 2),
            )

        logger.info(
            "Discovered %d supported source files across %d languages; %d unsupported files",
            len(supported_files),
            len(language_stats),
            len(unsupported_files),
        )

        return supported_files, unsupported_files, language_stats
