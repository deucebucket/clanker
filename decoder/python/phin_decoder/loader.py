"""
Dictionary loader for Phin-Lang.

Loads YAML dictionary files and caches them for repeated use.
Searches standard dictionary paths relative to the project root.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Optional


class DictionaryLoader:
    """Loads and caches Phin dictionary YAML files."""

    def __init__(self, search_paths: Optional[list] = None):
        """
        Initialize the loader.

        Args:
            search_paths: List of directories to search for dictionary files.
                          If None, uses the default project dictionaries/ path.
        """
        self._cache: Dict[str, dict] = {}

        if search_paths:
            self._search_paths = [Path(p) for p in search_paths]
        else:
            # Default: look for dictionaries/ relative to the project root
            project_root = Path(__file__).resolve().parent.parent.parent.parent
            self._search_paths = [project_root / "dictionaries"]

    def load(self, language: str) -> dict:
        """
        Load a dictionary by language code or name.

        Searches in order:
          1. human/<language>.yaml
          2. code/<language>.yaml
          3. other/<language>.yaml

        Args:
            language: Language code (e.g., 'en', 'zh') or name (e.g., 'python', 'rust').

        Returns:
            Parsed dictionary as a dict.

        Raises:
            FileNotFoundError: If no dictionary found for the given language.
        """
        if language in self._cache:
            return self._cache[language]

        subdirs = ["human", "code", "other"]
        for base in self._search_paths:
            for subdir in subdirs:
                path = base / subdir / f"{language}.yaml"
                if path.exists():
                    with open(path, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f)
                    self._cache[language] = data
                    return data

        raise FileNotFoundError(
            f"No dictionary found for language '{language}'. "
            f"Searched in: {[str(p) for p in self._search_paths]}"
        )

    def load_from_file(self, filepath: str) -> dict:
        """
        Load a dictionary from an explicit file path.

        Args:
            filepath: Absolute or relative path to a YAML dictionary file.

        Returns:
            Parsed dictionary as a dict.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Dictionary file not found: {filepath}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        lang = data.get("language", path.stem)
        self._cache[lang] = data
        return data

    def available_languages(self) -> list:
        """
        List all available dictionary languages.

        Returns:
            List of language codes/names found in search paths.
        """
        languages = []
        subdirs = ["human", "code", "other"]
        for base in self._search_paths:
            for subdir in subdirs:
                d = base / subdir
                if d.exists():
                    for f in d.glob("*.yaml"):
                        if not f.name.startswith("_"):
                            languages.append(f.stem)
        return sorted(set(languages))

    def clear_cache(self):
        """Clear all cached dictionaries."""
        self._cache.clear()
