"""
Path validation utilities for secure file operations
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple


class PathValidator:
    """Validates file paths and names for security"""

    MAX_FILENAME_LENGTH = 255

    @staticmethod
    def resolve_managed_file(
        path_value: object,
        roots: list[str | Path] | tuple[str | Path, ...],
        *,
        must_exist: bool = True,
    ) -> Path:
        """Resolve a client path only when it stays inside an app-managed root."""
        raw = str(path_value or "").strip()
        if not raw or "\x00" in raw:
            raise ValueError("File path is empty or invalid")
        raw_path = Path(raw).expanduser()
        resolved_roots = [Path(root).expanduser().resolve() for root in roots]
        candidates = (
            [raw_path.resolve()]
            if raw_path.is_absolute()
            else [(root / raw_path).resolve() for root in resolved_roots]
        )

        inside_managed_storage = False
        for candidate in candidates:
            for root in resolved_roots:
                try:
                    candidate.relative_to(root)
                except ValueError:
                    continue
                inside_managed_storage = True
                if not must_exist or (candidate.exists() and candidate.is_file()):
                    return candidate

        if not inside_managed_storage:
            raise ValueError("File path is outside the managed storage directory")
        if must_exist:
            raise FileNotFoundError("Managed file was not found")
        raise ValueError("File path is outside the managed storage directory")

    @staticmethod
    def validate_filename(filename: str) -> Tuple[bool, str]:
        """
        Validate filename for security issues

        Args:
            filename: The filename to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not filename:
            return False, "Filename cannot be empty"

        if filename in {".", ".."}:
            return False, "Invalid filename: dot path segments are not allowed"

        if any(ord(char) < 32 or ord(char) == 127 for char in filename):
            return False, "Invalid filename: control characters are not allowed"

        # Prevent directory traversal - check for path separators with '..'
        # This allows '...' or '....' but blocks '../' or '..\' patterns
        if filename.startswith('/') or filename.startswith('\\'):
            return False, "Invalid filename: absolute path not allowed"

        # Check for directory traversal patterns
        # Block: ../ or ..\ (with separators)
        if '/../' in filename or '\\..\\' in filename or '/..' in filename or '\\..' in filename:
            return False, "Invalid filename: directory traversal not allowed"

        # Reject both Unix and Windows separators regardless of host OS. On
        # macOS/Linux, os.path.basename("foo\\bar.txt") treats backslash as a
        # literal character, so relying on host path rules lets Windows-style
        # paths through.
        if '/' in filename or '\\' in filename:
            return False, "Invalid filename: path separators not allowed"

        # Check filename length
        if len(filename) > PathValidator.MAX_FILENAME_LENGTH:
            return False, f"Filename too long (max {PathValidator.MAX_FILENAME_LENGTH} characters)"

        # Prevent absolute paths
        if ':' in filename and len(filename) > 2 and filename[1] == ':':  # Windows absolute path
            return False, "Absolute paths not allowed"

        return True, ""

    @staticmethod
    def validate_filenames(filenames: list) -> Tuple[bool, str]:
        """
        Validate a list of filenames

        Args:
            filenames: List of filenames to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(filenames, list):
            return False, "Filenames must be a list"

        if len(filenames) == 0:
            return False, "No filenames provided"

        for filename in filenames:
            is_valid, error = PathValidator.validate_filename(filename)
            if not is_valid:
                return False, f"Invalid filename '{filename}': {error}"

        return True, ""
