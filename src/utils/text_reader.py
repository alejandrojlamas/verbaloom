"""Robust plain-text file reading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from src.utils.text_encoding import repair_mojibake


class TextReadError(ValueError):
    """Raised when a file cannot be decoded as usable text."""


def _looks_like_utf16(raw: bytes) -> bool:
    if len(raw) < 4:
        return False
    sample = raw[: min(len(raw), 4096)]
    nulls = sample.count(b"\x00")
    return nulls > max(2, len(sample) // 8)


def read_text_file_with_fallbacks(path: str | Path) -> Tuple[str, str]:
    """Read a text file using common book/archive encodings.

    Upload validation accepts non-UTF-8 text with a warning. The translation
    pipeline must mirror that leniency or Latin-1/Windows-1252 books can upload
    successfully and then fail before the first chunk is created.

    Returns:
        Tuple of ``(text, encoding_used)``.
    """
    file_path = Path(path)
    try:
        raw = file_path.read_bytes()
    except OSError as exc:
        raise TextReadError(f"Could not read text file: {exc}") from exc

    if not raw:
        return "", "empty"

    encoding_candidates: list[str] = []
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        encoding_candidates.append("utf-16")
    elif raw.startswith(b"\xef\xbb\xbf"):
        encoding_candidates.append("utf-8-sig")
    elif _looks_like_utf16(raw):
        encoding_candidates.extend(["utf-16", "utf-16-le", "utf-16-be"])

    encoding_candidates.extend(["utf-8", "utf-8-sig", "cp1252", "latin-1", "iso-8859-1"])

    seen = set()
    for encoding in encoding_candidates:
        if encoding in seen:
            continue
        seen.add(encoding)
        try:
            text = raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
        if "\x00" in text and encoding not in {"utf-16", "utf-16-le", "utf-16-be"}:
            continue
        return repair_mojibake(text.replace("\x00", "")), encoding

    try:
        text = raw.decode("utf-8", errors="replace")
    except Exception as exc:
        raise TextReadError(f"Could not decode text file: {exc}") from exc

    return repair_mojibake(text.replace("\x00", "")), "utf-8-replace"
