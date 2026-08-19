"""Bounded validation and extraction for uploaded ZIP-based documents."""

from __future__ import annotations

import shutil
import stat
import zipfile
from io import BytesIO
from pathlib import Path, PurePosixPath


MAX_ARCHIVE_MEMBERS = 10_000
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
MAX_ARCHIVE_MEMBER_BYTES = 128 * 1024 * 1024
MAX_ARCHIVE_COMPRESSION_RATIO = 200


class UnsafeArchiveError(ValueError):
    """Raised when an archive exceeds limits or contains an unsafe member."""


def _validated_member_path(name: str) -> PurePosixPath:
    if not name or "\x00" in name:
        raise UnsafeArchiveError("Archive contains an invalid member name")
    normalized = name.replace("\\", "/")
    member = PurePosixPath(normalized)
    if member.is_absolute() or any(part in {"", ".", ".."} for part in member.parts):
        raise UnsafeArchiveError(f"Archive member escapes the destination: {name!r}")
    if member.parts and ":" in member.parts[0]:
        raise UnsafeArchiveError(f"Archive member uses an absolute drive path: {name!r}")
    return member


def validate_zip_archive(
    archive: zipfile.ZipFile,
    *,
    max_members: int = MAX_ARCHIVE_MEMBERS,
    max_total_bytes: int = MAX_ARCHIVE_UNCOMPRESSED_BYTES,
    max_member_bytes: int = MAX_ARCHIVE_MEMBER_BYTES,
    max_ratio: int = MAX_ARCHIVE_COMPRESSION_RATIO,
) -> None:
    """Reject path traversal, links, encryption, and excessive expansion."""
    infos = archive.infolist()
    if len(infos) > max_members:
        raise UnsafeArchiveError(f"Archive contains too many members ({len(infos)} > {max_members})")

    total = 0
    for info in infos:
        _validated_member_path(info.filename)
        mode = (info.external_attr >> 16) & 0xFFFF
        if mode and stat.S_ISLNK(mode):
            raise UnsafeArchiveError(f"Archive contains a symbolic link: {info.filename!r}")
        if info.flag_bits & 0x1:
            raise UnsafeArchiveError(f"Archive contains an encrypted member: {info.filename!r}")
        if info.file_size < 0 or info.compress_size < 0:
            raise UnsafeArchiveError("Archive contains invalid size metadata")
        if info.file_size > max_member_bytes:
            raise UnsafeArchiveError(
                f"Archive member is too large after expansion: {info.filename!r}"
            )
        total += info.file_size
        if total > max_total_bytes:
            raise UnsafeArchiveError("Archive expands beyond the allowed total size")
        if info.file_size:
            if info.compress_size == 0 or info.file_size / info.compress_size > max_ratio:
                raise UnsafeArchiveError(
                    f"Archive member has an excessive compression ratio: {info.filename!r}"
                )


def safe_extractall(archive: zipfile.ZipFile, destination: str | Path) -> None:
    """Extract a previously untrusted archive after enforcing expansion bounds."""
    validate_zip_archive(archive)
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)

    for info in archive.infolist():
        member = _validated_member_path(info.filename)
        target = (root / Path(*member.parts)).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise UnsafeArchiveError(
                f"Archive member escapes the destination: {info.filename!r}"
            ) from exc
        if info.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info, "r") as source, target.open("wb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)


def validate_zip_bytes(payload: bytes) -> None:
    """Validate an in-memory ZIP-based document before a parser expands it."""
    with zipfile.ZipFile(BytesIO(payload), "r") as archive:
        validate_zip_archive(archive)
