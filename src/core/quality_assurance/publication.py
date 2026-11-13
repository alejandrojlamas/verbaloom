"""Private assembly and atomic, non-overwriting delivery of validated artifacts."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
from pathlib import Path


def create_staging_path(destination: str | Path) -> Path:
    """Keep the real basename/format, outside the public file-list directory."""
    destination = Path(destination).absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    directory = tempfile.mkdtemp(prefix=".publication-", dir=destination.parent)
    return Path(directory) / destination.name


def artifact_sha256(path: str | Path) -> str:
    artifact = Path(path)
    if not artifact.is_file():
        return ""
    digest = hashlib.sha256()
    with artifact.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Missing or invalid validated artifact checksum; validate again.")
    return value


def atomic_publish(
    source: str | Path, destination: str | Path, *, expected_sha256: str | None = None
) -> Path:
    """Publish a caller-validated snapshot without replacing any existing entry.

    Copy before linking so later repairs to the staged candidate cannot mutate a
    delivered file. The exclusive hard link is the commit point; a check followed
    by replace would still overwrite a concurrent delivery. Unsupported
    filesystems fail closed and retain the source, never fall back to overwrite.
    """
    source = Path(source)
    if expected_sha256 is not None:
        require_sha256(expected_sha256)
    destination = Path(destination).expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=".publishing-", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as output, source.open("rb") as input_file:
            shutil.copyfileobj(input_file, output)
            output.flush()
            os.fsync(output.fileno())
        if expected_sha256 is not None and artifact_sha256(temporary) != expected_sha256:
            raise ValueError(
                "The copied bytes differ from the validated artifact; validate again."
            )
        for index in range(10000):
            candidate = (
                destination
                if index == 0
                else destination.with_name(
                    f"{destination.stem} ({index}){destination.suffix}"
                )
            )
            try:
                os.link(temporary, candidate)
            except FileExistsError:
                continue
            return candidate
        raise FileExistsError("No unused publication filename after 10000 attempts")
    finally:
        temporary.unlink(missing_ok=True)


def publish_report_sidecars(source: Path, published: Path) -> None:
    """Expose reports under the committed artifact name, never native drafts."""
    for report in source.parent.iterdir():
        if (
            report.is_file()
            and report.suffix in {".md", ".json", ".html"}
            and report.name.startswith(source.stem + " - ")
        ):
            name = published.stem + report.name[len(source.stem) :]
            atomic_publish(report, published.with_name(name))
