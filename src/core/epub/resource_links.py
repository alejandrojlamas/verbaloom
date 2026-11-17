"""Convert EPUB resource URLs at the filesystem boundary, exactly once."""

import os
import posixpath
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

from lxml import etree


def resource_uri_path(href: str) -> str:
    """Decode a URI path, preserving literal plus signs and encoded percent signs.

    Only use this on resource URLs, never on already-decoded checkpoint paths.
    Callers remain responsible for containment and external-resource policy.
    """
    return unquote(urlsplit(href).path)


def relative_resource_url(resource: Path, document: Path) -> str:
    # Both paths must use the same physical base: macOS /var and /private/var
    # can name the same directory, but relpath does not resolve that alias.
    relative = Path(os.path.relpath(resource.resolve(), document.resolve().parent))
    return quote(relative.as_posix(), safe="/")


def declared_package_path(container: etree._Element) -> str:
    """Resolve the default container rootfile URL to its archive member name."""
    first = container.find("./{*}rootfiles/{*}rootfile")
    full_path = first.get("full-path") if first is not None else None
    if not full_path:
        raise ValueError("EPUB container has no default package document")
    parsed = urlsplit(full_path)
    path = posixpath.normpath(resource_uri_path(full_path))
    if parsed.scheme or parsed.netloc or path.startswith(("/", "../")) or path in (".", ".."):
        raise ValueError(f"EPUB package document must stay inside the archive: {full_path}")
    return path


def find_package_document(directory: Path) -> Path | None:
    """Select the default rendition declared by the EPUB container."""
    container = directory / "META-INF" / "container.xml"
    if container.is_file():
        full_path = declared_package_path(etree.parse(str(container)).getroot())
        package = directory / full_path
        if not package.resolve().is_relative_to(directory.resolve()):
            raise ValueError(f"EPUB package document must stay inside the archive: {full_path}")
        if not package.is_file():
            raise FileNotFoundError(f"EPUB package document is missing: {full_path}")
        return package

    # Old unpacked inputs sometimes omit container.xml. Only one possible
    # package can be selected without guessing which rendition was intended.
    candidates = [
        item for item in directory.rglob("*")
        if item.suffix.lower() == ".opf" and item.is_file()
    ]
    if len(candidates) > 1:
        raise ValueError("Multiple EPUB package documents without META-INF/container.xml")
    return candidates[0] if candidates else None
