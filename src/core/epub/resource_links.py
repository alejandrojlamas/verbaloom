"""Convert EPUB resource URLs at the filesystem boundary, exactly once."""

import os
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit


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
