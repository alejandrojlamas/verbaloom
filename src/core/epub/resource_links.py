"""Build portable EPUB resource URLs from extracted filesystem paths."""

import os
from pathlib import Path
from urllib.parse import quote


def relative_resource_url(resource: Path, document: Path) -> str:
    # Both paths must use the same physical base: macOS /var and /private/var
    # can name the same directory, but relpath does not resolve that alias.
    relative = Path(os.path.relpath(resource.resolve(), document.resolve().parent))
    return quote(relative.as_posix(), safe="/")
