"""Create audiobook EPUB companions without flattening publication structure.

The narration TXT is intentionally text-first.  The EPUB companion is not:
readers still need the cover, illustrations, captions, navigation, fonts, and
the exact visual placement supplied by the translated EPUB.  This module clones
that publication atomically and permits only a package-title suffix.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
from pathlib import Path, PurePosixPath
import posixpath
import re
import shutil
from tempfile import NamedTemporaryFile
from typing import Any
from zipfile import ZIP_STORED, ZipFile, ZipInfo

from lxml import etree

from .publication_gate import snapshot_epub


_CONTAINER_PATH = "META-INF/container.xml"
_IMAGE_SUFFIXES = {".avif", ".gif", ".jpeg", ".jpg", ".png", ".svg", ".webp"}
_CAPTION_CLASS_RE = re.compile(
    r"(?:caption|credit|figcap|figcaption|legend|pie(?:[-_ ]de[-_ ](?:foto|imagen))?)",
    re.IGNORECASE,
)
_XML_PARSER = etree.XMLParser(recover=False, remove_blank_text=False)


class AudiobookEpubPreservationError(RuntimeError):
    """Raised when a structured companion would lose publication assets."""


@dataclass
class StructuredAudiobookEpubReport:
    source_path: str
    output_path: str
    image_count: int = 0
    image_references: int = 0
    image_placements: int = 0
    captions_preserved: int = 0
    cover_preserved: bool = False
    spine_preserved: bool = False
    xhtml_preserved: bool = False
    mimetype_valid: bool = False
    title: str = ""
    errors: list[str] = field(default_factory=list)

    @property
    def publishable(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["publishable"] = self.publishable
        return payload


@dataclass(frozen=True)
class _ImagePlacement:
    document: str
    dom_path: str
    reference: str
    alt_text: str
    caption: str


def create_structured_audiobook_epub(
    source_epub: str | Path,
    output_epub: str | Path,
    *,
    title_suffix: str = "Audiolibro",
) -> StructuredAudiobookEpubReport:
    """Clone an EPUB while proving that its visual publication contract survives.

    No prose is regenerated and no LLM is called.  Every XHTML document and
    non-package resource is copied byte-for-byte.  The only permitted payload
    change is the first ``dc:title`` in the OPF, which receives ``title_suffix``.
    """

    source_path = Path(source_epub).expanduser().resolve()
    output_path = Path(output_epub).expanduser().resolve()
    if source_path == output_path:
        raise ValueError("Audiobook companion output must differ from its source EPUB")
    if source_path.suffix.casefold() != ".epub" or not source_path.is_file():
        raise ValueError(f"Structured audiobook source is not an EPUB: {source_path}")
    if output_path.suffix.casefold() != ".epub":
        raise ValueError(f"Structured audiobook output must be an EPUB: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        prefix=f".{output_path.stem}-",
        suffix=".epub",
        dir=output_path.parent,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)

    try:
        title = _clone_epub_with_title_suffix(
            source_path,
            temporary_path,
            title_suffix=title_suffix,
        )
        report = _validate_structured_clone(
            source_path,
            temporary_path,
            reported_output=output_path,
            title=title,
        )
        if not report.publishable:
            raise AudiobookEpubPreservationError("; ".join(report.errors))
        temporary_path.replace(output_path)
        return report
    finally:
        temporary_path.unlink(missing_ok=True)


def _clone_epub_with_title_suffix(
    source_path: Path,
    output_path: Path,
    *,
    title_suffix: str,
) -> str:
    with ZipFile(source_path, "r") as source:
        names = source.namelist()
        if "mimetype" not in names or _CONTAINER_PATH not in names:
            raise AudiobookEpubPreservationError(
                "The translated EPUB lacks mimetype or META-INF/container.xml"
            )
        opf_path = _opf_path(source)
        package_payload, title = _package_with_title_suffix(
            source.read(opf_path),
            title_suffix=title_suffix,
        )
        infos = {info.filename: info for info in source.infolist()}
        ordered_names = ["mimetype", *(name for name in names if name != "mimetype")]

        with ZipFile(output_path, "w") as destination:
            destination.comment = source.comment
            for name in ordered_names:
                info = infos[name]
                clone = _clone_zip_info(info)
                if name == "mimetype":
                    clone.compress_type = ZIP_STORED
                if name == opf_path:
                    destination.writestr(clone, package_payload)
                    continue
                with source.open(info, "r") as source_member:
                    with destination.open(clone, "w") as output_member:
                        shutil.copyfileobj(source_member, output_member, length=1024 * 1024)
    return title


def _opf_path(archive: ZipFile) -> str:
    root = etree.fromstring(archive.read(_CONTAINER_PATH), parser=_XML_PARSER)
    paths = root.xpath("//*[local-name()='rootfile']/@full-path")
    if not paths:
        raise AudiobookEpubPreservationError("EPUB container does not declare an OPF package")
    opf_path = posixpath.normpath(str(paths[0]).replace("\\", "/"))
    if opf_path.startswith("../") or opf_path not in archive.namelist():
        raise AudiobookEpubPreservationError(f"Invalid OPF package path: {opf_path}")
    return opf_path


def _package_with_title_suffix(payload: bytes, *, title_suffix: str) -> tuple[bytes, str]:
    root = etree.fromstring(payload, parser=_XML_PARSER)
    titles = root.xpath(
        "//*[local-name()='metadata']/*[local-name()='title']"
    )
    if not titles:
        raise AudiobookEpubPreservationError("EPUB package does not contain dc:title")
    base_title = " ".join((titles[0].text or "").split()) or "Libro"
    suffix = " ".join(str(title_suffix or "").split()).strip(" ()")
    rendered_suffix = f"({suffix})" if suffix else ""
    if rendered_suffix and rendered_suffix.casefold() not in base_title.casefold():
        titles[0].text = f"{base_title} {rendered_suffix}"
    title = " ".join((titles[0].text or base_title).split())
    return (
        etree.tostring(root, encoding="utf-8", xml_declaration=True),
        title,
    )


def _clone_zip_info(info: ZipInfo) -> ZipInfo:
    clone = ZipInfo(info.filename, date_time=info.date_time)
    clone.comment = info.comment
    clone.extra = info.extra
    clone.internal_attr = info.internal_attr
    clone.external_attr = info.external_attr
    clone.create_system = info.create_system
    clone.flag_bits = info.flag_bits
    clone.compress_type = info.compress_type
    return clone


def _validate_structured_clone(
    source_path: Path,
    candidate_path: Path,
    *,
    reported_output: Path,
    title: str,
) -> StructuredAudiobookEpubReport:
    source = snapshot_epub(source_path, recover=False)
    output = snapshot_epub(candidate_path, recover=False)
    source_placements = _image_placements(source_path)
    output_placements = _image_placements(candidate_path)
    xhtml_suffixes = {".htm", ".html", ".xhtml"}
    source_xhtml_hashes = _archive_hashes(source_path, suffixes=xhtml_suffixes)
    output_xhtml_hashes = _archive_hashes(candidate_path, suffixes=xhtml_suffixes)
    image_hashes_match = all(
        output.resource_hashes.get(name) == source.resource_hashes.get(name)
        for name in source.image_files
    )
    report = StructuredAudiobookEpubReport(
        source_path=str(source_path),
        output_path=str(reported_output),
        image_count=len(source.image_files),
        image_references=sum(len(values) for values in source.image_refs.values()),
        image_placements=len(source_placements),
        captions_preserved=sum(1 for placement in source_placements if placement.caption),
        cover_preserved=(
            output.cover_image == source.cover_image
            and output.cover_page == source.cover_page
            and bool(output.cover_image or not source.obvious_cover_image)
        ),
        spine_preserved=output.spine == source.spine,
        xhtml_preserved=output_xhtml_hashes == source_xhtml_hashes,
        mimetype_valid=output.mimetype_first_stored and output.mimetype_exact,
        title=title,
    )

    if set(output.entry_names) != set(source.entry_names):
        report.errors.append("archive resource set changed")
    if output.image_files != source.image_files or not image_hashes_match:
        report.errors.append("image resources changed or disappeared")
    if output.image_refs != source.image_refs:
        report.errors.append("image references changed")
    if output_placements != source_placements:
        report.errors.append("image position, alternative text, or caption changed")
    if not report.cover_preserved:
        report.errors.append("cover declaration or cover page was not preserved")
    if not report.spine_preserved:
        report.errors.append("spine order changed")
    if not report.xhtml_preserved:
        report.errors.append("XHTML content changed")
    if not report.mimetype_valid:
        report.errors.append("EPUB mimetype is not first, exact, and uncompressed")
    return report


def _image_placements(epub_path: Path) -> tuple[_ImagePlacement, ...]:
    placements: list[_ImagePlacement] = []
    with ZipFile(epub_path, "r") as archive:
        for name in archive.namelist():
            if PurePosixPath(name).suffix.casefold() not in {".htm", ".html", ".xhtml"}:
                continue
            root = etree.fromstring(archive.read(name), parser=_XML_PARSER)
            tree = root.getroottree()
            for image in root.xpath(
                "//*[local-name()='img' or local-name()='image' or local-name()='object']"
            ):
                reference = (
                    image.get("src")
                    or image.get("data")
                    or image.get("href")
                    or image.get("{http://www.w3.org/1999/xlink}href")
                    or ""
                )
                if not reference:
                    continue
                suffix = PurePosixPath(reference.split("#", 1)[0]).suffix.casefold()
                if image.tag.rsplit("}", 1)[-1] == "object" and suffix not in _IMAGE_SUFFIXES:
                    continue
                placements.append(
                    _ImagePlacement(
                        document=name,
                        dom_path=tree.getpath(image),
                        reference=reference,
                        alt_text=" ".join((image.get("alt") or "").split()),
                        caption=_nearby_caption(image),
                    )
                )
    return tuple(placements)


def _archive_hashes(epub_path: Path, *, suffixes: set[str]) -> dict[str, str]:
    with ZipFile(epub_path, "r") as archive:
        return {
            name: hashlib.sha256(archive.read(name)).hexdigest()
            for name in archive.namelist()
            if PurePosixPath(name).suffix.casefold() in suffixes
        }


def _nearby_caption(image: etree._Element) -> str:
    figure = next(
        (
            ancestor
            for ancestor in image.iterancestors()
            if ancestor.tag.rsplit("}", 1)[-1].casefold() == "figure"
        ),
        None,
    )
    if figure is not None:
        captions = figure.xpath(".//*[local-name()='figcaption']")
        if captions:
            return _node_text(captions[0])

    parent = image.getparent()
    if parent is None:
        return ""
    for sibling in (parent.getprevious(), parent.getnext()):
        if sibling is None:
            continue
        tag = sibling.tag.rsplit("}", 1)[-1].casefold()
        classes = str(sibling.get("class") or "")
        if tag == "figcaption" or _CAPTION_CLASS_RE.search(classes):
            return _node_text(sibling)
    return ""


def _node_text(node: etree._Element) -> str:
    return " ".join("".join(node.itertext()).split())
