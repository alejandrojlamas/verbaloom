import hashlib
import json
from pathlib import Path
import zipfile

from lxml import etree

from src.api.handlers import _create_audiobook_companion_outputs
from src.core.epub.audiobook_companion import create_structured_audiobook_epub


def _write_illustrated_epub(path: Path) -> None:
    container = """<?xml version="1.0" encoding="utf-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>"""
    package = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="book-id">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="book-id">urn:test:illustrated</dc:identifier>
    <dc:title>Libro ilustrado</dc:title>
    <dc:language>es</dc:language>
    <meta name="cover" content="cover-image"/>
  </metadata>
  <manifest>
    <item id="cover-image" href="images/cover.jpg" media-type="image/jpeg" properties="cover-image"/>
    <item id="cover" href="cover.xhtml" media-type="application/xhtml+xml"/>
    <item id="chapter" href="chapter.xhtml" media-type="application/xhtml+xml"/>
    <item id="illustration" href="images/scene.jpg" media-type="image/jpeg"/>
  </manifest>
  <spine>
    <itemref idref="cover"/>
    <itemref idref="chapter"/>
  </spine>
  <guide>
    <reference type="cover" title="Portada" href="cover.xhtml"/>
  </guide>
</package>"""
    cover = """<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" lang="es" xml:lang="es">
  <head><title>Portada</title></head>
  <body><div class="cover"><img src="images/cover.jpg" alt="Portada"/></div></body>
</html>"""
    chapter = """<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" lang="es" xml:lang="es">
  <head><title>Capítulo uno</title></head>
  <body>
    <h1>Capítulo uno</h1>
    <p>Texto traducido antes de la imagen.</p>
    <figure>
      <img src="images/scene.jpg" alt="Escena de la película"/>
      <figcaption>La protagonista observa la ciudad desde el tren.</figcaption>
    </figure>
    <p>Texto traducido después de la imagen.</p>
  </body>
</html>"""
    with zipfile.ZipFile(path, "w") as archive:
        mimetype = zipfile.ZipInfo("mimetype")
        mimetype.compress_type = zipfile.ZIP_STORED
        archive.writestr(mimetype, "application/epub+zip")
        archive.writestr(_zip_info("META-INF/container.xml"), container)
        archive.writestr(_zip_info("OEBPS/content.opf"), package)
        archive.writestr(_zip_info("OEBPS/cover.xhtml"), cover)
        archive.writestr(_zip_info("OEBPS/chapter.xhtml"), chapter)
        archive.writestr(_zip_info("OEBPS/images/cover.jpg"), b"cover-image-bytes")
        archive.writestr(_zip_info("OEBPS/images/scene.jpg"), b"scene-image-bytes")


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name)
    info.compress_type = zipfile.ZIP_DEFLATED
    return info


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_structured_audiobook_epub_preserves_cover_images_positions_and_caption(tmp_path):
    source = tmp_path / "source.epub"
    output = tmp_path / "output.epub"
    _write_illustrated_epub(source)

    report = create_structured_audiobook_epub(source, output)

    assert report.publishable is True
    assert report.image_count == 2
    assert report.image_references == 2
    assert report.image_placements == 2
    assert report.captions_preserved == 1
    assert report.cover_preserved is True
    assert report.spine_preserved is True
    assert report.xhtml_preserved is True
    assert report.mimetype_valid is True

    with zipfile.ZipFile(source) as original, zipfile.ZipFile(output) as companion:
        for name in (
            "OEBPS/images/cover.jpg",
            "OEBPS/images/scene.jpg",
            "OEBPS/cover.xhtml",
            "OEBPS/chapter.xhtml",
        ):
            assert _sha(companion.read(name)) == _sha(original.read(name))
        package = etree.fromstring(companion.read("OEBPS/content.opf"))
        assert package.xpath(
            "string(//*[local-name()='metadata']/*[local-name()='title'][1])"
        ) == "Libro ilustrado (Audiolibro)"
        assert companion.infolist()[0].filename == "mimetype"
        assert companion.infolist()[0].compress_type == zipfile.ZIP_STORED


def test_job_companion_writer_uses_structured_epub_instead_of_flattening(tmp_path):
    source = tmp_path / "Libro (Spanish).epub"
    _write_illustrated_epub(source)

    result = _create_audiobook_companion_outputs(
        output_path=str(source),
        output_format="epub",
        target_language="Spanish",
        output_dir=str(tmp_path),
    )

    epub_name = next(name for name in result["files"] if name.endswith(".epub"))
    report_name = next(name for name in result["files"] if name.endswith("report).json"))
    with zipfile.ZipFile(tmp_path / epub_name) as companion:
        assert companion.read("OEBPS/images/cover.jpg") == b"cover-image-bytes"
        assert companion.read("OEBPS/images/scene.jpg") == b"scene-image-bytes"
        chapter = companion.read("OEBPS/chapter.xhtml").decode("utf-8")
        assert "<figcaption>La protagonista observa la ciudad desde el tren.</figcaption>" in chapter
        assert chapter.index("Texto traducido antes") < chapter.index("images/scene.jpg")
        assert chapter.index("images/scene.jpg") < chapter.index("Texto traducido después")

    payload = json.loads((tmp_path / report_name).read_text(encoding="utf-8"))
    assert payload["structured_epub_preserved"] is True
    assert payload["images_preserved"] == 2
    assert payload["image_placements_preserved"] == 2
    assert payload["visual_captions_preserved"] == 1
    assert payload["cover_preserved"] is True
    assert payload["spine_preserved"] is True
    assert payload["xhtml_preserved"] is True
    assert payload["epub_mimetype_valid"] is True
