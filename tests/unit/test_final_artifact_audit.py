from pathlib import Path
import zipfile

import yaml
from lxml import etree

from src.core.book_profiles import create_profile

from src.core.final_artifact_audit import (
    _clean_inline_text,
    _is_sentence_initial_position,
    _reading_structure_findings,
    audit_and_clean_final_artifact,
    final_artifact_report_path,
)


XHTML = "http://www.w3.org/1999/xhtml"


def _xhtml(body: str, title: str = "10") -> str:
    return f"""<?xml version="1.0" encoding="utf-8"?>
<html xmlns="{XHTML}" lang="es">
  <head><title>{title}</title></head>
  <body>{body}</body>
</html>
"""


def _write_epub(
    path: Path,
    chapters: list[tuple[str, str, str]],
    nav_labels: list[str],
    *,
    book_title: str = "Test Book",
) -> None:
    manifest_items = []
    spine_items = []
    nav_items = []
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")
        zf.writestr(
            "META-INF/container.xml",
            """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles>
</container>
""",
        )
        for idx, (href, title, body) in enumerate(chapters, start=1):
            item_id = f"chap{idx}"
            manifest_items.append(
                f'<item id="{item_id}" href="{href}" media-type="application/xhtml+xml"/>'
            )
            spine_items.append(f'<itemref idref="{item_id}"/>')
            nav_label = nav_labels[idx - 1]
            nav_items.append(f'<li><a href="{href}">{nav_label}</a></li>')
            zf.writestr(f"OEBPS/{href}", _xhtml(body, title=title))
        zf.writestr(
            "OEBPS/nav.xhtml",
            _xhtml(
                '<nav epub:type="toc" xmlns:epub="http://www.idpf.org/2007/ops"><ol>'
                + "".join(nav_items)
                + "</ol></nav>",
                title=book_title,
            ),
        )
        zf.writestr(
            "OEBPS/content.opf",
            f"""<?xml version="1.0" encoding="utf-8"?>
<package version="3.0" xmlns="http://www.idpf.org/2007/opf" xmlns:dc="http://purl.org/dc/elements/1.1/">
  <metadata><dc:title>{book_title}</dc:title><dc:language>es</dc:language></metadata>
  <manifest><item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>{''.join(manifest_items)}</manifest>
  <spine>{''.join(spine_items)}</spine>
</package>
""",
        )


def _epub_text(path: Path, name: str) -> str:
    with zipfile.ZipFile(path) as zf:
        return zf.read(name).decode("utf-8")


def test_final_artifact_audit_cleans_epub_source_links_and_repeated_page_toc(tmp_path):
    epub = tmp_path / "book.epub"
    chapters = []
    labels = []
    page_labels = ["10", "11", "12", "10", "11", "12", "10", "11"]
    for idx, label in enumerate(page_labels, start=1):
        chapters.append(
            (
                f"chap-{idx:03d}.xhtml",
                label,
                "<p>Texto antes.</p>"
                "<p>[OceanofPDF.com](https://oceanofpdf.com)</p>"
                "<p>12.</p>"
                "<p>Texto despues con [nota](../Text/notas.xhtml#nt23).</p>",
            )
        )
        labels.append(label)
    _write_epub(epub, chapters, labels, book_title="_OceanofPDF.com_Test Book")

    report = audit_and_clean_final_artifact(epub, output_format="epub")

    assert report.changed
    assert report.source_artifacts_removed == 16
    assert report.numeric_toc_labels_rewritten == 8
    assert report.numeric_titles_rewritten == 8
    assert final_artifact_report_path(epub).exists()

    all_text = "\n".join(
        _epub_text(epub, name)
        for name in ("OEBPS/nav.xhtml", "OEBPS/content.opf", "OEBPS/chap-001.xhtml")
    )
    assert "OceanofPDF" not in all_text
    assert "https://oceanofpdf.com" not in all_text
    assert "notas.xhtml" not in all_text
    assert ">12.<" not in all_text
    assert "Seccion 1" in all_text
    chapter_root = etree.fromstring(
        _epub_text(epub, "OEBPS/chap-001.xhtml").encode("utf-8")
    )
    paragraphs = chapter_root.xpath("//*[local-name()='p']")
    assert len(paragraphs) == 4
    assert sum(
        "verbaloom-sanitized-artifact" in str(node.get("class") or "").split()
        for node in paragraphs
    ) == 2


def test_final_artifact_audit_keeps_unique_numeric_nav_labels(tmp_path):
    epub = tmp_path / "numbered-sections.epub"
    chapters = [
        (f"chap-{idx:03d}.xhtml", str(idx), f"<p>Seccion real {idx}</p>")
        for idx in range(1, 9)
    ]
    _write_epub(epub, chapters, [str(idx) for idx in range(1, 9)])

    report = audit_and_clean_final_artifact(epub, output_format="epub")

    assert not report.changed
    assert report.numeric_toc_labels_rewritten == 0
    assert report.numeric_titles_rewritten == 0
    nav = _epub_text(epub, "OEBPS/nav.xhtml")
    assert ">1<" in nav
    assert "Seccion 1" not in nav


def test_final_artifact_audit_cleans_txt_source_links(tmp_path):
    path = tmp_path / "book.txt"
    path.write_text(
        "Texto real.\n\n[OceanofPDF.com]\n(https://oceanofpdf.com)\n\n13.\n\nSigue.",
        encoding="utf-8",
    )

    report = audit_and_clean_final_artifact(path, output_format="txt")

    assert report.changed
    assert path.read_text(encoding="utf-8") == "Texto real.\n\nSigue."


def test_final_artifact_audit_preserves_legitimate_epub_url(tmp_path):
    epub = tmp_path / "book.epub"
    _write_epub(
        epub,
        [
            (
                "chap-001.xhtml",
                "Material complementario",
                '<p>Visite: <span class="inlineurl">'
                '<a href="https://www.wiley.com/college/block">'
                "www.wiley.com/college/block</a></span></p>",
            )
        ],
        ["Material complementario"],
    )

    report = audit_and_clean_final_artifact(epub, output_format="epub")

    chapter = _epub_text(epub, "OEBPS/chap-001.xhtml")
    assert report.source_artifacts_removed == 0
    assert "verbaloom-sanitized-artifact" not in chapter
    assert 'href="https://www.wiley.com/college/block"' in chapter
    assert ">www.wiley.com/college/block<" in chapter


def test_final_artifact_audit_cleans_epub_llm_protocol_leaks(tmp_path):
    epub = tmp_path / "book.epub"
    _write_epub(
        epub,
        [
            (
                "chap-001.xhtml",
                "Capitulo",
                "<p>Texto antes. &lt;/TRANSLATIONATION&gt; Texto despues.</p>",
            )
        ],
        ["Capitulo"],
    )

    report = audit_and_clean_final_artifact(epub, output_format="epub")

    assert report.changed
    assert report.text_rewrites >= 1
    chapter = _epub_text(epub, "OEBPS/chap-001.xhtml")
    assert "TRANSLATIONATION" not in chapter
    assert "Texto antes." in chapter
    assert "Texto despues." in chapter


def test_final_artifact_audit_localizes_note_backlinks_without_breaking_href(tmp_path):
    epub = tmp_path / "book.epub"
    _write_epub(
        epub,
        [
            (
                "chap-001.xhtml",
                "Notas",
                '<p><a href="chap-002.xhtml#note-1">'
                "GO TO NOTE REFERENCE IN TEXT</a></p>",
            ),
            ("chap-002.xhtml", "Texto", '<p id="note-1">Texto principal.</p>'),
        ],
        ["Notas", "Texto"],
    )

    report = audit_and_clean_final_artifact(
        epub,
        output_format="epub",
        target_language="Spanish",
    )

    assert report.changed
    chapter = _epub_text(epub, "OEBPS/chap-001.xhtml")
    assert "GO TO NOTE REFERENCE IN TEXT" not in chapter
    assert "IR A LA NOTA EN EL TEXTO" in chapter
    assert 'href="chap-002.xhtml#note-1"' in chapter


def test_final_artifact_audit_restores_source_proven_symbols_inside_names(tmp_path):
    source = tmp_path / "source.epub"
    output = tmp_path / "output.epub"
    _write_epub(
        source,
        [("chap-001.xhtml", "Capitulo", "<p>Ch*Tril llamó a Ek*Tiq.</p>")],
        ["Capitulo"],
    )
    _write_epub(
        output,
        [("chap-001.xhtml", "Capitulo", "<p>ChTril llamó a EkTiq.</p>")],
        ["Capitulo"],
    )

    report = audit_and_clean_final_artifact(
        output,
        output_format="epub",
        source_epub_path=source,
    )

    assert report.symbol_names_restored == 2
    chapter = _epub_text(output, "OEBPS/chap-001.xhtml")
    assert "Ch*Tril" in chapter
    assert "Ek*Tiq" in chapter


def test_final_artifact_audit_reflows_source_proven_page_split_paragraphs(tmp_path):
    source = tmp_path / "source.epub"
    output = tmp_path / "output.epub"
    _write_epub(
        source,
        [(
            "chap-001.xhtml",
            "Chapter",
            "<p>The narrator continued describing the long afternoon beside the harbor and the people who were still</p>"
            "<p>waiting there when the fishing boats finally returned through the fog.</p>"
            "<p>remem-</p><p>bering that journey changed them forever.</p>",
        )],
        ["Chapter"],
    )
    _write_epub(
        output,
        [(
            "chap-001.xhtml",
            "Capitulo",
            "<p>El narrador siguió describiendo la larga tarde junto al puerto y a las personas que todavía estaban</p>"
            "<p>esperando allí cuando los barcos pesqueros regresaron por fin entre la niebla.</p>"
            "<p>recor-</p><p>dar aquel viaje los cambió para siempre.</p>",
        )],
        ["Capitulo"],
    )

    first = audit_and_clean_final_artifact(
        output,
        output_format="epub",
        source_epub_path=source,
        write_report=False,
    )
    second = audit_and_clean_final_artifact(
        output,
        output_format="epub",
        source_epub_path=source,
        write_report=False,
    )

    chapter = _epub_text(output, "OEBPS/chap-001.xhtml")
    assert first.paragraph_continuations_reflowed == 2
    assert first.paragraph_continuations_dehyphenated == 1
    assert second.paragraph_continuations_reflowed == 0
    assert "todavía estaban esperando allí" in chapter
    assert "recordar aquel viaje" in chapter
    assert "verbaloom-merged-continuation" in chapter


def test_final_artifact_audit_applies_only_approved_exact_profile_terms(
    tmp_path,
    monkeypatch,
):
    profiles_root = tmp_path / "profiles"
    monkeypatch.setenv("BOOK_PROFILES_DIR", str(profiles_root))
    profile_dir = create_profile("book_terms", profiles_root=profiles_root)
    (profile_dir / "glossary" / "terms.yml").write_text(
        yaml.safe_dump({
            "entries": [{
                "source": "Source-Mind",
                "target": "Mente de Origen",
                "type": "concept",
                "status": "approved",
                "translation_policy": "translate_exact",
                "injection_policy": "translate_exact",
            }]
        }, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    epub = tmp_path / "book.epub"
    _write_epub(
        epub,
        [("chap-001.xhtml", "Capitulo", "<p>La Source-Mind respondió.</p>")],
        ["Capitulo"],
    )

    report = audit_and_clean_final_artifact(
        epub,
        output_format="epub",
        prompt_options={"editorial_mode": "book_profile", "profile_id": "book_terms"},
    )

    assert report.exact_glossary_terms_repaired == 1
    assert "Mente de Origen" in _epub_text(epub, "OEBPS/chap-001.xhtml")


def test_final_artifact_audit_does_not_rewrite_terms_inside_work_title_markup(
    tmp_path,
    monkeypatch,
):
    profiles_root = tmp_path / "profiles"
    monkeypatch.setenv("BOOK_PROFILES_DIR", str(profiles_root))
    profile_dir = create_profile("title_context_terms", profiles_root=profiles_root)
    (profile_dir / "glossary" / "terms.yml").write_text(
        yaml.safe_dump({
            "entries": [{
                "source": "Russian",
                "target": "ruso",
                "type": "term",
                "status": "approved",
                "translation_policy": "translate_exact",
                "injection_policy": "translate_exact",
            }]
        }, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    epub = tmp_path / "book.epub"
    _write_epub(
        epub,
        [(
            "chap-001.xhtml",
            "Capitulo",
            (
                "<p><cite>The Film Factory: Russian and Soviet Cinema in Documents</cite></p>"
                "<p>La experiencia Russian cambió el cine.</p>"
            ),
        )],
        ["Capitulo"],
    )

    report = audit_and_clean_final_artifact(
        epub,
        output_format="epub",
        prompt_options={
            "editorial_mode": "book_profile",
            "profile_id": "title_context_terms",
        },
    )
    chapter = _epub_text(epub, "OEBPS/chap-001.xhtml")

    assert report.exact_glossary_terms_repaired == 1
    assert "The Film Factory: Russian and Soviet Cinema in Documents" in chapter
    assert "La experiencia ruso cambió el cine." in chapter


def test_final_artifact_audit_cleans_repeated_image_description_labels_in_txt(tmp_path):
    path = tmp_path / "book.txt"
    path.write_text(
        "Descripcion de imagen: Descripción de imagen: La lámpara seguía encendida.",
        encoding="utf-8",
    )

    report = audit_and_clean_final_artifact(path, output_format="txt")

    assert report.changed
    assert report.reader_artifact_labels_removed == 2
    assert "image-description" not in "\n".join(report.reading_quality_warnings)
    assert path.read_text(encoding="utf-8") == "La lámpara seguía encendida."


def test_final_artifact_audit_cleans_repeated_image_description_labels_in_epub(tmp_path):
    epub = tmp_path / "book.epub"
    _write_epub(
        epub,
        [
            (
                "chap-001.xhtml",
                "Capitulo",
                "<p>Descripcion de imagen: Descripción de imagen: La lámpara seguía encendida.</p>",
            )
        ],
        ["Capitulo"],
    )

    report = audit_and_clean_final_artifact(epub, output_format="epub")

    assert report.changed
    chapter = _epub_text(epub, "OEBPS/chap-001.xhtml")
    assert "Descripcion de imagen" not in chapter
    assert "Descripción de imagen" not in chapter
    assert "La lámpara seguía encendida." in chapter


def test_final_artifact_audit_warns_on_mixed_dialogue_and_tts_punctuation(tmp_path):
    path = tmp_path / "book.txt"
    path.write_text(
        "\n".join([
            "—No puedo ir.",
            "—Volveré mañana.",
            "—Anna guardó silencio.",
            '"No puedo ir."',
            '"Volveré mañana."',
            '"Anna guardó silencio."',
            "—Monsieur, ya casi estamos... —",
            "Su sonrisa—. Nací aquí, madame.—",
            'Final."."',
        ]),
        encoding="utf-8",
    )

    report = audit_and_clean_final_artifact(path, output_format="txt")
    warnings = "\n".join(report.reading_quality_warnings)

    assert "Mixed dialogue marker conventions detected" in warnings
    assert "TTS-hostile punctuation residue detected" in warnings


def test_final_artifact_audit_reports_reading_structure_findings(tmp_path):
    path = tmp_path / "book.txt"
    path.write_text(
        "\n".join([
            "Capítulo I",
            "México recibió la noticia y Mexico volvió a aparecer sin acento.",
            "Consulta https://example.com/source para revisar el archivo externo.",
            "[referencia](https://example.com/source)",
            "Quedó una referencia plana (../Text/notas.xhtml#nt23) dentro del texto.",
            "[[23]](../Text/notas.xhtml#nt23)",
            "12",
            "13",
            "14",
            "Indice ........ 45",
            "Texto final para conservar.",
        ]),
        encoding="utf-8",
    )

    report = audit_and_clean_final_artifact(path, output_format="txt")
    markdown = report.to_markdown()

    assert report.link_artifact_findings == 1
    assert report.footnote_link_findings == 1
    assert report.pagination_artifact_findings >= 3
    assert report.chapter_heading_count >= 1
    assert report.possible_name_variant_groups >= 1
    assert "Links visibles sospechosos" in markdown
    assert "Grupos posibles de variantes de nombre" in markdown


def test_reading_structure_counts_each_link_once():
    findings = _reading_structure_findings("\n".join([
        "Consulta https://example.com/source.",
        "[referencia](https://example.com/source)",
        "(../Text/notas.xhtml#nt23)",
        "[[23]](../Text/notas.xhtml#nt23)",
    ]))

    assert findings["link_artifact_findings"] == 2
    assert findings["footnote_link_findings"] == 2


def test_final_artifact_audit_does_not_treat_sentence_words_as_name_variants(tmp_path):
    path = tmp_path / "book.txt"
    path.write_text(
        "Como empezó la mañana. Cómo terminó la tarde. Esta fue la duda. Está resuelta.",
        encoding="utf-8",
    )

    report = audit_and_clean_final_artifact(path, output_format="txt", write_report=False)

    assert report.possible_name_variant_groups == 0


def test_sentence_initial_position_uses_only_the_adjacent_boundary():
    text = "Nombre interior. “Inicio citado” sigue.\n   Nueva línea"

    assert _is_sentence_initial_position(text, 0)
    assert _is_sentence_initial_position(text, text.index("Inicio"))
    assert _is_sentence_initial_position(text, text.index("Nueva"))
    assert not _is_sentence_initial_position(text, text.index("interior"))
    assert not _is_sentence_initial_position(text, text.index("sigue"))


def test_final_artifact_audit_rejects_empty_readable_output(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")

    report = audit_and_clean_final_artifact(path, output_format="txt", write_report=False)

    assert not report.clean
    assert any("no contiene texto legible" in finding for finding in report.unresolved_findings)


def test_inline_cleanup_repairs_sentence_spacing_without_breaking_initials():
    text = "Terminó el viaje.Después volvió W.G. Sebald a St.Quentin y habló con el Sr.O'Hare."

    cleaned = _clean_inline_text(text)

    assert cleaned == (
        "Terminó el viaje. Después volvió W.G. Sebald a St. Quentin y habló con el Sr. O'Hare."
    )


def test_inline_cleanup_repairs_dates_roman_numerals_and_inline_punctuation_tail():
    assert _clean_inline_text("Terminó en 1921.Solo entonces volvió.") == (
        "Terminó en 1921. Solo entonces volvió."
    )
    assert _clean_inline_text("siglos XVIII y XIX.siglo") == "siglos XVIII y XIX. siglo"
    assert _clean_inline_text(" .Pero siguió") == ". Pero siguió"
    assert _clean_inline_text(" ?y respondió") == "? y respondió"
