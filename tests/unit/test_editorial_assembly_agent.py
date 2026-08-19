import zipfile

from src.core.editorial_assembly import EditorialAssemblyAgent
from src.core.output_formats import convert_output_file


def test_agent_removes_front_ocr_junk_and_isolates_toc_entries():
    text = """fYfWITÍlfSS

A-------------------------

MU

Los libros de Avon estan disponibles para compras al por mayor.

EYfWITDfSS

mm

EDITADO POR JOHN CAREY

OPM 10 9

Indice

Edward Grim

Sacrificios Humanos entre los Aztecas, c. 15x0 86 Jose de Acosta

Desembarco en Nueva Inglaterra, noviembre i6ro 174 William Bradford

Introduccion

Antes de editar un libro de reportaje, hay que decidir que es el reportaje.

Atrapados en el hielo artico, 1596. La odisea de los marineros holandeses empieza aqui."""

    plan = EditorialAssemblyAgent().assemble(text, "Eyewitness")
    titles = [section.title for section in plan.sections]
    full_text_parts = []
    for section in plan.sections:
        full_text_parts.extend([section.title, *section.paragraphs])
    full_text = "\n".join(full_text_parts)

    assert any("fYfWIT" in item for item in plan.removed_artifacts)
    assert "mm" not in full_text
    assert "OPM 10 9" not in full_text
    assert "Inicio" in titles
    assert "Indice" in titles
    assert "Introduccion" in titles
    assert "Atrapados en el hielo artico, 1596" in titles
    assert "Sacrificios Humanos entre los Aztecas, c. 1520" not in titles
    assert plan.toc_entries_detected >= 3


def test_text_to_epub_uses_agent_for_clean_nav_and_body_preservation(tmp_path):
    source = tmp_path / "source.txt"
    destination = tmp_path / "clean.epub"
    source.write_text(
        """fYfWITÍlfSS

A-------------------------

MU

Los libros de Avon estan disponibles para compras al por mayor.

mm

Indice

Edward Grim

Sacrificios Humanos entre los Aztecas, c. 15x0 86 Jose de Acosta

Introduccion

Antes de editar un libro de reportaje, hay que decidir que es el reportaje.

Atrapados en el hielo artico, 1596. La odisea de los marineros holandeses empieza aqui.""",
        encoding="utf-8",
    )

    convert_output_file(source, destination, "epub")

    with zipfile.ZipFile(destination) as zf:
        nav = zf.read("OEBPS/nav.xhtml").decode("utf-8")
        first_chapter = zf.read("OEBPS/chap-001.xhtml").decode("utf-8")

    assert "fYfWIT" not in nav
    assert "mm" not in nav
    assert "OPM" not in nav
    assert "Sacrificios Humanos entre los Aztecas" not in nav
    assert "Introduccion" in nav
    assert "Atrapados en el hielo artico, 1596" in nav
    assert "Los libros de Avon" in first_chapter


def test_sentence_with_date_is_demoted_not_used_as_heading(tmp_path):
    source = tmp_path / "source.txt"
    destination = tmp_path / "clean.epub"
    source.write_text(
        "Atrapados en el hielo artico, 1596. Cuerpo inicial.\n\n"
        "La primera entrega consistiria en la Santa Cruz, 100,000 dinares y "
        "1,600 prisioneros. Los hombres enviados revisaron todo completo.",
        encoding="utf-8",
    )

    convert_output_file(source, destination, "epub")

    with zipfile.ZipFile(destination) as zf:
        nav = zf.read("OEBPS/nav.xhtml").decode("utf-8")
        chapter = zf.read("OEBPS/chap-001.xhtml").decode("utf-8")

    assert "Atrapados en el hielo artico, 1596" in nav
    assert "La primera entrega consistiria" not in nav
    assert "La primera entrega consistiria" in chapter


def test_declared_heading_on_first_line_does_not_absorb_body_text():
    text = (
        "Parte 5\n"
        "Jennie queria dormir conmigo, pero le dije que descansaria mejor sola.\n\n"
        "El resto del capitulo continua aqui."
    )

    plan = EditorialAssemblyAgent().assemble(text, "Libro")

    assert [section.title for section in plan.sections] == ["Parte 5"]
    assert plan.sections[0].paragraphs[0].startswith("Jennie queria dormir conmigo")
