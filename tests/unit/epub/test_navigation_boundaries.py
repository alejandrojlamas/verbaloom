"""Only exact application-generated cover landmarks may extend navigation."""

import zipfile

import pytest

from src.core.epub.dom_boundaries import audit_epub_dom_boundaries

TOC = '<nav epub:type="toc"><h1>Contents</h1><ol><li><a href="one.xhtml">One</a></li></ol></nav>'
LANDMARKS = (
    '<nav epub:type="landmarks" hidden="hidden"><h2>Guide</h2><ol>'
    '<li><a epub:type="cover" href="cover.xhtml">Cover</a></li></ol></nav>'
)


def audit(tmp_path, source_body, output_body, *, declared=True, language="en"):
    for filename, body in (("source.epub", source_body), ("output.epub", output_body)):
        with zipfile.ZipFile(tmp_path / filename, "w") as archive:
            archive.writestr(
                "Text/nav.xhtml",
                (
                    '<html xmlns="http://www.w3.org/1999/xhtml" '
                    'xmlns:epub="http://www.idpf.org/2007/ops" '
                    f'lang="{language}"><body>{body}</body></html>'
                ),
            )
    return audit_epub_dom_boundaries(
        tmp_path / "source.epub",
        tmp_path / "output.epub",
        generated_cover_landmarks={"Text/nav.xhtml": "Text/cover.xhtml"}
        if declared
        else {},
    )


@pytest.mark.parametrize("language", ["en", "es"])
def test_exact_added_landmarks_preserve_boundary_validation(tmp_path, language):
    landmarks = LANDMARKS
    if language == "es":
        landmarks = landmarks.replace("Guide", "Gu\u00eda").replace("Cover", "Portada")
    report = audit(tmp_path, TOC, TOC + landmarks, language=language)
    assert report.clean
    assert report.allowed_element_additions == {
        "Text/nav.xhtml": {"nav": 1, "h2": 1, "ol": 1, "li": 1, "a": 1}
    }


@pytest.mark.parametrize(
    "mutation",
    [
        "undeclared",
        "visible",
        "wrong_target",
        "wrong_label",
        "extra_text",
        "wrong_namespace",
        "existing",
        "nested",
        "duplicate",
        "missing_toc",
        "tail",
    ],
)
def test_landmark_allowance_does_not_hide_damage(tmp_path, mutation):
    source, output = TOC, TOC + LANDMARKS
    if mutation == "visible":
        output = output.replace(' hidden="hidden"', "")
    elif mutation == "wrong_target":
        output = output.replace('href="cover.xhtml"', 'href="other.xhtml"')
    elif mutation == "wrong_label":
        output = output.replace("Cover</a>", "Invented story</a>")
    elif mutation == "extra_text":
        output = output.replace("<h2>", "<p>Invented story.</p><h2>")
    elif mutation == "wrong_namespace":
        output = output.replace(
            '<nav epub:type="landmarks"',
            '<nav xmlns="urn:foreign" epub:type="landmarks"',
        )
    elif mutation == "existing":
        source += LANDMARKS
        output = TOC
    elif mutation == "nested":
        output = TOC + "<section>" + LANDMARKS + "</section>"
    elif mutation == "duplicate":
        output += LANDMARKS
    elif mutation == "missing_toc":
        output = LANDMARKS
    elif mutation == "tail":
        output += "Invented story."
    report = audit(tmp_path, source, output, declared=mutation != "undeclared")
    assert not report.clean


def test_original_navigation_text_cannot_be_emptied(tmp_path):
    output = TOC.replace("Contents", "").replace(">One</a>", "></a>") + LANDMARKS
    assert not audit(tmp_path, TOC, output).clean
