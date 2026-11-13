"""Native EPUB orchestration/resources with synthetic HTTP and actual EPUBCheck."""

import asyncio
import hashlib
import io
import json
import re
import shutil
import subprocess
import zipfile
from types import SimpleNamespace

import httpx
import pytest
from lxml import etree
from PIL import Image

from src.core.adapters.translate_file import translate_file
from src.core.epub.publication_gate import audit_epub_publication
from src.core.pipeline_runtime import activate_pipeline, close_pipeline, request_role
from src.core.usage.store import TokenUsageStore
from src.persistence.checkpoint_manager import CheckpointManager
from tests.characterization.fake_llm import _extract_source_block
from tests.unit.test_benchmark_execution import CREDENTIALS
from tests.unit.test_source_audited_strategy import ASSESSMENT, config

TRANSLATIONS = {
    "Chapter One": "Cap\u00edtulo uno",
    "Chapter Two": "Cap\u00edtulo dos",
    "The traveler watched the quiet sea and remembered his childhood.": "El viajero contemplaba el mar tranquilo y recordaba su infancia.",
    "He did not open the door; he waited for Ana until midnight.": "No abri\u00f3 la puerta; esper\u00f3 a Ana hasta la medianoche.",
    "It was a cold morning.": "Era una ma\u00f1ana fr\u00eda.",
    "She counted 17 coins on the table.": "Cont\u00f3 17 monedas sobre la mesa.",
    "The map showed the road to the harbor.": "El mapa mostraba el camino al puerto.",
    "This note describes the oldest map in the archive.": "Esta nota describe el mapa m\u00e1s antiguo del archivo.",
    "Return to the passage": "Volver al pasaje",
    "The next day they returned to the quiet house.": "Al d\u00eda siguiente regresaron a la casa silenciosa.",
    "She kept the letter, but never answered it.": "Conserv\u00f3 la carta, pero nunca la contest\u00f3.",
    "Quantity": "Cantidad",
    "Price": "Precio",
    "Contents": "\u00cdndice",
    "A Journey": "Un viaje",
}
NS = {"h": "http://www.w3.org/1999/xhtml"}


def write_book(path, *, nav_in_spine=False):
    image = io.BytesIO()
    Image.new("RGB", (320, 480), "green").save(image, "PNG")

    def xhtml(title, body):
        return f"""<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" lang="en" xml:lang="en">
<head><title>{title}</title><link rel="stylesheet" type="text/css" href="style.css"/></head>
<body>{body}</body></html>"""

    resources = {
        "META-INF/container.xml": """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
<rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>""",
        "OEBPS/content.opf": """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="book-id">
<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
<dc:identifier id="book-id">urn:uuid:7574129d-0d2f-49b9-9d0f-10787adef08c</dc:identifier>
<dc:title>A Journey</dc:title><dc:language>en</dc:language><dc:creator>Fixture Author</dc:creator>
<meta property="dcterms:modified">2026-11-12T00:00:00Z</meta></metadata>
<manifest>
<item id="cover" href="cover.xhtml" media-type="application/xhtml+xml"/>
<item id="one" href="one.xhtml" media-type="application/xhtml+xml"/>
<item id="two" href="two.xhtml" media-type="application/xhtml+xml"/>
<item id="nav" href="nav.xhtml" properties="nav" media-type="application/xhtml+xml"/>
<item id="cover-image" href="cover.png" properties="cover-image" media-type="image/png"/>
<item id="map-image" href="map.png" media-type="image/png"/>
<item id="css" href="style.css" media-type="text/css"/>
</manifest><spine><itemref idref="cover"/><itemref idref="one"/><itemref idref="two"/></spine>
</package>""",
        "OEBPS/cover.xhtml": xhtml(
            "A Journey",
            '<section epub:type="cover"><img src="cover.png" alt="A Journey"/></section>',
        ),
        "OEBPS/one.xhtml": xhtml(
            "Chapter One",
            """<section epub:type="chapter">
<h1 id="ch1">Chapter One</h1>
<p id="p1">The traveler watched the quiet sea and remembered his childhood.</p>
<p id="p2">He did not open the door; he waited for Ana until midnight.</p>
<p id="p3"><strong>It was a cold morning.</strong> <em>She counted 17 coins on the table.</em></p>
<figure id="map"><img src="map.png" alt="Map"/><figcaption>The map showed the road to the harbor.</figcaption></figure>
<p id="reference"><a id="ref1" href="#note1" epub:type="noteref">1</a></p>
<aside id="note1" epub:type="footnote"><p>This note describes the oldest map in the archive.</p><a href="#ref1">Return to the passage</a></aside>
</section>""",
        ),
        "OEBPS/two.xhtml": xhtml(
            "Chapter Two",
            """<section epub:type="chapter">
<h1 id="ch2">Chapter Two</h1>
<p id="p4">The next day they returned to the quiet house.</p>
<p id="p5">She kept the letter, but never answered it.</p>
<table><thead><tr><th>Quantity</th><th>Price</th></tr></thead><tbody><tr><td>17</td><td>23.50</td></tr></tbody></table>
<pre id="formula"><code>E = mc^2</code></pre>
</section>""",
        ),
        "OEBPS/nav.xhtml": xhtml(
            "Contents",
            """<nav epub:type="toc" id="toc"><h1>Contents</h1><ol>
<li><a href="one.xhtml#ch1">Chapter One</a></li><li><a href="two.xhtml#ch2">Chapter Two</a></li>
</ol></nav>""",
        ),
        "OEBPS/style.css": "body { margin: 1em; } p { line-height: 1.5; } img { max-width: 100%; }",
        "OEBPS/cover.png": image.getvalue(),
        "OEBPS/map.png": image.getvalue(),
    }
    if nav_in_spine:
        resources["OEBPS/content.opf"] = resources["OEBPS/content.opf"].replace(
            "</spine>", '<itemref idref="nav"/></spine>'
        )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED
        )
        for name, data in resources.items():
            archive.writestr(name, data)


@pytest.fixture
def native_http(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []

    class HTTP:
        async def post(self, url, **kwargs):
            payload = kwargs["json"]
            user = payload["messages"][-1]["content"]
            system = payload["messages"][0]["content"]
            role = request_role.get()
            calls.append({"role": role, "model": payload["model"], "user": user})
            if "COMBINED_QUALITY_AUDIT_JSON" in system:
                fidelity = json.loads(ASSESSMENT.split(">", 1)[1].split("</", 1)[0])
                answer = json.dumps(
                    {
                        "editorial": {
                            "decision": "accept",
                            "confidence": 0.95,
                            "reason": "Synthetic fixture",
                            "issues": [],
                            "missing_from_source": [],
                            "added_not_in_source": [],
                            "weird_symbols": [],
                            "structure_score": 1,
                        },
                        "fidelity": fidelity,
                        "profile": {
                            "scores": {},
                            "overall_decision": "pass",
                            "issues": [],
                            "glossary_suggestions": [],
                            "summary": "Synthetic fixture",
                        },
                    }
                )
            elif role == "audit":
                answer = ASSESSMENT
            else:
                source = _extract_source_block(user)
                for original, translated in TRANSLATIONS.items():
                    source = source.replace(original, translated)
                answer = "<TRANSLATION>" + source + "</TRANSLATION>"
            return httpx.Response(
                200,
                request=httpx.Request("POST", url),
                json={
                    "choices": [
                        {"finish_reason": "stop", "message": {"content": answer}}
                    ],
                    "usage": {"prompt_tokens": 500, "completion_tokens": 200},
                    "model": payload["model"],
                    "id": "synthetic-epub-response",
                },
            )

    async def client(_):
        return HTTP()

    monkeypatch.setattr("src.core.llm.base.LLMProvider._get_client", client)
    monkeypatch.setattr(
        "src.core.llm.providers.deepseek.get_deepseek_pricing_status",
        lambda: SimpleNamespace(disabled=False),
    )
    return calls


@pytest.mark.asyncio
@pytest.mark.parametrize("nav_in_spine", [False, True])
@pytest.mark.parametrize(
    "preset",
    [
        "compatible",
        "openai_equivalent",
        "deepseek_source_audited",
        "openai_source_audited",
    ],
)
async def test_native_epub_preserves_resources_and_passes_real_epubcheck(
    tmp_path, native_http, preset, nav_in_spine
):
    executable = shutil.which("epubcheck")
    if executable is None:
        pytest.skip("Actual EPUBCheck is required; no successful-validator stub")
    source, output = tmp_path / "source.epub", tmp_path / "translated.epub"
    write_book(source, nav_in_spine=nav_in_spine)
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    before = await asyncio.to_thread(
        subprocess.run,
        [executable, str(source)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert before.returncode == 0, before.stdout + before.stderr
    settings = config(preset, refine=True, preserve_technical=True)
    settings["pipeline"]["budget"] = {"limit_usd": "5", "max_calls": 100}
    store = TokenUsageStore(tmp_path / "usage.db")
    manager = CheckpointManager(
        str(tmp_path / "jobs.db"), uploads_dir=tmp_path / "uploads"
    )
    manager.start_job("native", "epub", settings, str(source))
    token = activate_pipeline(
        settings["pipeline"], job_id="native", credentials=CREDENTIALS, store=store
    )
    logs = []
    checkpoint = None
    try:
        result = await translate_file(
            input_filepath=str(source),
            output_filepath=str(output),
            source_language="English",
            target_language="Spanish",
            model_name=settings["model"],
            llm_provider=settings["llm_provider"],
            deepseek_api_key=CREDENTIALS["deepseek"],
            openai_api_key=CREDENTIALS["openai"],
            prompt_options=settings["prompt_options"],
            max_tokens_per_chunk=450,
            checkpoint_manager=manager,
            translation_id="native",
            log_callback=lambda event, message, **data: logs.append((event, message)),
        )
        assert result is True, "\n".join(
            f"{event}: {message}" for event, message in logs[-15:]
        )
        checkpoint = manager.load_checkpoint("native")
    finally:
        await close_pipeline(token)
        manager.close()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_hash
    with zipfile.ZipFile(source) as original, zipfile.ZipFile(output) as translated:
        for name in ("cover.png", "map.png"):
            assert translated.read("OEBPS/" + name) == original.read("OEBPS/" + name)
        one = etree.fromstring(translated.read("OEBPS/one.xhtml"))
        two = etree.fromstring(translated.read("OEBPS/two.xhtml"))
        for root, ids in ((one, ("p1", "p2")), (two, ("p4", "p5"))):
            for ident in ids:
                text = root.xpath(f'string(//*[@id="{ident}"])')
                assert text in TRANSLATIONS.values(), (ident, text)
        assert (
            one.xpath("string(//h:strong)", namespaces=NS)
            == TRANSLATIONS["It was a cold morning."]
        )
        assert (
            one.xpath("string(//h:em)", namespaces=NS)
            == TRANSLATIONS["She counted 17 coins on the table."]
        )
        assert one.xpath("//h:figure/h:img/@src", namespaces=NS) == ["map.png"]
        assert (
            one.xpath("string(//h:figcaption)", namespaces=NS)
            == TRANSLATIONS["The map showed the road to the harbor."]
        )
        assert one.xpath('//*[@id="ref1"]/@href') == ["#note1"]
        assert one.xpath('//*[@id="note1"]/h:a/@href', namespaces=NS) == ["#ref1"]
        assert two.xpath("//h:td/text()", namespaces=NS) == ["17", "23.50"]
        assert two.xpath('string(//*[@id="formula"])') == "E = mc^2"
        visible = " ".join(one.itertext()) + " ".join(two.itertext())
        assert not re.search(r"TBLBLOCK|VERBALOOMBLOCK|\[id\d+\]", visible)
    report = await asyncio.to_thread(
        audit_epub_publication,
        source,
        output,
        source_language="English",
        target_language="Spanish",
        epubcheck_command=[executable],
    )
    assert report.publishable, report.errors
    assert report.epubcheck_errors == report.inherited_epubcheck_errors == 0
    assert report.preserved_images == 2
    assert report.broken_links == 0
    assert len(store.events()) == len(native_http)
    assert any(call["role"] == "audit" for call in native_http)

    # Reopen real persistence and runtime, as after a stopped worker. Assembly
    # recovery must not buy another translation of already approved files.
    assert checkpoint["checkpoint_complete"]
    calls_before = len(native_http)
    resumed = tmp_path / "resumed.epub"
    manager = CheckpointManager(
        str(tmp_path / "jobs.db"), uploads_dir=tmp_path / "uploads"
    )
    token = activate_pipeline(
        settings["pipeline"], job_id="native", credentials=CREDENTIALS, store=store
    )
    try:
        assert await translate_file(
            input_filepath=str(source),
            output_filepath=str(resumed),
            source_language="English",
            target_language="Spanish",
            model_name=settings["model"],
            llm_provider=settings["llm_provider"],
            deepseek_api_key=CREDENTIALS["deepseek"],
            openai_api_key=CREDENTIALS["openai"],
            prompt_options=settings["prompt_options"],
            max_tokens_per_chunk=450,
            checkpoint_manager=manager,
            translation_id="native",
            resume_from_index=checkpoint["resume_from_index"],
            log_callback=lambda event, message, **data: logs.append((event, message)),
        ), logs[-10:]
    finally:
        await close_pipeline(token)
        manager.close()
    assert len(native_http) == len(store.events()) == calls_before
    with zipfile.ZipFile(output) as first, zipfile.ZipFile(resumed) as recovered:
        assert set(first.namelist()) == set(recovered.namelist())
        for name in first.namelist():
            if name.endswith((".xhtml", ".png", ".css")):
                assert first.read(name) == recovered.read(name), name
    report = await asyncio.to_thread(
        audit_epub_publication,
        source,
        resumed,
        source_language="English",
        target_language="Spanish",
        epubcheck_command=[executable],
    )
    assert report.publishable, report.errors
