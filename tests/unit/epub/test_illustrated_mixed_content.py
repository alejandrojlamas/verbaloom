from lxml import etree
import pytest

from src.core.epub.body_serializer import replace_body_content
from src.core.epub.epub_translation_adapter import EpubTranslationAdapter
from src.core.epub.exceptions import XmlParsingError
from src.core.epub.structure_safe_fallback import refinement_preserves_paragraphs, split_structure_safe_parts
from src.core.epub.tag_preservation import TagPreserver
from tests.unit.epub.test_native_pipeline_e2e import native_http

XHTML = 'http://www.w3.org/1999/xhtml'
EPUB = 'http://www.idpf.org/2007/ops'
XLINK = 'http://www.w3.org/1999/xlink'
SVG = 'http://www.w3.org/2000/svg'


@pytest.mark.parametrize('prefix, declaration', [
    ('', ''), ('', f'xmlns="{XHTML}"'), ('h:', f'xmlns:h="{XHTML}"')
])
def test_illustration_path_and_caption_survive_translation(prefix, declaration):
    xml = f'''<{prefix}html {declaration} xmlns:epub="{EPUB}" xmlns:xlink="{XLINK}">
        <{prefix}body id="main" data-hint="a &gt; b"><{prefix}figure epub:type="illustration">
        <{prefix}img src="../Images/A  B.png" alt="Map"/><{prefix}figcaption>
        <{prefix}i>Map</{prefix}i> of the city.</{prefix}figcaption></{prefix}figure>
        <svg xmlns="{SVG}"><image xlink:href="../Images/Detail.png"/></svg>
        <{prefix}p>Afterword</{prefix}p></{prefix}body>
    </{prefix}html>'''
    doc = etree.XML(xml)
    adapter = EpubTranslationAdapter()
    source, context = adapter.extract_content(doc, None)
    masked, tags, _ = adapter.preserve_structure(source, context, None)
    output = adapter.reconstruct_content(
        [masked.replace('Map', 'Plano').replace(' of the city.', ' de A & B &lt;C&gt;.')
         .replace('Afterword', 'Notas')], tags, context,
    )
    assert adapter.finalize_output(output, doc, context, None)
    # Read the serialized bytes with the same strict namespace semantics as a reader.
    rebuilt = etree.fromstring(etree.tostring(doc))
    ns = '{' + XHTML + '}' if declaration else ''
    body, = rebuilt.findall(f'{ns}body')
    assert body.get('id') == 'main' and body.get('data-hint') == 'a > b'
    figure, graphic, paragraph = body
    assert figure.tag == ns + 'figure' and figure.get('{' + EPUB + '}type') == 'illustration'
    img, caption = figure
    assert img.tag == ns + 'img' and img.get('src') == '../Images/A  B.png' and img.get('alt') == 'Map'
    assert caption.tag == ns + 'figcaption' and caption[0].tag == ns + 'i'
    assert ''.join(caption.itertext()).strip() == 'Plano de A & B <C>.'
    assert graphic[0].get('{' + XLINK + '}href') == '../Images/Detail.png'
    assert paragraph.tag == ns + 'p' and paragraph.text == 'Notas'


def test_malformed_markup_cannot_replace_original_body():
    body = etree.XML('<body><figure><img/><figcaption>Original.</figcaption></figure></body>')
    before = etree.tostring(body)
    with pytest.raises(XmlParsingError):
        replace_body_content(body, '<figure><img><figcaption>Translation.</figcaption></figure>')
    assert etree.tostring(body) == before


@pytest.mark.parametrize('prefix', ['', 'h:'])
def test_caption_loss_detected_even_with_intact_inline_placeholders(prefix):
    html = (f'<{prefix}figure><{prefix}img/><{prefix}figcaption><{prefix}em>North</{prefix}em>'
            f' breccia</{prefix}figcaption></{prefix}figure><{prefix}p>Following text</{prefix}p>')
    source, tags = TagPreserver().preserve_tags(html)
    parts = split_structure_safe_parts(source, tags)
    assert len([p for p in parts if p.kind == 'content']) == 2
    assert refinement_preserves_paragraphs(source, source.replace('North', 'Brecha').replace(' breccia', ' norte'), tags)
    assert not refinement_preserves_paragraphs(source, source.replace('North', '').replace(' breccia', ''), tags)


def test_bilingual_chunks_split_inside_caption_interleave_complete_dom_blocks():
    from src.core.epub.xhtml_translator import _reconstruct_html

    doc = etree.XML(f'<html xmlns="{XHTML}"><body/></html>')
    body = doc[0]
    tags = {
        "[id0]": '<figure id="plate"><img src="a.jpg"/><figcaption id="legend"><i>',
        "[id1]": '</i></figcaption></figure><p id="next"><a href="#legend">',
        "[id2]": '</a></p>',
    }
    original_chunks = [
        {"text": "[id0]Northern ", "global_indices": [0]},
        {"text": "lights[id0]Back[id1]", "global_indices": [1, 2]},
    ]
    translated = ["[id0]Auroras", "boreales[id1]Regresar[id2]"]
    combined = _reconstruct_html(
        translated, tags, TagPreserver(), original_chunks=original_chunks, bilingual=True,
        body_element=body, source_language="English",
    )
    replace_body_content(body, combined)
    caption = body.find(".//{*}figcaption")
    assert caption.find("{*}span").text == "Northern lights"
    assert caption.find("{*}span").get("{http://www.w3.org/XML/1998/namespace}lang") == "en"
    assert caption.find("{*}i").text == "Auroras boreales"
    assert body.xpath(".//*[@id]/@id") == ["plate", "legend", "next"]
    assert body.xpath('.//*[local-name()="img"]/@src') == ["a.jpg"]
    assert body.xpath('.//*[local-name()="a"]/@href') == ["#legend"]


@pytest.mark.asyncio
async def test_bilingual_native_illustrated_epub_passes_epubcheck(tmp_path, native_http):
    import shutil
    import subprocess
    import zipfile

    from src.core.adapters.translate_file import translate_file
    from src.core.pipeline_runtime import activate_pipeline, close_pipeline
    from src.core.usage.store import TokenUsageStore
    from src.persistence.checkpoint_manager import CheckpointManager
    from tests.unit.epub.test_native_pipeline_e2e import CREDENTIALS, TRANSLATIONS, config, write_book

    executable = shutil.which("epubcheck")
    if not executable:
        pytest.skip("EPUBCheck is required for edition validation")
    source, output = tmp_path / "source.epub", tmp_path / "bilingual.epub"
    write_book(source)
    settings = config("compatible", refine=True, preserve_technical=True)
    manager = CheckpointManager(str(tmp_path / "jobs.db"), uploads_dir=tmp_path / "uploads")
    manager.start_job("bilingual", "epub", settings, str(source))
    token = activate_pipeline(
        settings["pipeline"], job_id="bilingual", credentials=CREDENTIALS, store=TokenUsageStore(tmp_path / "usage.db")
    )
    try:
        assert await translate_file(
            input_filepath=str(source), output_filepath=str(output), source_language="English",
            target_language="Spanish", model_name=settings["model"], llm_provider=settings["llm_provider"],
            deepseek_api_key=CREDENTIALS["deepseek"], openai_api_key=CREDENTIALS["openai"],
            prompt_options=settings["prompt_options"], max_tokens_per_chunk=450, bilingual_output=True,
            checkpoint_manager=manager, translation_id="bilingual",
        )
    finally:
        await close_pipeline(token)
        manager.close()
    check = subprocess.run([executable, str(output)], capture_output=True, text=True, timeout=60)
    assert check.returncode == 0, check.stdout + check.stderr
    with zipfile.ZipFile(output) as archive:
        chapter = etree.fromstring(archive.read("OEBPS/one.xhtml"))
        caption = chapter.find(".//{*}figcaption")
        original = "The map showed the road to the harbor."
        assert caption[0].text == original
        assert caption[0].tail == TRANSLATIONS[original]
        ids = chapter.xpath("//*[@id]/@id")
        assert len(ids) == len(set(ids))
        assert chapter.xpath('//*[local-name()="img"]/@src') == ["map.png"]
