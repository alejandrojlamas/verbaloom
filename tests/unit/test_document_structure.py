from src.core.document_structure import (
    DocumentBlockClassifier,
    DocumentIR,
    is_locator_index_block,
    is_locator_index_entry,
    is_locator_index_identity_block,
    is_locator_index_identity_fragment,
    normalize_document_structure,
    repair_structural_artifacts,
)
from src.core.post_processor import clean_translated_text


def test_normalizes_scientific_table_to_markdown_block():
    source = """Table 2: The Transformer achieves better BLEU scores.
Model
BLEU Training Cost (FLOPs)
EN-DE EN-FR EN-DE EN-FR
ByteNet [18] 23.75
Deep-Att + PosUnk [39] 39.2 1.0 · 1020
GNMT + RL [38] 24.6 39.92 2.3 · 1019 1.4 · 1020
Residual Dropout We apply dropout to the output of each sub-layer before normalization."""

    normalized, report = normalize_document_structure(source, source_type="pdf")

    assert report.tables_detected == 1
    assert "| Item | Value 1 |" in normalized
    assert "| ByteNet [18] | 23.75 |" in normalized
    assert "1.0 · 10^20" in normalized
    assert "2.3 · 10^19" in normalized
    assert "Residual Dropout We apply dropout" in normalized
    assert "Residual Dropout We apply dropout" not in normalized.split("|")[-2]
    assert report.block_type_counts["table"] == 1
    assert report.policy_counts["reconstruct"] == 1


def test_document_block_classifier_assigns_block_policies():
    source = """CAPÍTULO I

Texto narrativo con una escena normal y varias frases de desarrollo.

Contents
Chapter One ........ 7
Chapter Two ........ 15

E = mc^2

[OceanofPDF.com](https://oceanofpdf.com)"""

    blocks = DocumentBlockClassifier(source_type="pdf").classify_text(source)
    by_type = {block.type: block for block in blocks}

    assert by_type["title"].policy == "translate"
    assert by_type["narrative"].policy == "translate"
    assert by_type["toc"].policy == "reconstruct"
    assert by_type["formula"].policy == "preserve"
    assert by_type["watermark"].policy == "exclude"


def test_classifier_recognizes_flattened_lexical_glossary_entries():
    source = (
        "Each entry ends with a book and line reference. "
        "Acastus a-kas´-tus ): king of Dulichium.14.340. "
        "Achaean a-kee´-an ): inhabitants of Achaea.1.272. "
        "Acheron a´-ker-on ): a mythical river.10.516. "
        "Achilles a-kil´-eez ): a Greek warrior.3.106."
    )

    block_type, policy, confidence, strategy, _notes = (
        DocumentBlockClassifier(source_type="epub").classify_block([source])
    )

    assert block_type == "glossary"
    assert policy == "translate"
    assert confidence >= 0.85
    assert strategy == "translate_definitions_preserve_lexical_structure"


def test_classifier_recognizes_dense_pronunciation_key():
    source = (
        "PRONUNCIATION KEY a as in cat ah as in father ai as in light "
        "ee as in street u as in us you as in you zh as in vision"
    )

    block_type, policy, _confidence, _strategy, _notes = (
        DocumentBlockClassifier(source_type="epub").classify_block([source])
    )

    assert block_type == "glossary"
    assert policy == "translate"


def test_classifier_recognizes_flattened_epub_endnotes_with_bare_urls():
    source = (
        "Notes Epigraphs “It is said”: Joseph Weizenbaum, "
        "“ELIZA—a Computer Program for the Study of Natural Language "
        "Communication Between Man and Machine,” Communications of the ACM 9, "
        "no. 1 (January 1966): 36–45, doi.org/10.1145/365153.365168 "
        "GO TO NOTE REFERENCE IN TEXT “Successful people create companies”: "
        "Sam Altman, “Successful People,” March 7, 2013, "
        "blog.samaltman.com/successful-people"
    )

    block_type, policy, confidence, strategy, _notes = (
        DocumentBlockClassifier(source_type="epub").classify_block([source])
    )

    assert block_type == "critical_apparatus"
    assert policy == "preserve"
    assert confidence >= 0.70
    assert strategy == "preserve_bibliographic_apparatus"


def test_critical_apparatus_wins_over_numeric_table_heuristic():
    source = "\n".join([
        "Chapter 5: Scale of Ambition",
        "“How about now?”: Cade Metz, Genius Makers (Dutton, 2021), 93; "
        "“On Working with Ilya,” posted May 20, 2024, YouTube, 45 min., 45 sec., "
        "youtu.be/n4IQOBka8bc",
        "GO TO NOTE REFERENCE IN TEXT",
        "He stunned Hinton: Author interview with Geoffrey Hinton, November 2023.",
        "GO TO NOTE REFERENCE IN TEXT",
        "At times he grew: Metz, Genius Makers, 94.",
        "GO TO NOTE REFERENCE IN TEXT",
        "“One doesn’t bet”: MIT Technology Review, October 26, 2023, "
        "technologyreview.com/article",
    ])

    block_type, policy, confidence, strategy, _notes = (
        DocumentBlockClassifier(source_type="epub").classify_block(
            source.splitlines()
        )
    )

    assert block_type == "critical_apparatus"
    assert policy == "preserve"
    assert confidence >= 0.70
    assert strategy == "preserve_bibliographic_apparatus"


def test_single_complete_backlinked_citation_is_critical_apparatus():
    block_type, policy, confidence, strategy, _notes = (
        DocumentBlockClassifier(source_type="text").classify_block([
            "GO TO NOTE REFERENCE IN TEXT",
            (
                "Jeffrey Ding and Jenny W. Xiao, Recent Trends in China’s Large "
                "Language Model Landscape, April 28, 2023, "
                "cdn.governance.ai/Trends_in_Chinas_LLMs.pdf"
            ),
        ])
    )

    assert block_type == "critical_apparatus"
    assert policy == "preserve"
    assert confidence >= 0.70
    assert strategy == "preserve_bibliographic_apparatus"


def test_classifier_recognizes_plain_publisher_year_reading_list():
    source = (
        "Interpersonal Peacemaking: Confrontations and Third-Party Consultation. "
        "Reading, Mass.: Addison-Wesley, 1969.\n"
        "Weisbord, Marvin. Discovering Common Ground: How Future Search "
        "Conferences Bring People Together. "
        "San Francisco: Berrett-Koehler, 1992.\n"
        "Whitmore, John. Coaching for Performance. "
        "London: Nicholas Brealey, 2009."
    )

    block_type, policy, confidence, strategy, _notes = (
        DocumentBlockClassifier(source_type="epub").classify_block(
            source.splitlines()
        )
    )

    assert block_type == "critical_apparatus"
    assert policy == "preserve"
    assert confidence >= 0.70
    assert strategy == "preserve_bibliographic_apparatus"


def test_classifier_recognizes_legacy_comma_delimited_bibliography():
    source = "\n".join([
        "S. A. Handford, Penguin, 1951",
        "Cameron, James, What a Way to Run the Tribe, Macmillan, 1968",
        "Churchill, Winston, My Early Life, Heinemann, 1930",
        "Coleridge, Samuel Taylor, Letters, E. L. Griggs (ed.), "
        "Oxford University Press, 1956-71",
    ])

    block_type, policy, confidence, strategy, _notes = (
        DocumentBlockClassifier(source_type="epub").classify_block(
            source.splitlines()
        )
    )

    assert block_type == "critical_apparatus"
    assert policy == "preserve"
    assert confidence >= 0.70
    assert strategy == "preserve_bibliographic_apparatus"


def test_classifier_does_not_treat_year_terminated_prose_as_bibliography():
    source = "\n".join([
        "The firm moved to Boston, expanded rapidly, 1951.",
        "The family returned home, exhausted and discouraged, 1968.",
    ])

    block_type, policy, _confidence, _strategy, _notes = (
        DocumentBlockClassifier(source_type="epub").classify_block(
            source.splitlines()
        )
    )

    assert block_type == "narrative"
    assert policy == "translate"


def test_locator_index_block_recognizes_name_and_page_runs():
    lines = [
        "Bramwell, James G., 595",
        "Bride, Harold, 435",
        "Buckingham, Duchess of (1620), 172",
        "Byron, George Gordon, Lord, 302",
        "Campbell, Sir Colin, 339,347",
    ]

    assert is_locator_index_block(lines) is True
    assert is_locator_index_entry("Bramwell, James G., 595") is True
    assert is_locator_index_entry("Campbell, Sir Colin, 339,347") is True
    assert is_locator_index_entry("Camel!, William, 211") is True
    assert is_locator_index_entry("Croy. Lord, 71") is True
    assert is_locator_index_entry("Elizabeth 1,149,156") is True
    assert is_locator_index_entry("Canterbury, 96") is False
    assert is_locator_index_entry("children, protection of, 83") is False


def test_locator_index_identity_block_accepts_split_table_fragments():
    entries = [
        "Louis-Charles, Dauphin of France, 246",
        "Lucan, Lord, 336",
        "Montgomery, Field-Marshal Bernard,",
        "Richard 1,35",
        "Roosevelt, Theodore, 407",
    ]

    assert is_locator_index_identity_fragment(entries[2]) is True
    assert is_locator_index_identity_fragment(
        "Maud’huy, General de, 450 Méneval, Baron Claude Francois de,"
    ) is True
    assert is_locator_index_identity_fragment(". ardine, Douglas, 505") is True
    assert is_locator_index_identity_fragment(". osephus, 14") is True
    assert is_locator_index_identity_fragment(
        "Hugo, Victor, 328 ahangir, the Great Mogul, 168,171 ames 1,172"
    ) is True
    assert is_locator_index_identity_fragment(
        "Morrison, lan. 559 Morton, Sir Thomas, 173"
    ) is True
    assert is_locator_index_identity_fragment(
        "N^xrleon, Bonaparte, 254,278,285"
    ) is True
    assert is_locator_index_identity_fragment(
        "Schnirdel, Hu Ider ike, 92"
    ) is True
    assert is_locator_index_identity_fragment(
        "Munro, H.H.fSaki’), 469 Munro, Ross, 566"
    ) is True
    assert is_locator_index_identity_fragment("Rios, Pedro de la, 111") is True
    assert is_locator_index_identity_block(entries) is True


def test_locator_index_identity_block_rejects_translatable_subject_entry():
    entries = [
        "Louis-Charles, Dauphin of France, 246",
        "Lucan, Lord, 336",
        "children, protection of, 83",
        "Richard 1,35",
        "Roosevelt, Theodore, 407",
    ]

    assert is_locator_index_identity_fragment(entries[2]) is False
    assert is_locator_index_identity_fragment(
        ". children, protection of, 83"
    ) is False
    assert is_locator_index_identity_fragment(
        "Hugo, Victor, 328 children, protection of, 83"
    ) is False
    assert is_locator_index_identity_fragment(
        "History, The New age, 83"
    ) is False
    assert is_locator_index_identity_block(entries) is False


def test_locator_index_block_rejects_numeric_narrative_lines():
    lines = [
        "The firm moved to Boston, expanded rapidly, 1951.",
        "The family returned home, exhausted and discouraged, 1968.",
        "The account appeared in print, many years later, 1972.",
    ]

    assert is_locator_index_block(lines) is False
    assert is_locator_index_entry(lines[0]) is False


def test_document_ir_exposes_llm_text_blocks_and_report():
    ir = DocumentIR.from_text(
        "CAPÍTULO I\n\nTexto real.\n\n[OceanofPDF.com](https://oceanofpdf.com)",
        source_type="epub",
    )

    assert ir.llm_text.startswith("CAPÍTULO I")
    assert "OceanofPDF" not in ir.llm_text
    assert ir.report.excluded_blocks >= 1
    assert any(block.policy == "exclude" for block in ir.blocks)
    assert ir.to_dict()["report"]["source_type"] == "epub"


def test_normalize_document_structure_excludes_junk_links_and_reports_policy_counts():
    source = """Texto real antes.

[OceanofPDF.com](https://oceanofpdf.com)

https://oceanofpdf.com

Texto real despues."""

    normalized, report = normalize_document_structure(source, source_type="epub")

    assert "Texto real antes." in normalized
    assert "Texto real despues." in normalized
    assert "OceanofPDF" not in normalized
    assert "https://oceanofpdf.com" not in normalized
    assert report.excluded_blocks >= 2
    assert report.policy_counts["exclude"] >= 2
    assert report.block_type_counts["watermark"] >= 1


def test_classifier_does_not_exclude_legitimate_inline_url_paragraph():
    source = (
        "Consulta https://example.com/apendice para ver el material complementario "
        "antes de continuar con el capítulo."
    )

    normalized, report = normalize_document_structure(source, source_type="txt")

    assert "material complementario" in normalized
    assert "https://example.com/apendice" in normalized
    assert report.excluded_blocks == 0
    assert report.block_type_counts["narrative"] == 1


def test_classifier_cleans_standalone_markdown_link_without_dropping_label():
    normalized, report = normalize_document_structure(
        "Texto previo.\n\n[Apéndice documental](https://example.com/apendice)\n\nTexto posterior.",
        source_type="txt",
    )

    assert "Texto previo." in normalized
    assert "Apéndice documental" in normalized
    assert "https://example.com/apendice" not in normalized
    assert "Texto posterior." in normalized
    assert report.cleaned_blocks == 1
    assert report.policy_counts["clean"] == 1


def test_pipe_table_without_caption_is_reported_as_table():
    normalized, report = normalize_document_structure(
        "| Modelo | BLEU |\n| --- | --- |\n| Base | 27.3 |",
        source_type="txt",
    )

    assert "| Modelo | BLEU |" in normalized
    assert report.tables_detected == 1
    assert report.block_type_counts["table"] == 1
    assert report.has_structured_blocks is True


def test_classifier_keeps_standalone_chapter_numbers():
    normalized, report = normalize_document_structure(
        "I\n\nEn un lugar de la Mancha.\n\n12\n\nOtro apartado.",
        source_type="txt",
    )

    assert normalized.startswith("I")
    assert "\n12\n" in f"\n{normalized}\n"
    assert report.excluded_blocks == 0


def test_classifier_excludes_explicit_page_markers():
    normalized, report = normalize_document_structure(
        "Page 12\n\nTexto real.\n\n14 / 200",
        source_type="pdf",
    )

    assert "Texto real." in normalized
    assert "Page 12" not in normalized
    assert "14 / 200" not in normalized
    assert report.block_type_counts["header_footer"] == 2
    assert report.excluded_blocks == 2


def test_narrative_only_blocks_do_not_activate_structured_layout():
    normalized, report = normalize_document_structure(
        "Este es un párrafo narrativo normal.\n\nEste es otro párrafo normal.",
        source_type="txt",
    )

    assert "párrafo narrativo" in normalized
    assert report.block_type_counts["narrative"] == 2
    assert report.has_structured_blocks is False


def test_compacts_visual_token_runs_without_dropping_special_tokens():
    source = """Attention Visualizations
Input-Input Layer5
It
is
in
this
spirit
that
a
majority
of
governments
passed
laws
.
<EOS>
<pad>
<pad>
Figure 3: An example of the attention mechanism."""

    normalized, report = normalize_document_structure(source, source_type="pdf")

    assert report.figure_text_blocks_detected == 1
    assert "Figure visual text:" in normalized
    assert "It is in this spirit" in normalized
    assert "<EOS>" in normalized
    assert "<pad>" in normalized
    assert "\nIt\nis\nin\nthis\n" not in normalized
    assert "Figure 3:" in normalized


def test_repairs_structural_numeric_spacing():
    cleaned, repairs = repair_structural_artifacts(
        "ByteNet [18] 23. 75\nDeep-Att 1. 0 · 1020\nPdrop = 0. 1"
    )

    assert repairs >= 3
    assert "23.75" in cleaned
    assert "1.0 · 10^20" in cleaned
    assert "0.1" in cleaned


def test_postprocessor_applies_structural_numeric_cleanup():
    assert clean_translated_text("BLEU 23. 75 y Pdrop = 0. 1") == "BLEU 23.75 y Pdrop = 0.1"
