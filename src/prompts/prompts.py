import re
from typing import List, NamedTuple, Optional, Tuple

from src.config import (
    INPUT_TAG_IN,
    INPUT_TAG_OUT,
    PLACEHOLDER_PREFIX,
    PLACEHOLDER_SUFFIX,
    TRANSLATE_TAG_IN,
    TRANSLATE_TAG_OUT,
    create_placeholder,
)
from src.core.book_profiles import build_profile_instruction_block, profile_enabled
from src.core.locale_quality import is_mexican_spanish_target, is_spanish_target
from src.prompts.examples import (
    TAG0,
    build_placeholder_section,
    get_output_format_example,
    get_subtitle_example,
)
from src.prompts.security import UNTRUSTED_BOOK_CONTENT_SECTION

# Tags for placeholder correction responses
CORRECTED_TAG_IN = "<CORRECTED_TAG_IN>"
CORRECTED_TAG_OUT = "<CORRECTED_TAG_OUT>"


class PromptPair(NamedTuple):
    """A pair of system and user prompts for LLM translation."""
    system: str
    user: str


# ============================================================================
# SHARED PROMPT SECTIONS
# ============================================================================

def _get_output_format_section(
    translate_tag_in: str,
    translate_tag_out: str,
    input_tag_in: str,
    input_tag_out: str,
    additional_rules: str = "",
    example_format: str = "Your translated text here"
) -> str:
    """
    Generate standardized output format instructions.

    Args:
        translate_tag_in: Opening tag for translation output
        translate_tag_out: Closing tag for translation output
        input_tag_in: Opening tag for input text
        input_tag_out: Closing tag for input text
        additional_rules: Optional additional formatting rules
        example_format: Example text to show in correct format

    Returns:
        str: Formatted output format instructions
    """
    additional_rules_text = f"\n{additional_rules}" if additional_rules else ""

    return f"""# OUTPUT FORMAT

**CRITICAL OUTPUT RULES:**
1. Translate ONLY the text between "{input_tag_in}" and "{input_tag_out}" tags
2. Your response MUST start with {translate_tag_in} (first characters, no text before)
3. Your response MUST end with {translate_tag_out} (last characters, no text after)
4. Include NOTHING before {translate_tag_in} and NOTHING after {translate_tag_out}
5. Do NOT add meta-explanations, comments, separate notes, or greetings{additional_rules_text}

**INCORRECT examples (DO NOT do this):**
❌ "Here is the translation: {translate_tag_in}Text...{translate_tag_out}"
❌ "{translate_tag_in}Text...{translate_tag_out} (Additional comment)"
❌ "Sure! {translate_tag_in}Text...{translate_tag_out}"
❌ "Text..." (missing tags entirely)
❌ "{translate_tag_in}Text..." (missing closing tag)

**CORRECT format (ONLY this):**
✅ {translate_tag_in}
{example_format}
{translate_tag_out}
"""


# Numbering starts at 6 because rules 1-5 are emitted by _get_output_format_section.
_SUBTITLE_FORMAT_RULES = (
    "\n6. Each subtitle has an index marker: [index]text - PRESERVE these markers exactly"
    "\n7. Keep ONE [index] per subtitle - do NOT merge or split subtitles"
    "\n8. Maintain line breaks between indexed subtitles"
    "\n9. Preserve inline tags (<i>, <b>, <u>, <font ...>, {\\an8}, etc.) and any \\n line breaks INSIDE a subtitle exactly as in the source"
)


def _is_spanish_target(target_language: str) -> bool:
    return is_spanish_target(target_language)


def _role_language(target_language: str, prompt_options: Optional[dict] = None) -> str:
    if is_mexican_spanish_target(target_language, prompt_options):
        return "Mexican Spanish"
    return target_language


_AUTO_SOURCE_LANGUAGE_PROMPT_VALUES = {
    "",
    "auto",
    "autodetect",
    "auto-detect",
    "automatic",
    "detect automatically",
    "detectar automaticamente",
    "detectar automáticamente",
}


def _source_language_prompt_label(source_language: str) -> str:
    normalized = str(source_language or "").strip().lower()
    if normalized in _AUTO_SOURCE_LANGUAGE_PROMPT_VALUES:
        return "the source language of each passage"
    return str(source_language or "").strip() or "the source language of each passage"


def _target_locale_section(target_language: str, prompt_options: Optional[dict] = None) -> str:
    """Return stable locale/style instructions for broad target languages."""
    prompt_options = prompt_options or {}
    if not is_mexican_spanish_target(target_language, prompt_options):
        return ""
    if str(prompt_options.get("text_transform_mode") or "").strip().lower() == "modernize":
        return """
# SPANISH LOCALE

Write in natural Mexican Spanish / neutral Latin American Spanish when modernizing ordinary prose.
- Rewrite historically marked dialogue, titles, and ceremonial distance as natural contemporary editorial prose chosen by context.
- Preserve relationships, voice, and social distance through current language; do not preserve obsolete surface grammar as a substitute for voice.
- For Mexican/LatAm output, do not leave "vosotros", "os", "vuestro/vuestra", "podéis", "queréis", "tenéis", "habéis", "sois", "estáis", or similar second-person plural forms unless the active profile glossary explicitly requires them.
- Convert archaic/Peninsular address by context: use "usted", "ustedes", "su/sus", "lo/la/le/los/las/les", or a natural rewritten sentence while preserving who speaks to whom and the social distance.
- Replace old conjunction "e" with "y" except where modern Spanish genuinely requires "e" before an /i/ sound.
- Do not add local slang or contemporary idioms unless the source tone clearly supports it.
- Final self-check: the output must read like current publishable editorial Spanish, not like lightly respelled colonial or Golden Age Spanish.
""".strip()

    return """
# SPANISH LOCALE

Write in natural Mexican Spanish / neutral Latin American Spanish.
- Literary or archaic source language must become elevated Mexican Spanish, not Peninsular/archaic Spanish.
- Forbidden output forms: "vosotros", "vosotras", "os", "sois", "estáis", "habéis", "rendisteis", "vuestro", "vuestra", "vuestros", "vuestras", "chitón", "hala", "vale" as filler, "avisaos", "avispaos", "precipitaos", and similar "-aos/-eos" imperatives.
- Prefer Mexican/LatAm phrasing that still sounds literary and publishable: "ustedes" when plural address is needed, "se rindieron" over "os rendisteis", "sus" over "vuestros", "cállense/guarden silencio" over "chitón", "estén atentos" over "avispaos", and "precipítense" over "precipitaos".
- Do not add local slang unless the source tone clearly calls for colloquial speech.
- Final self-check: scan the entire output before answering and remove every Peninsular/archaic Spanish form listed above.
""".strip()


def _literary_style_contract_section(target_language: str, prompt_options: Optional[dict] = None) -> str:
    """Return a stable cross-chunk style contract for Spanish literary output."""
    options = prompt_options or {}
    if options.get("literary_style_contract") is False:
        return ""
    if not is_spanish_target(target_language):
        return ""
    return """
# LITERARY STYLE CONTRACT

Use this contract consistently across every chunk of Spanish literary prose.
- Use the em dash/raya (—) for spoken dialogue. Do not switch dialogue to English quotation marks (" ") or guillemets (« ») unless the source explicitly requires a quoted quotation.
- Reserve « » for interior thought, quoted titles, or quotations inside narration when needed; do not mix them with raya as the main dialogue system.
- For interruptions and speech tags, prefer clean paired rays such as "—texto —dijo Anna— y siguió." Do not leave orphan dashes after ellipses.
- Keep names, places, chapter titles, and forms of address consistent with the active glossary/profile. If no glossary entry says otherwise, choose one canonical form within the chunk and keep it.
- Remove reader-visible processing labels such as "Descripción de imagen:" or "Descripcion de imagen:"; keep the narrative content that follows them.

Examples:
- Spoken dialogue: —No puedo ir —dijo Anna—. Volveré mañana.
- Interior thought: «No puedo ir», pensó Anna.
""".strip()


_TEXT_TRANSFORM_MODE_SPECS = {
    "modernize": {
        "label": "Modernizar",
        "goal": (
            "Convert archaic, old-fashioned, or heavily period-styled prose into "
            "faithful contemporary prose in the same language. Content fidelity is "
            "non-negotiable, and the result must read like a current published "
            "book when the source grammar is obsolete."
        ),
        "rules": [
            "Modernize only stable language: spelling, obsolete forms, and syntax whose meaning is unambiguous.",
            "Preserve paragraph breaks, headings, dialogue dashes, quotations, names, dates, facts, and narrative sequence.",
            "Modernize formal/courtly address inside quotations and dialogue by context; preserve social distance and speaker relationship, not obsolete verb endings.",
            "Do not change who is addressed or the respect level; choose a natural contemporary equivalent instead of keeping archaic grammar.",
            "Preserve literary dignity and historical voice through rhythm, focus, irony, and register; do not preserve period flavor by leaving obsolete surface forms.",
            "If a phrase is famous, quoted, deliberately archaic, or semantically risky, keep the meaning close but still make the syntax readable unless the active profile explicitly preserves the form.",
            "If block markers like [id900000] appear, preserve every marker exactly once, in the same order, and keep each marked block separate.",
        ],
        "high_strength_goal": (
            "Perform a complete intralingual modernization: rewrite archaic "
            "orthography, morphology, and syntax into natural contemporary "
            "prose while preserving every fact, name, scene, relationship, "
            "and narrative effect. A version that merely cleans spelling but "
            "keeps the old sentence scaffolding is insufficient."
        ),
        "high_strength_rules": [
            "Preserve all facts, actions, proper names, numbers, objects, places, relationships, and narrative order.",
            "Actively restructure sentences whose syntax sounds archaic today; keep the content, change the scaffolding.",
            "Reproduce comic, ironic, or solemn effects with contemporary literary devices, not by retaining obsolete surface forms.",
            "Modernize dialogue and address forms into contemporary equivalents chosen by context; preserve who speaks, who is addressed, respect, distance, and narrative effect even when obsolete verb morphology must change.",
            "For Spanish modernization, eliminate residual old-Spanish grammar such as archaic conjunctions before non-/i/ sounds, contracted archaisms, enclitic verb order, and Peninsular second-person plural forms in es-MX output.",
            "Do not summarize, censor, soften, explain, or add information.",
            "Preserve paragraph breaks, headings, dialogue dashes, and quotation boundaries.",
            "If block markers like [id900000] appear, preserve every marker exactly once, in the same order, and keep each marked block separate.",
        ],
    },
    "simplify": {
        "label": "Explicar",
        "goal": (
            "Rewrite dense, technical, academic, or difficult prose as clear "
            "university-level prose in common language. Explain complex ideas, "
            "tables, formulas, and arguments intelligently while preserving rigor, "
            "nuance, and all substantive content."
        ),
        "allows_explanatory_expansion": True,
        "rules": [
            "Write in polished university-level prose: clear, adult, precise, and readable, never childish or reductive.",
            "Integrate brief explanations into the prose when they help a reader understand a concept, argument, table, formula, or specialized term.",
            "You may add clarifying bridges, definitions, and interpretive context when they are strongly supported by the source; do not invent facts, numbers, citations, examples, or conclusions.",
            "Preserve every substantive claim, name, date, number, citation, formula, table value, sequence, and argumentative relationship.",
            "Keep necessary technical terms, then explain them in common language instead of deleting or oversimplifying them.",
            "For tables, preserve the data and explain what the rows, columns, comparisons, or patterns mean in readable prose; do not drop rows or summarize away important values.",
            "For formulas, preserve the formula or notation and explain what it means, what the variables represent when the source makes that clear, and how it supports the surrounding argument.",
            "Do not turn the output into a study guide, bullet outline, or separate notes unless the source itself is structured that way.",
            "Do not soften controversial claims, censor material, remove nuance, or replace the author's argument with your own opinion.",
        ],
    },
    "humanize": {
        "label": "Humanizar",
        "goal": (
            "Remove stiff, generic, or obviously AI-like phrasing and make the prose feel "
            "natural, human, and publishable in the same language."
        ),
        "rules": [
            "Improve rhythm, transitions, and sentence variety without changing facts.",
            "Remove robotic repetition, over-explaining, and unnatural cadence.",
            "Do not make the text more casual unless the original tone supports it.",
        ],
    },
    "mexican_spanish": {
        "label": "Adaptar a mexicano",
        "goal": (
            "Adapt the text to natural Mexican Spanish / neutral Latin American Spanish "
            "while preserving register and meaning."
        ),
        "rules": [
            "Remove Peninsular forms such as vosotros, os, sois, vuestro, chitón, hala, and vale as filler.",
            "Use Mexican/LatAm phrasing that still fits the genre and period.",
            "Do not add slang, jokes, or local color unless the source already calls for it.",
        ],
    },
    "academic_clarity": {
        "label": "Claridad académica",
        "goal": (
            "Make academic or essayistic prose clearer, more orderly, and easier to follow "
            "without changing the argument."
        ),
        "rules": [
            "Clarify long sentences and ambiguous references.",
            "Keep citations, numbers, terminology, equations, and claims intact.",
            "Do not add conclusions or evidence not present in the source.",
        ],
    },
    "literary_polish": {
        "label": "Pulido literario",
        "goal": (
            "Polish prose at a serious editorial level: rhythm, diction, paragraph flow, "
            "and voice, while staying close to the source."
        ),
        "rules": [
            "Respect the author's voice, point of view, and atmosphere.",
            "Prefer precise, elegant edits over decorative rewriting.",
            "Preserve dialogue style, paragraph breaks, and tonal shifts.",
        ],
    },
    "ocr_structure": {
        "label": "Corregir OCR y estructura",
        "goal": (
            "Repair OCR/scanner noise, broken lines, bad spacing, collapsed paragraphs, "
            "and malformed headings while preserving the same text."
        ),
        "rules": [
            "Remove meaningless glyph garbage and page artifacts.",
            "Restore paragraph breaks, headings, lists, quotes, and formula blocks conservatively.",
            "Do not rewrite good prose just to show improvement.",
        ],
    },
}


def build_text_transform_instructions(
    prompt_options: Optional[dict],
    target_language: str = "English",
) -> str:
    """Return monolingual transformation instructions for refinement mode."""
    options = prompt_options or {}
    raw_mode = str(options.get("text_transform_mode") or "").strip().lower()
    if not raw_mode:
        return ""

    spec = _TEXT_TRANSFORM_MODE_SPECS.get(raw_mode)
    custom_label = str(options.get("text_transform_label") or "").strip()
    high_strength = str(
        options.get("modernization_strength") or ""
    ).strip().lower() in {"high", "full", "aggressive"}
    if spec is None:
        label = custom_label or raw_mode.replace("_", " ").title()
        goal = "Apply the selected same-language text transformation faithfully."
        rules = []
    else:
        label = custom_label or spec["label"]
        if high_strength and spec.get("high_strength_goal"):
            goal = spec["high_strength_goal"]
            rules = spec.get("high_strength_rules") or spec["rules"]
        else:
            goal = spec["goal"]
            rules = spec["rules"]

    rules_text = "\n".join(f"- {rule}" for rule in rules)
    if rules_text:
        rules_text = f"\n\nMode-specific rules:\n{rules_text}"

    profile_section = build_profile_instruction_block(
        options,
        phase=raw_mode or "modernize",
        target_language=target_language,
    )

    if profile_enabled(options) and raw_mode == "modernize":
        base = f"""
TEXT TRANSFORMATION MODE: {label}

These mode instructions activate a book-scoped editorial profile. The active
profile controls modernization strength, target locale, voice policy, glossary,
audit dimensions, and repair criteria.

You are transforming text within the SAME language: the output language remains {target_language}.

Goal: Produce the profile-defined contemporary literary version, not a merely
orthographic cleanup.

Non-negotiable constraints:
- Stay faithful to the source text's real content; do not invent, censor, omit, summarize, or add unsupported facts.
- Preserve names, dates, numbers, citations, formulas, scene order, argument order, paragraph intent, and speaker relationships.
- Modernize language to the active profile's strength. If the profile asks for high modernization, rewrite old syntax into natural contemporary syntax while preserving voice and content.
- Preserve authorial voice through equivalent contemporary literary effects, not by retaining obsolete surface forms.
- Use only the active profile and its approved glossary for book-specific decisions.
- Output only the transformed text, without notes or explanations.
""".strip()
    else:
        explanation_policy = ""
        if spec and spec.get("allows_explanatory_expansion"):
            explanation_policy = "\n- The active mode permits explanatory rewriting: integrate supported explanations into the prose when they improve understanding."
        output_boundary = (
            "Output only the transformed text, without separate notes, prefaces, or meta-commentary."
            if spec and spec.get("allows_explanatory_expansion")
            else "Output only the transformed text, without notes or explanations."
        )

        base = f"""
TEXT TRANSFORMATION MODE: {label}

These mode instructions override the default "light refinement only" posture when they conflict.
You are transforming text within the SAME language: the output language remains {target_language}.

Goal: {goal}

Non-negotiable constraints:
- Stay faithful to the source text's real content; do not invent, censor, omit, or add unsupported facts.
- Preserve names, dates, numbers, citations, formulas, scene order, argument order, and meaningful paragraph structure.
- Keep the author's intent and register unless the selected mode explicitly asks for a controlled shift.
- Preserve direct quotations and dialogue treatment; do not change grammatical person or normalize away a meaningful historical voice.
- If a passage already satisfies this mode's goal, leave it close to unchanged; otherwise apply the mode directly.{explanation_policy}
- {output_boundary}{rules_text}
""".strip()
    return "\n\n".join(part for part in (base, profile_section) if part)


def _transform_allows_explanatory_expansion(prompt_options: Optional[dict]) -> bool:
    mode = str((prompt_options or {}).get("text_transform_mode") or "").strip().lower()
    spec = _TEXT_TRANSFORM_MODE_SPECS.get(mode) or {}
    return bool(spec.get("allows_explanatory_expansion"))


_MARKDOWN_TABLE_ROW_RE = re.compile(r"(?m)^\s*\|.*\|\s*$")
_NUMERIC_ONLY_LINE_RE = re.compile(
    r"(?m)^\s*(?:"
    r"\(?[A-Z]\)?|"
    r"[-+]?\d+(?:[.,]\d+)?(?:\s*(?:%|M|B|K))?|"
    r"[-+]?\d+(?:[.,]\d+)?\s*[+/-]\s*[.,]?\d+|"
    r"[-+]?\d+(?:[.,]\d+)?\s*(?:x|×|·)\s*10\^?[-+]?\d+"
    r")\s*$"
)
_STRUCTURAL_CONTEXT_CUE_RE = re.compile(
    r"\b("
    r"table|tabla|figure|figura|formula|fórmula|equation|ecuaci[oó]n|"
    r"caption|leyenda|bleu|rouge|meteor|cider|nist|glue|mnli|wikiSQL|"
    r"softmax|lrate|learning rate|perplexity|perplejidad|optimizer|"
    r"hyperparameter|hiperpar[aá]metro|dataset|benchmark"
    r")\b|[|=∑Σ√βγδ∆∈≤≥≈]",
    re.IGNORECASE,
)


def _clip_context_head(text: str, limit: int = 900) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    cut = text.rfind("\n\n", 0, limit)
    if cut < limit // 2:
        cut = text.rfind("\n", 0, limit)
    if cut < limit // 2:
        cut = limit
    return text[:cut].strip() + "\n[...]"


def _clip_context_tail(text: str, limit: int = 900) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    start = len(text) - limit
    cut = text.find("\n\n", start)
    if cut < 0 or cut > len(text) - (limit // 2):
        cut = text.find("\n", start)
    if cut < 0 or cut > len(text) - (limit // 2):
        cut = start
    return "[...]\n" + text[cut:].strip()


def _needs_structural_context(*texts: str) -> bool:
    joined = "\n".join(t for t in texts if t and t.strip())
    if not joined:
        return False
    if _MARKDOWN_TABLE_ROW_RE.search(joined):
        return True
    if len(_NUMERIC_ONLY_LINE_RE.findall(joined)) >= 5:
        return True
    return bool(_STRUCTURAL_CONTEXT_CUE_RE.search(joined))


def _build_surrounding_context_block(
    context_before: str,
    context_after: str,
    main_text: str,
    *,
    heading: str = "SURROUNDING SOURCE CONTEXT",
) -> str:
    """Return a small non-output context block for tables, formulas, and figures."""
    if not (context_before or context_after):
        return ""
    if not _needs_structural_context(main_text, context_before, context_after):
        return ""

    before = _clip_context_tail(context_before)
    after = _clip_context_head(context_after)
    parts = [
        f"# {heading}",
        "",
        "Use this only to understand table continuations, formulas, figure captions, headings, and section boundaries.",
        "Do not translate, transform, copy, summarize, or output this context unless the same content also appears between the input tags.",
    ]
    if before:
        parts.extend(["", "Previous context:", before])
    if after:
        parts.extend(["", "Next context:", after])
    parts.append("")
    return "\n".join(parts)


# ============================================================================
# OPTIONAL PROMPT SECTIONS
# ============================================================================

# Technical content preservation section (for technical documents)
TECHNICAL_CONTENT_SECTION = """
**Technical Content (DO NOT TRANSLATE):**
- Code snippets and syntax: `function()`, `variable_name`, `class MyClass`
- Command lines: `npm install`, `git commit -m "message"`
- File paths: `/usr/bin/`, `C:/Users/Documents/`
- URLs: `https://example.com`, `www.site.org`
- Programming identifiers, API names, and technical terms"""

# Text cleanup section (for OCR or poorly formatted source/draft texts)
TEXT_CLEANUP_SECTION = """
# TEXT CLEANUP (Source Defects Correction)

The source text may contain OCR errors, formatting artifacts, or typographic defects.
**CORRECT THESE ISSUES during translation or refinement:**

- **Line breaks**: Fix broken words (e.g., "trans-\\nlation" → "translation")
- **Spacing**: Remove double spaces, fix missing spaces after punctuation
- **Punctuation**: Correct misplaced or missing punctuation marks
- **Paragraph flow**: Merge incorrectly split paragraphs, preserve intentional breaks

**DO NOT** add content, remove meaningful text, or alter the author's style."""


STRUCTURED_LAYOUT_SECTION = """
# STRUCTURED TABLE / FIGURE BLOCKS

The source may contain normalized document-structure blocks produced before chunking.

- Preserve Markdown table syntax exactly: keep every `|`, row, column boundary, and header separator.
- Translate captions, headings, and textual cell content, but preserve numbers, citations, formulas, model names, acronyms, and mathematical notation exactly unless normal target-language typography requires only spacing changes.
- Do not merge table rows into prose.
- Do not add, remove, reorder, or summarize table rows.
- If a block begins with "Figure visual text:", translate it as figure text while preserving special tokens such as `<EOS>`, `<pad>`, labels, numbers, and captions.
""".strip()


def _build_optional_prompt_sections(prompt_options: dict) -> str:
    """
    Build optional prompt sections based on the provided options.

    Args:
        prompt_options: Dictionary containing prompt customization flags:
            - preserve_technical_content: DEPRECATED - Technical content is now protected
              via placeholder system (no prompt section needed)
            - text_cleanup: Include OCR/typographic defect correction instructions

    Returns:
        str: Concatenated optional sections to include in the system prompt
    """
    if prompt_options is None:
        prompt_options = {}

    sections = []

    # Technical content preservation is now handled by the placeholder system
    # (TagPreserver with protect_technical=True), so no prompt instructions are needed.
    # The LLM never sees technical content - it's hidden in placeholders like [id0], [id1].
    # Leaving this commented for reference:
    # if prompt_options.get('preserve_technical_content', False):
    #     sections.append(TECHNICAL_CONTENT_SECTION)

    # Inline-markdown preservation for text-first DOCX/EPUB plain mode. The
    # extractors encode bold/italic/links as markdown so the LLM can preserve
    # them without seeing raw markup.
    if prompt_options.get('inline_markdown', False):
        from src.common.inline_markdown import INLINE_MARKDOWN_PROMPT_SECTION
        sections.append(INLINE_MARKDOWN_PROMPT_SECTION)

    # Text cleanup for OCR or poorly formatted sources
    if prompt_options.get('text_cleanup', False):
        sections.append(TEXT_CLEANUP_SECTION)

    if prompt_options.get('structured_layout_active', False):
        sections.append(STRUCTURED_LAYOUT_SECTION)

    # Join sections with double newline for proper separation
    return '\n\n'.join(sections)


# ============================================================================
# TRANSLATION PROMPT FUNCTIONS
# ============================================================================

def generate_translation_prompt(
    main_content: str,
    context_before: str,
    context_after: str,
    previous_translation_context: str,
    source_language: str = "English",
    target_language: str = "English",
    translate_tag_in: str = TRANSLATE_TAG_IN,
    translate_tag_out: str = TRANSLATE_TAG_OUT,
    has_placeholders: bool = True,
    prompt_options: dict = None,
    placeholder_format: Optional[Tuple[str, str]] = None,
    glossary_block: str = "",
    continuity_block: str = "",
) -> PromptPair:
    """
    Generate the translation prompt with all contextual elements.

    Args:
        main_content: The text to translate
        context_before: Text appearing before main_content for context
        context_after: Text appearing after main_content for context
        previous_translation_context: Previously translated text for consistency
        source_language: Source language name
        target_language: Target language name
        translate_tag_in: Opening tag for translation output
        translate_tag_out: Closing tag for translation output
        has_placeholders: If True, includes placeholder preservation instructions (for EPUB HTML tags)
        prompt_options: Optional dict with prompt customization options:
            - preserve_technical_content: If True, includes instructions to NOT translate
              code, paths, URLs, etc. (for technical documents)
        placeholder_format: Optional tuple of (prefix, suffix) for placeholders.
            e.g., ('[', ']') for [0] format or ('[[', ']]') for [[0]] format.
            If None, uses default [[0]] format
        glossary_block: Optional per-chunk glossary instructions.
        continuity_block: Optional compact literary continuity memory.

    Returns:
        PromptPair: A named tuple with 'system' and 'user' prompts
    """
    # Initialize prompt_options if not provided
    if prompt_options is None:
        prompt_options = {}

    # Extract custom instructions if provided
    custom_instructions = prompt_options.get('custom_instructions', '')
    source_language_label = _source_language_prompt_label(source_language)
    role_language = _role_language(target_language, prompt_options)

    # Get target-language-specific example text for output format
    example_texts = {
        "chinese": "您翻译的文本在这里" if not has_placeholders else f"您翻译的文本在这里，所有{TAG0}标记都精确保留",
        "french": "Votre texte traduit ici" if not has_placeholders else f"Votre texte traduit ici, tous les marqueurs {TAG0} sont préservés exactement",
        "spanish": "Su texto traducido aquí" if not has_placeholders else f"Su texto traducido aquí, todos los marcadores {TAG0} se preservan exactamente",
        "german": "Ihr übersetzter Text hier" if not has_placeholders else f"Ihr übersetzter Text hier, alle {TAG0}-Markierungen werden genau beibehalten",
        "japanese": "翻訳されたテキストはこちら" if not has_placeholders else f"翻訳されたテキストはこちら、すべての{TAG0}マーカーは正確に保持されます",
        "italian": "Il tuo testo tradotto qui" if not has_placeholders else f"Il tuo testo tradotto qui, tutti i marcatori {TAG0} sono conservati esattamente",
        "portuguese": "Seu texto traduzido aqui" if not has_placeholders else f"Seu texto traduzido aqui, todos os marcadores {TAG0} são preservados exatamente",
        "russian": "Ваш переведенный текст здесь" if not has_placeholders else f"Ваш переведенный текст здесь, все маркеры {TAG0} сохранены точно",
        "korean": "번역된 텍스트는 여기에" if not has_placeholders else f"번역된 텍스트는 여기에, 모든 {TAG0} 마커는 정확히 보존됩니다",
    }

    # Try to match target language to get appropriate example
    target_lang_lower = target_language.lower()
    example_format_text = example_texts.get(target_lang_lower, "Your translated text here")

    # Build the output format section outside the f-string to avoid backslash issues in Python 3.11
    output_format_section = _get_output_format_section(
        translate_tag_in,
        translate_tag_out,
        INPUT_TAG_IN,
        INPUT_TAG_OUT,
        additional_rules="",
        example_format=example_format_text
    )

    # Build placeholder preservation section dynamically based on languages
    if has_placeholders:
        placeholder_section = build_placeholder_section(source_language_label, target_language, placeholder_format)
    else:
        placeholder_section = ""

    # Build optional prompt sections based on prompt_options
    optional_sections = _build_optional_prompt_sections(prompt_options)
    locale_section = _target_locale_section(target_language, prompt_options)
    literary_style_contract = _literary_style_contract_section(target_language, prompt_options)
    transform_mode = str(prompt_options.get("text_transform_mode") or "").strip().lower()
    transform_instructions = build_text_transform_instructions(prompt_options, target_language)
    transform_extra_rule = (
        "Do not add separate notes, prefaces, censorship, or unsupported facts; "
        "integrated explanatory rewriting is allowed only when the active mode asks for it."
        if _transform_allows_explanatory_expansion(prompt_options)
        else "Do not add explanations, notes, summaries, censorship, or unsupported facts"
    )
    profile_section = build_profile_instruction_block(
        prompt_options,
        phase="modernize" if transform_mode == "modernize" else "translation",
        target_language=target_language,
    )

    # Build custom instructions section
    custom_instructions_section = ""
    if custom_instructions and custom_instructions.strip():
        custom_instructions_section = f"""# ⚠️ MANDATORY STYLE INSTRUCTIONS - ABSOLUTE PRIORITY ⚠️

**These instructions override ALL other guidelines. Non-compliance = FAILURE.**

{custom_instructions.strip()}

⚠️ Apply to EVERY word you translate. Zero exceptions. ⚠️

"""

    if transform_mode:
        mode_label = str(
            prompt_options.get("text_transform_label")
            or _TEXT_TRANSFORM_MODE_SPECS.get(transform_mode, {}).get("label")
            or transform_mode.replace("_", " ").title()
        ).strip()
        system_prompt = f"""You are a professional {role_language} same-language transformation editor.

{UNTRUSTED_BOOK_CONTENT_SECTION}

{custom_instructions_section}# TRANSFORMATION TASK

Transform {source_language_label} text into the requested {target_language} version using mode: {mode_label}.
This is not a translation-between-languages prompt and not a light copyedit prompt.
This is an active rewriting task: if the mode/profile calls for modernization,
"the prose is already good" is NOT a reason to skip the transformation.

**PRIORITY ORDER:**
1. Preserve every real unit of source content: facts, names, numbers, chronology, dialogue roles, arguments, formulas, and citations
2. Apply the active transformation mode/profile at the requested strength
3. Produce natural, publishable {target_language} prose
4. Preserve meaningful structure: paragraphs, headings, lists, quotations, and displayed formulas
5. {transform_extra_rule}

**QUALITY CHECK:**
- Does the output satisfy the active transformation mode/profile?
- Is every source detail still present?
- Does it sound natural to a native {target_language} reader?
- If this is modernization, did you modernize syntax and diction rather than only spelling?

**LAYOUT PRESERVATION:**
- Keep meaningful text layout, spacing, line breaks, and indentation
- Preserve placeholder markers exactly when present
- Write the transformed output in {target_language.upper()}
{optional_sections}
{locale_section}
{literary_style_contract}
{transform_instructions}
{placeholder_section}

# FINAL REMINDER: YOUR OUTPUT LANGUAGE

**YOU MUST OUTPUT {target_language.upper()}.**
Do NOT answer in any other language.

{output_format_section}"""

        previous_translation_block_text = ""
        if previous_translation_context and previous_translation_context.strip():
            previous_translation_block_text = f"""# CONTEXT - Previous Transformed Passage

For consistency and natural flow, here's what came immediately before:

{previous_translation_context}

"""

        continuity_section = f"{continuity_block}\n" if continuity_block and continuity_block.strip() else ""
        glossary_section = f"{glossary_block}\n" if glossary_block and glossary_block.strip() else ""
        surrounding_context_section = _build_surrounding_context_block(
            context_before,
            context_after,
            main_content,
        )

        user_prompt = f"""{previous_translation_block_text}{surrounding_context_section}{continuity_section}{glossary_section}# TEXT TO TRANSFORM

{INPUT_TAG_IN}
{main_content}
{INPUT_TAG_OUT}

REMINDER: Output ONLY your transformed text in this exact format:
{translate_tag_in}
your transformed text here
{translate_tag_out}

Start with {translate_tag_in} and end with {translate_tag_out}. Nothing before or after.

Provide the transformed text now:"""

        return PromptPair(system=system_prompt.strip(), user=user_prompt.strip())

    # SYSTEM PROMPT - Role and instructions (stable across requests)
    system_prompt = f"""You are a professional {role_language} translator and writer.

{UNTRUSTED_BOOK_CONTENT_SECTION}

{custom_instructions_section}# TRANSLATION PRINCIPLES

Translate {source_language_label} to {target_language}. Output only the translation.

**PRIORITY ORDER:**
1. Preserve exact names, facts, numbers, dates, chronology, and causal links
2. Preserve every explicit measurement's original numeric value and unit; a localized conversion may follow in parentheses, but must not replace or round away the source measurement
3. Match original tone and formality
4. Use natural {target_language} phrasing - never word-for-word
5. Fix grammar/spelling errors in output
6. Translate idioms to {target_language} equivalents
7. Translate source-language chapter titles, standalone work-title mentions,
   series labels, roles, and descriptive headings unless the active approved
   glossary explicitly marks a conventional name for preservation. In notes,
   references, and bibliographies, preserve the complete authentic title of a
   cited work, journal, publisher, platform, and registry identifier exactly;
   translate the surrounding explanatory prose. Never produce a partly
   translated or mixed-language bibliographic title. Quoted content from
   interviews, speeches, social-media posts, letters, and messages is prose,
   not a work title: translate that quoted content completely.

**QUALITY CHECK:**
- Does it sound natural to a native {target_language} speaker?
- Are all details from the original included?
- Does punctuation follow {target_language} conventions?

If unsure between literal and natural phrasing: **choose natural**.

**LAYOUT PRESERVATION:**
- Keep the exact text layout, spacing, line breaks, and indentation
- **WRITE YOUR TRANSLATION IN {target_language.upper()} - THIS IS MANDATORY**
{optional_sections}
{locale_section}
{literary_style_contract}
{profile_section}
{placeholder_section}

# FINAL REMINDER: YOUR OUTPUT LANGUAGE

**YOU MUST TRANSLATE INTO {target_language.upper()}.**
Your entire translation output must be written in {target_language}.
Do NOT write in {source_language_label} or any other language - ONLY {target_language.upper()}.

{output_format_section}"""

    # USER PROMPT - Context and content to translate (varies per request)
    previous_translation_block_text = ""
    if previous_translation_context and previous_translation_context.strip():
        previous_translation_block_text = f"""# CONTEXT - Previous Paragraph

For consistency and natural flow, here's what came immediately before:

{previous_translation_context}

"""

    # Glossary block lives in the user prompt: it changes per chunk, so
    # keeping it out of the system prompt lets the system prompt stay
    # stable and cacheable across chunks.
    continuity_section = f"{continuity_block}\n" if continuity_block and continuity_block.strip() else ""
    glossary_section = f"{glossary_block}\n" if glossary_block and glossary_block.strip() else ""
    surrounding_context_section = _build_surrounding_context_block(
        context_before,
        context_after,
        main_content,
    )

    user_prompt = f"""{previous_translation_block_text}{surrounding_context_section}{continuity_section}{glossary_section}# TEXT TO TRANSLATE

{INPUT_TAG_IN}
{main_content}
{INPUT_TAG_OUT}

REMINDER: Output ONLY your translation in this exact format:
{translate_tag_in}
your translation here
{translate_tag_out}

Start with {translate_tag_in} and end with {translate_tag_out}. Nothing before or after.

Provide your translation now:"""

    return PromptPair(system=system_prompt.strip(), user=user_prompt.strip())


# ============================================================================
# GLOSSARY: NER EXTRACTION PROMPT (Phase 2)
# ============================================================================

NER_TAG_IN = "<NER_JSON>"
NER_TAG_OUT = "</NER_JSON>"


def generate_ner_extraction_prompt(
    text: str,
    source_language: str = "Chinese",
    target_language: str = "English",
) -> PromptPair:
    """
    Build a prompt that asks the LLM to extract recurring proper-noun entities
    (characters, locations, organizations/sects, items) from a sample of source
    text, along with a suggested target-language translation for each.

    Output is wrapped in <NER_JSON>...</NER_JSON> with a strict schema. The
    parser is permissive (handles markdown fences, missing tags, partial JSON).
    """
    system_prompt = f"""You are a literary entity extractor. Your job is to read a passage written in {source_language} and identify recurring proper nouns that a translator would want to keep consistent across an entire book.

{UNTRUSTED_BOOK_CONTENT_SECTION}

# CATEGORIES (use exactly these labels)

- "character"     — named persons (李凡, Li Fan, Captain Ahab)
- "location"      — places, regions, buildings (青玄宗大殿, Mount Tai)
- "organization"  — sects, schools, clans, factions, companies (青玄宗, Heavenly Sword Gate)
- "item"          — named artifacts, weapons, treasures, techniques (混沌珠, Excalibur)
- "title"         — honorifics or named ranks tied to a person (Elder, 长老, Master)
- "other"         — anything else worth keeping consistent (events, magical formulas)

# RULES

1. Extract ONLY proper nouns or named concepts that look likely to recur. Skip generic words.
2. Do NOT translate common nouns or descriptive phrases — only named entities.
3. For each entity, propose ONE canonical {target_language} translation. Use the standard romanization or the most natural literary rendering. Keep the proposal concise.
4. Deduplicate: if the same entity appears multiple times in the passage, list it once.
5. If you are unsure about an entry, omit it rather than guessing.
6. Preserve the original {source_language} form exactly as it appears in the text (no extra spaces, no normalization).

# OUTPUT FORMAT

Return ONLY a JSON array wrapped between {NER_TAG_IN} and {NER_TAG_OUT}. No prose, no explanations.

Each array element MUST be an object with these keys:
  - "source"   (string, required) — the entity in {source_language}
  - "target"   (string, required) — the proposed {target_language} translation
  - "category" (string, required) — one of the labels listed above

Example:
{NER_TAG_IN}
[
  {{"source": "李凡", "target": "Li Fan", "category": "character"}},
  {{"source": "青玄宗", "target": "Qingxuan Sect", "category": "organization"}}
]
{NER_TAG_OUT}

If no entities are found, return an empty array: {NER_TAG_IN}[]{NER_TAG_OUT}.

Do NOT wrap the JSON in markdown code fences. Do NOT add commentary before or after the tags."""

    user_prompt = f"""# SOURCE TEXT ({source_language})

{INPUT_TAG_IN}
{text}
{INPUT_TAG_OUT}

Extract the recurring named entities now. Output the JSON array between {NER_TAG_IN} and {NER_TAG_OUT}, nothing else."""

    return PromptPair(system=system_prompt.strip(), user=user_prompt.strip())


def generate_refinement_prompt(
    draft_translation: str,
    context_before: str = "",
    context_after: str = "",
    previous_refined_context: str = "",
    target_language: str = "English",
    translate_tag_in: str = TRANSLATE_TAG_IN,
    translate_tag_out: str = TRANSLATE_TAG_OUT,
    has_placeholders: bool = True,
    prompt_options: dict = None,
    placeholder_format: Optional[Tuple[str, str]] = None,
    additional_instructions: str = "",
    glossary_block: str = "",
    continuity_block: str = "",
) -> PromptPair:
    """
    Generate a refinement prompt for final editorial review.

    This is used for a second pass where the LLM copyedits a first-pass
    translation, focusing on correctness, natural flow, and restraint.

    Args:
        draft_translation: The first-pass translation to refine
        context_before: Previously refined text for context (default: "")
        context_after: Text appearing after for context (default: "")
        previous_refined_context: Last refined text for consistency (default: "")
        target_language: Target language name
        translate_tag_in: Opening tag for translation output
        translate_tag_out: Closing tag for translation output
        has_placeholders: If True, includes placeholder preservation instructions
        prompt_options: Optional dict with prompt customization options
        placeholder_format: Optional tuple of (prefix, suffix) for placeholders.
            e.g., ('[', ']') for [0] format or ('[[', ']]') for [[0]] format.
            If None, uses default [[0]] format
        additional_instructions: Additional refinement instructions to include in the prompt (default: "")
        glossary_block: Optional per-chunk glossary instructions.
        continuity_block: Optional compact literary continuity memory.

    Returns:
        PromptPair: A named tuple with 'system' and 'user' prompts
    """
    if prompt_options is None:
        prompt_options = {}
    role_language = _role_language(target_language, prompt_options)
    transform_mode = str(prompt_options.get("text_transform_mode") or "").strip().lower()

    # Get target-language-specific example text for output format
    example_texts = {
        "chinese": "您润色后的文本在这里",
        "french": "Votre texte affiné ici",
        "spanish": "Su texto refinado aquí",
        "german": "Ihr verfeinerter Text hier",
        "japanese": "洗練されたテキストはこちら",
        "italian": "Il tuo testo raffinato qui",
        "portuguese": "Seu texto refinado aqui",
        "russian": "Ваш улучшенный текст здесь",
        "korean": "다듬어진 텍스트는 여기에",
    }

    target_lang_lower = target_language.lower()
    example_format_text = example_texts.get(target_lang_lower, "Your refined text here")

    output_format_section = _get_output_format_section(
        translate_tag_in,
        translate_tag_out,
        INPUT_TAG_IN,
        INPUT_TAG_OUT,
        additional_rules="",
        example_format=example_format_text
    )

    # Build placeholder preservation section if needed
    if has_placeholders:
        placeholder_section = build_placeholder_section(target_language, target_language, placeholder_format)
    else:
        placeholder_section = ""

    # Build optional prompt sections
    optional_sections = _build_optional_prompt_sections(prompt_options)
    locale_section = _target_locale_section(target_language, prompt_options)
    literary_style_contract = _literary_style_contract_section(target_language, prompt_options)
    profile_phase = "voice_restoration" if transform_mode else "translation_refinement"
    profile_section = build_profile_instruction_block(
        prompt_options,
        phase=profile_phase,
        target_language=target_language,
    )
    profile_override_section = ""
    if (
        profile_enabled(prompt_options)
        and str(prompt_options.get("text_transform_mode") or "").strip().lower() == "modernize"
    ):
        profile_override_section = """
# ACTIVE BOOK PROFILE OVERRIDE

The active book profile is higher priority than the default light-copyedit
posture in this refinement prompt.

If the profile or audit asks for high modernization, perform real sentence-level
and paragraph-level restructuring where needed. Do not limit yourself to small
wording edits when syntax, naturalness, locale, or profile voice are below the
target. Preserve all meaning and structure, but modernize the prose according to
the profile.
""".strip()
    output_must_be_extra_rule = (
        "Free of separate notes, prefaces, censorship, or unsupported additions; "
        "supported explanations must be integrated into the prose."
        if _transform_allows_explanatory_expansion(prompt_options)
        else "Free of explanations, notes, summaries, censorship, or unsupported additions"
    )

    # Add additional instructions section if provided
    additional_instructions_section = ""
    if additional_instructions and additional_instructions.strip():
        additional_instructions_section = f"""

# ADDITIONAL REFINEMENT INSTRUCTIONS

{additional_instructions.strip()}"""

    if transform_mode:
        mode_label = str(
            prompt_options.get("text_transform_label")
            or _TEXT_TRANSFORM_MODE_SPECS.get(transform_mode, {}).get("label")
            or transform_mode.replace("_", " ").title()
        ).strip()
        system_prompt = f"""You are an elite {role_language} literary transformation editor.

{UNTRUSTED_BOOK_CONTENT_SECTION}

# YOUR TASK: SAME-LANGUAGE TRANSFORMATION

You will receive a {target_language} text and must perform the active same-language transformation: {mode_label}.
This is a first-class transformation task, not a light copyedit pass.
This is an active rewriting task: if the mode/profile calls for modernization,
"the prose is already good" is NOT a reason to skip the transformation.

**YOUR OUTPUT MUST BE:**
- A complete transformed version in {target_language}
- Faithful to every real unit of meaning in the input
- Governed by the active transformation mode, book profile, and approved glossary
- {output_must_be_extra_rule}

# TRANSFORMATION PRINCIPLES

**PRIORITY ORDER:**
1. Preserve source content, facts, names, numbers, speaker relationships, sequence, and meaningful structure
2. Apply the active transformation/profile at the requested strength
3. Repair mechanical defects, OCR artifacts, bad spacing, and punctuation when present
4. Preserve authorial voice through equivalent target-language effects, not by mechanically preserving obsolete surface forms
5. Keep terms consistent with the active glossary; do not import rules from other profiles

**WHAT TO CHANGE:**
- Language features targeted by the active mode/profile
- Awkward syntax, obsolete phrasing, OCR damage, and unnatural cadence when the mode asks for it
- Paragraph flow and punctuation when they obstruct the active editorial goal

**WHAT TO PRESERVE:**
- All factual content and meaning
- Character names and proper nouns
- Technical terms, citations, formulas, numbers, and dates
- Dialogue roles, treatment relationships, and narrative point of view
- Headings, section numbers, list items, block quotes, and displayed equations as distinct blocks

# STRUCTURE, ARTIFACTS, AND FORMULAS

**Document structure is part of the text.**
- Do NOT collapse the whole passage into one block.
- Preserve blank lines between real paragraphs unless the active profile explicitly repairs broken OCR layout.
- Keep obvious headings or section labels on their own line, with a blank line after them.
- Keep lists as lists; do not merge list items into prose.

**Remove artifacts; never invent replacement symbols.**
- Delete stray replacement boxes or unsupported-glyph markers such as ■, □, █, ▪, or �.
- Do NOT multiply a suspicious symbol from the draft.
- Do NOT introduce decorative separators, bullets, boxes, or placeholder-looking symbols unless they belong to the document structure.

**Formula handling.**
- Preserve variables, Greek letters, numbers, operators, citations, and units.
- If a formula is readable, keep it readable in plain text or LaTeX-like notation.
- If OCR damaged a formula, make only conservative repairs such as spacing and notation clarity; do not guess missing values.
- Keep formulas separate from running prose when they were displayed or equation-like in the draft.
{optional_sections}
{locale_section}
{literary_style_contract}
{profile_section}
{profile_override_section}
{placeholder_section}
{additional_instructions_section}

# CRITICAL REMINDER

You are transforming text in {target_language.upper()} according to the active mode.
Do not fall back to light proofreading if the active mode/profile calls for sentence-level restructuring.

**PLACEHOLDER PRESERVATION IS ABSOLUTELY CRITICAL:**
If the input contains ANY placeholders (like [id0], [id1], etc.), preserve them EXACTLY.
Removing or corrupting placeholders will corrupt the document structure.

{output_format_section}"""

        previous_context_block = ""
        if previous_refined_context and previous_refined_context.strip():
            previous_context_block = f"""# CONTEXT - Previous Transformed Passage

For consistency and natural flow, here's what came immediately before:

{previous_refined_context}

"""

        continuity_section = f"{continuity_block}\n" if continuity_block and continuity_block.strip() else ""
        glossary_section = f"{glossary_block}\n" if glossary_block and glossary_block.strip() else ""
        surrounding_context_section = _build_surrounding_context_block(
            context_before,
            context_after,
            draft_translation,
            heading="SURROUNDING DRAFT CONTEXT",
        )

        user_prompt = f"""{previous_context_block}{surrounding_context_section}{continuity_section}{glossary_section}# TEXT TO TRANSFORM

The following {target_language} text must be transformed with mode: {mode_label}.
Apply the active mode/profile directly. Preserve all real content.

{INPUT_TAG_IN}
{draft_translation}
{INPUT_TAG_OUT}

REMINDER: Output ONLY your transformed text in this exact format:
{translate_tag_in}
your transformed text here
{translate_tag_out}

Start with {translate_tag_in} and end with {translate_tag_out}. Nothing before or after.

Provide the transformed version now:"""

        return PromptPair(system=system_prompt.strip(), user=user_prompt.strip())

    # SYSTEM PROMPT for refinement
    system_prompt = f"""You are an elite {role_language} literary editor, proofreader, and prose stylist.

{UNTRUSTED_BOOK_CONTENT_SECTION}

# YOUR TASK: FINAL EDITORIAL REVIEW

You will receive a DRAFT {target_language} text that may need cleanup and light improvement.
Your job is to act like a final human editor: correct the text when it benefits
from correction, and leave it close to unchanged when it is already good.

**THE INPUT IS:**
- A {target_language} translation or OCR-derived text
- It may contain awkward LLM phrasing, mistranslation-like roughness, OCR artifacts, line-break damage, or spacing defects
- Some passages may already be good and should be preserved with minimal edits

**YOUR OUTPUT MUST BE:**
- Fluent, natural {target_language} prose
- Clean, readable {target_language} prose
- Faithful to the draft's content, tone, paragraph order, and level of formality

# REFINEMENT PRINCIPLES

**PRIORITY ORDER:**
1. **Preserve meaning and content** - Do not add, remove, summarize, or reinterpret facts
2. **Editorial judgment** - Correct grammar, punctuation, awkward phrasing, and style only when needed
3. **Repair mechanical defects** - Fix OCR artifacts, broken words, bad spacing, and obvious punctuation errors
4. **Improve only awkward wording** - Prefer small edits over wholesale rewriting
5. **Consistency** - Preserve names, terms, capitalization choices, and paragraph order

**WHAT TO FIX:**
- Broken OCR line wraps and hyphenated words
- Missing spaces after punctuation and repeated spaces
- Obvious grammar or spelling errors
- Awkward literal phrases when a small edit makes them natural
- LLM-ish stiffness, repetition, or unnatural cadence when it weakens the prose
- Source-language titles, labels, or ordinary terms left untranslated without an
  explicit approved glossary/profile preservation instruction

**WHAT TO PRESERVE:**
- All factual content and meaning
- Character names and proper nouns
- Technical terms (if any)
- Paragraph boundaries unless they are clearly OCR damage
- Headings, section numbers, list items, block quotes, and displayed equations as distinct blocks
- Inline formulas as inline formulas, and displayed formulas on their own line
- Good sentences that do not need editing

# STRUCTURE, ARTIFACTS, AND FORMULAS

**Document structure is part of the text.**
- Do NOT collapse the whole passage into one block.
- Preserve blank lines between real paragraphs.
- Keep obvious headings or section labels on their own line, with a blank line after them.
- Keep lists as lists; do not merge list items into prose.

**Remove artifacts; never invent replacement symbols.**
- Delete stray replacement boxes or unsupported-glyph markers such as ■, □, █, ▪, or �.
- Do NOT multiply a suspicious symbol from the draft. If one box appears where a character is unreadable, remove the box and keep the surrounding sentence/formula readable.
- Do NOT introduce decorative separators, bullets, boxes, or placeholder-looking symbols unless they already belong to the document structure.

**Formula handling.**
- Preserve variables, Greek letters, numbers, operators, citations, and units.
- If a formula is readable, keep it readable in plain text or LaTeX-like notation.
- If OCR damaged a formula, make only conservative repairs such as spacing and notation clarity (`10^-9`, `d_model^-0.5`, `β1 = 0.9`); do not guess missing values.
- Keep formulas separate from running prose when they were displayed or equation-like in the draft.
{optional_sections}
{locale_section}
{literary_style_contract}
{profile_section}
{profile_override_section}
{placeholder_section}
{additional_instructions_section}

# CRITICAL REMINDER

You are NOT translating - you are copyediting in {target_language.upper()}.
If a sentence is already clear and natural, keep it close to the original.
Do not make the text more ornate just to show improvement. Editorial restraint is part of the task.

**⚠️ PLACEHOLDER PRESERVATION IS ABSOLUTELY CRITICAL:**
If the input contains ANY placeholders (like [id0], [id1], etc.), you MUST preserve them EXACTLY.
Removing or corrupting placeholders will corrupt the document structure.
Your refinement MUST maintain the exact same placeholders in the exact same positions.

{output_format_section}"""

    # USER PROMPT
    previous_context_block = ""
    if previous_refined_context and previous_refined_context.strip():
        previous_context_block = f"""# CONTEXT - Previous Refined Paragraph

For consistency and natural flow, here's what came immediately before:

{previous_refined_context}

"""

    # Glossary block injected here (per-chunk dynamic) so the system prompt
    # stays cacheable across chunks.
    continuity_section = f"{continuity_block}\n" if continuity_block and continuity_block.strip() else ""
    glossary_section = f"{glossary_block}\n" if glossary_block and glossary_block.strip() else ""

    user_prompt = f"""{previous_context_block}{continuity_section}{glossary_section}# DRAFT TO REFINE

The following {target_language} text is ready for final editorial review.
Correct grammar, punctuation, style, and mechanical defects only where needed; preserve content, voice, and good passages.

{INPUT_TAG_IN}
{draft_translation}
{INPUT_TAG_OUT}

REMINDER: Output ONLY your refined text in this exact format:
{translate_tag_in}
your refined text here
{translate_tag_out}

Start with {translate_tag_in} and end with {translate_tag_out}. Nothing before or after.

Provide your refined version now:"""

    return PromptPair(system=system_prompt.strip(), user=user_prompt.strip())


def generate_subtitle_refinement_block_prompt(
    subtitle_blocks: List[Tuple[int, str]],
    previous_refined_block: str = "",
    target_language: str = "English",
    translate_tag_in: str = TRANSLATE_TAG_IN,
    translate_tag_out: str = TRANSLATE_TAG_OUT,
    additional_instructions: str = "",
    glossary_block: str = "",
) -> PromptPair:
    """
    Generate a refinement prompt for multiple subtitles in a single LLM call.

    Mirrors generate_subtitle_block_prompt but rewrites each draft subtitle into
    polished target-language prose while preserving the [index] markers.

    Args:
        subtitle_blocks: List of tuples (local_index, draft_translated_text)
        previous_refined_block: Last refined block for continuity
        target_language: Target language
        translate_tag_in: Opening tag for refinement output
        translate_tag_out: Closing tag for refinement output
        additional_instructions: Extra refinement guidance
        glossary_block: Optional glossary block

    Returns:
        PromptPair: A named tuple with 'system' and 'user' prompts
    """
    subtitle_additional_rules = _SUBTITLE_FORMAT_RULES
    subtitle_example_format = "[0]Première ligne affinée\n[1]Deuxième ligne affinée"
    subtitle_output_format_section = _get_output_format_section(
        translate_tag_in,
        translate_tag_out,
        INPUT_TAG_IN,
        INPUT_TAG_OUT,
        additional_rules=subtitle_additional_rules,
        example_format=subtitle_example_format,
    )

    additional_instructions_section = ""
    if additional_instructions and additional_instructions.strip():
        additional_instructions_section = f"""

# ADDITIONAL REFINEMENT INSTRUCTIONS

{additional_instructions.strip()}"""
    locale_section = _target_locale_section(target_language)

    system_prompt = f"""You are an elite {target_language} subtitle editor and dialogue stylist.

{UNTRUSTED_BOOK_CONTENT_SECTION}

# YOUR TASK: REFINE A BLOCK OF SUBTITLES

You will receive a block of DRAFT {target_language} subtitles, each prefixed with an [index] marker.
Your job is to REWRITE each subtitle with natural, idiomatic {target_language} dialogue while
preserving the index markers and the one-subtitle-per-marker structure.

**THE INPUT IS:**
- A block of draft {target_language} subtitles, possibly literal or awkward
- Each subtitle is prefixed with [N] where N is its local index

**YOUR OUTPUT MUST BE:**
- The same number of subtitles, each prefixed with the SAME [N] marker
- Fluent, natural spoken {target_language} suited to subtitling

# REFINEMENT PRINCIPLES

**PRIORITY ORDER:**
1. **Natural dialogue** - sound like real {target_language} speech, not translation
2. **Reading speed** - keep subtitle length viewer-friendly
3. **Continuity** - terminology and tone consistent across the block
4. **Preserve meaning** - keep the original meaning intact while improving style

**WHAT TO FIX:**
- Awkward literal phrasing -> natural {target_language} expressions
- Repetitive vocabulary that is clearly an artefact of literal translation -> varied word choices
- Unnatural word order -> proper {target_language} syntax

**WHAT TO PRESERVE:**
- The [index] markers exactly as given
- All factual content and meaning
- Character names and proper nouns
- The one-subtitle-per-[index] structure (no merging, no splitting)
- Intentional repetitions (e.g. "No. No. No.") and dialogue dashes ("- ...\\n- ...") when present in the draft
- Inline formatting tags and any \\n line breaks inside a subtitle{additional_instructions_section}
{locale_section}

# CRITICAL REMINDERS

You are NOT translating - you are REWRITING in {target_language.upper()}.
The input is already in {target_language}, but possibly poorly written.
Your output must be polished, natural {target_language} dialogue.

**Index markers are MANDATORY:** every input [N] must appear exactly once in the output,
in the same order, followed by the refined text for that subtitle.

{subtitle_output_format_section}"""

    previous_refined_block_text = ""
    if previous_refined_block and previous_refined_block.strip():
        previous_refined_block_text = f"""# CONTEXT - Previous Refined Block

For continuity and consistency, here's the previous refined block:

{previous_refined_block}

"""

    formatted_subtitles = [f"[{idx}]{text}" for idx, text in subtitle_blocks]
    formatted_subtitles_text = "\n".join(formatted_subtitles)

    glossary_section = f"{glossary_block}\n" if glossary_block and glossary_block.strip() else ""

    user_prompt = f"""{previous_refined_block_text}{glossary_section}# SUBTITLES TO REFINE

{INPUT_TAG_IN}
{formatted_subtitles_text}
{INPUT_TAG_OUT}

REMINDER: Output format must be:
{translate_tag_in}
[0]refined subtitle 0
[1]refined subtitle 1
{translate_tag_out}

Start with {translate_tag_in} and end with {translate_tag_out}. Nothing before or after.

Provide your refined block now:"""

    return PromptPair(system=system_prompt.strip(), user=user_prompt.strip())


def generate_subtitle_block_prompt(
    subtitle_blocks: List[Tuple[int, str]],
    previous_translation_block: str,
    source_language: str = "English",
    target_language: str = "English",
    translate_tag_in: str = TRANSLATE_TAG_IN,
    translate_tag_out: str = TRANSLATE_TAG_OUT,
    custom_instructions: str = "",
    glossary_block: str = "",
) -> PromptPair:
    """
    Generate translation prompt for multiple subtitle blocks with index markers.

    Args:
        subtitle_blocks: List of tuples (index, text) for subtitles to translate
        previous_translation_block: Previous translated block for context
        source_language: Source language
        target_language: Target language
        translate_tag_in: Opening tag for translation output
        translate_tag_out: Closing tag for translation output
        custom_instructions: Additional custom translation instructions

    Returns:
        PromptPair: A named tuple with 'system' and 'user' prompts
    """
    source_language_label = _source_language_prompt_label(source_language)

    # Build the output format section outside the f-string to avoid backslash issues in Python 3.11
    subtitle_additional_rules = _SUBTITLE_FORMAT_RULES
    subtitle_example_format = "[1]第一行翻译文本\n[2]第二行翻译文本"
    subtitle_output_format_section = _get_output_format_section(
        translate_tag_in,
        translate_tag_out,
        INPUT_TAG_IN,
        INPUT_TAG_OUT,
        additional_rules=subtitle_additional_rules,
        example_format=subtitle_example_format
    )

    # Build custom instructions section if provided
    custom_instructions_section = ""
    if custom_instructions and custom_instructions.strip():
        custom_instructions_section = f"""

# ⚠️ MANDATORY STYLE INSTRUCTIONS - ABSOLUTE PRIORITY ⚠️

**These instructions override ALL other guidelines. Non-compliance = FAILURE.**

{custom_instructions.strip()}

⚠️ Apply to EVERY subtitle. Zero exceptions. ⚠️
"""
    locale_section = _target_locale_section(target_language)

    # SYSTEM PROMPT - Role and instructions for subtitle translation
    system_prompt = f"""You are a professional {target_language} subtitle translator and dialogue adaptation specialist.

{UNTRUSTED_BOOK_CONTENT_SECTION}

# CRITICAL: TARGET LANGUAGE IS {target_language.upper()}

**YOUR SUBTITLE TRANSLATION MUST BE WRITTEN ENTIRELY IN {target_language.upper()}.**

You are translating subtitles FROM {source_language_label} TO {target_language}.
Your output must be in {target_language} ONLY - do NOT use any other language.

# SUBTITLE TRANSLATION PRINCIPLES

**Quality Standards:**
- Translate dialogues naturally and conversationally for {target_language} viewers
- Adapt expressions, slang, and cultural references appropriately
- Keep subtitle length readable (typically 40-42 characters per line)
- Restructure sentences naturally (avoid word-by-word translation)
- Maintain speaker's tone, personality, and emotion
- **WRITE YOUR TRANSLATION IN {target_language.upper()} - THIS IS MANDATORY**

**Subtitle-Specific Rules:**
- Prioritize clarity and reading speed over literal accuracy
- Condense when necessary without losing meaning
- Use natural, spoken {target_language} (not formal written style)
- Preserve intentional repetitions (e.g. "No. No. No.") and dialogue dashes ("- ...\\n- ...") from the source
- Preserve inline formatting tags (<i>, <b>, <font ...>, {{\\an8}}, etc.) and any \\n line breaks inside a subtitle{custom_instructions_section}
{locale_section}

# FINAL REMINDER: YOUR OUTPUT LANGUAGE

**YOU MUST TRANSLATE INTO {target_language.upper()}.**
Your entire subtitle translation must be written in {target_language}.
Do NOT write in {source_language_label} or any other language - ONLY {target_language.upper()}.

{subtitle_output_format_section}"""

    # USER PROMPT - Context and subtitles to translate
    previous_translation_block_text = ""
    if previous_translation_block and previous_translation_block.strip():
        previous_translation_block_text = f"""# CONTEXT - Previous Subtitle Block

For continuity and consistency, here's the previous subtitle block:

{previous_translation_block}

"""

    # Format subtitle blocks with indices
    formatted_subtitles = [f"[{idx}]{text}" for idx, text in subtitle_blocks]

    # Join subtitles outside f-string to avoid Python 3.11 backslash issues
    formatted_subtitles_text = "\n".join(formatted_subtitles)

    # Glossary block in user prompt (dynamic per chunk).
    glossary_section = f"{glossary_block}\n" if glossary_block and glossary_block.strip() else ""

    user_prompt = f"""{previous_translation_block_text}{glossary_section}# SUBTITLES TO TRANSLATE

{INPUT_TAG_IN}
{formatted_subtitles_text}
{INPUT_TAG_OUT}

REMINDER: Output format must be:
{translate_tag_in}
[1]translated subtitle 1
[2]translated subtitle 2
{translate_tag_out}

Start with {translate_tag_in} and end with {translate_tag_out}. Nothing before or after.

Provide your translation now:"""

    return PromptPair(system=system_prompt.strip(), user=user_prompt.strip())


# ============================================================================
# PLACEHOLDER CORRECTION PROMPT
# ============================================================================

def generate_placeholder_correction_prompt(
    original_text: str,
    translated_text: str,
    specific_errors: str,
    source_language: str,
    target_language: str,
    expected_count: int,
    placeholder_format: Optional[Tuple[str, str]] = None
) -> PromptPair:
    """
    Generate a prompt for correcting placeholder errors in a translation.

    This prompt is used when a translation has placeholder issues (missing,
    duplicated, mutated, or out of order). It asks the LLM to fix ONLY the
    placeholder positions without modifying the translated text.

    Args:
        original_text: Source text with correct placeholders
        translated_text: Translation with placeholder errors
        specific_errors: Detailed error description (generated by build_specific_error_details)
        source_language: Source language name (e.g., "English")
        target_language: Target language name (e.g., "French")
        expected_count: Number of placeholders expected (0 to expected_count-1)
        placeholder_format: Optional tuple of (prefix, suffix) for placeholders.
            e.g., ('[', ']') for [0] format or ('[[', ']]') for [[0]] format.
            If None, uses default [[0]] format

    Returns:
        PromptPair: A named tuple with 'system' and 'user' prompts
    """
    # Use custom format if provided, otherwise use defaults
    if placeholder_format:
        prefix, suffix = placeholder_format
    else:
        prefix, suffix = PLACEHOLDER_PREFIX, PLACEHOLDER_SUFFIX

    # Generate dynamic placeholder examples using the correct format
    def make_placeholder(idx: int) -> str:
        return f"{prefix}{idx}{suffix}"

    max_index = expected_count - 1 if expected_count > 0 else 0
    placeholder_format_str = f"{prefix}N{suffix}"
    example_range = f"{make_placeholder(0)} to {make_placeholder(max_index)}"
    placeholder_list = ", ".join(make_placeholder(i) for i in range(min(3, expected_count)))
    if expected_count > 3:
        placeholder_list += ", etc."

    # SYSTEM PROMPT
    system_prompt = f"""You are a technical placeholder correction specialist.

{UNTRUSTED_BOOK_CONTENT_SECTION}

## YOUR TASK

A {source_language} to {target_language} translation was performed, but the placeholders were corrupted.
You must fix the placeholder positions to match the original text structure.

## PLACEHOLDER FORMAT

**CORRECT format:** {make_placeholder(0)}, {make_placeholder(1)}, {make_placeholder(2)}, etc.
- Brackets: {prefix} and {suffix}
- Sequential numbering starting from 0
- Expected range for this text: {example_range}

**FORMAT VARIATIONS:**
The system uses different placeholder formats based on text content:
- [id0], [id1], [id2]... (default - semantic markers, highest accuracy)
- /0, /1, /2... (when text contains brackets)
- $0$, $1$, $2$... (when text contains brackets and slashes)
- [[0]], [[1]], [[2]]... (legacy format)

All formats follow the same rules: preserve exact format, maintain sequential order, keep position.

## HOW TO POSITION PLACEHOLDERS

Placeholders represent HTML/XML tags. To position them correctly:

1. **Look at the ORIGINAL text** to see what content each placeholder surrounds
2. **Find the equivalent content** in the translation
3. **Place the placeholder at the same logical position** around that content

**Example:**
- Original: "{make_placeholder(0)}Hello{make_placeholder(1)} world"
- If translation is "Bonjour monde", the placeholders mark "Hello"
- Correct: "{make_placeholder(0)}Bonjour{make_placeholder(1)} monde"

## VALIDATION RULES

1. **EXACT COUNT**: Must contain exactly {expected_count} placeholders
2. **SEQUENTIAL ORDER**: Placeholders must appear in order: {placeholder_list}
3. **NO DUPLICATES**: Each placeholder must appear exactly once
4. **NO MUTATIONS**: Use ONLY the {placeholder_format_str} format
5. **POSITION MATCHING**: Each placeholder must surround the translated equivalent of what it surrounded in the original

## CRITICAL INSTRUCTIONS

- Analyze the ORIGINAL to understand what each placeholder marks
- Position placeholders around the SAME semantic content in the translation
- Do NOT add or remove words from the translation
- Keep the {target_language} text intact, only fix placeholder positions

## OUTPUT FORMAT

Your response MUST start with {CORRECTED_TAG_IN} and end with {CORRECTED_TAG_OUT}.
Include NOTHING before or after these tags."""

    # USER PROMPT
    user_prompt = f"""## ORIGINAL TEXT ({source_language}) - Reference for placeholder positions:

<ORIGINAL_TAG_IN>
{original_text}
<ORIGINAL_TAG_OUT>

## TRANSLATION WITH ERRORS ({target_language}):

<TRANSLATION_TAG_IN>
{translated_text}
<TRANSLATION_TAG_OUT>

## DETECTED ERRORS:

{specific_errors}

## YOUR TASK:

Reposition the placeholders {example_range} in the translation above.
Keep the translated text unchanged - only fix placeholder positions.

Provide your corrected version now:"""

    return PromptPair(system=system_prompt.strip(), user=user_prompt.strip())


# ============================================================================
# ALIAS FOR BACKWARDS COMPATIBILITY
# ============================================================================

def generate_post_processing_prompt(
    translated_text: str,
    target_language: str = "English",
    context_before: str = "",
    context_after: str = "",
    additional_instructions: str = "",
    has_placeholders: bool = True,
    prompt_options: dict = None,
    placeholder_format: Optional[Tuple[str, str]] = None,
    glossary_block: str = "",
    continuity_block: str = "",
) -> PromptPair:
    """
    Alias for generate_refinement_prompt with parameter name mapping.

    This function exists for backwards compatibility and to provide a more intuitive
    API for post-processing/refinement use cases.

    Args:
        translated_text: The draft translation to refine (mapped to draft_translation)
        target_language: Target language name
        context_before: Previously refined text for context
        context_after: Text appearing after for context
        additional_instructions: Additional refinement instructions
        has_placeholders: If True, includes placeholder preservation instructions
        prompt_options: Optional dict with prompt customization options
        placeholder_format: Optional tuple of (prefix, suffix) for placeholders
        glossary_block: Optional per-chunk glossary instructions.
        continuity_block: Optional compact literary continuity memory.

    Returns:
        PromptPair: A named tuple with 'system' and 'user' prompts
    """
    return generate_refinement_prompt(
        draft_translation=translated_text,
        context_before=context_before,
        context_after=context_after,
        previous_refined_context="",  # Not used in post-processing calls
        target_language=target_language,
        has_placeholders=has_placeholders,
        prompt_options=prompt_options,
        placeholder_format=placeholder_format,
        additional_instructions=additional_instructions,
        glossary_block=glossary_block,
        continuity_block=continuity_block,
    )
