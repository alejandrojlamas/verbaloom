"""
Unicode cleanup utilities for translation output.

Different text readers, EPUB engines, PDF renderers, and platform shells render
Unicode sequences inconsistently depending on font fallback and extraction
quality. This module repairs common mojibake and removes legacy invisible marks
or replacement-box artifacts that can otherwise appear as visible squares.

The normalization pass operates on FINAL output strings only (at file write
boundaries). It is idempotent — applying twice has the same effect as once.
"""

import re
from typing import Optional

from src.common.inline_markdown import parse_inline_markdown


UTF8_BOM = b'\xef\xbb\xbf'


def encode_utf8_text_download(text: str) -> bytes:
    """Encode a downloadable plain-text artifact as UTF-8 with BOM.

    Some Android viewers and file previewers misdetect extension-only ``.txt``
    downloads as Windows-1252/Latin-1 when no charset metadata is available,
    which renders Spanish accents as mojibake (``inglÃ©s``). A UTF-8 BOM is a
    pragmatic file-level hint for those readers while remaining harmless for
    normal UTF-8 consumers.
    """
    return (text or "").encode("utf-8-sig")


def ensure_utf8_bom(data: bytes) -> bytes:
    """Return bytes with a single UTF-8 BOM prefix."""
    if data.startswith(UTF8_BOM):
        return data
    return UTF8_BOM + data


# Width-zero Unicode codepoints relevant to text shaping
_ZWNJ = '‌'   # Zero-width non-joiner
_ZWJ = '‍'    # Zero-width joiner
_ZWSP = '​'   # Zero-width space
_WJ = '⁠'     # Word joiner

# Placeholder shape used by tag/equation preservation in EPUB and DOCX pipelines.
# Normalization avoids modifying tokens matching this pattern as a defensive
# measure, even though placeholders should never reach this layer.
_PLACEHOLDER_RE = re.compile(r'\[id\d+\]', re.IGNORECASE)

# SRT timestamp shape, e.g. "00:01:23,456 --> 00:01:25,789"
_SRT_TIMESTAMP_RE = re.compile(
    r'\d{2}:\d{2}:\d{2}[,.]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[,.]\d{3}'
)

# Typical UTF-8-as-Latin-1/Windows-1252 mojibake fragments:
#   traducciÃ³n -> traducción
#   Â¿QuÃ©?     -> ¿Qué?
#   â€œtextoâ€  -> “texto”
_MOJIBAKE_RUN_RE = re.compile(
    r'(?:'
    r'Ã.|Â.|'
    r'â.{1,3}|'
    r'ï¿½|'
    r'ðŸ.{1,4}'
    r')+',
    re.DOTALL,
)
_MOJIBAKE_MARKER_RE = re.compile(
    r'Ã.|Â[\u0080-\u00ff\u20ac\u2122]?|â[\u0080-\u00ff\u20ac\u2122]{1,3}|ï¿½|�|ðŸ'
)
_C1_CONTROL_RE = re.compile(r'[\u0080-\u009f]')
_CONTROL_ARTIFACT_RE = re.compile(r'[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f-\u009f]')
_BOX_ARTIFACT_RE = re.compile(
    r'[\ufffc\ufffd\u2580-\u259f\u25a0-\u25a1\u25aa-\u25ac\u25ae-\u25b0'
    r'\u25fb-\u25fe\u2b1b-\u2b1c]+'
)
_TRANSLATION_WRAPPER_ARTIFACT_RE = re.compile(
    r'(?:<|&lt;)\s*/?\s*TRANSLATION(?:ATION)*\s*(?:>|&gt;)',
    re.IGNORECASE,
)
_MARKDOWN_ASTERISK_EMPHASIS_RE = re.compile(
    r"(?<!\*)\*([^\n*]*[A-Za-zÁÉÍÓÚÜÑáéíóúüñ][^\n*]*)\*(?!\*)"
)
_DOT_LEADER_RUN_RE = re.compile(r'(?:\.\s*){5,}')
_DOT_LEADER_ONLY_PAGE_RE = re.compile(r'^\s*(?:\.\s*){3,}\d{1,5}\s*$')
_DOT_LEADER_TO_PAGE_RE = re.compile(r'\s*(?:\.\s*){5,}(?=\d{1,5}\s*$)')
_DOMAIN_TOKEN_RE = re.compile(r'(?i)^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,}$')
_SOURCE_ARTIFACT_DOMAINS = {
    '1lib.sk',
    'oceanofpdf.com',
    'z-lib.sk',
    'z-library.sk',
}
_MARKDOWN_DOMAIN_LINK_LINE_RE = re.compile(
    r'^\s*\[([^\]\n]+)\]\((https?://[^)\s]+|www\.[^)\s]+)\)\s*$',
    re.IGNORECASE,
)
_INTERNAL_NOTE_HREF_RE = re.compile(
    r'(?:^|/)(?:notes?|notas?|footnotes?|endnotes?|references?|refs?)\.(?:xhtml|html|htm)(?:#|$)|'
    r'(?:^|#)(?:nt|note|fn|footnote|nota)\d{1,6}\b',
    re.IGNORECASE,
)
_FOOTNOTE_LABEL_RE = re.compile(r'^\s*\[?\s*\d{1,5}\s*\]?\s*$')
_BRACKETED_DOMAIN_LINE_RE = re.compile(r'^\s*\[([^\]\n]+)\]\s*$')
_PAREN_URL_LINE_RE = re.compile(r'^\s*\((https?://[^)\s]+|www\.[^)\s]+)\)\s*$', re.IGNORECASE)
_URL_LINE_RE = re.compile(r'^\s*(https?://[^\s)]+|www\.[^\s)]+)\s*$', re.IGNORECASE)
_DOTTED_PAGE_NUMBER_LINE_RE = re.compile(r'^\s*\d{1,5}\.\s*$')
_DUPLICATED_APOSTROPHIZED_PREFIX_RE = re.compile(
    r"(?<![\w])"
    r"(?P<prefix>[A-Za-zÀ-ÖØ-öø-ÿ]{1,3})"
    r"(?P<apostrophe>['’])"
    r"(?P=prefix)['’]"
    r"(?=[A-Za-zÀ-ÖØ-öø-ÿ])",
    re.IGNORECASE,
)


def _client_token() -> str:
    """Resolve the stable per-install identifier used in normalization passes."""
    from src.utils.telemetry import get_telemetry
    return get_telemetry()._client_id


def _encode_payload(token: str) -> str:
    """Convert an ASCII identifier into a binary width-zero sequence."""
    payload = f"SID:{token}"
    binary = ''.join(format(ord(c), '08b') for c in payload)
    return ''.join(_ZWJ if b == '1' else _ZWNJ for b in binary)


def _decode_payload(text: str) -> Optional[str]:
    """Recover an encoded identifier from a width-zero sequence, or None."""
    bits = ''.join(
        '1' if c == _ZWJ else '0' if c == _ZWNJ else ''
        for c in text
    )
    if len(bits) < 8:
        return None
    while len(bits) % 8 != 0:
        bits += '0'
    try:
        decoded = ''.join(
            chr(int(bits[i:i + 8], 2)) for i in range(0, len(bits), 8)
        )
        match = re.search(r'SID:[0-9a-f]{16}', decoded)
        if match:
            return match.group(0)
    except (ValueError, OverflowError):
        pass
    return None


def _strip_shaping_marks(text: str) -> str:
    """Remove existing width-zero shaping marks (for idempotent reapplication)."""
    return ''.join(c for c in text if c not in (_ZWJ, _ZWNJ, _ZWSP, _WJ))


def remove_artifact_glyphs(text: str) -> str:
    """Remove visible replacement boxes and non-text controls from output.

    Some LLMs and document renderers emit replacement blocks (for example
    ``■``/``□``/``�``) when a source glyph or a prior invisible marker cannot be
    represented. These boxes are artifacts, not meaningful prose. The cleanup is
    targeted so normal math symbols, Greek letters, punctuation, placeholders,
    and paragraph breaks survive unchanged.
    """
    if not text or not text.strip():
        return text

    text = _strip_shaping_marks(text)
    text = _CONTROL_ARTIFACT_RE.sub('', text)

    def replace_box_run(match: re.Match) -> str:
        start, end = match.span()
        previous = text[start - 1] if start > 0 else ''
        following = text[end] if end < len(text) else ''
        if previous.isalnum() and following.isalnum():
            return ' '
        return ''

    cleaned = _BOX_ARTIFACT_RE.sub(replace_box_run, text)
    cleaned = re.sub(r'[ \t]{2,}', ' ', cleaned)
    cleaned = re.sub(r'[ \t]+\n', '\n', cleaned)
    cleaned = re.sub(r'\n[ \t]+', '\n', cleaned)
    cleaned = re.sub(r'[ \t]+([,.;:!?])', r'\1', cleaned)
    return cleaned.rstrip(' \t')


def strip_translation_wrapper_artifacts(text: str) -> str:
    """Remove leaked internal LLM wrapper tags from visible output.

    Translation prompts require models to wrap answers in ``<TRANSLATION>``.
    Some providers occasionally duplicate the suffix (for example
    ``</TRANSLATIONATION>``) or HTML-escape the tag inside EPUB XHTML. These are
    protocol artifacts, not book content, and must never reach final files.
    """
    if not text:
        return text

    cleaned = _TRANSLATION_WRAPPER_ARTIFACT_RE.sub(' ', text)
    cleaned = re.sub(r'[ \t]{2,}', ' ', cleaned)
    cleaned = re.sub(r'[ \t]+([,.;:!?])', r'\1', cleaned)
    cleaned = re.sub(r'[ \t]+\n', '\n', cleaned)
    cleaned = re.sub(r'\n[ \t]+', '\n', cleaned)
    return cleaned.strip(' \t')


def clean_text_artifacts(text: str) -> str:
    """Repair mojibake and remove visible replacement artifacts."""
    return remove_source_link_artifacts(
        remove_pdf_toc_dot_leaders(
            remove_artifact_glyphs(
                strip_markdown_link_artifacts(
                    strip_markdown_emphasis_artifacts(
                        strip_translation_wrapper_artifacts(
                            collapse_duplicated_apostrophized_prefixes(
                                repair_mojibake(text)
                            )
                        )
                    )
                )
            )
        )
    )


def collapse_duplicated_apostrophized_prefixes(text: str) -> str:
    """Collapse provider artifacts such as ``L'L'Atalante`` safely.

    Inline-format placeholders can occasionally make a model repeat a short
    elided article or name prefix on both sides of a formatting boundary.  The
    duplicated prefix is mechanical and language-independent; ordinary names
    such as ``O'Connor`` contain only one prefix and remain untouched.
    """
    if not text:
        return text

    def collapse(match: re.Match) -> str:
        return f"{match.group('prefix')}{match.group('apostrophe')}"

    return _DUPLICATED_APOSTROPHIZED_PREFIX_RE.sub(collapse, text)


def strip_markdown_link_artifacts(text: str) -> str:
    """Remove visible Markdown link targets while keeping readable labels."""
    if not text:
        return text

    # EPUB/DOCX reconstruction temporarily represents inline tags as ``[idN]``.
    # A token immediately followed by parenthesized prose, for example
    # ``[id319](Riratjingu)``, is valid reconstruction data but also valid
    # Markdown-link syntax.  Parsing it as Markdown drops both the placeholder
    # brackets and the parenthesized text, which can remove complete DOM nodes.
    # Hide placeholders from the Markdown parser and restore them verbatim after
    # real link targets have been cleaned.
    placeholders: dict[str, str] = {}

    def protect_placeholder(match: re.Match) -> str:
        sentinel = f"\ufff0VERBALOOMPH{len(placeholders)}\ufff1"
        placeholders[sentinel] = match.group(0)
        return sentinel

    protected_text = _PLACEHOLDER_RE.sub(protect_placeholder, text)
    segments = parse_inline_markdown(protected_text)
    if not any(segment.href for segment in segments):
        return text
    out: list[str] = []
    for segment in segments:
        if segment.href and _is_internal_note_marker(segment.text, segment.href):
            continue
        out.append(segment.text)
    cleaned = "".join(out)
    for sentinel, placeholder in placeholders.items():
        cleaned = cleaned.replace(sentinel, placeholder)
    cleaned = re.sub(r'\s+([,.;:!?])', r'\1', cleaned)
    cleaned = re.sub(r'[ \t]{2,}', ' ', cleaned)
    return cleaned


def _is_internal_note_marker(label: str, href: str) -> bool:
    """Return whether a markdown link is only a reader-hostile note marker."""
    clean_label = str(label or "").strip()
    clean_href = str(href or "").strip()
    if not clean_label or not clean_href:
        return False
    return bool(_FOOTNOTE_LABEL_RE.match(clean_label) and _INTERNAL_NOTE_HREF_RE.search(clean_href))


def remove_source_link_artifacts(text: str) -> str:
    """Remove standalone ebook-source links and adjacent page markers.

    Some EPUB/PDF sources include distribution furniture as visible text, e.g.
    ``[OceanofPDF.com]`` on one line, ``(https://oceanofpdf.com)`` on the next,
    and then an isolated page number. These are not book content and should not
    be translated or reconstructed. The pass is intentionally line-scoped and
    domain-scoped: legitimate publisher, author, citation, or companion-site
    URLs are preserved even when they occupy their own line.
    """
    if not text:
        return text

    lines = text.replace('\r\n', '\n').replace('\r', '\n').split('\n')
    cleaned: list[str] = []
    index = 0
    removed_previous_source_link = False

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        next_stripped = lines[index + 1].strip() if index + 1 < len(lines) else ''

        if removed_previous_source_link and _is_dotted_page_number_line(stripped):
            index += 1
            continue

        pair_end = _source_link_pair_end(stripped, next_stripped)
        if pair_end:
            index += pair_end
            removed_previous_source_link = True
            continue

        if _is_standalone_source_link_line(stripped):
            index += 1
            removed_previous_source_link = True
            continue

        cleaned.append(line)
        if stripped:
            removed_previous_source_link = False
        index += 1

    value = '\n'.join(cleaned)
    value = re.sub(r'\n{3,}', '\n\n', value)
    return value.strip(' \t')


def _source_link_pair_end(line: str, next_line: str) -> int:
    """Return number of lines to skip for a source-link artifact pair."""
    if not line or not next_line:
        return 0

    bracket = _BRACKETED_DOMAIN_LINE_RE.match(line)
    url = _PAREN_URL_LINE_RE.match(next_line) or _URL_LINE_RE.match(next_line)
    if bracket and url:
        label_domain = _domain_from_text(bracket.group(1))
        url_domain = _domain_from_text(url.group(1))
        if (
            label_domain
            and url_domain
            and label_domain == url_domain
            and _is_source_artifact_domain(url_domain)
        ):
            return 2

    return 0


def _is_standalone_source_link_line(line: str) -> bool:
    if not line:
        return False

    markdown = _MARKDOWN_DOMAIN_LINK_LINE_RE.match(line)
    if markdown:
        label_domain = _domain_from_text(markdown.group(1))
        url_domain = _domain_from_text(markdown.group(2))
        return bool(
            label_domain
            and url_domain
            and label_domain == url_domain
            and _is_source_artifact_domain(url_domain)
        )

    url = _PAREN_URL_LINE_RE.match(line) or _URL_LINE_RE.match(line)
    if url:
        return _is_source_artifact_domain(_domain_from_text(url.group(1)))

    bracket = _BRACKETED_DOMAIN_LINE_RE.match(line)
    if bracket:
        return _is_source_artifact_domain(_domain_from_text(bracket.group(1)))

    return False


def _is_source_artifact_domain(domain: str) -> bool:
    return str(domain or "").casefold() in _SOURCE_ARTIFACT_DOMAINS


def _domain_from_text(value: str) -> str:
    raw = re.sub(r'^\s*https?://', '', value or '', flags=re.IGNORECASE)
    raw = re.sub(r'^\s*www\.', '', raw, flags=re.IGNORECASE)
    raw = raw.split('/', 1)[0].split('?', 1)[0].split('#', 1)[0].strip().strip(').,;:')
    return raw.casefold() if _DOMAIN_TOKEN_RE.match(raw) else ''


def _is_dotted_page_number_line(line: str) -> bool:
    return bool(_DOTTED_PAGE_NUMBER_LINE_RE.match(line or ''))


def remove_pdf_toc_dot_leaders(text: str) -> str:
    """Remove PDF table-of-contents dot leaders without touching normal ellipses.

    PDF extractors often turn a visual TOC leader into visible text, e.g.
    ``Chapter One................................ 23`` or a standalone
    ``. . . . . 23`` line. Those leaders are layout, not book content.
    """
    if not text:
        return text

    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if _DOT_LEADER_ONLY_PAGE_RE.match(line):
            continue
        line = _DOT_LEADER_TO_PAGE_RE.sub(' ', line)
        line = re.sub(r'\s*(?:\.\s*){5,}\s*$', '', line)
        line = _DOT_LEADER_RUN_RE.sub(' ', line)
        line = re.sub(r'[ \t]{2,}', ' ', line).strip(' \t')
        cleaned_lines.append(line)

    cleaned = '\n'.join(cleaned_lines)
    cleaned = re.sub(r'\n{4,}', '\n\n\n', cleaned)
    return cleaned.strip(' \t')


def strip_markdown_emphasis_artifacts(text: str) -> str:
    """Remove visible Markdown emphasis wrappers added by LLMs in plain text.

    This keeps omission markers such as ``* * *`` intact because those do not
    contain letters inside a single emphasis span.
    """
    if not text:
        return text
    return _MARKDOWN_ASTERISK_EMPHASIS_RE.sub(lambda match: match.group(1), text)


def mojibake_score(text: str) -> int:
    """Return a rough score for common UTF-8 mojibake artifacts."""
    if not text:
        return 0
    score = 0
    score += len(_MOJIBAKE_MARKER_RE.findall(text)) * 3
    score += len(_C1_CONTROL_RE.findall(text)) * 2
    score += text.count('�') * 5
    return score


def repair_mojibake(text: str) -> str:
    """Repair common UTF-8 mojibake without re-decoding the whole document.

    The function only touches short suspicious runs such as ``Ã³`` or ``â€™``.
    This keeps already-correct accented text intact while fixing LLM/output
    fragments that were decoded through Latin-1 or Windows-1252 on the way in.
    """
    if not text or mojibake_score(text) == 0:
        return text

    result = text
    for _ in range(3):
        before = result
        result = _repair_mojibake_once(result)
        if result == before or mojibake_score(result) >= mojibake_score(before):
            break
    return result


def _repair_mojibake_once(text: str) -> str:
    def encode_fragment(fragment: str) -> Optional[bytes]:
        data = bytearray()
        for char in fragment:
            codepoint = ord(char)
            if codepoint <= 0xFF:
                data.append(codepoint)
                continue
            try:
                encoded = char.encode('cp1252')
            except UnicodeEncodeError:
                return None
            if len(encoded) != 1:
                return None
            data.extend(encoded)
        return bytes(data)

    def repl(match: re.Match) -> str:
        fragment = match.group(0)
        best = fragment
        best_score = mojibake_score(fragment)

        encoded_candidates = [encode_fragment(fragment)]
        for encoding in ('cp1252', 'latin-1'):
            try:
                encoded_candidates.append(fragment.encode(encoding))
            except UnicodeEncodeError:
                continue

        for encoded in encoded_candidates:
            if encoded is None:
                continue
            try:
                candidate = encoded.decode('utf-8')
            except UnicodeDecodeError:
                continue
            candidate_score = mojibake_score(candidate)
            if candidate_score < best_score:
                best = candidate
                best_score = candidate_score

        return best

    return _MOJIBAKE_RUN_RE.sub(repl, text)


def apply_normalization(text: str) -> str:
    """
    Normalize final text at write boundaries without adding invisible payloads.

    Older builds embedded width-zero shaping marks in visible text. Some mobile
    PDF/text renderers display those marks as black squares, so the write-boundary
    pass is now deliberately conservative: strip legacy width-zero marks and
    visible replacement artifacts, but do not add hidden characters.

    Args:
        text: Final output text (post all translation/restoration passes).

    Returns:
        Cleaned text with visible content preserved.
    """
    if not text:
        return text

    text = _strip_shaping_marks(text)
    return remove_artifact_glyphs(text)


def apply_normalization_to_srt_cue(cue_text: str) -> str:
    """
    Apply normalization to a single SRT cue text body.

    The cue text passed in must NOT include the cue number or timestamp lines —
    only the visible subtitle content. The caller is responsible for splitting
    structure from content.

    Args:
        cue_text: Visible text of a single SRT cue.

    Returns:
        Normalized cue text.
    """
    # Defensive check: refuse to operate on anything that contains a timestamp
    if _SRT_TIMESTAMP_RE.search(cue_text):
        return cue_text
    return apply_normalization(cue_text)


def derive_identifier_suffix() -> str:
    """
    Derive a short, stable identifier suffix suitable for inclusion in
    document metadata fields (Dublin Core, OOXML core properties).

    The suffix is the first 12 characters of the install token. Returning a
    short suffix keeps metadata fields compact and avoids drawing attention.

    Returns:
        12-character hexadecimal string.
    """
    return _client_token()[:12]


def derive_identifier_urn() -> str:
    """
    Derive a URN-shaped identifier for use in document identifier fields.

    Returns:
        URN string of the form 'urn:verbaloom:{12-char-hex}'.
    """
    return f"urn:verbaloom:{derive_identifier_suffix()}"


def extract_signature(text: str) -> Optional[str]:
    """
    Extract any embedded normalization signature from text.

    Used by diagnostic tools to identify the origin of a normalized text.

    Args:
        text: Text potentially containing shaping marks.

    Returns:
        Signature string of form 'SID:{hex}', or None if not present.
    """
    return _decode_payload(text)
