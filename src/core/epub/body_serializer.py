"""
Body serialization for simplified EPUB processing

This module handles extracting and replacing the content of <body> elements
in XHTML documents, enabling full-document translation instead of per-element processing.
"""
from copy import deepcopy
import re
from typing import Optional, Tuple

from lxml import etree

from src.utils.unified_logger import info, LogType
from .exceptions import XmlParsingError, BodyExtractionError

_SMALL_CAPS_CLASS = re.compile(r"small[0-9]*|small[-_]?caps?|smallcaps[0-9]*", re.I)
_DROP_CAP_CLASS = re.compile(r"drop[-_]?cap(?:ital)?s?[0-9]*|lettrine[0-9]*|initial[-_]letter", re.I)
_HEADINGS = {f"h{i}" for i in range(1, 7)}


def _join_decorative_initials(body: etree._Element) -> None:
    """Make typographically split words atomic for translation, keeping the DOM."""
    for block in body.iter():
        if not isinstance(block.tag, str):
            continue
        tag = etree.QName(block).localname
        if tag not in _HEADINGS and tag != "p":
            continue
        if any(
            isinstance(ancestor.tag, str) and etree.QName(ancestor).localname in {"pre", "code"}
            for ancestor in block.iterancestors()
        ):
            continue
        # Keep page anchors in place, including those before a chapter title.
        children = list(block)
        preceding, attribute = block, "text"
        while children:
            first = children[0]
            if (
                (getattr(preceding, attribute) or "").strip()
                or not isinstance(first.tag, str) or etree.QName(first).localname != "a"
                or first.text or len(first) or not first.get("id") or set(first.attrib) != {"id"}
            ):
                break
            preceding, attribute = children.pop(0), "tail"
        if not children:
            continue
        prefix = getattr(preceding, attribute) or ""

        if tag in _HEADINGS and all(
            isinstance(child.tag, str) and etree.QName(child).localname == "span"
            and not len(child) and set(child.attrib) == {"class"}
            and all(_SMALL_CAPS_CLASS.fullmatch(name) for name in child.get("class").split())
            and child.get("class").strip()
            for child in children
        ):
            # Require a word broken between an uppercase initial and its span.
            # Without that evidence, these are ordinary editorial styles.
            left = prefix
            split_word = False
            for child in children:
                right = child.text or ""
                initial = re.search(r"(?<!\w)([^\W\d_])$", left)
                if initial and initial.group().isupper() and right[:1].isalpha():
                    split_word = True
                left = child.tail or ""
            if split_word:
                setattr(preceding, attribute, prefix + "".join(
                    (child.text or "") + (child.tail or "") for child in children
                ))
                for child in children:
                    child.text = child.tail = None
            continue

        first = children[0]
        initial = prefix.strip()
        if (
            tag == "p" and len(initial) == 1 and initial.isalpha() and initial.isupper()
            and isinstance(first.tag, str) and etree.QName(first).localname == "span"
            and not len(first) and set(first.attrib) == {"class"}
            and first.get("class").strip()
            and all(_SMALL_CAPS_CLASS.fullmatch(name) for name in first.get("class").split())
            and (first.text or "").lstrip()[:1].isalpha()
        ):
            # An opening initial can precede its small-caps span instead of
            # occupying a drop-cap span. Keep the complete opening in one
            # editable text node, including a real space after a pronoun or
            # article. Otherwise the isolated initial can become protected
            # markup and remain in the source language after translation.
            first.text = prefix + first.text
            setattr(preceding, attribute, None)
            continue
        if (
            prefix.strip(" \t\r\n\u00a0\"'\u00ab\u2018\u201c") or not isinstance(first.tag, str)
            or etree.QName(first).localname != "span" or len(first)
            or set(first.attrib) != {"class"}
            or not any(_DROP_CAP_CLASS.fullmatch(name) for name in first.get("class").split())
        ):
            continue
        letter = (first.text or "").strip()
        rest = first.tail or ""
        if len(letter) == 1 and letter.isupper() and rest.lstrip()[:1].isalpha():
            # Respect a real space (e.g. a drop cap on the article "A").
            # A fused lexical word is otherwise sent intact, not split by tags.
            first.tail = letter + rest
            first.text = None


def normalize_whitespace(html: str) -> str:
    """
    Normalize excessive whitespace in HTML content while preserving structural breaks.

    This handles the common case where EPUB source files have arbitrary line breaks
    and indentation that are just formatting artifacts, not meaningful content.

    For example, transforms:
        "La technologie est en constante évolution.
            Au fur et à mesure que les ordinateurs"
    Into:
        "La technologie est en constante évolution. Au fur et à mesure que les ordinateurs"

    But preserves structural breaks by injecting newlines after block-level closing tags.

    Rules:
    - Normalize whitespace only WITHIN text content (inside paragraphs, etc.)
    - Inject newlines after closing block tags (</p>, </li>, </div>, etc.)
    - Preserve content inside <pre>, <code>, <script>, <style> tags unchanged
    - Preserve <br> and <br/> tags (they represent intentional line breaks)

    Args:
        html: Raw HTML string with potential excessive whitespace

    Returns:
        HTML with normalized whitespace
    """
    # Protect content in preformatted tags by replacing with placeholders
    preserved_blocks = []

    def preserve_block(match):
        preserved_blocks.append(match.group(0))
        return f"__PRESERVED_BLOCK_{len(preserved_blocks) - 1}__"

    # Preserve <pre>, <code>, <script>, <style> blocks (case insensitive)
    html = re.sub(
        r'<(pre|code|script|style)[^>]*>.*?</\1>',
        preserve_block,
        html,
        flags=re.DOTALL | re.IGNORECASE
    )

    # Preserve <br> tags (they represent intentional line breaks)
    html = re.sub(r'<br\s*/?\s*>', preserve_block, html, flags=re.IGNORECASE)

    # Step 1: Normalize line endings
    html = html.replace('\r\n', '\n').replace('\r', '\n')

    # Step 2: Inject newlines after block-level closing tags
    # This ensures structural separation is maintained even if source has no newlines
    block_tags = r'</(?:p|div|li|h[1-6]|blockquote|section|article|header|footer|nav|aside|ol|ul|table|tr|td|th|dt|dd)>'
    html = re.sub(f'({block_tags})[ \t]*', r'\1\n', html, flags=re.IGNORECASE)

    # Step 3: Replace remaining single newlines (within text content) with a single space
    # But preserve newlines that are right after a closing tag (just added in step 2)
    # Pattern: newline NOT preceded by > (closing tag)
    html = re.sub(r'(?<!>)[ \t]*\n(?!\n)[ \t]*', ' ', html)

    # Step 4: Collapse multiple spaces into one
    html = re.sub(r' {2,}', ' ', html)

    # Step 5: Clean up whitespace around newlines
    html = re.sub(r' \n', '\n', html)  # Remove space before newline
    html = re.sub(r'\n ', '\n', html)  # Remove space after newline

    # Restore preserved blocks
    for i, block in enumerate(preserved_blocks):
        html = html.replace(f"__PRESERVED_BLOCK_{i}__", block)

    return html


def extract_body_html(
    doc_root: etree._Element,
    normalize: bool = True,
    *,
    join_decorative_initials: bool = False,
) -> Tuple[str, Optional[etree._Element]]:
    """
    Extract the HTML content of <body> as a string.

    Args:
        doc_root: Root of the parsed XHTML document
        normalize: If True, normalize excessive whitespace (default: True)
        join_decorative_initials: Move ornamental initial letters into readable
            words for the model without mutating the original DOM.

    Returns:
        Tuple (body_inner_html, body_element)
        Returns ("", None) if no body element found
    """
    # Try XHTML namespace first, then fallback to no namespace
    body = doc_root.find('.//{http://www.w3.org/1999/xhtml}body')
    if body is None:
        body = doc_root.find('.//body')

    if body is None:
        return "", None

    # Serialize the inner content of body (without the <body> tag itself)
    serialized_body = body
    if join_decorative_initials:
        serialized_body = deepcopy(body)
        _join_decorative_initials(serialized_body)
    inner_html = etree.tostring(serialized_body, encoding='unicode', method='html')

    # Remove the outer <body> tags
    # <body class="x">content</body> → content
    inner_html = re.sub(r'^<body[^>]*>', '', inner_html)
    inner_html = re.sub(r'</body>$', '', inner_html)

    inner_html = inner_html.strip()

    # Normalize whitespace to remove arbitrary line breaks from source formatting
    if normalize:
        inner_html = normalize_whitespace(inner_html)

    return inner_html, body


def replace_body_content(body_element: etree._Element, new_html: str) -> None:
    """
    Replace the content of <body> with new translated HTML.

    Args:
        body_element: The <body> element to modify
        new_html: New translated HTML content
    """
    # Parse the new content FIRST before clearing the body
    # This prevents data loss if parsing fails
    # Wrap in a temp element to handle multiple root elements
    wrapped = f"<temp xmlns='http://www.w3.org/1999/xhtml'>{new_html}</temp>"

    # Try XML parser first with huge_tree option to handle large documents
    # This preserves exact XML structure without HTML normalization
    parser = etree.XMLParser(
        recover=True,           # Recover from errors
        encoding='utf-8',
        huge_tree=True,         # Allow parsing very large trees
        remove_blank_text=False # Preserve whitespace exactly as-is
    )

    parse_method = "XML"
    parse_warnings = []

    try:
        temp = etree.fromstring(wrapped.encode('utf-8'), parser)
        if parser.error_log:
            parse_warnings.append(f"recovered from {len(parser.error_log)} errors")
    except etree.XMLSyntaxError as e:
        # Fallback: try without namespace
        wrapped_no_ns = f"<temp>{new_html}</temp>"
        try:
            temp = etree.fromstring(wrapped_no_ns.encode('utf-8'), parser)
            parse_method = "XML (no namespace)"
        except Exception as e2:
            # Last resort: HTML parser (but this may alter structure!)
            temp = etree.HTML(wrapped_no_ns)
            parse_method = "HTML fallback"
            if temp is not None:
                temp_elem = temp.find('.//temp')
                if temp_elem is not None:
                    temp = temp_elem
                else:
                    raise XmlParsingError(
                        "Could not find <temp> element after HTML parsing",
                        original_error=e2,
                        content_preview=new_html[:200]
                    )
            else:
                raise XmlParsingError(
                    "All parsing methods failed",
                    original_error=e2,
                    content_preview=new_html[:200]
                )

    # NOW clear the body (only after successful parsing)
    body_element.text = None
    for child in list(body_element):
        body_element.remove(child)

    # Copy content into body
    body_element.text = temp.text
    child_count = 0
    for child in temp:
        body_element.append(child)
        child_count += 1

    # Single consolidated log message
    warnings_str = f" ({', '.join(parse_warnings)})" if parse_warnings else ""
    info(f"📄 Body reconstructed: {len(new_html)} chars → {child_count} elements [{parse_method}{warnings_str}]")
