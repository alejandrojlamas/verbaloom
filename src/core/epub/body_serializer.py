"""
Body serialization for simplified EPUB processing

This module handles extracting and replacing the content of <body> elements
in XHTML documents, enabling full-document translation instead of per-element processing.
"""
from copy import deepcopy
from html.entities import html5
import re
from typing import Optional, Tuple

from lxml import etree

from src.utils.unified_logger import info
from .exceptions import XmlParsingError

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


def normalize_whitespace(html: str, *, legacy: bool = False) -> str:
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

    marker = "__PRESERVED_BLOCK_"
    if not legacy:
        while marker in html:
            marker += "_"

    def preserve_block(match):
        preserved_blocks.append(match.group(0))
        return f"{marker}{len(preserved_blocks) - 1}__"

    # Preserve <pre>, <code>, <script>, <style> blocks (case insensitive)
    html = re.sub(
        (r'<(pre|code|script|style)[^>]*>.*?</\1>' if legacy else
         r'<((?:[\w.-]+:)?(?:pre|code|script|style))(?=[\s>])[^>]*>.*?</\1>'),
        preserve_block,
        html,
        flags=re.DOTALL | re.IGNORECASE
    )

    # Preserve <br> tags (they represent intentional line breaks)
    html = re.sub(r'<br\s*/?\s*>', preserve_block, html, flags=re.IGNORECASE)

    if not legacy:
        # Keep quoted attributes byte-for-byte, including spaces in a resource
        # path. Leave angle brackets in place for the tag-boundary rules below.
        html = re.sub(r"(?<=<)[\w.:-]+\s+[^<>]*(?=>)", preserve_block, html)

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

    if legacy:
        for i, block in enumerate(preserved_blocks):
            html = html.replace(f"{marker}{i}__", block)
        return html
    return re.sub(
        re.escape(marker) + r"(\d+)__",
        lambda match: preserved_blocks[int(match.group(1))], html,
    )


def extract_body_html(
    doc_root: etree._Element,
    normalize: bool = True,
    *,
    join_decorative_initials: bool = False,
    legacy_html_serialization: bool = False,
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

    serialized_body = body if legacy_html_serialization and not join_decorative_initials else deepcopy(body)
    if join_decorative_initials:
        _join_decorative_initials(serialized_body)
    if legacy_html_serialization:
        # Byte-for-byte input identity of checkpoints written by the HTML serializer.
        inner_html = etree.tostring(serialized_body, encoding='unicode', method='html')
        inner_html = re.sub(r'^<body[^>]*>', '', inner_html)
        inner_html = re.sub(r'</body>$', '', inner_html)
    else:
        # Explicit closing tags keep the established placeholder grouping and
        # checkpoint identity of ordinary XHTML. HTML serialization can turn a
        # literal image URL into another URL and emit unclosed void elements.
        for element in serialized_body.iter():
            if isinstance(element.tag, str) and not len(element) and element.text is None:
                element.text = ""
        serialized = etree.tostring(serialized_body, encoding='unicode', with_tail=False)
        # XML serialization escapes '>' inside attributes. The body may have
        # any namespace prefix, and its tail is outside the translated content.
        inner_html = serialized[serialized.index(">") + 1:serialized.rindex("</")]

    inner_html = inner_html.strip()

    # Normalize whitespace to remove arbitrary line breaks from source formatting
    if normalize:
        inner_html = normalize_whitespace(inner_html, legacy=legacy_html_serialization)

    return inner_html, body


def _parse_fragment(body_element: etree._Element, new_html: str) -> etree._Element:
    # Child elements and attributes inherit declarations from the real body,
    # including an absent default namespace in legacy EPUBs.
    wrapper = etree.tostring(etree.Element("temp", nsmap=body_element.nsmap), encoding="unicode")
    wrapped = wrapper[:-2] + ">" + new_html + "</temp>"
    parser = etree.XMLParser(huge_tree=True, resolve_entities=False, no_network=True)
    try:
        return etree.fromstring(wrapped.encode("utf-8"), parser)
    except etree.XMLSyntaxError as exc:
        # Recovery can silently put a caption or the next paragraph inside an
        # image. Leave the original body intact for the normal recovery path.
        raise XmlParsingError(str(exc), original_error=exc, content_preview=new_html[:200]) from exc


def replace_body_content(body_element: etree._Element, new_html: str) -> None:
    """Replace a body atomically after strict XML parsing."""
    temp = _parse_fragment(body_element, new_html)
    body_element.text = temp.text
    for child in list(body_element):
        body_element.remove(child)
    body_element.extend(temp)
    info(f"📄 Body reconstructed: {len(new_html)} chars → {len(body_element)} elements [XML]")


_ENTITY = re.compile(r"&(?:#(?:[0-9]+|x[0-9a-fA-F]+);|[a-zA-Z][a-zA-Z0-9]*;)?")


def escape_translated_xml_text(text: str) -> str:
    """Escape model text before reinserting protected markup, keeping existing entities."""
    def entity(match):
        literal = match.group()
        if literal in {"&amp;", "&lt;", "&gt;", "&quot;", "&apos;"} or literal.startswith("&#"):
            return literal
        if literal[1:] in html5:
            return html5[literal[1:]].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return "&amp;" + literal[1:]

    return _ENTITY.sub(entity, text).replace("<", "&lt;").replace(">", "&gt;")


# These containers admit interleaved prose. Never create HTML spans inside
# embedded graphics, mathematical expressions, or element-only table/list rows.
_FLOW = {
    "body", "article", "aside", "section", "main", "header", "footer", "nav",
    "div", "p", "address", "blockquote", "figure", "figcaption", "caption",
    "li", "dt", "dd", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6", "pre",
}
_STRUCTURAL = _FLOW | {"ul", "ol", "dl", "table", "tr", "thead", "tbody", "tfoot"}
_SEPARATE = {"img", "picture", "svg", "math", "audio", "video", "object", "script", "style", "hr"}


def bilingual_body_fragment(source_html, target_html, body, source_language_code=None):
    """Pair complete prose runs while retaining the target's single indexed DOM."""
    source = _parse_fragment(body, source_html)
    target = _parse_fragment(body, target_html)

    def visit(left, right, parent_name):
        left_nodes, right_nodes = list(left), list(right)
        runs = []
        start = 0
        for i, (src, dst) in enumerate(zip(left_nodes, right_nodes, strict=True)):
            if src.tag != dst.tag:
                raise XmlParsingError("Bilingual XHTML element alignment failed")
            if not isinstance(dst.tag, str):
                continue
            qname = etree.QName(dst)
            if qname.namespace not in (None, "http://www.w3.org/1999/xhtml") or qname.localname in _SEPARATE:
                boundary = True
            else:
                boundary = qname.localname in _STRUCTURAL
                if boundary:
                    visit(src, dst, qname.localname)
            if boundary:
                runs.append((start, i))
                start = i + 1
        runs.append((start, len(left_nodes)))
        if parent_name not in _FLOW:
            return

        def run_text(node, children, first, end):
            pieces = [node.text if first == 0 else children[first - 1].tail]
            for child in children[first:end]:
                if isinstance(child.tag, str):
                    pieces.extend(child.itertext())
                pieces.append(child.tail)
            return "".join(piece or "" for piece in pieces)

        # Start at the end so each insertion index still denotes the original DOM.
        for first, end in reversed(runs):
            original = run_text(left, left_nodes, first, end)
            translation = run_text(right, right_nodes, first, end)
            if not original.strip() or original.strip() == translation.strip():
                continue
            ns = etree.QName(right).namespace if right is not target else etree.QName(body).namespace
            name = "div" if parent_name == "body" else "span"
            prefix = etree.Element(etree.QName(ns, name))
            prefix.set("class", "original")
            prefix.set("style", "display: block; color: #666; font-size: 0.9em;")
            if source_language_code:
                prefix.set("{http://www.w3.org/XML/1998/namespace}lang", source_language_code)
            prefix.text = original
            owner, slot = (right, "text") if first == 0 else (right_nodes[first - 1], "tail")
            prefix.tail = getattr(owner, slot)
            setattr(owner, slot, None)
            right.insert(first, prefix)

    visit(source, target, "body")
    # Element text needs XML encoding too. Keep any prefixes declared on the
    # real body in scope when returning the serialized fragment.
    target.text = target.text or ""
    wrapped = etree.tostring(target, encoding="unicode", with_tail=False)
    return wrapped[wrapped.index(">") + 1:wrapped.rindex("</")]
