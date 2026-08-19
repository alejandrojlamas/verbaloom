"""Deterministic language evidence for mixed metadata and translated prose."""

from __future__ import annotations

import re


_LANGUAGE_ALIASES = {
    "de": "german",
    "deutsch": "german",
    "alemán": "german",
    "aleman": "german",
    "en": "english",
    "es": "spanish",
    "español": "spanish",
    "espanol": "spanish",
    "fr": "french",
    "français": "french",
    "francais": "french",
}
_LANGUAGE_CONNECTIVE_RE: dict[str, re.Pattern[str]] = {
    "english": re.compile(
        r"\b(?:the|and|of|to|in|that|is|was|for|with|as|on|by|from|this|it|"
        r"be|are|were|or|an|at|which|not|have|has|had|but|they|their|see|"
        r"also|about|according|chapter|page|pages|book|books|edition)\b",
        re.IGNORECASE,
    ),
    "spanish": re.compile(
        r"\b(?:el|la|los|las|un|una|unos|unas|de|del|que|en|para|con|por|"
        r"se|no|su|sus|al|es|son|era|fue|fueron|est[aá]|estaba|hab[ií]a|"
        r"pero|pues|cuando|entonces|tambi[eé]n|m[aá]s|menos|como|desde|"
        r"hasta|sobre|y|e|o|a|hay|luego|otro|otra|otros|otras|v[eé]ase|"
        r"v[eé]anse|seg[uú]n|cap[ií]tulo|p[aá]gina|p[aá]ginas|libro|"
        r"libros|edici[oó]n|entrevista|recomiendo|dijo)\b",
        re.IGNORECASE,
    ),
    "french": re.compile(
        r"\b(?:le|la|les|des|du|de|et|que|qui|dans|pour|avec|sur|est|sont|"
        r"une|un|ce|cette|par|pas|plus|mais|comme|voir|aussi|selon|chapitre|"
        r"page|pages|livre|livres|[eé]dition)\b",
        re.IGNORECASE,
    ),
    "german": re.compile(
        r"\b(?:der|die|das|den|dem|des|ein|eine|einer|und|oder|aber|ist|"
        r"sind|war|waren|mit|von|zu|im|in|auf|f[uü]r|nicht|dass|als|auch|"
        r"siehe|laut|kapitel|seite|seiten|buch|b[uü]cher|ausgabe)\b",
        re.IGNORECASE,
    ),
}
_LANGUAGE_REFERENCE_CUE_RE: dict[str, re.Pattern[str]] = {
    "english": re.compile(
        r"\b(?:see|see also|according to|edited by|illustrated by|translated by|"
        r"coauthored with|edition|interview|chapter|pages?)\b",
        re.IGNORECASE,
    ),
    "spanish": re.compile(
        r"\b(?:v[eé]ase|v[eé]anse|v[eé]ase tambi[eé]n|seg[uú]n|editado por|"
        r"ilustrad[oa] por|traducid[oa] por|en coautor[ií]a con|"
        r"edici[oó]n|entrevista|cap[ií]tulo|p[aá]ginas?|recomiendo|sobre)\b",
        re.IGNORECASE,
    ),
    "french": re.compile(
        r"\b(?:voir|voir aussi|selon|dirig[eé] par|illustr[eé] par|traduit par|"
        r"[eé]crit avec|[eé]dition|entretien|"
        r"chapitre|pages?|sur)\b",
        re.IGNORECASE,
    ),
    "german": re.compile(
        r"\b(?:siehe|siehe auch|laut|herausgegeben von|illustriert von|"
        r"[uü]bersetzt von|gemeinsam mit|ausgabe|interview|"
        r"kapitel|seiten?|[uü]ber)\b",
        re.IGNORECASE,
    ),
}
_BIBLIOGRAPHIC_SHAPE_RE = re.compile(
    r"(?:\((?:1[4-9]|20)\d{2}[a-z]?\)|\b(?:pp?\.|cap\.|ed\.)\s*\d*|"
    r"\bISBN\b|;\s*[A-ZÁÉÍÓÚÜÑÀ-ÖØ-Þ])",
    re.IGNORECASE,
)
_CITATION_LOCATOR_RE = re.compile(
    r"(?ix)"
    r"(?:https?://|www\.|doi(?:\.org|:)|\b[a-z0-9.-]+\.(?:com|org|net|edu|gov)/)"
    r"|\b(?:1[4-9]\d{2}|20\d{2})\b"
    r"|\b(?:n\.\s*[ºo]\s*\d+|vol\.\s*\d+|pp?\.\s*\d+|"
    r"temporada\s+\d+|episodio\s+\d+)\b"
    r"|(?:^|[\s,(])\d{1,4}\s*[–-]\s*\d{1,4}(?:[\s,.)]|$)"
)
_INDEX_LOCATOR_RE = re.compile(r"\b\d{1,4}(?:\s*[–-]\s*\d{1,4})?\b")
_LANGUAGE_DISTINCTIVE_RE = {
    "spanish": re.compile(r"[áéíóúüñ¿¡]", re.IGNORECASE),
    "french": re.compile(r"[àâçéèêëîïôùûüÿœ]", re.IGNORECASE),
    "german": re.compile(r"[äöüß]", re.IGNORECASE),
}
_LANGUAGE_CITATION_METADATA_RE = {
    "spanish": re.compile(
        r"\b(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|"
        r"septiembre|octubre|noviembre|diciembre|temporada|episodio|"
        r"presentador(?:a)?|anfitri[oó]n|tesis doctoral|consultad[oa]|anexo)\b|n\.\s*º",
        re.IGNORECASE,
    ),
    "french": re.compile(
        r"\b(?:janvier|f[eé]vrier|mars|avril|mai|juin|juillet|ao[uû]t|"
        r"septembre|octobre|novembre|d[eé]cembre|saison|[eé]pisode)\b",
        re.IGNORECASE,
    ),
    "german": re.compile(
        r"\b(?:januar|februar|m[aä]rz|april|mai|juni|juli|august|"
        r"september|oktober|november|dezember|staffel|folge)\b",
        re.IGNORECASE,
    ),
}
_SURFACE_WORD_RE = re.compile(
    r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñÀ-ÖØ-öø-ÿ'’.-]+",
    re.UNICODE,
)
_WORK_TITLE_CONNECTORS = {
    "a", "an", "and", "as", "at", "by", "da", "das", "de", "del",
    "der", "des", "di", "do", "dos", "du", "e", "el", "en", "et",
    "for", "from", "in", "into", "la", "las", "le", "les", "los", "of",
    "on", "or", "out", "para", "por", "the", "to", "un", "una", "une",
    "und", "von", "with", "y",
}
_WORK_TITLE_TYPE_PREFIXES = {
    "article",
    "book",
    "episode",
    "film",
    "journal",
    "magazine",
    "movie",
    "newspaper",
    "novel",
    "podcast",
    "program",
    "report",
    "series",
    "song",
}
_CITATION_METADATA_TOKENS = {
    "chap", "chapter", "d", "ed", "eds", "ll", "m", "no", "p", "pp",
    "re", "s", "t", "ve", "vol",
}
_NUMBERED_REFERENCE_RE = re.compile(
    r"^\s*(?:\[\s*\d{1,4}\s*\]|\d{1,4}\s*[.)])\s+"
)
_REFERENCE_DOCUMENT_HINT_RE = re.compile(
    r"(?:^|[/_.-])(?:notes?|endnotes?|footnotes?)(?:[/_.-]|$)",
    re.IGNORECASE,
)
_BIBLIOGRAPHY_DOCUMENT_HINT_RE = re.compile(
    r"(?:^|[/_.-])(?:references?|bibliograph(?:y|ies)|works[_ -]?cited|"
    r"further[_ -]*readings?|(?:recommended|suggested)[_ -]*readings?)"
    r"(?:[/_.-]|$)",
    re.IGNORECASE,
)
_COPYRIGHT_DOCUMENT_HINT_RE = re.compile(
    r"(?:^|[/_.-])(?:copyright|colophon|legal|credits?)(?:[/_.-]|$)",
    re.IGNORECASE,
)
_OTHER_WORKS_DOCUMENT_HINT_RE = re.compile(
    r"(?:^|[/_.-])(?:alsoby|also[_ -]*by|other[_ -]*works?|books?[_ -]*by)"
    r"(?:[/_.-]|$)",
    re.IGNORECASE,
)
_INDEX_DOCUMENT_HINT_RE = re.compile(
    r"(?:^|[/_.-])(?:index|indice|índice)(?:[/_.-]|$)",
    re.IGNORECASE,
)
_PRAISE_DOCUMENT_HINT_RE = re.compile(
    r"(?:^|[/_.-])(?:praise|endorsements?|elogios?)(?:[/_.-]|$)",
    re.IGNORECASE,
)
_ATTRIBUTION_ROLE_RE = re.compile(
    r"\b(?:autor(?:a)?|bestseller|coach|consultor(?:a)?|director(?:a)?|"
    r"fundador(?:a)?|jefe|miembro|profesor(?:a)?|socio|socia|president(?:e|a)|"
    r"ceo|cfo|cto|fellow|dean|editor|board|stanford|harvard|mit)\b",
    re.IGNORECASE,
)
_DISTINCT_SOURCE_PRONOUNS = {
    "english": {"we", "they", "you", "our", "their", "them"},
    "french": {"nous", "vous", "ils", "elles", "leur", "leurs"},
    "german": {"ich", "du", "wir", "ihr", "euch", "mein", "dein", "unser"},
    "spanish": {"nosotros", "ustedes", "ellos", "ellas", "nuestro", "nuestra"},
}


def normalize_language_name(value: str) -> str:
    key = str(value or "").strip().casefold()
    return _LANGUAGE_ALIASES.get(key, key)


def has_target_language_contextual_evidence(
    text: str,
    *,
    target_language: str,
) -> bool:
    """Recognize translated connective prose around preserved names and titles."""
    target_key = normalize_language_name(target_language)
    connective_re = _LANGUAGE_CONNECTIVE_RE.get(target_key)
    cue_re = _LANGUAGE_REFERENCE_CUE_RE.get(target_key)
    if connective_re is None or cue_re is None:
        return False

    if looks_like_translated_reference_context(
        text,
        target_language=target_key,
    ):
        return True

    words = _SURFACE_WORD_RE.findall(text or "")
    if len(words) < 8:
        return False
    connective_count = len(connective_re.findall(text or ""))
    reference_cue_count = len(cue_re.findall(text or ""))
    if (
        _BIBLIOGRAPHIC_SHAPE_RE.search(text or "")
        and reference_cue_count >= 1
        and connective_count >= 2
    ):
        return True
    capitalized_count = sum(
        1
        for word in words
        if word[:1].isupper() and not word.isupper()
    )
    name_ratio = capitalized_count / max(1, len(words))
    has_target_reference_context = (
        reference_cue_count >= 2
        and connective_count >= 3
    )
    has_target_name_list_context = (
        connective_count >= 4
        and capitalized_count >= 6
    )
    return bool(
        name_ratio >= 0.35
        and (
            has_target_reference_context
            or has_target_name_list_context
        )
    )


def looks_like_translated_reference_context(
    text: str,
    *,
    target_language: str,
) -> bool:
    """Return True only for translated citation locators or compact indexes."""
    target_key = normalize_language_name(target_language)
    connective_re = _LANGUAGE_CONNECTIVE_RE.get(target_key)
    cue_re = _LANGUAGE_REFERENCE_CUE_RE.get(target_key)
    if connective_re is None or cue_re is None:
        return False
    return bool(
        _has_translated_citation_locator(
            text or "",
            target_language=target_key,
            connective_re=connective_re,
            cue_re=cue_re,
        )
        or _looks_like_index_locator(text or "")
    )


def _has_translated_citation_locator(
    text: str,
    *,
    target_language: str,
    connective_re: re.Pattern[str],
    cue_re: re.Pattern[str],
) -> bool:
    """Recognize translated note locators followed by preserved work identity.

    Endnotes commonly translate only the short locator before the first colon
    or question mark while preserving the cited title, outlet, URL, and
    publication metadata.
    Whole-block language detection can therefore report the source language
    even though the reader-facing prose was translated correctly. Requiring a
    target-language prefix plus citation-shaped metadata keeps ordinary
    untranslated prose and untranslated bibliographies outside this exemption.
    """
    value = re.sub(r"\s+", " ", text or "").strip()
    separator = re.search(r"[:?!](?=\s)", value)
    if separator is None:
        return False

    prefix = value[:separator.start()]
    suffix = value[separator.end():]
    prefix_words = _SURFACE_WORD_RE.findall(prefix)
    suffix_words = _SURFACE_WORD_RE.findall(suffix)
    if not 2 <= len(prefix_words) <= 32 or len(suffix_words) < 2:
        return False
    citation_shape = bool(
        _CITATION_LOCATOR_RE.search(value)
        or re.search(r"[“”\"«»]", suffix)
        or suffix.count(",") + suffix.count(";") >= 2
    )
    if not citation_shape:
        return False
    if suffix.count(",") + suffix.count(";") < 1 and not re.search(
        r"[“”\"«»()]|\b(?:doi|ISBN|ECF)\b",
        suffix,
        re.IGNORECASE,
    ):
        return False

    distinctive_re = _LANGUAGE_DISTINCTIVE_RE.get(target_language)
    metadata_re = _LANGUAGE_CITATION_METADATA_RE.get(target_language)
    target_prefix_evidence = (
        len(connective_re.findall(prefix)) >= 1
        or len(cue_re.findall(prefix)) >= 1
        or bool(distinctive_re and distinctive_re.search(prefix))
        or bool(metadata_re and metadata_re.search(value))
    )
    return bool(target_prefix_evidence)


def _looks_like_index_locator(text: str) -> bool:
    """Recognize compact index entries made mostly of names and page numbers."""
    value = re.sub(r"\s+", " ", text or "").strip()
    locators = _INDEX_LOCATOR_RE.findall(value)
    words = [
        word
        for word in _SURFACE_WORD_RE.findall(value)
        if any(char.isalpha() for char in word)
    ]
    if len(locators) < 5 or not 1 <= len(words) <= 12:
        return False
    lexical_words = [
        word
        for word in words
        if word.casefold() not in _WORK_TITLE_CONNECTORS
    ]
    return bool(
        lexical_words
        and all(
            word[:1].isupper()
            or word.isupper()
            or any(char.isupper() for char in word[1:])
            for word in lexical_words
        )
    )


def looks_like_preserved_title_or_citation_sequence(tokens: list[str]) -> bool:
    """Return True for title-cased names/titles with citation metadata."""
    values = [str(token or "").strip(".'’-–—") for token in tokens]
    values = [value for value in values if value]
    if not 3 <= len(values) <= 42:
        return False

    anchors = 0
    for value in values:
        folded = value.casefold()
        if value[:1].isupper() or (len(value) >= 2 and value.isupper()):
            anchors += 1
            continue
        if folded in _WORK_TITLE_TYPE_PREFIXES:
            continue
        if folded in _WORK_TITLE_CONNECTORS or folded in _CITATION_METADATA_TOKENS:
            continue
        return False
    return anchors >= 2


def looks_like_structured_language_metadata(
    text: str,
    *,
    document_hint: str = "",
    target_language: str = "",
) -> bool:
    """Recognize non-prose identity metadata in structurally known sections.

    Work titles, journal names, organization names and contributor credentials
    often remain in their published language.  They must not be counted as
    untranslated narrative, but this exemption is deliberately tied to a
    notes, bibliography, index or praise section and to a narrow block shape.
    """
    value = re.sub(r"\s+", " ", text or "").strip()
    hint = str(document_hint or "")
    if not value or len(value) > 1800:
        return False

    words = _SURFACE_WORD_RE.findall(value)
    if not words:
        return False

    citation_shape = bool(
        _CITATION_LOCATOR_RE.search(value)
        or re.search(r"[“”\"«»]", value)
        or value.count(",") + value.count(";") >= 2
    )
    if (
        _REFERENCE_DOCUMENT_HINT_RE.search(hint)
        and _NUMBERED_REFERENCE_RE.match(value)
        and citation_shape
    ):
        return True
    if (
        _BIBLIOGRAPHY_DOCUMENT_HINT_RE.search(hint)
        and citation_shape
        and value.count(",") >= 1
    ):
        return True

    target_key = normalize_language_name(target_language)
    target_cue_re = _LANGUAGE_REFERENCE_CUE_RE.get(target_key)
    target_connective_re = _LANGUAGE_CONNECTIVE_RE.get(target_key)
    target_metadata_evidence = bool(
        target_cue_re
        and target_connective_re
        and target_cue_re.search(value)
        and len(target_connective_re.findall(value)) >= 1
    )
    if _COPYRIGHT_DOCUMENT_HINT_RE.search(hint):
        return bool(
            len(words) <= 70
            and target_metadata_evidence
            and (
                ":" in value
                or "/" in value
                or re.search(r"\b(?:ISBN|ed\.|edici[oó]n)\b", value, re.I)
            )
        )
    if _OTHER_WORKS_DOCUMENT_HINT_RE.search(hint):
        return bool(
            len(words) <= 60
            and target_metadata_evidence
            and (
                ":" in value
                or looks_like_preserved_title_or_citation_sequence(words)
            )
        )

    if _INDEX_DOCUMENT_HINT_RE.search(hint):
        return bool(
            _looks_like_index_locator(value)
            or looks_like_preserved_title_or_citation_sequence(words)
        )

    if _PRAISE_DOCUMENT_HINT_RE.search(hint):
        is_attribution = value.startswith(("—", "–", "-"))
        punctuation_count = value.count(",") + value.count(";")
        return bool(
            is_attribution
            and len(words) <= 42
            and punctuation_count >= 1
            and _ATTRIBUTION_ROLE_RE.search(value)
            and not re.search(r"[.!?]\s+[A-ZÁÉÍÓÚÜÑ]", value)
        )

    return False


def untranslated_source_pronouns(
    source_text: str,
    candidate_text: str,
    *,
    source_language: str,
    target_language: str,
) -> list[str]:
    """Find lower-case source pronouns embedded in target-language grammar."""
    source_key = normalize_language_name(source_language)
    target_key = normalize_language_name(target_language)
    pronouns = _DISTINCT_SOURCE_PRONOUNS.get(source_key, set())
    target_connectives = _LANGUAGE_CONNECTIVE_RE.get(target_key)
    if not pronouns or target_connectives is None or source_key == target_key:
        return []

    source_folded = str(source_text or "").casefold()
    findings: list[str] = []
    for match in re.finditer(r"\b[a-z]+\b", candidate_text or ""):
        token = match.group(0)
        if token not in pronouns:
            continue
        if not re.search(rf"\b{re.escape(token)}\b", source_folded):
            continue
        window = (candidate_text or "")[
            max(0, match.start() - 100):match.end() + 100
        ]
        if len(target_connectives.findall(window)) < 2:
            continue
        if token not in findings:
            findings.append(token)
    return findings
