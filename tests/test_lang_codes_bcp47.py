"""
BCP 47 / IANA conformance for the language-name -> ISO code mapping.

The HTML `lang` and `xml:lang` attributes accept BCP 47 language tags (RFC 5646),
which are built on the IANA Language Subtag Registry. Every code we emit must be
a non-deprecated primary language subtag.

This test guards against three regressions:
  1. Typos / made-up codes ('eng' instead of 'en').
  2. Use of deprecated subtags ('iw' instead of 'he', 'in' instead of 'id').
  3. 3-letter codes added without an explicit allowlist entry — those should be
     rare exceptions (only when no ISO 639-1 code exists).

The reference snapshot below was extracted from
https://www.iana.org/assignments/language-subtag-registry on 2026-05-08
(every Type:language Subtag of length 2 that is not Deprecated).

Re-fetching is straightforward; see plan/regenerate_iana_snapshot.py if the
registry changes upstream.
"""
from src.core.epub.lang_support import LANGUAGE_NAME_TO_CODE


# Snapshot of all 185 non-deprecated ISO 639-1 (2-letter) primary language
# subtags from the IANA Language Subtag Registry.
ISO_639_1_SUBTAGS: frozenset[str] = frozenset({
    'aa', 'ab', 'ae', 'af', 'ak', 'am', 'an', 'ar', 'as', 'av', 'ay', 'az',
    'ba', 'be', 'bg', 'bh', 'bi', 'bm', 'bn', 'bo', 'br', 'bs', 'ca', 'ce',
    'ch', 'co', 'cr', 'cs', 'cu', 'cv', 'cy', 'da', 'de', 'dv', 'dz', 'ee',
    'el', 'en', 'eo', 'es', 'et', 'eu', 'fa', 'ff', 'fi', 'fj', 'fo', 'fr',
    'fy', 'ga', 'gd', 'gl', 'gn', 'gu', 'gv', 'ha', 'he', 'hi', 'ho', 'hr',
    'ht', 'hu', 'hy', 'hz', 'ia', 'id', 'ie', 'ig', 'ii', 'ik', 'io', 'is',
    'it', 'iu', 'ja', 'jv', 'ka', 'kg', 'ki', 'kj', 'kk', 'kl', 'km', 'kn',
    'ko', 'kr', 'ks', 'ku', 'kv', 'kw', 'ky', 'la', 'lb', 'lg', 'li', 'ln',
    'lo', 'lt', 'lu', 'lv', 'mg', 'mh', 'mi', 'mk', 'ml', 'mn', 'mr', 'ms',
    'mt', 'my', 'na', 'nb', 'nd', 'ne', 'ng', 'nl', 'nn', 'no', 'nr', 'nv',
    'ny', 'oc', 'oj', 'om', 'or', 'os', 'pa', 'pi', 'pl', 'ps', 'pt', 'qu',
    'rm', 'rn', 'ro', 'ru', 'rw', 'sa', 'sc', 'sd', 'se', 'sg', 'sh', 'si',
    'sk', 'sl', 'sm', 'sn', 'so', 'sq', 'sr', 'ss', 'st', 'su', 'sv', 'sw',
    'ta', 'te', 'tg', 'th', 'ti', 'tk', 'tl', 'tn', 'to', 'tr', 'ts', 'tt',
    'tw', 'ty', 'ug', 'uk', 'ur', 'uz', 've', 'vi', 'vo', 'wa', 'wo', 'xh',
    'yi', 'yo', 'za', 'zh', 'zu',
})

# Codes we accept that are NOT 2-letter ISO 639-1 (typically because no
# 639-1 code exists). Each must be a valid IANA primary language subtag.
# Add to this set only with an inline rationale comment.
ALLOWED_NON_639_1_SUBTAGS: frozenset[str] = frozenset({
    'fil',  # Filipino — no ISO 639-1 code; ISO 639-2/3 is 'fil'. Registered in IANA.
})

# Common deprecated subtags that callers might be tempted to use; reject them
# explicitly to surface the real reason if someone reintroduces one.
DEPRECATED_FORBIDDEN_SUBTAGS: frozenset[str] = frozenset({
    'iw',   # Hebrew (deprecated; use 'he')
    'in',   # Indonesian (deprecated; use 'id')
    'ji',   # Yiddish (deprecated; use 'yi')
    'jw',   # Javanese (deprecated; use 'jv')
    'mo',   # Moldavian (deprecated; use 'ro')
})


def test_every_emitted_code_is_a_valid_bcp47_primary_subtag():
    """Every value in the map must be either a valid ISO 639-1 code or in the
    explicit allowlist of non-639-1 exceptions."""
    invalid = []
    for name, code in LANGUAGE_NAME_TO_CODE.items():
        c = code.lower()
        if c in ISO_639_1_SUBTAGS:
            continue
        if c in ALLOWED_NON_639_1_SUBTAGS:
            continue
        invalid.append((name, code))

    assert not invalid, (
        f"Invalid BCP 47 language tags in LANGUAGE_NAME_TO_CODE: {invalid}. "
        f"Either fix the code, or add it to ALLOWED_NON_639_1_SUBTAGS with a rationale."
    )


def test_no_deprecated_codes():
    """Reject deprecated IANA subtags. Catches the common 'iw'/'in' traps."""
    used = {c.lower() for c in LANGUAGE_NAME_TO_CODE.values()}
    leaked = used & DEPRECATED_FORBIDDEN_SUBTAGS
    assert not leaked, (
        f"Deprecated IANA subtags in LANGUAGE_NAME_TO_CODE: {leaked}. "
        f"See https://www.iana.org/assignments/language-subtag-registry"
    )


def test_codes_are_lowercase_per_bcp47_convention():
    """BCP 47 isn't strictly case-sensitive but recommends lowercase for
    primary language subtags. We rely on lowercase keys for lookups too."""
    bad = [(n, c) for n, c in LANGUAGE_NAME_TO_CODE.items() if c != c.lower()]
    assert not bad, f"Non-lowercase codes: {bad}"


def test_names_are_lowercase_for_case_insensitive_lookup():
    """Keys must be lowercase — `get_language_code` lowercases input before lookup."""
    bad = [n for n in LANGUAGE_NAME_TO_CODE if n != n.lower()]
    assert not bad, f"Non-lowercase names: {bad}"


def test_known_critical_pairs():
    """Spot-check the entries most likely to be miswritten."""
    expected = {
        'english': 'en',
        'spanish': 'es',     # the language from issue #159
        'french': 'fr',
        'german': 'de',
        'chinese': 'zh',
        'japanese': 'ja',
        'arabic': 'ar',
        'hebrew': 'he',      # NOT 'iw'
        'persian': 'fa',
        'farsi': 'fa',       # alias must agree
        'indonesian': 'id',  # NOT 'in'
        'yiddish': 'yi',     # NOT 'ji'
    }
    for name, code in expected.items():
        assert LANGUAGE_NAME_TO_CODE.get(name) == code, (
            f"{name!r} -> expected {code!r}, got {LANGUAGE_NAME_TO_CODE.get(name)!r}"
        )
