"""Locale/style checks for target-language quality guards."""

from __future__ import annotations

from collections import Counter
import re
from typing import Mapping, Optional


MEXICAN_SPANISH_VARIANTS = {
    "mexican",
    "mexico",
    "mx",
    "latam",
    "latin-american",
    "latin_american",
}

GENERIC_SPANISH_VARIANTS = {"none", "generic", "neutral", "standard"}

_SPANISH_TARGETS = {"spanish", "español", "espanol", "es", "castellano"}

_MEXICAN_SPANISH_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("chiton", re.compile(r"\bchit[oó]n\b", re.IGNORECASE)),
    ("avispaos", re.compile(r"\bavisaos\b|\bavispaos\b", re.IGNORECASE)),
    ("vosotros", re.compile(r"\bvosotros\b|\bvosotras\b", re.IGNORECASE)),
    ("os_pronoun", re.compile(r"\bos\b", re.IGNORECASE)),
    ("sois", re.compile(r"\bsois\b", re.IGNORECASE)),
    ("estais", re.compile(r"\best[aá]is\b", re.IGNORECASE)),
    ("habeis", re.compile(r"\bhab[eé]is\b|\bhabr[eé]is\b", re.IGNORECASE)),
    (
        "vosotros_verb",
        re.compile(
            r"\b(?:pod[eé]is|quer[eé]is|ten[eé]is|sab[eé]is|hac[eé]is|"
            r"dec[ií]s|veis|vais|ven[ií]s|cre[eé]is|fu[eé]rais|fuisteis|"
            r"mir[aá]is|tom[aá]is|estuvierais|hubierais)\b",
            re.IGNORECASE,
        ),
    ),
    ("rendisteis", re.compile(r"\brendisteis\b", re.IGNORECASE)),
    (
        "vuestro",
        re.compile(
            r"\b(?:vuestro|vuestra|vuestros|vuestras)\b"
            r"(?!\s+(?:alteza|majestad|excelencia|santidad|señor[ií]a)\b)",
            re.IGNORECASE,
        ),
    ),
    ("hala_vale", re.compile(r"\bhala\b|\bvale\b", re.IGNORECASE)),
    (
        "vosotros_imperative",
        re.compile(
            r"\b(?:precipitaos|rendeos|callaos|marchaos|apresuraos|guardaos|"
            r"entregaos|deteneos|poneos|haceos|quedaos|acercaos|alejaos|"
            r"quitaos|sentaos|levantaos|preparaos|armaos|largaos)\b",
            re.IGNORECASE,
        ),
    ),
    ("zumo", re.compile(r"\bzumos?\b", re.IGNORECASE)),
    ("cutre", re.compile(r"\bcutres?\b", re.IGNORECASE)),
    (
        "localizaciones_locaciones",
        re.compile(r"\blocalizaci[oó]n(?:es)?\b", re.IGNORECASE),
    ),
    ("ordenador", re.compile(r"\bordenador(?:es)?\b", re.IGNORECASE)),
    ("movil_phone", re.compile(r"\bm[oó]vil(?:es)?\b", re.IGNORECASE)),
    ("chaval", re.compile(r"\bchaval(?:es)?\b|\bchavala(?:s)?\b", re.IGNORECASE)),
    ("currar_curro", re.compile(r"\bcurrar\b|\bcurr[ao]s?\b", re.IGNORECASE)),
    ("guay", re.compile(r"\bguay\b", re.IGNORECASE)),
    ("flipar", re.compile(r"\bflip(?:ar|o|as|a|amos|an|ado|aba|aban)\b", re.IGNORECASE)),
    ("mola", re.compile(r"\bmola(?:r|s|n|ba|ban|do|ría|ria)?\b", re.IGNORECASE)),
    ("hostia_interjection", re.compile(r"\bhostias?\b|\bostias?\b", re.IGNORECASE)),
    ("gilipollas", re.compile(r"\bgilipollas?\b", re.IGNORECASE)),
    (
        "coger_regional",
        re.compile(
            r"\bcog(?:er|iendo|ido|ida|idos|idas|e|es|emos|en|í|iste|ió|imos|isteis|ieron|"
            r"ía|ías|íamos|íais|ían)(?:me|te|se|lo|la|los|las|le|les|nos)?\b|"
            r"\bcoj(?:o|a|as|amos|áis|an)(?:me|te|se|lo|la|los|las|le|les|nos)?\b",
            re.IGNORECASE,
        ),
    ),
)

_MEXICAN_SPANISH_ISSUE_CATALOG: dict[str, dict[str, str]] = {
    "chiton": {
        "label": "Interjeccion peninsular/arcaica",
        "suggestion": "Reescribir como 'guarden silencio', 'cállense' o una formulación literaria mexicana por contexto.",
        "category": "regional_register",
    },
    "avispaos": {
        "label": "Imperativo peninsular",
        "suggestion": "Usar una forma mexicana natural como 'estén atentos', 'pónganse listos' o equivalente por escena.",
        "category": "regional_register",
    },
    "vosotros": {
        "label": "Pronombre de segunda persona plural peninsular",
        "suggestion": "Usar 'ustedes' o reestructurar la frase con tratamiento mexicano contemporáneo.",
        "category": "grammar_register",
    },
    "os_pronoun": {
        "label": "Pronombre 'os'",
        "suggestion": "Reescribir con 'se', 'los', 'las', 'les' o una construcción natural mexicana.",
        "category": "grammar_register",
    },
    "sois": {
        "label": "Verbo peninsular de vosotros",
        "suggestion": "Usar 'son' o una formulación equivalente según el tratamiento del pasaje.",
        "category": "grammar_register",
    },
    "estais": {
        "label": "Verbo peninsular de vosotros",
        "suggestion": "Usar 'están' o una formulación equivalente según el tratamiento del pasaje.",
        "category": "grammar_register",
    },
    "habeis": {
        "label": "Verbo peninsular de vosotros",
        "suggestion": "Usar 'han', 'habrán' o una formulación equivalente según el tiempo verbal.",
        "category": "grammar_register",
    },
    "vosotros_verb": {
        "label": "Conjugación peninsular de vosotros",
        "suggestion": "Usar conjugación de 'ustedes' o reescribir con una voz mexicana natural.",
        "category": "grammar_register",
    },
    "rendisteis": {
        "label": "Pretérito peninsular de vosotros",
        "suggestion": "Usar 'se rindieron' o equivalente por contexto.",
        "category": "grammar_register",
    },
    "vuestro": {
        "label": "Posesivo peninsular",
        "suggestion": "Usar 'su/sus' o reformular para evitar ambigüedad.",
        "category": "grammar_register",
    },
    "hala_vale": {
        "label": "Muletilla/interjección peninsular",
        "suggestion": "Reescribir como una muletilla o reacción mexicana natural, o eliminarla si sólo rellena.",
        "category": "regional_idiom",
    },
    "vosotros_imperative": {
        "label": "Imperativo peninsular de vosotros",
        "suggestion": "Usar imperativo de ustedes como 'precipítense', 'cállense' o equivalente por escena.",
        "category": "grammar_register",
    },
    "zumo": {
        "label": "Léxico peninsular",
        "suggestion": "Usar 'jugo' cuando se refiere a una bebida; si es metafórico, adaptar la imagen sin perder tono.",
        "category": "regional_lexicon",
    },
    "cutre": {
        "label": "Coloquialismo peninsular",
        "suggestion": "Elegir por contexto: 'corriente', 'de mala muerte', 'pobre', 'mal hecho' o una solución literaria mexicana.",
        "category": "regional_lexicon",
    },
    "localizaciones_locaciones": {
        "label": "Calco regional de ubicaciones/locaciones",
        "suggestion": "Usar 'locaciones', 'lugares' o 'ubicaciones' según el sentido; evitar 'localizaciones' para escenarios.",
        "category": "translation_drift",
    },
    "ordenador": {
        "label": "Léxico peninsular",
        "suggestion": "Usar 'computadora' salvo que el contexto histórico o técnico exija otra cosa.",
        "category": "regional_lexicon",
    },
    "movil_phone": {
        "label": "Léxico peninsular probable",
        "suggestion": "Usar 'celular' cuando se refiere a un teléfono; conservar 'móvil' sólo si significa movimiento u objeto móvil.",
        "category": "regional_lexicon",
    },
    "chaval": {
        "label": "Coloquialismo peninsular",
        "suggestion": "Usar 'muchacho', 'chavo', 'joven' u otra solución que respete época, clase social y voz narrativa.",
        "category": "regional_idiom",
    },
    "currar_curro": {
        "label": "Coloquialismo peninsular",
        "suggestion": "Usar 'trabajar', 'chamba' o una forma literaria mexicana según registro y época.",
        "category": "regional_idiom",
    },
    "guay": {
        "label": "Coloquialismo peninsular",
        "suggestion": "Reescribir como 'bien', 'genial', 'padre' o una alternativa sobria por contexto editorial.",
        "category": "regional_idiom",
    },
    "flipar": {
        "label": "Coloquialismo peninsular",
        "suggestion": "Usar 'alucinar', 'sorprenderse', 'quedarse impactado' o una solución literaria por contexto.",
        "category": "regional_idiom",
    },
    "mola": {
        "label": "Coloquialismo peninsular",
        "suggestion": "Usar 'me gusta', 'está bien', 'está padre' o una alternativa editorial mexicana.",
        "category": "regional_idiom",
    },
    "hostia_interjection": {
        "label": "Interjección peninsular",
        "suggestion": "Adaptar la exclamación al registro mexicano y al personaje; no usar un reemplazo automático universal.",
        "category": "regional_idiom",
    },
    "gilipollas": {
        "label": "Insulto peninsular",
        "suggestion": "Adaptar insulto por época, clase social e intensidad: 'idiota', 'imbécil' u otra opción de voz mexicana.",
        "category": "regional_idiom",
    },
    "coger_regional": {
        "label": "Verbo regional sensible en México",
        "suggestion": "Revisar sentido y reemplazar por 'tomar', 'agarrar', 'recoger' u otra opción si en México produce doble sentido.",
        "category": "regional_semantics",
    },
}

_SPANISH_MODERNIZATION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "archaic_e_conjunction",
        re.compile(r"\be\s+(?!(?:i|hi)[a-záéíóúüñ])(?=[a-záéíóúüñ])", re.IGNORECASE),
    ),
    (
        "contracted_archaism",
        re.compile(
            r"\b(?:"
            + "|".join((
                "de" "sta",
                "de" "ste",
                "de" "stas",
                "de" "stos",
                "de" "sto",
                "de" "lla",
                "de" "llas",
                "de" "llo",
                "de" "llos",
                "des" "que",
                "den" "de",
                "a" "queste",
                "a" "questa",
                "a" "questos",
                "a" "questas",
                "an" "s[ií]",
                "ag" "ora",
                "mesmo",
                "mesma",
                "mesmos",
                "mesmas",
                "fasta",
                "non",
                "ca",
            ))
            + r")\b",
            re.IGNORECASE,
        ),
    ),
    (
        "archaic_enclitic",
        re.compile(
            r"\b(?:serv[ií]ase|tra[ií]anle|ten[ií]anle|hab[ií]anse|"
            r"sal[ií]ase|ech[aá]banle|ech[aá]base|dij[oó]le|torn[oó]se|"
            r"mand[oó]le|pregunt[oó]le|respond[ií]ole|dij[eé]ronle|"
            r"fu[eé]ronse|dec[ií]ale|hac[ií]ase|pon[ií]anle)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "archaic_syntax_phrase",
        re.compile(
            r"\b(?:la\s+color|cada\s+un\s+d[ií]a|lejas\s+tierras|"
            r"desde\s+a\s+\w+|entrar\s+de\s+rota\s+batida|"
            r"tener(?:lo|le)?\s+en\s+merced|(?:se\s+lo\s+)?ten[ií]a(?:n)?\s+en\s+merced)\b",
            re.IGNORECASE,
        ),
    ),
)


def is_spanish_target(target_language: str) -> bool:
    return (target_language or "").strip().lower() in _SPANISH_TARGETS


def resolve_spanish_variant(prompt_options: Optional[Mapping] = None) -> str:
    prompt_options = prompt_options or {}
    return str(
        prompt_options.get("spanish_variant")
        or prompt_options.get("target_locale")
        or "mexican"
    ).strip().lower()


def is_mexican_spanish_target(
    target_language: str,
    prompt_options: Optional[Mapping] = None,
) -> bool:
    if not is_spanish_target(target_language):
        return False
    variant = resolve_spanish_variant(prompt_options)
    if variant in GENERIC_SPANISH_VARIANTS:
        return False
    return variant in MEXICAN_SPANISH_VARIANTS


def _mexican_issue_match_is_exempt(
    code: str,
    match: re.Match[str],
    text: str,
) -> bool:
    before = text[max(0, match.start() - 80):match.start()]
    after = text[match.end():match.end() + 80]
    value = match.group(0).casefold()
    if code == "hala_vale" and value == "vale":
        if re.search(r"\bmás(?:\s+(?:te|le|les|nos))?\s+$", before, re.IGNORECASE):
            return True
        if re.match(r"\s+(?:la\s+pena|un\b|una\b|\d|más\b|menos\b)", after, re.IGNORECASE):
            return True
        # "Vale" is regional drift here only as a discourse marker.  Verbal
        # uses such as "esto vale" are ordinary Mexican Spanish.
        left = before.rstrip()[-1:] if before.rstrip() else ""
        right = after.lstrip()[:1] if after.lstrip() else ""
        return left not in {"", ",", ";", ":", "¿", "¡", "—", "-", "("} or right not in {
            "", ".", ",", ";", ":", "?", "!", "…", "—", "-", ")"
        }
    if code == "movil_phone":
        left_clause = re.split(r"[.!?…;]", before)[-1]
        right_clause = re.split(r"[.!?…;]", after)[0]
        context = f"{left_clause} {right_clause}".casefold()
        phone_cues = (
            "teléfono", "telefono", "celular", "smartphone", "llamada", "mensaje",
            "marcó", "marco", "llamar", "número", "numero", "contacto", "batería",
        )
        if any(cue in context for cue in phone_cues):
            return False
        return not bool(
            re.search(
                r"\b(?:el|los|un|unos|su|sus|mi|mis|tu|tus|este|estos|ese|esos|aquel|aquellos)\s+$",
                before,
                re.IGNORECASE,
            )
        )
    if code == "vuestro" and match.group(0)[:1].isupper():
        # Capitalized ritual/formulaic voices often use Vuestra + a named
        # concept.  They are not evidence of accidental Peninsular narration.
        return bool(re.match(r"\s+[A-ZÁÉÍÓÚÜÑ]", after))
    return False


def _iter_mexican_issue_matches(
    code: str,
    pattern: re.Pattern[str],
    text: str,
):
    for match in pattern.finditer(text or ""):
        if not _mexican_issue_match_is_exempt(code, match, text or ""):
            yield match


def count_mexican_spanish_issues(text: str) -> dict[str, int]:
    """Count non-Mexican editorial Spanish forms that should not pass output QA."""
    counts: Counter[str] = Counter()
    for code, pattern in _MEXICAN_SPANISH_PATTERNS:
        matches = list(_iter_mexican_issue_matches(code, pattern, text or ""))
        if matches:
            counts[code] += len(matches)
    return dict(counts)


def mexican_spanish_issue_catalog() -> dict[str, dict[str, str]]:
    """Return metadata used by repair prompts, output dashboards, and glossary QA."""
    return {code: dict(metadata) for code, metadata in _MEXICAN_SPANISH_ISSUE_CATALOG.items()}


def mexican_spanish_issue_codes() -> set[str]:
    return {code for code, _pattern in _MEXICAN_SPANISH_PATTERNS}


def total_mexican_spanish_issues(text: str) -> int:
    return sum(count_mexican_spanish_issues(text).values())


def count_spanish_modernization_residue(text: str) -> dict[str, int]:
    """Count old-Spanish residue that should not survive strong modernization."""
    counts: Counter[str] = Counter()
    for code, pattern in _SPANISH_MODERNIZATION_PATTERNS:
        matches = pattern.findall(text or "")
        if matches:
            counts[code] += len(matches)
    for code, count in count_mexican_spanish_issues(text).items():
        if count:
            counts[code] += count
    return dict(counts)


def collect_spanish_modernization_residue_examples(
    text: str,
    *,
    max_per_type: int = 5,
    max_total: int = 14,
) -> dict[str, list[str]]:
    """Collect concrete old-Spanish residue hits from the current draft.

    These are not global equivalence rules. They are per-chunk evidence used to
    tell the repair model what still appears in its own candidate.
    """
    if not text:
        return {}
    examples: dict[str, list[str]] = {}
    total = 0
    mexican_codes = mexican_spanish_issue_codes()
    for code, pattern in (*_SPANISH_MODERNIZATION_PATTERNS, *_MEXICAN_SPANISH_PATTERNS):
        if total >= max_total:
            break
        seen: set[str] = set()
        hits: list[str] = []
        matches = (
            _iter_mexican_issue_matches(code, pattern, text)
            if code in mexican_codes
            else pattern.finditer(text)
        )
        for match in matches:
            value = re.sub(r"\s+", " ", match.group(0)).strip()
            key = value.casefold()
            if not value or key in seen:
                continue
            seen.add(key)
            hits.append(value)
            total += 1
            if len(hits) >= max_per_type or total >= max_total:
                break
        if hits:
            examples[code] = hits
    return examples


def collect_mexican_spanish_issue_examples(
    text: str,
    *,
    max_per_type: int = 5,
    max_total: int = 20,
) -> dict[str, list[str]]:
    """Collect exact local evidence for Mexican editorial Spanish QA.

    Returned examples are compact, bounded, and safe to show in the pre-download
    dashboard or inject into a repair prompt.
    """
    if not text:
        return {}
    examples: dict[str, list[str]] = {}
    total = 0
    for code, pattern in _MEXICAN_SPANISH_PATTERNS:
        if total >= max_total:
            break
        seen: set[str] = set()
        hits: list[str] = []
        for match in _iter_mexican_issue_matches(code, pattern, text):
            value = re.sub(r"\s+", " ", match.group(0)).strip()
            key = value.casefold()
            if not value or key in seen:
                continue
            seen.add(key)
            hits.append(value)
            total += 1
            if len(hits) >= max_per_type or total >= max_total:
                break
        if hits:
            examples[code] = hits
    return examples


def total_spanish_modernization_residue(text: str) -> int:
    return sum(count_spanish_modernization_residue(text).values())


def format_mexican_spanish_issues(issue_counts: Mapping[str, int]) -> str:
    if not issue_counts:
        return "none"
    return ", ".join(f"{code}={count}" for code, count in sorted(issue_counts.items()))


def format_spanish_modernization_residue(issue_counts: Mapping[str, int]) -> str:
    if not issue_counts:
        return "none"
    return ", ".join(f"{code}={count}" for code, count in sorted(issue_counts.items()))


def format_spanish_modernization_residue_examples(examples: Mapping[str, list[str]]) -> str:
    if not examples:
        return "none"
    parts: list[str] = []
    for code, hits in sorted(examples.items()):
        compact = ", ".join(f'"{hit}"' for hit in hits[:5])
        if compact:
            parts.append(f"{code}: {compact}")
    return "; ".join(parts) if parts else "none"


def build_mexican_spanish_repair_instructions(
    issue_counts: Optional[Mapping[str, int]] = None,
    *,
    examples: Optional[Mapping[str, list[str]]] = None,
) -> str:
    issue_summary = format_mexican_spanish_issues(issue_counts or {})
    example_summary = format_spanish_modernization_residue_examples(examples or {})
    return f"""
The draft contains Peninsular/archaic or non-Mexican Spanish forms that are not acceptable for this project.
Fix ONLY the regional register while preserving meaning, paragraphs, names, punctuation intent, tags, and literary force.

Detected issue types: {issue_summary}
Detected examples: {example_summary}

Mandatory Mexican/LatAm conversions:
- Never output "vosotros", "vosotras", or the pronoun "os"; use "ustedes", "se", "los/las/les", or a natural Mexican construction.
- Do not use "vuestro/vuestra/vuestros/vuestras" as ordinary possession; use "su/sus" or rewrite naturally. Preserve conventional historical honorifics such as "Vuestra Alteza", "Vuestra Majestad", "Vuestra Excelencia", "Vuestra Santidad" and "Vuestra Señoría" when the source and profile require them.
- Never output "habéis/habréis/sois/estáis/rendisteis"; use Mexican/LatAm verb forms such as "han", "habrán", "son", "están", "se rindieron".
- Never output old second-person plural forms such as "podéis", "queréis", "tenéis", "fuérais", "fuisteis", "vuestro/vuestra", or "os"; choose a natural contemporary treatment by context.
- Never output "-aos/-eos" imperatives such as "avispaos" or "precipitaos"; use forms like "estén atentos" or "precipítense".
- Replace "chitón" with natural Mexican literary wording such as "guarden silencio", "cállense", or a sentence-specific equivalent.
- Replace Peninsular vocabulary such as "zumo", "cutre", "ordenador", "chaval", "currar/curro", "guay", "flipar", "mola", "gilipollas", and interjectional "hostia" with Mexican editorial wording appropriate to the character, period, social register, and sentence.
- Avoid "localizaciones" when the meaning is locations/filming sites/places; use "locaciones", "lugares", or "ubicaciones" according to context.
- Review "móvil" and "coger" carefully: keep them only when the local Mexican meaning is intended; otherwise use "celular", "tomar", "agarrar", "recoger", or another context-specific solution.

Before returning, scan the full output and make sure none of the forbidden forms remains outside a justified conventional honorific.
""".strip()


def build_spanish_modernization_repair_instructions(
    issue_counts: Optional[Mapping[str, int]] = None,
    *,
    examples: Optional[Mapping[str, list[str]]] = None,
) -> str:
    issue_summary = format_spanish_modernization_residue(issue_counts or {})
    example_summary = format_spanish_modernization_residue_examples(examples or {})
    return f"""
The draft still reads like old Spanish instead of current editorial Spanish.
Fix the modernization residue while preserving every fact, name, number,
relationship, scene, order, paragraph intent, and authorial force.

Detected residue types: {issue_summary}
Concrete residual forms found in this draft: {example_summary}

Mandatory modernization rules:
- Replace archaic conjunction "e" with "y" except where modern Spanish genuinely requires "e" before an /i/ sound.
- Modernize old contracted demonstratives, prepositional contractions, temporal adverbs, and obsolete particles when they are only surface archaism.
- Rewrite old enclitic syntax such as "servíase", "traíanle", "habíanse", "díjole", "tornóse", and similar forms into natural contemporary syntax.
- For Mexican/LatAm output, replace ordinary "os", "vuestro/vuestra", "podéis", "queréis", "tenéis", "habéis", "sois", "estáis", "fuérais", and related second-person plural forms with a context-faithful contemporary treatment. Preserve conventional historical honorifics such as "Vuestra Alteza" when they are semantically required.
- Do not flatten the prose into a school summary. Keep literary dignity and historical voice through rhythm and point of view, not through obsolete grammar.

Before returning, scan the full output and remove every residual old-Spanish surface form unless the active profile glossary explicitly says to preserve it.
""".strip()
