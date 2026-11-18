"""Business goals for book profile preparation.

The generic profile engine should not decide that every book needs the same
kind of glossary.  These policies describe what a prepared profile is trying to
achieve, how much material it should review, and which decisions are safe to
approve automatically for that goal.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class ProfileGoalRules:
    key: str
    label: str
    description: str
    glossary_goal: str
    discovery_focus: tuple[str, ...]
    reviewer_rules: tuple[str, ...]
    preferred_entry_types: tuple[str, ...]
    translatable_categories: tuple[str, ...]
    preserve_categories: tuple[str, ...]
    local_terms_floor: int
    local_terms_cap: int
    local_terms_scale: float
    review_terms_floor: int
    review_terms_cap: int
    review_fraction: float
    review_batch_size: int
    approved_floor: int
    approved_cap: int
    pending_floor: int
    pending_cap: int
    same_target_translatable_policy: str = "demote_to_pending"
    source_equals_target_note: str = (
        "If a candidate is translatable into the target language, source==target "
        "is not a faithful decision; keep it pending or translate it."
    )
    output_rules: tuple[str, ...] = ()

    def local_terms_limit(self, text_chars: int, *, requested: int | None = None, hard_cap: int = 5000) -> int:
        if requested is not None:
            return max(1, min(int(requested), hard_cap))
        auto = int(self.local_terms_floor + math.sqrt(max(1, text_chars)) * self.local_terms_scale)
        return max(self.local_terms_floor, min(auto, self.local_terms_cap, hard_cap))

    def review_terms_limit(self, *, text_chars: int = 0, candidate_count: int = 0) -> int:
        if candidate_count <= 0:
            return 0
        size_component = int(math.sqrt(max(1, text_chars)) * 0.34)
        variety_component = int(candidate_count * self.review_fraction)
        auto = max(self.review_terms_floor, size_component, variety_component)
        return max(1, min(auto, self.review_terms_cap, candidate_count))

    def approved_limit(self, *, text_chars: int = 0, candidate_count: int = 0) -> int:
        variety = int(math.sqrt(max(1, candidate_count)) * 12)
        size = int(math.log10(max(10, text_chars)) * 28)
        auto = max(self.approved_floor, self.approved_floor + variety + size)
        return max(1, min(auto, self.approved_cap))

    def pending_limit(self, *, text_chars: int = 0, candidate_count: int = 0) -> int:
        variety = int(math.sqrt(max(1, candidate_count)) * 22)
        size = int(math.log10(max(10, text_chars)) * 42)
        auto = max(self.pending_floor, self.pending_floor + variety + size)
        return max(1, min(auto, self.pending_cap))

    def to_config(self) -> dict[str, Any]:
        return {
            "goal": self.key,
            "label": self.label,
            "description": self.description,
            "glossary_goal": self.glossary_goal,
            "discovery_focus": list(self.discovery_focus),
            "reviewer_rules": list(self.reviewer_rules),
            "output_rules": list(self.output_rules),
            "same_target_translatable_policy": self.same_target_translatable_policy,
        }

    def prompt_brief(self) -> str:
        focus = "\n".join(f"- {item}" for item in self.discovery_focus)
        rules = "\n".join(f"- {item}" for item in self.reviewer_rules)
        return (
            f"Business goal: {self.label} ({self.key})\n"
            f"Goal: {self.glossary_goal}\n\n"
            f"Discovery focus:\n{focus}\n\n"
            f"Reviewer rules:\n{rules}\n"
            f"- {self.source_equals_target_note}"
        ).strip()


_COMMON_TRANSLATABLE = (
    "concept",
    "glossary",
    "idiom",
    "key_term",
    "lexical_archaism",
    "orthographic_variant",
    "phrase",
    "syntax_pattern",
    "technical",
    "technical_term",
)

_COMMON_PRESERVE = (
    "acronym",
    "character",
    "location",
    "organization",
    "proper_noun",
    "title",
)

_GOALS: dict[str, ProfileGoalRules] = {
    "faithful_translation": ProfileGoalRules(
        key="faithful_translation",
        label="Traducción fiel",
        description="Consistencia terminológica y editorial para traducir sin omitir, suavizar ni dejar términos traducibles en la lengua fuente.",
        glossary_goal="Establish stable equivalents for recurring names, terms and phrases while keeping different senses distinct.",
        discovery_focus=(
            "Track names, aliases and forms of address together with the speaker and referent that distinguish them.",
            "Find recurring technical or cultural terms with a defined sense; collect contrasting uses if the same word changes meaning.",
            "Record verbal echoes, idioms and cited titles whose identity matters across passages.",
        ),
        reviewer_rules=(
            "Approve an equivalent when the cited usage supports the same referent, sense and register in the target language.",
            "Keep a context-dependent rendering when an idiom, title or form of address changes with the speaker or situation.",
            "Distinguish an authentic cited title from the surrounding translatable description. Leave uncertain identities pending.",
        ),
        output_rules=(
            "Translate each assertion and implication idiomatically, carrying over the clues to who did what to whom and when. Leave identities open wherever the source does.",
            "Match the force of the original, including negation, doubt, understatement and deliberate ambiguity.",
            "Use approved equivalents in their documented sense. Preserve a repeated phrase when its return matters to the passage.",
        ),
        source_equals_target_note="An identical spelling can be the attested target-language form or a quoted original. Require evidence for that use.",
        preferred_entry_types=("proper_noun", "canonical_proper_noun", "technical_term", "concept", "term", "title", "acronym", "idiom"),
        translatable_categories=_COMMON_TRANSLATABLE,
        preserve_categories=_COMMON_PRESERVE,
        local_terms_floor=220,
        local_terms_cap=4500,
        local_terms_scale=1.65,
        review_terms_floor=260,
        review_terms_cap=900,
        review_fraction=0.45,
        review_batch_size=70,
        approved_floor=160,
        approved_cap=700,
        pending_floor=260,
        pending_cap=1000,
    ),
    "modernization": ProfileGoalRules(
        key="modernization",
        label="Modernización literaria",
        description="Modernización intralingüística por obra: actualizar época lingüística sin borrar voz, personajes ni intención.",
        glossary_goal="Map obsolete senses, spellings and sentence patterns to contemporary usage while preserving the period being described.",
        discovery_focus=(
            "Pair each dated word with evidence for its sense at this point in the work, especially when its modern sense differs.",
            "Track address forms by speaker, listener and occasion; note changes of intimacy, deference or hostility.",
            "Distinguish period-wide syntax from an individual speaker's habitual phrasing or a quotation from an older text.",
        ),
        reviewer_rules=(
            "Approve a current equivalent only for the documented historical sense; leave competing readings contextual.",
            "Use the relationship to judge a treatment. A mechanical pronoun swap can change social distance.",
            "Keep historically specific objects, offices and institutions identifiable. Record deliberate archaism as a voice or quotation decision.",
        ),
        output_rules=(
            "Modernize dated syntax, word order and diction to the selected strength in the target language. Ordinary current wording can stand.",
            "Express deference, intimacy and irony through contemporary usage while preserving differences between speakers.",
            "Keep the historical setting, events, material objects and marked quotations recognizable. Avoid anachronistic comparisons.",
        ),
        source_equals_target_note="Retain already-current wording and deliberate historical quotations. A dated form needs a context-specific reason to be retained.",
        preferred_entry_types=("lexical_archaism", "orthographic_variant", "address_form", "syntax_pattern", "character_voice", "iconic_phrase", "proper_noun", "title"),
        translatable_categories=_COMMON_TRANSLATABLE + ("address_form", "character_voice"),
        preserve_categories=_COMMON_PRESERVE + ("iconic_phrase",),
        local_terms_floor=260,
        local_terms_cap=5000,
        local_terms_scale=1.9,
        review_terms_floor=320,
        review_terms_cap=1000,
        review_fraction=0.52,
        review_batch_size=70,
        approved_floor=150,
        approved_cap=560,
        pending_floor=320,
        pending_cap=1300,
    ),
    "audiobook": ProfileGoalRules(
        key="audiobook",
        label="Faithful audiobook",
        description="Traducción fiel optimizada para escucha: continuidad, limpieza auditiva, notas manejables y pies de imagen integrables.",
        glossary_goal="Keep people, terms and references recognizable on first hearing while retaining the full written work.",
        discovery_focus=(
            "Identify names, abbreviations and homographs whose spoken interpretation depends on a definition or a speaker in the text.",
            "Locate phrases in which clause length, a delayed subject or ambiguous attribution can obscure who is speaking.",
            "Distinguish substantive captions, notes and table labels from page furniture using the text and its placement.",
        ),
        reviewer_rules=(
            "Use an expansion or pronunciation cue only when supported by the source or approved glossary; keep spellings canonical in print.",
            "Treat captions, credits, pagination and URLs as document structure. A recurring layout pattern is not a glossary replacement.",
            "Keep definitions and qualifications from notes linked to the claims they qualify. Flag uncertain readings for review.",
        ),
        output_rules=(
            "Shape clauses into natural breath groups. Keep the source's clues to who is speaking, addressed or mentioned audible, even when their identities remain open.",
            "Preserve meaningful pauses and repetitions. Use source-supported names and expansions, retaining written numbers, units and identifiers.",
            "Translate captions, notes and table labels in their own places. Keep attribution, note links and the relationship of each value to its label.",
        ),
        source_equals_target_note="Keep an identical written form when the target language uses it for this referent; confirm unfamiliar spellings against the source.",
        preferred_entry_types=("proper_noun", "canonical_proper_noun", "technical_term", "concept", "caption_pattern", "note_pattern", "reference_pattern", "title", "acronym"),
        translatable_categories=_COMMON_TRANSLATABLE + ("caption", "caption_pattern", "note_pattern", "reference_pattern"),
        preserve_categories=_COMMON_PRESERVE,
        local_terms_floor=220,
        local_terms_cap=3800,
        local_terms_scale=1.55,
        review_terms_floor=240,
        review_terms_cap=760,
        review_fraction=0.42,
        review_batch_size=65,
        approved_floor=130,
        approved_cap=520,
        pending_floor=300,
        pending_cap=1200,
    ),
    "academic_translation": ProfileGoalRules(
        key="academic_translation",
        label="Académico/técnico",
        description="Traducción fiel de textos técnicos o académicos con terminología, tablas, fórmulas, siglas y citas consistentes.",
        glossary_goal="Resolver terminología técnica, conceptos, siglas, métodos, tablas, fórmulas y nombres de trabajos para evitar traducciones inconsistentes.",
        discovery_focus=(
            "términos técnicos, modelos, métodos, medidas, variables y fórmulas",
            "siglas, datasets, instituciones, autores, títulos de papers y bibliografía",
            "conceptos que admiten traducción directa y no deben quedar en inglés por inercia",
            "tablas, captions, notas y referencias que requieren reconstrucción fiel",
            "unidades, números, símbolos y convenciones que deben preservarse",
        ),
        reviewer_rules=(
            "Prioriza traducciones técnicas aceptadas en el idioma destino.",
            "No conserves términos ingleses si existe traducción técnica normal.",
            "Preserva siglas y símbolos, pero traduce su expansión si aparece como texto común.",
            "Marca fórmulas, variables o notación como preserve_exact solo cuando sean notación real.",
        ),
        preferred_entry_types=("technical_term", "concept", "acronym", "formula", "proper_noun", "title", "term"),
        translatable_categories=_COMMON_TRANSLATABLE + ("formula_caption", "method", "dataset"),
        preserve_categories=_COMMON_PRESERVE + ("formula",),
        local_terms_floor=300,
        local_terms_cap=5000,
        local_terms_scale=1.95,
        review_terms_floor=340,
        review_terms_cap=1100,
        review_fraction=0.55,
        review_batch_size=75,
        approved_floor=220,
        approved_cap=850,
        pending_floor=320,
        pending_cap=1400,
    ),
    "explanatory_rewrite": ProfileGoalRules(
        key="explanatory_rewrite",
        label="Explicación universitaria",
        description="Reescritura explicativa en lenguaje común con precisión conceptual y nivel universitario.",
        glossary_goal="Identify the concepts and argumentative steps that an adult reader outside the field needs to understand this passage.",
        discovery_focus=(
            "Find terms whose everyday sense differs from their use here, and the source definitions or contrasts that resolve the difference.",
            "Identify unstated referents and compressed steps whose relationships can be recovered from the author's own argument.",
            "Connect a table heading, unit or formula symbol with the sentence that explains what it measures and under which conditions.",
        ),
        reviewer_rules=(
            "Approve a short gloss when the source supports both the definition and its scope. Mark an unresolved concept as pending.",
            "Keep the technical term available alongside its plain-language sense when later passages depend on the distinction.",
            "Use an example the author gives and retain its limits. A plausible analogy or outside explanation is insufficient evidence for a glossary rule.",
        ),
        output_rules=(
            "Write for an intelligent adult new to this subject. At the first difficult use, retain the key term and unpack its meaning using the passage.",
            "Make a compressed relationship explicit when the source establishes it. Keep assumptions, hypotheses, findings and exceptions distinguishable.",
            "Explain what a formula or table compares using its own labels and units. Preserve every value and qualification; introduce no new examples or conclusions.",
        ),
        source_equals_target_note="A precise technical name can remain unchanged while its source-supported meaning is explained in the surrounding prose.",
        preferred_entry_types=("concept", "technical_term", "term", "proper_noun", "title", "idiom"),
        translatable_categories=_COMMON_TRANSLATABLE,
        preserve_categories=_COMMON_PRESERVE,
        local_terms_floor=180,
        local_terms_cap=3300,
        local_terms_scale=1.35,
        review_terms_floor=200,
        review_terms_cap=620,
        review_fraction=0.36,
        review_batch_size=70,
        approved_floor=100,
        approved_cap=380,
        pending_floor=240,
        pending_cap=900,
    ),
    "literary_polish": ProfileGoalRules(
        key="literary_polish",
        label="Pulido literario",
        description="Revisión de estilo manteniendo fidelidad, voz, continuidad y consistencia literaria.",
        glossary_goal="Record the voice, motifs and word choices that a line editor should recognize before revising a passage.",
        discovery_focus=(
            "Identify shifts among narrator, dialogue and interior thought through syntax, diction or a change in form of address.",
            "Track repeated images, refrains and recurring concrete details, including variations that change their effect.",
            "Distinguish an awkward literal construction from deliberate dialect, interruption, irony or syntactic roughness.",
        ),
        reviewer_rules=(
            "Require a passage-specific example before treating a verbal habit as a character's voice. Leave isolated quirks contextual.",
            "Reserve fixed renderings for stable names and deliberate verbal echoes; preserve a meaningful variation of a motif.",
            "Approve idiomatic alternatives that retain the same image, register and viewpoint. Leave effective wording in place.",
        ),
        output_rules=(
            "Edit locally for idiom, cadence and precise diction. Leave a line that already carries the intended effect.",
            "Keep viewpoint, narrative distance and the rhythm of each speaker distinct, including fragments, pauses and interruptions.",
            "Retain the original image, degree of emphasis and deliberate repetition. Keep subtext implicit where the author leaves it implicit.",
        ),
        source_equals_target_note="An already-effective phrase may remain unchanged. A fixed glossary entry still needs evidence of a recurring editorial choice.",
        preferred_entry_types=("proper_noun", "character_voice", "iconic_phrase", "idiom", "term", "title", "concept"),
        translatable_categories=_COMMON_TRANSLATABLE + ("character_voice",),
        preserve_categories=_COMMON_PRESERVE + ("iconic_phrase",),
        local_terms_floor=220,
        local_terms_cap=4200,
        local_terms_scale=1.6,
        review_terms_floor=260,
        review_terms_cap=820,
        review_fraction=0.44,
        review_batch_size=70,
        approved_floor=140,
        approved_cap=560,
        pending_floor=300,
        pending_cap=1100,
    ),
}

_ALIASES = {
    "translation": "faithful_translation",
    "translate": "faithful_translation",
    "faithful": "faithful_translation",
    "faithful_translation": "faithful_translation",
    "modernize": "modernization",
    "modernizar": "modernization",
    "modernization": "modernization",
    "contemporize": "modernization",
    "audiobook": "audiobook",
    "audio": "audiobook",
    "academic": "academic_translation",
    "technical": "academic_translation",
    "academic_translation": "academic_translation",
    "simplify": "explanatory_rewrite",
    "explain": "explanatory_rewrite",
    "explanatory": "explanatory_rewrite",
    "explanatory_rewrite": "explanatory_rewrite",
    "literary": "literary_polish",
    "literary_polish": "literary_polish",
    "style": "literary_polish",
}


def declared_profile_goal(config: Mapping[str, Any]) -> ProfileGoalRules | None:
    """Read a profile's saved purpose, including explicit settings from older profiles."""
    business_rules = config.get("business_rules")
    goal = (
        config.get("profile_goal")
        or (business_rules.get("goal") if isinstance(business_rules, Mapping) else "")
        or config.get("goal")
    )
    if goal:
        # A custom goal owns its policy even if it also requests an audio companion.
        return _GOALS.get(_ALIASES.get(str(goal).strip().casefold(), ""))

    audio = config.get("audiobook") or config.get("audio_sanitization")
    if isinstance(audio, Mapping):
        audio = audio.get("enabled") is True or audio.get("generate_companion") is True
    if audio is True or config.get("audit_dimensions") == "translation_audiobook_full":
        return _GOALS["audiobook"]

    mode = str(config.get("transform_mode") or config.get("mode") or "").strip().casefold()
    return _GOALS.get(_ALIASES.get(mode, ""))


def resolve_profile_goal(
    value: str = "",
    *,
    transform_mode: str = "",
    source_name: str = "",
    profile_config: Mapping[str, Any] | None = None,
) -> ProfileGoalRules:
    config = profile_config if isinstance(profile_config, Mapping) else {}
    raw = (
        str(value or "").strip()
        or str(config.get("profile_goal") or config.get("goal") or "").strip()
        or str(transform_mode or config.get("transform_mode") or "").strip()
    )
    key = _ALIASES.get(raw.casefold(), "")
    if not key:
        haystack = f"{source_name} {raw}".casefold()
        if "audio" in haystack:
            key = "audiobook"
        elif any(token in haystack for token in ("paper", "thesis", "tesis", "academic", "technical")):
            key = "academic_translation"
        elif any(token in haystack for token in ("modern", "moderniz", "contempor")):
            key = "modernization"
        else:
            key = "faithful_translation"
    return _GOALS.get(key, _GOALS["faithful_translation"])


def profile_goal_options() -> list[dict[str, str]]:
    return [
        {"value": rules.key, "label": rules.label, "description": rules.description}
        for rules in _GOALS.values()
    ]
