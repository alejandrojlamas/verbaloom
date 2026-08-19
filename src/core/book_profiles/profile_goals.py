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
        glossary_goal="Traducir con fidelidad y consistencia: nombres canónicos, términos técnicos/culturales traducidos cuando corresponda, y preservación solo cuando sea una decisión editorial real.",
        discovery_focus=(
            "nombres de personas, lugares, pueblos, instituciones, obras y variantes canónicas",
            "términos técnicos, culturales, históricos o de dominio que necesitan traducción estable",
            "expresiones recurrentes cuyo sentido no debe variar entre capítulos",
            "títulos de obras, cargos, tratamientos y relaciones sociales",
            "términos que parecen nombres propios pero podrían ser conceptos traducibles",
        ),
        reviewer_rules=(
            "Fidelidad no significa conservar literalmente la lengua fuente.",
            "Aprueba source==target solo para nombres propios, siglas, códigos, obras o entidades que realmente se preservan.",
            "Si un término técnico o conceptual tiene traducción directa al idioma destino, aprueba translate_exact o déjalo pendiente con una traducción sugerida.",
            "Mantén nombres de personajes consistentes, pero permite forma canónica en la lengua destino cuando exista.",
        ),
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
        glossary_goal="Detectar arcaísmos, tratamientos, fórmulas, voces y patrones sintácticos que requieren decisiones de modernización consistentes.",
        discovery_focus=(
            "arcaísmos léxicos y ortográficos recurrentes",
            "tratamientos, fórmulas de cortesía y distancia social",
            "patrones sintácticos antiguos que deben modernizarse por contexto",
            "voces de narrador y personajes que no deben homogeneizarse",
            "frases icónicas o efectos de voz que conviene recrear, no aplanar",
        ),
        reviewer_rules=(
            "No conviertas arcaísmo superficial en regla global; apruébalo solo en este perfil.",
            "No preserves source==target cuando impide la modernización esperada.",
            "Distingue rasgo de voz de obstáculo lingüístico antiguo.",
            "Los tratamientos requieren decisión contextual; no apruebes equivalencias automáticas dudosas.",
        ),
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
        glossary_goal="Preparar una traducción fiel que se escuche bien: nombres consistentes, términos traducibles, notas/captions/rótulos tratados sin romper la narración.",
        discovery_focus=(
            "nombres propios y títulos que deben pronunciarse o preservarse de forma consistente",
            "pies de imagen informativos que pueden reescribirse como prosa escuchable",
            "créditos visuales, URLs, numeración, notas y referencias que deben moverse o limpiarse",
            "términos de dominio que deben traducirse para no sonar como restos de la fuente",
            "patrones de tablas, listas o aparatos críticos que necesitan reconstrucción auditiva",
        ),
        reviewer_rules=(
            "La salida debe seguir siendo traducción fiel, no resumen.",
            "No apruebes URLs, paginación, créditos de imagen o basura visual como glosario inyectable.",
            "Distingue pies de imagen informativos de créditos visuales.",
            "Los términos traducibles deben traducirse; preserve_exact solo para nombres, siglas o títulos reales.",
        ),
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
        glossary_goal="Identificar conceptos complejos, términos culturales/técnicos, tablas y referencias que deben explicarse con precisión sin simplificación infantil.",
        discovery_focus=(
            "conceptos difíciles que necesitan explicación consistente",
            "términos técnicos o culturales que no deben banalizarse",
            "tablas, ejemplos, notas y referencias que requieren interpretación clara",
            "nombres y obras que deben mantenerse reconocibles",
        ),
        reviewer_rules=(
            "No conviertas términos clave en nombres preservados si deben explicarse o traducirse.",
            "Aprueba pocos preserve_exact; prioriza contextual y translate_exact.",
            "El glosario debe ayudar a explicar con precisión, no a simplificar de más.",
        ),
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
        glossary_goal="Preparar señales de voz, nombres, motivos, frases recurrentes y decisiones de estilo para un pulido consistente.",
        discovery_focus=(
            "nombres, lugares, relaciones y motivos recurrentes",
            "voces narrativas y registros de personajes",
            "frases recurrentes, imágenes, símbolos e ironías que deben mantenerse",
            "términos regionales o culturales que afectan tono y naturalidad",
        ),
        reviewer_rules=(
            "No apruebes cambios que homogenicen voces.",
            "Mantén términos de voz como decisiones contextuales salvo que sean mecánicamente seguros.",
            "Preserva nombres reales; traduce conceptos o títulos cuando el perfil lo requiera.",
        ),
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
