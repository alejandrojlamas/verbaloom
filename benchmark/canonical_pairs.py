"""
Canonical (source, target) language pair sets for the benchmark.

These three tiers are the **fixed contract** that makes scores comparable
across models, contributors, and time. Don't substitute the lists silently —
if you change them, bump the rubric version in `docs/JUDGE_RUBRIC.md` and
publish the change.

Design principle: **unidirectional**. We pick the direction with strongest
real-world demand for each language. We do NOT include both `en→fr` and
`fr→en`; the wiki shows one direction per cell.

Selection rationale (see `docs/BENCHMARK_WORKFLOW.md`):

- `en→X` covers users *consuming* English content in their native language
  (the dominant casual-translation flow).
- `X→en` covers fan-translation / academic-publishing flows (manga, k-lit,
  Chinese webnovels, papers).
- Cross-Asian (e.g. `ja→zh-Hans`, `zh-Hans→ja`) covers documented anime/
  manga/light-novel industry flows.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Quick — 8 pairs. Used by default. Fast iteration on a new model (~45
# translations, ~10 min of judge time). Picked via market study at the time
# of v2 launch; do not silently add/remove.
# ---------------------------------------------------------------------------

QUICK_PAIRS: list[tuple[str, str]] = [
    ("en", "zh-Hans"),   # #1 in real demand: Chinese users importing foreign content
    ("en", "es"),        # 500M+ speakers, high-volume baseline
    ("en", "fr"),        # quality baseline (DeepL is excellent here)
    ("en", "vi"),        # underserved by mainstream tools, growing market
    ("ja", "en"),        # manga / light novel community
    ("ko", "en"),        # k-literature: +285% in 2024
    ("zh-Hans", "en"),   # Chinese webnovel / academic flow
    ("ja", "zh-Hans"),   # manga industry flow into China
]


# ---------------------------------------------------------------------------
# Standard — 16 pairs = Quick + 8 outbound English to major target languages.
# ~125 translations per model, ~30 min of judge time. Recommended default
# when a model is being evaluated for general-purpose use.
# ---------------------------------------------------------------------------

_STANDARD_ADDITIONS: list[tuple[str, str]] = [
    ("en", "de"),        # Germanic, large EU market
    ("en", "pt"),        # 270M speakers (Brazil + Portugal)
    ("en", "ja"),        # English content into Japanese (anime, tech, business)
    ("en", "ko"),        # English content into Korean (k-pop subs, business)
    ("en", "ru"),        # post-Western-tools market, large diaspora
    ("en", "it"),        # Italian publishing, classical literature
    ("en", "ar"),        # Arabic-speaking world, major underserved market
    ("en", "hi"),        # 1.5B+ speakers, fastest-growing translation demand
]

STANDARD_PAIRS: list[tuple[str, str]] = QUICK_PAIRS + _STANDARD_ADDITIONS


# ---------------------------------------------------------------------------
# Full — 28 pairs = Standard + 12 broad-coverage additions. ~245 translations
# per model, ~60 min of judge time. For deep evaluation when budget allows.
# Adds linguistic family / script diversity and underserved markets.
# ---------------------------------------------------------------------------

_FULL_ADDITIONS: list[tuple[str, str]] = [
    # European diversity
    ("en", "nl"),        # Dutch — Netherlands publishing
    ("en", "pl"),        # Polish — 40M speakers, distinct Slavic from Russian
    ("en", "sv"),        # Swedish — Nordic gateway
    ("en", "da"),        # Danish — completes Scandinavian
    ("en", "el"),        # Greek — Hellenic, distinct script
    ("en", "tr"),        # Turkish — 80M, agglutinative, large diaspora

    # Asian — DeepL gaps and underserved markets
    ("en", "th"),        # Thai — 70M, missing from DeepL
    ("en", "id"),        # Indonesian — 280M, missing from DeepL (the biggest gap)
    ("en", "bn"),        # Bengali — 270M, very underserved
    ("en", "ta"),        # Tamil — 80M, Dravidian (different family from Hindi)

    # RTL / Semitic
    ("en", "he"),        # Hebrew — high-value tech/academic market

    # Cross-Asian (unidirectional)
    ("zh-Hans", "ja"),   # Chinese light novels into Japanese (industry flow)
]

FULL_PAIRS: list[tuple[str, str]] = STANDARD_PAIRS + _FULL_ADDITIONS


# ---------------------------------------------------------------------------
# Public lookup
# ---------------------------------------------------------------------------

PAIR_SETS: dict[str, list[tuple[str, str]]] = {
    "quick": QUICK_PAIRS,
    "standard": STANDARD_PAIRS,
    "full": FULL_PAIRS,
}


def get_pair_set(name: str) -> list[tuple[str, str]]:
    """
    Return the canonical pair list for the given tier name.

    Raises:
        KeyError: if name is not 'quick', 'standard', or 'full'.
    """
    key = name.lower().strip()
    if key not in PAIR_SETS:
        raise KeyError(
            f"Unknown pair set '{name}'. Available: {sorted(PAIR_SETS)}"
        )
    return list(PAIR_SETS[key])


def format_pair_set(name: str) -> str:
    """Render a pair set as the space-separated string the CLI expects."""
    return " ".join(f"{src}:{tgt}" for src, tgt in get_pair_set(name))


def summary() -> str:
    """Human-readable summary of the three tiers."""
    lines: list[str] = []
    for name, pairs in PAIR_SETS.items():
        lines.append(f"{name:>8}: {len(pairs)} pairs — {' '.join(f'{s}:{t}' for s, t in pairs)}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
