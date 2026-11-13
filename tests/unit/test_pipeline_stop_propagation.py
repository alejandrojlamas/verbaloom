"""Operational stops must not become successful local-only profiles or retries."""

import pytest

from src.core.book_profiles.preparation import prepare_book_profile_from_text
from src.core.book_profiles.term_review import review_profile_terms
from src.core.llm.exceptions import ProviderActionRequiredError, RateLimitError
from src.core.usage.budget import BudgetLimitError


class StoppedProvider:
    def __init__(self, error):
        self.error = error
        self.calls = 0

    async def generate(self, *args, **kwargs):
        self.calls += 1
        raise self.error


@pytest.mark.parametrize(
    "error",
    [
        ProviderActionRequiredError(
            "Review needed", reason="provider_refusal", provider="openai"
        ),
        BudgetLimitError("Budget reached"),
        RateLimitError("Wait for provider", retry_after=5),
    ],
)
@pytest.mark.asyncio
async def test_term_review_preserves_operational_stop_and_does_not_split(error):
    provider = StoppedProvider(error)
    candidates = [
        {
            "source": f"Technical Concept {i}",
            "category": "technical",
            "occurrences": 8,
            "confidence": 0.9,
            "contexts": [f"Technical Concept {i} describes the mechanism."],
        }
        for i in range(24)
    ]
    with pytest.raises(type(error)) as caught:
        await review_profile_terms(
            candidates,
            profile_id="fixture",
            language="Spanish",
            target_locale="es-ES",
            llm_provider=provider,
        )
    assert caught.value is error
    assert provider.calls == 1


@pytest.mark.asyncio
async def test_profile_discovery_does_not_publish_success_after_budget_stop(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    provider = StoppedProvider(BudgetLimitError("Budget reached"))
    events = []
    with pytest.raises(BudgetLimitError):
        await prepare_book_profile_from_text(
            "The narrator walked down the empty street. " * 60,
            source_name="fixture.txt",
            source_language="English",
            language="Spanish",
            target_locale="es-ES",
            llm_provider=provider,
            max_llm_chunks=2,
            progress_callback=events.append,
        )
    assert provider.calls == 1
    assert not any(e.get("stage") == "completed" for e in events)


@pytest.mark.asyncio
async def test_epub_phase_two_preserves_budget_stop_instead_of_generic_quality_failure(
    monkeypatch,
):
    from src.core.epub import xhtml_translator
    from src.core.epub.translation_metrics import TranslationMetrics

    source = "[id0]First paragraph.[id1]Second paragraph.[id2]"
    stop = BudgetLimitError("Budget reached")
    calls = []

    async def request(text, **kwargs):
        calls.append(text)
        if text == source:
            return
        raise stop

    monkeypatch.setattr(xhtml_translator, "generate_translation_request", request)
    with pytest.raises(BudgetLimitError) as caught:
        await xhtml_translator.translate_chunk_with_fallback(
            chunk_text=source,
            local_tag_map={"[id0]": "<p>", "[id1]": "</p><p>", "[id2]": "</p>"},
            global_indices=[0, 1, 2],
            source_language="English",
            target_language="Spanish",
            model_name="fixture",
            llm_client=object(),
            stats=TranslationMetrics(total_chunks=1),
            max_retries=1,
            prompt_options={"fidelity_supervisor_mode": "off"},
            runtime_state={},
        )
    assert caught.value is stop
    assert len(calls) == 2
