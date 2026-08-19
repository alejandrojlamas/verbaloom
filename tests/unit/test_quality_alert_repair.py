import re

import pytest

from src.core.llm import LLMResponse
from src.core.translator import (
    _count_quality_alerts,
    _maybe_repair_quality_alerts,
    _resolve_profile_audit_model,
    _resolve_quality_alert_model,
)


class _FakeProvider:
    def __init__(self, model="deepseek-v4-pro"):
        self.model = model


class _FakeClient:
    def __init__(self, content: str, model="deepseek-v4-pro"):
        self.provider = _FakeProvider(model)
        self.content = content
        self.make_request_calls = []
        self.generate_calls = []

    def _get_provider(self):
        return self.provider

    async def make_request(self, prompt, model=None, timeout=None, system_prompt=None):
        self.make_request_calls.append({
            "model": model,
            "prompt": prompt,
            "system_prompt": system_prompt,
        })
        if model:
            self.provider.model = model
        return LLMResponse(
            content=self.content,
            prompt_tokens=11,
            completion_tokens=7,
            context_used=18,
            context_limit=1_000_000,
        )

    async def generate(self, prompt, system_prompt=None, timeout=None):
        self.generate_calls.append({
            "prompt": prompt,
            "system_prompt": system_prompt,
        })
        return LLMResponse(
            content=self.content,
            prompt_tokens=5,
            completion_tokens=5,
            context_used=10,
            context_limit=1_000_000,
        )

    def extract_translation(self, response):
        match = re.search(r"<TRANSLATION>(.*?)</TRANSLATION>", response, re.S)
        return match.group(1).strip() if match else None


def test_quality_alert_model_defaults_to_deepseek_pro_for_deepseek_primary():
    assert _resolve_quality_alert_model("deepseek-v4-pro", {}) == "deepseek-v4-pro"
    assert _resolve_quality_alert_model("deepseek-chat", {}) == "deepseek-v4-pro"
    assert _resolve_quality_alert_model("gpt-4.1", {}) == "gpt-4.1"


def test_profile_audit_model_off_alias_falls_back_without_name_error():
    assert _resolve_profile_audit_model(
        "deepseek-v4-pro",
        {"profile_audit_model": "off"},
    ) == "deepseek-v4-pro"


def test_quality_alert_model_can_be_overridden_or_disabled():
    assert (
        _resolve_quality_alert_model(
            "deepseek-v4-pro",
            {"quality_alert_model": "deepseek-chat"},
        )
        == "deepseek-chat"
    )
    assert (
        _resolve_quality_alert_model(
            "deepseek-v4-pro",
            {"quality_alert_model": "same"},
        )
        == "deepseek-v4-pro"
    )


def test_quality_alert_counter_includes_regional_and_literary_alerts():
    counts = _count_quality_alerts(
        "Chitón. ¡Oh! ustedes, mis agujas. _¡Así_, renuncio.",
        target_language="Spanish",
        prompt_options={"spanish_variant": "mexican"},
    )

    assert counts["chiton"] == 1
    assert counts["awkward_oh_ustedes"] == 1
    assert counts["markdown_emphasis_residue"] == 1


@pytest.mark.asyncio
async def test_quality_alert_repair_uses_pro_and_restores_primary_model():
    client = _FakeClient(
        "<TRANSLATION>Guarden silencio. ¡Oh, mis agujas! ¡Así, renuncio.</TRANSLATION>",
        model="deepseek-chat",
    )

    repaired, response = await _maybe_repair_quality_alerts(
        "Chitón. ¡Oh! ustedes, mis agujas. _¡Así_, renuncio.",
        target_language="Spanish",
        model="deepseek-chat",
        client=client,
        prompt_options={"spanish_variant": "mexican"},
    )

    assert repaired == "Guarden silencio. ¡Oh, mis agujas! ¡Así, renuncio."
    assert response.context_used == 18
    assert client.make_request_calls[0]["model"] == "deepseek-v4-pro"
    assert client.provider.model == "deepseek-chat"
    assert _count_quality_alerts(
        repaired,
        target_language="Spanish",
        prompt_options={"spanish_variant": "mexican"},
    ) == {}


@pytest.mark.asyncio
async def test_quality_alert_repair_can_use_primary_model_when_configured():
    client = _FakeClient("<TRANSLATION>Guarden silencio.</TRANSLATION>")

    repaired, _ = await _maybe_repair_quality_alerts(
        "Chitón.",
        target_language="Spanish",
        model="deepseek-v4-pro",
        client=client,
        prompt_options={"quality_alert_model": "same"},
    )

    assert repaired == "Guarden silencio."
    assert client.make_request_calls == []
    assert len(client.generate_calls) == 1


@pytest.mark.asyncio
async def test_quality_alert_repair_cleans_protocol_wrappers_before_accepting():
    client = _FakeClient(
        "<TRANSLATION><PROFILE_TERM_REVIEW_JSON>Guarden silencio.</PROFILE_TERM_REVIEW_JSON></TRANSLATION>"
    )

    repaired, _ = await _maybe_repair_quality_alerts(
        "Chitón.",
        target_language="Spanish",
        model="deepseek-v4-pro",
        client=client,
        prompt_options={"spanish_variant": "mexican"},
    )

    assert repaired == "Guarden silencio."


@pytest.mark.asyncio
async def test_quality_alert_repair_rejects_lost_asterisk_marker_even_if_alert_clears():
    client = _FakeClient(
        "<TRANSLATION>Guarden silencio. Reina una confusión absoluta.</TRANSLATION>"
    )
    original = "Chitón. * * * Reina una confusión absoluta."

    repaired, response = await _maybe_repair_quality_alerts(
        original,
        target_language="Spanish",
        model="deepseek-v4-pro",
        client=client,
        prompt_options={"spanish_variant": "mexican"},
    )

    assert repaired == original
    assert response.context_used == 10


@pytest.mark.asyncio
async def test_quality_alert_repair_rejects_dialogue_quote_regression():
    client = _FakeClient(
        "<TRANSLATION>“Guarden silencio. Aquí viene.”</TRANSLATION>"
    )
    original = "—Chitón. Aquí viene."

    repaired, _ = await _maybe_repair_quality_alerts(
        original,
        target_language="Spanish",
        model="deepseek-v4-pro",
        client=client,
        prompt_options={"spanish_variant": "mexican"},
    )

    assert repaired == original
