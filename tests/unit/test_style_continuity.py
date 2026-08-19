import re

import pytest

from src.core.llm.base import LLMResponse
from src.core.style_continuity import build_style_continuity_hint
from src.core.translator import _make_llm_request_with_adaptive_context, _make_refinement_request


def test_style_continuity_hint_is_compact_and_metric_based():
    reference = (
        "Ella abrió la puerta. Miró el patio. Nadie contestó.\n\n"
        "Entonces caminó despacio, con una calma que parecía sostener toda la escena. "
    ) * 10

    hint = build_style_continuity_hint(reference)

    assert "# STYLE CONTINUITY HINT" in hint
    assert "do not copy facts or wording" in hint
    assert len(hint) < 800
    assert "Ella abrió la puerta" not in hint


def test_style_continuity_hint_can_be_disabled():
    reference = ("Una frase breve. Otra frase breve. " * 40)

    assert build_style_continuity_hint(reference, prompt_options={"style_continuity": False}) == ""


class _FakeRefinementClient:
    def __init__(self):
        self.prompt = ""
        self.system_prompt = ""

    async def make_request(self, prompt, model=None, **kwargs):
        self.prompt = prompt
        self.system_prompt = kwargs.get("system_prompt") or ""
        return LLMResponse(
            content="<TRANSLATION>Texto refinado final.</TRANSLATION>",
            prompt_tokens=10,
            completion_tokens=5,
            context_used=15,
            context_limit=4096,
        )

    def extract_translation(self, content):
        match = re.search(r"<TRANSLATION>(.*?)</TRANSLATION>", content, re.S)
        return match.group(1).strip() if match else None


class _FakeTranslationClient(_FakeRefinementClient):
    async def generate(self, prompt, **kwargs):
        self.prompt = prompt
        self.system_prompt = kwargs.get("system_prompt") or ""
        return LLMResponse(
            content="<TRANSLATION>Texto traducido final.</TRANSLATION>",
            prompt_tokens=12,
            completion_tokens=6,
            context_used=18,
            context_limit=4096,
        )


@pytest.mark.asyncio
async def test_refinement_prompt_includes_style_continuity_hint_when_context_exists():
    client = _FakeRefinementClient()
    previous = (
        "Ella abrió la puerta. Miró el patio. Nadie contestó.\n\n"
        "Entonces caminó despacio, con una calma que parecía sostener toda la escena. "
    ) * 10

    refined, _response = await _make_refinement_request(
        draft_translation="Texto refinado final.",
        context_before="",
        context_after="",
        previous_refined_context=previous,
        target_language="Spanish",
        model="fake-model",
        llm_client=client,
        log_callback=None,
        has_placeholders=False,
        prompt_options={},
        source_text="Texto refinado final.",
    )

    assert refined == "Texto refinado final."
    assert "# STYLE CONTINUITY HINT" in client.prompt


@pytest.mark.asyncio
async def test_translation_prompt_includes_style_continuity_hint_when_previous_context_exists():
    client = _FakeTranslationClient()
    previous = (
        "Ella abrió la puerta. Miró el patio. Nadie contestó.\n\n"
        "Entonces caminó despacio, con una calma que parecía sostener toda la escena. "
    ) * 10

    translated, _content, _response = await _make_llm_request_with_adaptive_context(
        main_content="She opened the door.",
        context_before="",
        context_after="",
        previous_translation_context=previous,
        source_language="English",
        target_language="Spanish",
        model="fake-model",
        llm_client=client,
        log_callback=None,
        has_placeholders=False,
        prompt_options={},
    )

    assert translated == "Texto traducido final."
    assert "# STYLE CONTINUITY HINT" in client.prompt
