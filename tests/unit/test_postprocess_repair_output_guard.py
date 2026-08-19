import pytest

from src.core.postprocess_repair import repair_flagged_chunks


class _FakeResponse:
    content = "<TRANSLATIONATION>\nTexto limpio reparado.\n</TRANSLATIONATION>\nReturn only the translated text."


class _FakeClient:
    async def make_request(self, *args, **kwargs):
        return _FakeResponse()

    def extract_translation(self, response):
        return response


@pytest.mark.asyncio
async def test_postprocess_repair_cleans_llm_protocol_before_accepting_candidate():
    result = await repair_flagged_chunks(
        refined_parts=["Texto limpio reparado.\n<TRANSLATIONATION>"],
        structured_chunks=[{"_source_text": "Texto limpio reparado."}],
        target_language="Spanish",
        model_name="fake-model",
        api_endpoint="",
        llm_provider="deepseek",
        prompt_options={"postprocess_repair_enabled": True},
        llm_client=_FakeClient(),
    )

    assert result.repaired_indices == [0]
    assert result.parts == ["Texto limpio reparado."]
    assert "TRANSLATION" not in result.parts[0]
    assert "Return only" not in result.parts[0]
