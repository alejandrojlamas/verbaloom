"""
Unit tests for translator fallback context isolation (issue #170 fix)
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, Mock

from src.core.translator import _make_llm_request_with_adaptive_context
from src.core.llm.base import LLMResponse
from src.core.llm.exceptions import ContentRiskError


class TestTranslatorFallbackContext:
    """Test that raw fallback responses do not contaminate chunk context chain."""

    @pytest.fixture
    def mock_llm_client(self):
        client = Mock()
        client.extract_translation = Mock(side_effect=lambda text: None)
        return client

    @pytest.mark.asyncio
    async def test_successful_extraction_has_no_fallback_flag(self, mock_llm_client):
        """When tags are found, was_fallback must be False."""
        mock_llm_client.generate = AsyncMock(return_value=LLMResponse(
            content="<TRANSLATION>Bonjour</TRANSLATION>",
            prompt_tokens=10,
            completion_tokens=5,
            context_used=15,
            context_limit=2048,
            was_truncated=False,
        ))
        mock_llm_client.extract_translation = Mock(return_value="Bonjour")
        prompt_options = {}

        translated, _, response = await _make_llm_request_with_adaptive_context(
            main_content="Hello",
            context_before="",
            context_after="",
            previous_translation_context="",
            source_language="English",
            target_language="French",
            model="test-model",
            llm_client=mock_llm_client,
            log_callback=None,
            has_placeholders=False,
            prompt_options=prompt_options,
        )

        assert translated == "Bonjour"
        assert response.was_fallback is False
        records = prompt_options["_candidate_results"]
        assert records[-1]["phase"] == "translation"
        assert records[-1]["decision"] == "accepted"
        assert records[-1]["token_cost"]["total_tokens"] == 15

    @pytest.mark.asyncio
    async def test_plain_text_fallback_sets_fallback_flag(self, mock_llm_client):
        """When extraction fails for plain text, was_fallback must be True."""
        # Response must NOT contain the input text exactly, otherwise echo detection rejects it
        mock_llm_client.generate = AsyncMock(return_value=LLMResponse(
            content="Here is the translation: Bonjour le monde",
            prompt_tokens=10,
            completion_tokens=5,
            context_used=15,
            context_limit=2048,
            was_truncated=False,
        ))
        prompt_options = {}

        translated, _, response = await _make_llm_request_with_adaptive_context(
            main_content="Hello world",
            context_before="",
            context_after="",
            previous_translation_context="",
            source_language="English",
            target_language="French",
            model="test-model",
            llm_client=mock_llm_client,
            log_callback=None,
            has_placeholders=False,
            prompt_options=prompt_options,
        )

        assert translated == "Bonjour le monde"
        assert response.was_fallback is True
        phases = [record["phase"] for record in prompt_options["_candidate_results"]]
        assert "translation_extraction" in phases
        assert "translation_fallback" in phases

    @pytest.mark.asyncio
    async def test_epub_no_fallback_on_failure(self, mock_llm_client):
        """When has_placeholders=True, failed extraction must return None (no raw fallback)."""
        mock_llm_client.generate = AsyncMock(return_value=LLMResponse(
            content="Here is the translation: Hello world",
            prompt_tokens=10,
            completion_tokens=5,
            context_used=15,
            context_limit=2048,
            was_truncated=False,
        ))

        translated, _, response = await _make_llm_request_with_adaptive_context(
            main_content="Hello world",
            context_before="",
            context_after="",
            previous_translation_context="",
            source_language="English",
            target_language="French",
            model="test-model",
            llm_client=mock_llm_client,
            log_callback=None,
            has_placeholders=True,
        )

        assert translated is None
        assert response.was_fallback is False

    @pytest.mark.asyncio
    async def test_provider_truncation_is_never_accepted_without_context_manager(
        self,
        mock_llm_client,
    ):
        mock_llm_client.generate = AsyncMock(return_value=LLMResponse(
            content="<TRANSLATION>Bonjour partiel</TRANSLATION>",
            prompt_tokens=10,
            completion_tokens=128,
            context_used=138,
            context_limit=2048,
            was_truncated=True,
        ))
        mock_llm_client.extract_translation = Mock(return_value="Bonjour partiel")

        translated, _, response = await _make_llm_request_with_adaptive_context(
            main_content="Hello, this complete paragraph must not be cut in half.",
            context_before="",
            context_after="",
            previous_translation_context="",
            source_language="English",
            target_language="French",
            model="test-model",
            llm_client=mock_llm_client,
            log_callback=None,
            has_placeholders=False,
        )

        assert translated is None
        assert response.was_truncated is True

    @pytest.mark.asyncio
    async def test_context_manager_implicit_truncation_retry(self, mock_llm_client):
        """If response starts with <TRANSLATION> but has no closing tag, retry with larger context."""
        call_count = 0
        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return LLMResponse(
                    content="<TRANSLATION>\nPartial text without closing tag",
                    prompt_tokens=10,
                    completion_tokens=5,
                    context_used=15,
                    context_limit=2048,
                    was_truncated=False,
                )
            return LLMResponse(
                content="<TRANSLATION>Completed</TRANSLATION>",
                prompt_tokens=10,
                completion_tokens=5,
                context_used=15,
                context_limit=4096,
                was_truncated=False,
            )

        mock_llm_client.generate = side_effect
        # Second call succeeds
        mock_llm_client.extract_translation = Mock(side_effect=lambda text: None if "Partial" in text else "Completed")

        context_manager = Mock()
        context_manager.should_retry_with_larger_context = Mock(return_value=True)
        context_manager.increase_context = Mock()
        context_manager.get_context_size = Mock(return_value=4096)

        translated, _, response = await _make_llm_request_with_adaptive_context(
            main_content="Hello",
            context_before="",
            context_after="",
            previous_translation_context="",
            source_language="English",
            target_language="French",
            model="test-model",
            llm_client=mock_llm_client,
            log_callback=None,
            has_placeholders=False,
            context_manager=context_manager,
        )

        assert call_count == 2
        assert translated == "Completed"
        assert context_manager.increase_context.called

    @pytest.mark.asyncio
    async def test_content_filter_retries_as_smaller_semantic_units(self):
        source = (
            "The traveler crossed the silent city before dawn and carefully recorded every "
            "street, conversation, landmark, and event that occurred during the difficult "
            "journey. Nothing was omitted from his account, because each detail would matter "
            "to the people waiting for news at home.\n\n"
            "Later that morning, the traveler reached the harbor and described the ships, "
            "the workers, the weather, and the long line of families waiting beside the sea. "
            "He finished the report in the same order in which the events had happened and "
            "sent the complete account to his companions."
        )
        first_translation = (
            "El viajero cruzó la ciudad silenciosa antes del amanecer y registró con cuidado "
            "cada calle, conversación, punto de referencia y suceso ocurrido durante el difícil "
            "trayecto. No omitió nada de su relato, porque cada detalle sería importante para "
            "quienes esperaban noticias en casa."
        )
        second_translation = (
            "Más tarde esa mañana, el viajero llegó al puerto y describió los barcos, a los "
            "trabajadores, el clima y la larga fila de familias que esperaban junto al mar. "
            "Terminó el informe en el mismo orden en que habían ocurrido los hechos y envió el "
            "relato completo a sus compañeros."
        )
        client = Mock()
        client.generate = AsyncMock(side_effect=[
            ContentRiskError("Content Exists Risk", provider="deepseek"),
            LLMResponse(content=f"<TRANSLATION>{first_translation}</TRANSLATION>"),
            LLMResponse(content=f"<TRANSLATION>{second_translation}</TRANSLATION>"),
        ])
        client.extract_translation = Mock(
            side_effect=lambda text: text.split("<TRANSLATION>", 1)[1].split("</TRANSLATION>", 1)[0]
        )
        events = []

        translated, actual_content, _response = await _make_llm_request_with_adaptive_context(
            main_content=source,
            context_before="",
            context_after="",
            previous_translation_context="",
            source_language="English",
            target_language="Spanish",
            model="deepseek-v4-pro",
            llm_client=client,
            log_callback=lambda event, message, **_kwargs: events.append((event, message)),
            has_placeholders=False,
            prompt_options={},
        )

        assert actual_content == source
        assert translated == f"{first_translation}\n{second_translation}"
        assert client.generate.await_count == 3
        assert any(event == "content_risk_split_retry" for event, _message in events)

    @pytest.mark.asyncio
    async def test_content_filter_microfragments_are_reassembled_in_target_language(self):
        source = (
            "The traveler crossed the silent city before dawn and carefully recorded every "
            "street, conversation, landmark, and event that occurred during the difficult "
            "journey. Nothing was omitted from his account, because each detail would matter "
            "to the people waiting for news at home.\n\n"
            "Later that morning, the traveler reached the harbor and described the ships, "
            "the workers, the weather, and the long line of families waiting beside the sea. "
            "He finished the report in the same order in which the events had happened and "
            "sent the complete account to his companions."
        )
        micro_translation = (
            "El viajero cruzó la ciudad antes del amanecer y registró cuidadosamente cada "
            "calle y cada conversación."
        )
        remaining_translation = (
            "No omitió ningún detalle de su relato, porque todo sería importante para quienes "
            "esperaban noticias en casa. Más tarde llegó al puerto y describió los barcos, a "
            "los trabajadores, el clima y a las familias junto al mar. Terminó el informe en "
            "el mismo orden de los hechos y envió el relato completo a sus compañeros."
        )
        stitched = f"{micro_translation} {remaining_translation}"
        client = Mock()
        client.generate = AsyncMock(side_effect=[
            ContentRiskError("Content Exists Risk", provider="deepseek"),
            ContentRiskError("Content Exists Risk", provider="deepseek"),
            LLMResponse(content=f"<TRANSLATION>{micro_translation}</TRANSLATION>"),
            LLMResponse(content=f"<TRANSLATION>{remaining_translation}</TRANSLATION>"),
            LLMResponse(content=f"<TRANSLATION>{stitched}</TRANSLATION>"),
        ])
        client.extract_translation = Mock(
            side_effect=lambda text: text.split("<TRANSLATION>", 1)[1].split("</TRANSLATION>", 1)[0]
        )
        events = []

        translated, _actual_content, _response = await _make_llm_request_with_adaptive_context(
            main_content=source,
            context_before="",
            context_after="",
            previous_translation_context="",
            source_language="English",
            target_language="Spanish",
            model="deepseek-v4-pro",
            llm_client=client,
            log_callback=lambda event, message, **_kwargs: events.append((event, message)),
            has_placeholders=False,
            prompt_options={},
        )

        assert translated == stitched
        assert client.generate.await_count == 5
        assert any(event == "content_risk_stitch_accepted" for event, _message in events)
