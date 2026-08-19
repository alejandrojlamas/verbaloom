import pytest

from src.core.llm.base import LLMResponse
from src.core.post_processor import clean_translated_text
from src.core.translator import generate_translation_request
from src.utils.text_encoding import clean_text_artifacts, repair_mojibake


class FakeLLMClient:
    def __init__(self, content: str):
        self.content = content

    async def generate(self, prompt: str, system_prompt: str = None):
        return LLMResponse(content=self.content)

    def extract_translation(self, response: str):
        start = response.find("<TRANSLATION>")
        end = response.find("</TRANSLATION>")
        if start == -1 or end == -1:
            return None
        return response[start + len("<TRANSLATION>"):end]


def test_repairs_common_spanish_utf8_mojibake():
    broken = (
        "modelos dominantes de transducciÃ³n de secuencias se basan en redes "
        "neuronales. Los modelos con mejor rendimiento tambiÃ©n conectan el "
        "codificador y el decodificador a travÃ©s de un mecanismo de atenciÃ³n."
    )

    repaired = repair_mojibake(broken)

    assert "transducción" in repaired
    assert "también" in repaired
    assert "través" in repaired
    assert "atención" in repaired
    assert "Ã" not in repaired


def test_repairs_windows_1252_punctuation_mojibake():
    broken = "El modelo â€œTransformerâ€\u009d logrÃ³ mejores resultados â€” rÃ¡pido."

    assert repair_mojibake(broken) == "El modelo “Transformer” logró mejores resultados — rápido."


def test_repairs_double_encoded_mojibake():
    assert repair_mojibake("traducciÃƒÂ³n automÃƒÂ¡tica") == "traducción automática"


def test_preserves_already_correct_spanish_text():
    correct = "El niño pidió traducción automática en español."

    assert repair_mojibake(correct) == correct


def test_post_processor_repairs_mojibake_before_saving():
    assert clean_translated_text("traducciÃ³n &amp; atenciÃ³n") == "traducción & atención"


def test_post_processor_removes_replacement_square_artifacts():
    broken = (
        "aprenden■■■■■■■■■■■■■■■■■■\n"
        "por diferentes tareas. β1 = 0,9, β2 = 0,98 y ε = 10■■."
    )

    cleaned = clean_translated_text(broken)

    assert "■" not in cleaned
    assert "aprenden\npor diferentes tareas" in cleaned
    assert "β1 = 0,9" in cleaned
    assert "β2 = 0,98" in cleaned


def test_clean_text_artifacts_strips_legacy_width_zero_marks():
    text = "Texto limpio\u200d\u200c\u200b\u2060 con marcas invisibles."

    assert clean_text_artifacts(text) == "Texto limpio con marcas invisibles."


@pytest.mark.asyncio
async def test_first_pass_translation_returns_repaired_text():
    client = FakeLLMClient("<TRANSLATION>transducciÃ³n y atenciÃ³n ■■■</TRANSLATION>")

    translated = await generate_translation_request(
        main_content="sequence transduction and attention",
        context_before="",
        context_after="",
        previous_translation_context="",
        source_language="English",
        target_language="Spanish",
        model="fake-model",
        llm_client=client,
    )

    assert translated == "transducción y atención"
