import pytest

from src.core.translation_memory import (
    TranslationMemoryCache,
    TranslationMemoryRequest,
    prompt_fingerprint,
)
from src.core.adapters.format_adapter import FormatAdapter
from src.core.adapters.generic_translator import GenericTranslator
from src.core.adapters.translation_unit import TranslationUnit
from src.persistence.checkpoint_manager import CheckpointManager


class _MemoryTestAdapter(FormatAdapter):
    def __init__(self, input_file_path, output_file_path, config):
        super().__init__(input_file_path, output_file_path, config)
        self.saved = {}

    async def prepare_for_translation(self) -> bool:
        return True

    def get_translation_units(self):
        return [
            TranslationUnit("chunk_0", "First source unit.", metadata={"chunk_index": 0}),
            TranslationUnit("chunk_1", "Second source unit.", metadata={"chunk_index": 1}),
        ]

    async def save_unit_translation(self, unit_id: str, translated_content: str) -> bool:
        self.saved[unit_id] = translated_content
        return True

    async def reconstruct_output(self, bilingual: bool = False) -> bytes:
        return "\n".join(self.saved[key] for key in sorted(self.saved)).encode("utf-8")

    async def resume_from_checkpoint(self, checkpoint_data):
        return 0

    async def cleanup(self):
        return None

    @property
    def format_name(self) -> str:
        return "txt"


def _request(**overrides):
    data = {
        "source_text": "The quick brown fox jumps over the lazy dog.",
        "source_language": "English",
        "target_language": "Spanish",
        "provider": "deepseek",
        "model": "deepseek-v4-pro",
        "prompt_options": {
            "target_locale": "es-MX",
            "profile_id": "book_a",
            "glossary_terms": {"fox": "zorro"},
        },
        "format_name": "txt",
        "unit_id": "chunk_0",
        "context_before": "Before.",
        "context_after": "After.",
        "previous_translation_context": "Previo.",
    }
    data.update(overrides)
    return TranslationMemoryRequest(**data)


def test_translation_memory_reuses_exact_matching_request(tmp_path):
    cache = TranslationMemoryCache(tmp_path / "memory.db")
    request = _request()

    key = cache.put(request, "El veloz zorro cafe salta sobre el perro perezoso.")

    entry = cache.get(request)
    assert entry is not None
    assert entry.key == key
    assert entry.translated_text.startswith("El veloz zorro")
    assert entry.hits == 1


def test_translation_memory_key_changes_with_profile_or_context(tmp_path):
    cache = TranslationMemoryCache(tmp_path / "memory.db")
    request = _request()
    cache.put(request, "Traduccion A")

    different_profile = _request(prompt_options={
        "target_locale": "es-MX",
        "profile_id": "book_b",
        "glossary_terms": {"fox": "zorro"},
    })
    different_context = _request(previous_translation_context="Otro contexto.")

    assert cache.get(different_profile) is None
    assert cache.get(different_context) is None


def test_translation_memory_delete_removes_exact_entry(tmp_path):
    cache = TranslationMemoryCache(tmp_path / "memory.db")
    request = _request()
    cache.put(request, "Traduccion temporal")

    assert cache.get(request) is not None
    assert cache.delete(request) is True
    assert cache.get(request) is None
    assert cache.delete(request) is False


def test_prompt_fingerprint_redacts_secrets_and_hashes_large_payloads():
    fp_a = prompt_fingerprint({
        "target_locale": "es-MX",
        "deepseek_api_key": "secret-a",
        "glossary_terms": {"alpha": "alfa"},
    })
    fp_b = prompt_fingerprint({
        "target_locale": "es-MX",
        "deepseek_api_key": "secret-b",
        "glossary_terms": {"alpha": "alfa"},
    })
    fp_c = prompt_fingerprint({
        "target_locale": "es-MX",
        "deepseek_api_key": "secret-b",
        "glossary_terms": {"alpha": "alfa cambiada"},
    })

    assert fp_a == fp_b
    assert fp_b != fp_c


async def _run_memory_integration(tmp_path, monkeypatch, translation_id: str, calls: list[str]):
    input_path = tmp_path / f"{translation_id}.txt"
    output_path = tmp_path / f"{translation_id}_out.txt"
    input_path.write_text("First source unit.\nSecond source unit.", encoding="utf-8")

    async def fake_generate_translation_request(**kwargs):
        calls.append(kwargs["main_content"])
        return f"ES:{kwargs['main_content']}"

    monkeypatch.setattr(
        "src.core.translator.generate_translation_request",
        fake_generate_translation_request,
    )
    adapter = _MemoryTestAdapter(
        str(input_path),
        str(output_path),
        {"prompt_options": {"translation_memory_enabled": True}},
    )
    checkpoint_manager = CheckpointManager(db_path=str(tmp_path / f"{translation_id}.db"))
    translator = GenericTranslator(adapter, checkpoint_manager, translation_id)
    logs = []
    success = await translator.translate(
        source_language="English",
        target_language="Spanish",
        model_name="fake-model",
        llm_provider="fake-provider",
        log_callback=lambda event, message: logs.append(event),
    )
    return success, output_path.read_text(encoding="utf-8"), logs


@pytest.mark.asyncio
async def test_generic_translator_reuses_translation_memory_between_jobs(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSLATION_MEMORY_DB", str(tmp_path / "memory.db"))
    calls: list[str] = []

    success, output, _logs = await _run_memory_integration(
        tmp_path,
        monkeypatch,
        "first_job",
        calls,
    )
    assert success is True
    assert output == "ES:First source unit.\nES:Second source unit."
    assert calls == ["First source unit.", "Second source unit."]

    calls.clear()
    success, output, logs = await _run_memory_integration(
        tmp_path,
        monkeypatch,
        "second_job",
        calls,
    )
    assert success is True
    assert output == "ES:First source unit.\nES:Second source unit."
    assert calls == []
    assert logs.count("translation_memory_hit") == 2


@pytest.mark.asyncio
async def test_generic_translator_ignores_wrong_language_translation_memory(tmp_path, monkeypatch):
    memory_path = tmp_path / "memory.db"
    monkeypatch.setenv("TRANSLATION_MEMORY_DB", str(memory_path))
    cache = TranslationMemoryCache(memory_path)
    cache.put(
        TranslationMemoryRequest(
            source_text="First source unit.",
            context_before="",
            context_after="",
            previous_translation_context="",
            source_language="English",
            target_language="Spanish",
            provider="fake-provider",
            model="fake-model",
            prompt_options={},
            format_name="txt",
            unit_id="chunk_0",
            phase="translation",
        ),
        (
            "Ἄνδρα μοι ἔννεπε Μοῦσα πολύτροπον ὃς μάλα πολλὰ "
            "πλάγχθη ἐπεὶ Τροίης ἱερὸν πτολίεθρον ἔπερσεν"
        ),
    )
    calls: list[str] = []

    success, output, logs = await _run_memory_integration(
        tmp_path,
        monkeypatch,
        "bad_cache_job",
        calls,
    )

    assert success is True
    assert output == "ES:First source unit.\nES:Second source unit."
    assert calls == ["First source unit.", "Second source unit."]
    assert "translation_memory_target_language_rejected" in logs


@pytest.mark.asyncio
async def test_generic_translator_cleans_protocol_leaks_before_save_and_cache(tmp_path, monkeypatch):
    memory_path = tmp_path / "memory.db"
    monkeypatch.setenv("TRANSLATION_MEMORY_DB", str(memory_path))
    calls: list[str] = []

    async def fake_generate_translation_request(**kwargs):
        calls.append(kwargs["main_content"])
        return f"ES:{kwargs['main_content']} </TRANSLATIONATION>"

    monkeypatch.setattr(
        "src.core.translator.generate_translation_request",
        fake_generate_translation_request,
    )
    input_path = tmp_path / "protocol_leak.txt"
    output_path = tmp_path / "protocol_leak_out.txt"
    input_path.write_text("First source unit.\nSecond source unit.", encoding="utf-8")
    adapter = _MemoryTestAdapter(
        str(input_path),
        str(output_path),
        {"prompt_options": {"translation_memory_enabled": True}},
    )
    checkpoint_manager = CheckpointManager(db_path=str(tmp_path / "protocol_leak.db"))
    translator = GenericTranslator(adapter, checkpoint_manager, "protocol-leak-job")
    logs = []

    success = await translator.translate(
        source_language="English",
        target_language="Spanish",
        model_name="fake-model",
        llm_provider="fake-provider",
        log_callback=lambda event, message, **kwargs: logs.append(event),
    )

    assert success is True
    output = output_path.read_text(encoding="utf-8")
    assert "TRANSLATIONATION" not in output
    assert "llm_output_guard" in logs

    calls.clear()
    second_output_path = tmp_path / "protocol_leak_second_out.txt"
    second_adapter = _MemoryTestAdapter(
        str(input_path),
        str(second_output_path),
        {"prompt_options": {"translation_memory_enabled": True}},
    )
    second_translator = GenericTranslator(
        second_adapter,
        CheckpointManager(db_path=str(tmp_path / "protocol_leak_second.db")),
        "protocol-leak-second-job",
    )
    success = await second_translator.translate(
        source_language="English",
        target_language="Spanish",
        model_name="fake-model",
        llm_provider="fake-provider",
        log_callback=lambda event, message, **kwargs: logs.append(event),
    )

    assert success is True
    assert calls == []
    assert "TRANSLATIONATION" not in second_output_path.read_text(encoding="utf-8")
