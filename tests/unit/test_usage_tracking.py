import pytest
import sqlite3
from datetime import datetime, timezone

from src.api.blueprints.usage_routes import _build_live_jobs
from src.core.llm.base import LLMResponse
from src.core.usage.store import TokenUsageStore
from src.core.usage.tracking_provider import UsageTrackingProvider


OFF_PEAK_TIMESTAMP = datetime(2026, 9, 12, 12, tzinfo=timezone.utc).timestamp()
PEAK_TIMESTAMP = datetime(2026, 9, 14, 2, tzinfo=timezone.utc).timestamp()


def test_usage_store_records_cost_and_summary(tmp_path, monkeypatch):
    monkeypatch.setattr("src.core.usage.store.time.time", lambda: OFF_PEAK_TIMESTAMP)
    store = TokenUsageStore(tmp_path / "usage.db")

    store.record_call(
        provider="deepseek",
        model="deepseek-v4-pro",
        prompt="hola",
        response_content="mundo",
        prompt_tokens=1_000_000,
        completion_tokens=500_000,
        status="ok",
        context={
            "translation_id": "job-1",
            "process_type": "translation",
            "book_name": "Libro de prueba",
        },
    )

    summary = store.summary()
    assert summary["totals"]["calls"] == 1
    assert summary["totals"]["total_tokens"] == 1_500_000
    assert round(summary["totals"]["total_cost_usd"], 4) == 1.65
    assert summary["by_book"][0]["book_name"] == "Libro de prueba"
    assert summary["by_translation"][0]["translation_id"] == "job-1"
    assert summary["by_phase"][0]["phase"] == "llm_call"


def test_usage_store_records_deepseek_cache_aware_cost(tmp_path, monkeypatch):
    monkeypatch.setattr("src.core.usage.store.time.time", lambda: OFF_PEAK_TIMESTAMP)
    store = TokenUsageStore(tmp_path / "usage.db")

    store.record_call(
        provider="deepseek",
        model="deepseek-v4-pro",
        prompt="hola",
        response_content="",
        prompt_tokens=1_000_000,
        completion_tokens=0,
        prompt_cache_hit_tokens=900_000,
        prompt_cache_miss_tokens=100_000,
        status="ok",
        context={"translation_id": "job-cache", "book_name": "Libro cache"},
    )

    summary = store.summary()
    totals = summary["totals"]
    assert totals["prompt_cache_hit_tokens"] == 900_000
    assert totals["prompt_cache_miss_tokens"] == 100_000
    assert summary["by_book"][0]["prompt_cache_hit_tokens"] == 900_000
    assert summary["by_translation"][0]["prompt_cache_miss_tokens"] == 100_000
    assert 0 < totals["total_cost_usd"] < 0.66


def test_usage_store_records_actual_peak_tier_for_completed_calls(tmp_path, monkeypatch):
    monkeypatch.setattr("src.core.usage.store.time.time", lambda: PEAK_TIMESTAMP)
    store = TokenUsageStore(tmp_path / "usage.db")

    store.record_call(
        provider="deepseek",
        model="deepseek-flash",
        prompt="hola",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        status="ok",
    )

    event = store.events()[0]
    assert event["pricing_source"] == "deepseek_official_2026-09-10_peak"
    assert event["total_cost_usd"] == pytest.approx(1.5)


def test_usage_store_does_not_invent_billable_tokens_for_failed_calls(tmp_path):
    store = TokenUsageStore(tmp_path / "usage.db")

    store.record_call(
        provider="deepseek",
        model="deepseek-flash",
        prompt="This prompt was blocked before a provider response.",
        status="error",
        metadata={"error": "DeepSeekPeakPricingError"},
    )

    event = store.events()[0]
    assert event["prompt_tokens"] == 0
    assert event["completion_tokens"] == 0
    assert event["total_tokens"] == 0
    assert event["total_cost_usd"] == 0
    assert event["estimated_tokens"] == 0


def test_usage_store_prefers_provider_reported_total(tmp_path):
    store = TokenUsageStore(tmp_path / "usage.db")

    store.record_call(
        provider="deepseek",
        model="deepseek-flash",
        prompt="hola",
        response_content="mundo",
        prompt_tokens=12,
        completion_tokens=7,
        total_tokens=21,
        status="ok",
    )

    assert store.events()[0]["total_tokens"] == 21


def test_usage_store_never_undercounts_an_inconsistent_provider_total(tmp_path):
    store = TokenUsageStore(tmp_path / "usage.db")

    store.record_call(
        provider="deepseek",
        model="deepseek-flash",
        prompt="hola",
        prompt_tokens=12,
        completion_tokens=7,
        total_tokens=18,
        status="ok",
    )

    assert store.events()[0]["total_tokens"] == 19


@pytest.mark.asyncio
async def test_usage_tracking_provider_records_without_changing_response(tmp_path, monkeypatch):
    store = TokenUsageStore(tmp_path / "usage.db")

    import src.core.usage.tracking_provider as tracking_module

    monkeypatch.setattr(tracking_module, "default_usage_store", lambda: store)

    class FakeProvider:
        model = "deepseek-flash"

        async def generate(self, prompt, timeout=1, system_prompt=None, temperature=None):
            self.temperature = temperature
            return LLMResponse(
                content="respuesta",
                prompt_tokens=12,
                completion_tokens=7,
                prompt_cache_hit_tokens=9,
                prompt_cache_miss_tokens=3,
            )

        def extract_translation(self, response):
            return response

        async def close(self):
            pass

    provider = UsageTrackingProvider(FakeProvider(), "deepseek")
    response = await provider.generate("prompt", system_prompt="system", temperature=0.1)

    assert response.content == "respuesta"
    events = store.events()
    assert len(events) == 1
    assert events[0]["provider"] == "deepseek"
    assert events[0]["model"] == "deepseek-flash"
    assert events[0]["total_tokens"] == 19
    assert events[0]["prompt_cache_hit_tokens"] == 9
    assert events[0]["prompt_cache_miss_tokens"] == 3
    assert provider._wrapped.temperature == 0.1


@pytest.mark.asyncio
async def test_usage_tracking_provider_infers_phase_and_propagates_model(tmp_path, monkeypatch):
    store = TokenUsageStore(tmp_path / "usage.db")

    import src.core.usage.tracking_provider as tracking_module

    monkeypatch.setattr(tracking_module, "default_usage_store", lambda: store)

    class FakeProvider:
        model = "deepseek-flash"
        context_window = 4096

        async def generate(self, prompt, timeout=1, system_prompt=None):
            return LLMResponse(
                content="respuesta",
                prompt_tokens=5,
                completion_tokens=3,
            )

        def extract_translation(self, response):
            return response

        async def close(self):
            pass

    wrapped = FakeProvider()
    provider = UsageTrackingProvider(wrapped, "deepseek")
    provider.model = "deepseek-v4-pro"
    provider.context_window = 8192

    await provider.generate(
        "Audita esta modernización intralingüística y devuelve JSON con \"overall_decision\".",
        system_prompt="Audita esta modernización.",
    )

    event = store.events()[0]
    assert wrapped.model == "deepseek-v4-pro"
    assert wrapped.context_window == 8192
    assert event["model"] == "deepseek-v4-pro"
    assert event["phase"] == "fidelity_audit"


def test_live_usage_projection_uses_jobs_and_simulated_events(tmp_path):
    store = TokenUsageStore(tmp_path / "usage.db")
    jobs_db = tmp_path / "jobs.db"
    with sqlite3.connect(jobs_db) as conn:
        conn.execute(
            """
            CREATE TABLE translation_jobs (
                translation_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                file_type TEXT NOT NULL,
                config JSON NOT NULL,
                progress JSON NOT NULL,
                translation_context JSON,
                server_session_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                paused_at TIMESTAMP,
                completed_at TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            INSERT INTO translation_jobs (
                translation_id, status, file_type, config, progress
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                "live-1",
                "running",
                "epub",
                '{"original_filename":"Libro vivo.epub","llm_provider":"deepseek","model":"deepseek-v4-pro","operation_mode":"transform","text_process":"modernize"}',
                '{"total_chunks":100,"completed_chunks":25,"failed_chunks":0,"percent":25}',
            ),
        )

    for phase, prompt_tokens, completion_tokens in [
        ("modernization", 120_000, 30_000),
        ("fidelity_audit", 20_000, 5_000),
    ]:
        store.record_call(
            provider="deepseek",
            model="deepseek-v4-pro",
            prompt="simulated prompt",
            response_content="simulated response",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            context={
                "translation_id": "live-1",
                "process_type": "transform_modernize",
                "phase": phase,
                "book_name": "Libro vivo.epub",
            },
        )

    summary = store.summary(limit=10)
    live_jobs = _build_live_jobs(jobs_db, summary, limit=10)

    assert len(live_jobs) == 1
    live = live_jobs[0]
    assert live["translation_id"] == "live-1"
    assert live["progress_percent"] == 25
    assert live["total_chunks"] == 100
    assert live["total_tokens"] == 175_000
    assert live["recent_total_tokens"] == 175_000
    assert live["projected_total_cost_usd"] > live["total_cost_usd"]
    assert live["projected_remaining_cost_usd"] > 0
    assert live["cost_per_processed_chunk_usd"] > 0
    assert {row["phase"] for row in live["phase_breakdown"]} == {"modernization", "fidelity_audit"}
    assert all("cost_share_pct" in row for row in live["phase_breakdown"])
