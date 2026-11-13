"""Real provider/tracking boundary, synthetic HTTP and isolated experiment ledger."""

import json

import httpx
import pytest

from src.core.benchmark_judge import evaluate_candidate
from src.core.pipeline_config import pipeline_preset
from src.core.pipeline_runtime import activate_pipeline, close_pipeline
from src.core.usage.budget import BudgetLimitError, SpendBudget
from src.core.usage.context import get_usage_context
from src.core.usage.store import TokenUsageStore
from tests.unit.test_editorial_benchmark import clean_assessment, corpus


@pytest.fixture
def setup(tmp_path, monkeypatch):
    store = TokenUsageStore(tmp_path / "usage.db")
    SpendBudget(store).configure("experiment:pilot", limit_usd="5", max_calls=1)
    calls, saved = [], []
    state = {"answer": json.dumps(clean_assessment()), "finish": "stop"}

    class Client:
        async def post(self, url, **kwargs):
            assert SpendBudget(store).snapshot("experiment:pilot")["reserved_usd"] > 0
            calls.append(kwargs)
            return httpx.Response(
                200,
                request=httpx.Request("POST", url),
                json={
                    "choices": [
                        {
                            "finish_reason": state["finish"],
                            "message": {"content": state["answer"]},
                        }
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 30},
                },
            )

    async def get_client(_):
        return Client()

    monkeypatch.setattr(
        "src.core.llm.providers.openai.OpenAICompatibleProvider._get_client", get_client
    )
    return store, calls, saved, state


def policy():
    config = pipeline_preset(
        "openai_equivalent", {"llm_provider": "deepseek", "model": "deepseek-flash"}
    )
    config["roles"]["audit"].update(model="gpt-6-luna")
    return config


def generated():
    return {
        "text": "A complete candidate.",
        "generators": [{"provider": "deepseek", "model": "deepseek-v4-pro"}],
    }


@pytest.mark.asyncio
async def test_evaluator_uses_shared_attempt_budget_and_keeps_raw_response(setup):
    store, calls, saved, _ = setup
    value = corpus()
    token = activate_pipeline(
        policy(),
        job_id="judge",
        credentials={"openai": "YOUR_API_KEY_HERE"},
        store=store,
        experiment_scope="experiment:pilot",
        roles=("audit",),
    )
    try:
        result = await evaluate_candidate(
            value,
            value["units"][0]["unit_id"],
            generated(),
            experiment_scope="experiment:pilot",
            response_sink=saved.append,
        )
        assert result["accepted"]
        assert len(saved) == len(calls) == 1
        assert saved[0]["response"]["raw_usage"]
        with pytest.raises(BudgetLimitError):
            await evaluate_candidate(
                value,
                value["units"][0]["unit_id"],
                generated(),
                experiment_scope="experiment:pilot",
                response_sink=saved.append,
            )
    finally:
        await close_pipeline(token)
    assert len(calls) == 1
    event = store.events()[0]
    assert event["process_type"] == "editorial_benchmark"
    assert event["process_id"] == "experiment:pilot"
    assert event["estimated_tokens"] == 0
    assert get_usage_context() == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault",
    ["self_judge", "missing_provenance", "budget", "sink", "malformed", "truncated"],
)
async def test_invalid_evaluation_never_approves_and_does_not_retry(setup, fault):
    store, calls, saved, state = setup
    value, candidate = corpus(), generated()
    scope, sink = "experiment:pilot", saved.append
    if fault == "self_judge":
        candidate["generators"].append({"provider": "openai", "model": "gpt-6-luna"})
    elif fault == "missing_provenance":
        candidate["generators"] = []
    elif fault == "budget":
        scope = "experiment:other"
    elif fault == "sink":
        sink = None
    elif fault == "malformed":
        state["answer"] = '{"verdict":"pass"}'
    else:
        state["finish"] = "length"
    token = activate_pipeline(
        policy(),
        job_id="judge",
        credentials={"openai": "YOUR_API_KEY_HERE"},
        store=store,
        experiment_scope="experiment:pilot",
        roles=("audit",),
    )
    try:
        with pytest.raises(ValueError):
            await evaluate_candidate(
                value,
                value["units"][0]["unit_id"],
                candidate,
                experiment_scope=scope,
                response_sink=sink,
            )
    finally:
        await close_pipeline(token)
    expected = 1 if fault in {"malformed", "truncated"} else 0
    assert len(calls) == len(saved) == expected


def test_deepseek_alias_is_not_an_independent_judge():
    from src.core.benchmark_judge import _model_identity

    assert _model_identity("deepseek", "deepseek-flash") == _model_identity(
        "deepseek", "deepseek-v4-flash"
    )


@pytest.mark.asyncio
async def test_missing_historical_profile_blocks_evaluation_before_http(setup):
    from src.core.editorial_benchmark import corpus_digest

    store, calls, saved, _ = setup
    value = corpus()
    value["book"]["profile"] = {"status": "unresolved"}
    value["content_sha256"] = corpus_digest(value)
    token = activate_pipeline(
        policy(),
        job_id="judge",
        credentials={"openai": "YOUR_API_KEY_HERE"},
        store=store,
        experiment_scope="experiment:pilot",
        roles=("audit",),
    )
    try:
        with pytest.raises(ValueError, match="historical profile"):
            await evaluate_candidate(
                value,
                value["units"][0]["unit_id"],
                generated(),
                experiment_scope="experiment:pilot",
                response_sink=saved.append,
            )
    finally:
        await close_pipeline(token)
    assert calls == saved == []
