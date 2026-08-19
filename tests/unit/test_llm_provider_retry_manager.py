"""
Every LLMProvider subclass gets its own RetryManager (circuit breaker +
backoff), scoped to that provider instance's lifetime -- typically one
whole translation job. This must be a fresh instance per provider object,
not a class-level/module-level singleton, otherwise one job's provider
outage would trip the circuit for unrelated concurrent jobs using the
same provider class.
"""
from src.core.adapters.retry_manager import RetryManager
from src.core.llm.providers.deepseek import DeepSeekProvider


def test_provider_gets_its_own_retry_manager_instance():
    provider = DeepSeekProvider(api_key="test-key", model="deepseek-v4-pro")
    assert isinstance(provider._retry_manager, RetryManager)


def test_two_provider_instances_do_not_share_circuit_breaker_state():
    provider_a = DeepSeekProvider(api_key="key-a", model="deepseek-v4-pro")
    provider_b = DeepSeekProvider(api_key="key-b", model="deepseek-v4-pro")

    provider_a._retry_manager._circuit_breaker.failure_threshold = 2
    provider_a._retry_manager.record_attempt_result(False)
    provider_a._retry_manager.record_attempt_result(False)

    assert provider_a._retry_manager.get_circuit_state() == "open"
    assert provider_b._retry_manager.get_circuit_state() == "closed"
    assert provider_b._retry_manager.circuit_allows_attempt() is True
