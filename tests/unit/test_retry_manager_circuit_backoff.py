"""
Tests for the small public API added to RetryManager so LLM providers can
use it as a shared circuit-breaker + backoff-delay utility, without going
through execute_with_retry() (which wraps every exception in
RetryExhaustedError -- incompatible with the RateLimitError/ContentRiskError/
ContextOverflowError/InsufficientCreditsError propagation contract that
translator.py and handlers.py already rely on).
"""
import time

from src.core.adapters.retry_manager import RetryConfig, RetryManager, RetryStrategy


def _manager(**circuit_kwargs):
    return RetryManager(
        default_config=RetryConfig(
            initial_delay=2.0,
            max_delay=20.0,
            backoff_factor=2.0,
            jitter=0.0,
            strategy=RetryStrategy.EXPONENTIAL,
        ),
        enable_circuit_breaker=True,
        **circuit_kwargs,
    )


def test_circuit_allows_attempt_when_closed():
    manager = _manager()
    assert manager.circuit_allows_attempt() is True


def test_circuit_opens_after_failure_threshold_and_blocks_attempts():
    manager = _manager()
    manager._circuit_breaker.failure_threshold = 3
    for _ in range(3):
        manager.record_attempt_result(False)

    assert manager.get_circuit_state() == "open"
    assert manager.circuit_allows_attempt() is False


def test_circuit_half_opens_after_timeout_and_closes_on_success():
    manager = _manager()
    manager._circuit_breaker.failure_threshold = 2
    manager._circuit_breaker.timeout = 0.05
    manager.record_attempt_result(False)
    manager.record_attempt_result(False)
    assert manager.get_circuit_state() == "open"
    assert manager.circuit_allows_attempt() is False

    time.sleep(0.08)
    # First check after the timeout flips it to half-open and allows exactly
    # one probe attempt.
    assert manager.circuit_allows_attempt() is True
    assert manager.get_circuit_state() == "half_open"

    manager.record_attempt_result(True)
    manager.record_attempt_result(True)
    assert manager.get_circuit_state() == "closed"
    assert manager.circuit_allows_attempt() is True


def test_isolated_failures_interspersed_with_successes_never_trip_circuit():
    """A handful of unrelated blips, each followed by a success, must not
    open the circuit -- only a sustained run of failures should."""
    manager = _manager()
    for _ in range(20):
        manager.record_attempt_result(False)
        manager.record_attempt_result(True)

    assert manager.get_circuit_state() == "closed"
    assert manager.circuit_allows_attempt() is True


def test_delay_for_attempt_matches_manual_calculate_delay():
    manager = _manager()
    for attempt in (1, 2, 3, 4):
        expected = manager._calculate_delay(attempt, manager.default_config)
        assert manager.delay_for_attempt(attempt) == expected


def test_delay_for_attempt_grows_and_is_capped():
    manager = _manager()
    assert manager.delay_for_attempt(1) == 2.0
    assert manager.delay_for_attempt(2) == 4.0
    assert manager.delay_for_attempt(3) == 8.0
    assert manager.delay_for_attempt(10) == 20.0  # capped at max_delay


def test_manager_without_circuit_breaker_always_allows_attempts():
    manager = RetryManager(enable_circuit_breaker=False)
    for _ in range(50):
        manager.record_attempt_result(False)
    assert manager.circuit_allows_attempt() is True
    assert manager.get_circuit_state() is None
