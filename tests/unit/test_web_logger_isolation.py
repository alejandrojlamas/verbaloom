from src.utils.unified_logger import LogLevel, LogType, UnifiedLogger, setup_web_logger


def test_setup_web_logger_does_not_share_callbacks_between_jobs():
    calls_a = []
    calls_b = []

    logger_a = setup_web_logger(calls_a.append, calls_a.append)
    logger_b = setup_web_logger(calls_b.append, calls_b.append)

    assert logger_a is not logger_b

    logger_a.info("first job", LogType.GENERAL, {"job": "a"})
    logger_b.info("second job", LogType.GENERAL, {"job": "b"})

    assert [entry["data"]["job"] for entry in calls_a] == ["a", "a"]
    assert [entry["data"]["job"] for entry in calls_b] == ["b", "b"]


def test_debug_console_llm_logs_omit_raw_prompt_and_response_by_default(monkeypatch):
    monkeypatch.delenv("TBL_VERBOSE_LLM_LOGS", raising=False)
    logger = UnifiedLogger(console_output=False, min_level=LogLevel.DEBUG)

    request_message = logger._format_console_message(
        LogLevel.DEBUG,
        "LLM Request",
        LogType.LLM_REQUEST,
        {
            "system_prompt": "SECRET SYSTEM PROMPT",
            "user_prompt": "FULL BOOK CHUNK",
        },
    )
    response_message = logger._format_console_message(
        LogLevel.DEBUG,
        "LLM Response",
        LogType.LLM_RESPONSE,
        {"response": "FULL RAW MODEL RESPONSE"},
    )

    combined = f"{request_message}\n{response_message}"
    assert "SECRET SYSTEM PROMPT" not in combined
    assert "FULL BOOK CHUNK" not in combined
    assert "FULL RAW MODEL RESPONSE" not in combined
    assert "Prompt body omitted" in combined
    assert "Response body omitted" in combined
