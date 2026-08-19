from src.api.blueprints.translation_routes import _client_safe_config, _client_safe_logs


def test_client_safe_config_redacts_provider_keys_and_internal_runtime_objects():
    config = {
        "model": "deepseek-v4-pro",
        "deepseek_api_key": "sk-secret",
        "prompt_options": {
            "translation_memory_enabled": True,
            "_fidelity_report": object(),
            "_source_guard_refs_loaded": True,
            "_candidate_results": [{"text_snippet": "runtime only"}],
            "nested_token": "token-secret",
        },
    }

    safe = _client_safe_config(config)

    assert safe["model"] == "deepseek-v4-pro"
    assert safe["deepseek_api_key"] == "[redacted]"
    assert "_fidelity_report" not in safe["prompt_options"]
    assert "_source_guard_refs_loaded" not in safe["prompt_options"]
    assert "_candidate_results" not in safe["prompt_options"]
    assert safe["prompt_options"]["translation_memory_enabled"] is True
    assert safe["prompt_options"]["nested_token"] == "[redacted]"


def test_client_safe_logs_omit_verbose_prompt_and_response_payloads():
    logs = [
        {
            "message": "LLM call complete",
            "data": {
                "prompt": "x" * 1200,
                "response": "y" * 900,
                "total_tokens": 1234,
            },
        }
    ]

    safe = _client_safe_logs(logs)

    assert safe[0]["message"] == "LLM call complete"
    assert safe[0]["data"]["prompt"] == "[omitted 1200 chars]"
    assert safe[0]["data"]["response"] == "[omitted 900 chars]"
    assert safe[0]["data"]["total_tokens"] == 1234
