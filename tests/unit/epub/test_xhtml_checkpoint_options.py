from __future__ import annotations

from src.core.epub.xhtml_translator import _checkpoint_prompt_options


def test_checkpoint_prompt_options_drops_runtime_candidate_results():
    options = {
        "target_locale": "es-MX",
        "_fidelity_report": object(),
        "_editorial_quality_report": object(),
        "_candidate_results": [{"phase": "translation", "text_snippet": "runtime only"}],
    }

    safe = _checkpoint_prompt_options(options)

    assert safe["target_locale"] == "es-MX"
    assert "_fidelity_report" not in safe
    assert "_editorial_quality_report" not in safe
    assert "_candidate_results" not in safe
    assert "_candidate_results" in options
