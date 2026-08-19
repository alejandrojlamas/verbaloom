import json
from pathlib import Path

from src.core.final_source_sample_audit import audit_final_output_against_source_samples
from src.core.llm_output_guard import guard_llm_output


CORPUS_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "llm_regression_corpus.json"


def _load_cases():
    return json.loads(CORPUS_PATH.read_text(encoding="utf-8"))


def test_llm_regression_corpus_is_valid():
    cases = _load_cases()

    assert len(cases) >= 5
    assert len({case["id"] for case in cases}) == len(cases)
    assert {case["kind"] for case in cases} >= {
        "output_guard",
        "idempotence",
        "style_drift",
        "source_sample",
    }


def test_llm_output_guard_regression_cases():
    for case in _load_cases():
        if case["kind"] != "output_guard":
            continue
        result = guard_llm_output(case["input"], phase=case["id"])

        assert result.text == case["expected_text"], case["id"]
        assert case["expected_issue"] in {issue.code for issue in result.issues}


def test_llm_output_cleanup_regression_cases_are_idempotent():
    for case in _load_cases():
        if case["kind"] not in {"output_guard", "idempotence"}:
            continue
        first = guard_llm_output(case["input"], phase=case["id"])
        second = guard_llm_output(first.text, phase=case["id"])

        assert second.text == first.text, case["id"]
        if "expected_text" in case:
            assert first.text == case["expected_text"], case["id"]


def test_style_drift_regression_cases_warn_without_rewriting():
    for case in _load_cases():
        if case["kind"] != "style_drift":
            continue
        reference = (case["reference"] + " ") * 10
        candidate = (case["candidate"] + " ") * 10
        result = guard_llm_output(candidate, phase=case["id"], style_reference=reference)

        assert result.text == candidate.strip(), case["id"]
        assert case["expected_issue"] in {issue.code for issue in result.issues}


def test_final_source_sample_regression_cases(tmp_path):
    for case in _load_cases():
        if case["kind"] != "source_sample":
            continue
        source = tmp_path / f"{case['id']}_source.txt"
        output = tmp_path / f"{case['id']}_output.txt"
        source.write_text((case["source"] + " ") * 12, encoding="utf-8")
        output.write_text((case["output"] + " ") * 12, encoding="utf-8")

        report = audit_final_output_against_source_samples(
            source,
            output,
            source_language="English",
            target_language="Spanish",
            write_report=False,
        )

        assert case["expected_issue"] in {issue.code for issue in report.issues}, case["id"]
