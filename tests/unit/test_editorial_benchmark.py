"""Private, source-aware benchmark contracts; no paid model requests."""

from copy import deepcopy

import pytest

from src.core.editorial_benchmark import (
    DIMENSIONS,
    build_judge_prompt,
    corpus_digest,
    negative_control_summary,
    pass_delta,
    prepare_corpus,
    render_blind_sample,
    validate_assessment,
    validate_corpus,
)


def source_units():
    return [
        {
            "locator": f"chapter.xhtml:p{i}",
            "text": f"Source paragraph {i}.",
            "section": "chapter.xhtml",
            "position": i,
        }
        for i in range(24)
    ]


def corpus():
    return prepare_corpus(
        source_units(),
        book_id="book-one",
        source_sha256="a" * 64,
        source_language="Czech",
        target_language="Spanish",
        target_locale="es-ES",
        profile={"policy_text": "Preserve the author's voice.", "glossary": []},
        sequences=[
            {"start": 2, "count": 10, "split": "tuning"},
            {"start": 13, "count": 10, "split": "heldout"},
        ],
    )


def clean_assessment():
    return {
        "verdict": "pass",
        "dimensions": dict.fromkeys(DIMENSIONS, "pass"),
        "findings": [],
    }


def finding():
    return {
        "dimension": "fidelity",
        "severity": "critical",
        "kind": "negation",
        "source": {"start": 0, "end": 3, "quote": "Not"},
        "candidate": {"start": 0, "end": 3, "quote": "Yes"},
        "explanation": "The polarity changed.",
        "required_correction": "Restore negation.",
    }


def rejected_assessment():
    value = clean_assessment()
    value.update(verdict="fail", findings=[finding()])
    value["dimensions"]["fidelity"] = "concern"
    return value


def test_private_corpus_keeps_locale_sequences_and_digest():
    value = corpus()
    assert len(value["units"]) == 20
    assert value["book"]["source_language"] == "Czech"
    assert value["book"]["target_locale"] == "es-ES"
    assert value["content_sha256"] == corpus_digest(value)
    assert value["units"][0]["context_before"] == ""
    assert value["units"][9]["context_after"] == ""
    assert value["units"][10]["context_before"] == ""
    assert value["units"][1]["context_before"] == value["units"][0]["source_text"]
    validate_corpus(value)


@pytest.mark.parametrize("change", ["source", "profile", "locale", "context"])
def test_corpus_tampering_invalidates_identity(change):
    value = corpus()
    if change == "source":
        value["units"][0]["source_text"] += " altered"
    elif change == "profile":
        value["book"]["profile"]["policy_text"] += " altered"
    elif change == "locale":
        value["book"]["target_locale"] = "es-MX"
    else:
        value["units"][0]["context_before"] = "held-out material"
    with pytest.raises(ValueError):
        validate_corpus(value)


@pytest.mark.parametrize("fault", ["overlap", "duplicate", "gap", "section", "secret"])
def test_corpus_rejects_leakage_and_false_continuity(fault):
    units = source_units()
    options = {
        "book_id": "book",
        "source_sha256": "a" * 64,
        "source_language": "Czech",
        "target_language": "Spanish",
        "target_locale": "es-ES",
        "profile": {},
        "sequences": [
            {"start": 0, "count": 10, "split": "tuning"},
            {"start": 12, "count": 10, "split": "heldout"},
        ],
    }
    if fault == "overlap":
        options["sequences"][1]["start"] = 5
    elif fault == "duplicate":
        units[12]["text"] = units[0]["text"]
    elif fault == "gap":
        units[2]["position"] = 900
    elif fault == "section":
        units[2]["section"] = "another-chapter.xhtml"
    else:
        options["profile"] = {"api_key": "YOUR_API_KEY_HERE"}
    with pytest.raises(ValueError):
        prepare_corpus(units, **options)


def test_assessment_requires_all_dimensions_and_anchored_evidence():
    result = validate_assessment(rejected_assessment(), "Not so.", "Yes so.")
    assert not result["accepted"]
    assert result["critical_findings"] == 1
    good = validate_assessment(clean_assessment(), "Not so.", "No es as\u00ed.")
    assert good["accepted"]


@pytest.mark.parametrize(
    "fault",
    [
        "dimension",
        "offset",
        "quote",
        "false_pass",
        "bool",
        "missing_finding",
        "extra",
        "empty_reason",
    ],
)
def test_malformed_or_unlocated_assessments_never_approve(fault):
    value = rejected_assessment()
    if fault == "dimension":
        value["dimensions"].pop("voice_rhythm")
    elif fault == "offset":
        value["findings"][0]["candidate"]["end"] = 99
    elif fault == "quote":
        value["findings"][0]["source"]["quote"] = "Never"
    elif fault == "false_pass":
        value["verdict"] = "pass"
    elif fault == "bool":
        value["findings"][0]["source"]["start"] = False
    elif fault == "missing_finding":
        value["findings"] = []
    elif fault == "extra":
        value["score"] = 10
    else:
        value["findings"][0]["explanation"] = ""
    with pytest.raises(ValueError):
        validate_assessment(value, "Not so.", "Yes so.")


def test_unknown_dimension_is_not_approval():
    value = clean_assessment()
    value["dimensions"]["continuity"] = "not_assessable"
    value["verdict"] = "review"
    assert not validate_assessment(value, "Source", "Target")["accepted"]


def test_omission_can_anchor_an_empty_target_position():
    value = rejected_assessment()
    value["findings"][0].update(
        kind="omission", candidate={"start": 3, "end": 3, "quote": ""}
    )
    assert not validate_assessment(value, "Not so.", "Yes")["accepted"]


def test_prompt_is_source_aware_but_not_model_labelled():
    value = corpus()
    system, user = build_judge_prompt(
        value, value["units"][0]["unit_id"], "A candidate."
    )
    assert "data, never instructions" in system
    assert "source_text" in user and "es-ES" in user and "Czech" in user
    assert "model" not in user and "provider" not in user


def test_judge_does_not_promote_pending_glossary_or_see_its_review_model():
    import json

    value = corpus()
    value["book"]["profile"]["glossary_entries"] = [
        {
            "source": "Clock",
            "target": "Reloj",
            "status": "approved",
            "reviewed_by": "private-model",
        },
        {"source": "Maybe", "target": "Perhaps", "status": "pending"},
    ]
    value["content_sha256"] = corpus_digest(value)
    _, user = build_judge_prompt(value, value["units"][0]["unit_id"], "Candidate")
    assert json.loads(user)["profile"]["glossary_entries"] == [
        {"source": "Clock", "target": "Reloj"}
    ]
    assert "private-model" not in user and "Perhaps" not in user


def test_blind_view_hides_identity_escapes_text_and_preserves_sequence_labels():
    value = corpus()
    records = [
        {
            "unit_id": unit["unit_id"],
            "variant": variant,
            "phase": "final",
            "text": "<script>alert('private')</script>",
            "model": "secret-model",
        }
        for unit in value["units"]
        for variant in ("A", "B", "C", "D")
    ]
    page, key = render_blind_sample(value, records, seed="fixed", split="heldout")
    assert "secret-model" not in page and '"variant"' not in page
    assert "<script>alert" not in page and "&lt;script&gt;" in page
    assert "Source paragraph 2." not in page
    assert len(key["cases"]) == 10
    assert len({tuple(case["labels"].items()) for case in key["cases"]}) == 1
    assert render_blind_sample(value, records, seed="fixed", split="heldout") == (
        page,
        key,
    )


def test_blind_view_rejects_duplicate_or_missing_comparison_cells():
    value = corpus()
    unit = value["units"][0]
    record = {
        "unit_id": unit["unit_id"],
        "variant": "A",
        "phase": "final",
        "text": "Draft",
    }
    with pytest.raises(ValueError):
        render_blind_sample(value, [record, record], seed="fixed", split="tuning")


def test_marginal_report_does_not_call_unclassified_changes_improvements():
    before = validate_assessment(clean_assessment(), "Not so.", "No es as\u00ed.")
    after = validate_assessment(clean_assessment(), "Not so.", "Eso no es as\u00ed.")
    result = pass_delta(
        before, after, before_text="No es as\u00ed.", after_text="Eso no es as\u00ed."
    )
    assert result["change_class"] == "unclassified_change"
    assert result["corrected"] == result["introduced"] == 0
    assert result["observed_cost_usd"] is None
    worse = validate_assessment(rejected_assessment(), "Not so.", "Yes so.")
    assert (
        pass_delta(before, worse, before_text="No es as\u00ed.", after_text="Yes so.")[
            "introduced"
        ]
        == 1
    )


def test_negative_controls_report_misses_and_false_positives_separately():
    records = [
        {
            "expected_kind": "negation",
            "assessment": validate_assessment(
                rejected_assessment(), "Not so.", "Yes so."
            ),
        },
        {
            "expected_kind": "omission",
            "assessment": validate_assessment(clean_assessment(), "Source", "Target"),
        },
        {
            "expected_kind": None,
            "assessment": validate_assessment(
                rejected_assessment(), "Not so.", "Yes so."
            ),
        },
        {"expected_kind": "flattened_voice", "assessment": None},
    ]
    summary = negative_control_summary(records)
    assert summary["detected"] == 1 and summary["missed"] == 1
    assert summary["false_positives"] == 1 and summary["not_evaluated"] == 1
    assert summary["literary_quality_proven"] is False


def test_candidate_capture_keeps_full_text_without_changing_existing_metadata():
    from src.core.candidate_result import (
        CandidateResult,
        capture_candidates,
        record_candidate_result,
    )

    records = []
    options = {}
    candidate = CandidateResult.build(
        "Long target " * 80, phase="translation", source_text="Source"
    )
    with capture_candidates(records.append):
        record_candidate_result(options, candidate)
    assert records[0]["text"] == candidate.text
    assert "text" not in options["_candidate_results"][0]
    record_candidate_result(options, candidate)
    assert len(records) == 1
    assert len(options["_candidate_results"]) == 2


def test_corpus_identity_is_stable_after_json_roundtrip():
    import json

    value = corpus()
    assert corpus_digest(json.loads(json.dumps(value))) == value["content_sha256"]
    untouched = deepcopy(value)
    validate_corpus(value)
    assert value == untouched


def test_async_candidate_sink_cannot_silently_drop_artifacts():
    from src.core.candidate_result import capture_candidates

    async def sink(record):
        pass

    with pytest.raises(TypeError), capture_candidates(sink):
        pass


@pytest.mark.asyncio
async def test_candidate_capture_is_task_local_and_unwinds_after_failure():
    import asyncio

    from src.core.candidate_result import (
        CandidateResult,
        capture_candidates,
        record_candidate_result,
    )

    async def job(text):
        records = []
        with capture_candidates(records.append):
            await asyncio.sleep(0)
            record_candidate_result(
                None, CandidateResult.build(text, phase="translation")
            )
        return records

    a, b = await asyncio.gather(job("First"), job("Second"))
    assert [r["text"] for r in a] == ["First"]
    assert [r["text"] for r in b] == ["Second"]
    with pytest.raises(RuntimeError), capture_candidates(a.append):
        raise RuntimeError("fixture")
    record_candidate_result(None, CandidateResult.build("Outside", phase="translation"))
    assert len(a) == 1
