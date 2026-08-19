import re

import pytest
import yaml

from src.core import postprocess_repair
from src.core.book_profiles.loader import create_profile
from src.core.postprocess_repair import (
    detect_repair_issues,
    postprocess_repair_enabled,
    repair_flagged_chunks,
)
from src.core.refine import txt_refiner


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.closed = False

    async def make_request(self, prompt, model=None, timeout=None, system_prompt=None):
        self.calls.append({
            "prompt": prompt,
            "model": model,
            "timeout": timeout,
            "system_prompt": system_prompt,
        })
        if not self.responses:
            return None
        return FakeResponse(self.responses.pop(0))

    def extract_translation(self, response):
        match = re.search(r"<TRANSLATION>(.*?)</TRANSLATION>", response, re.S)
        return match.group(1).strip() if match else None

    async def close(self):
        self.closed = True


def _create_test_profile(tmp_path, monkeypatch, profile_id="profile_modernize_test"):
    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    profile_dir = create_profile(profile_id)
    profile_path = profile_dir / "profile.yml"
    config = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    config["target_locale"] = "es-MX"
    config["modernization_strength"] = "high"
    config["detectors"] = [
        {
            "code": "treatment_residual",
            "pattern": r"\bvuestras\s+mercedes\b",
            "severity": "high",
            "message": "Old treatment remains for this profile.",
            "applies_to": "candidate",
        }
    ]
    profile_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return profile_id


def test_postprocess_repair_enabled_only_for_profile_modernize():
    assert not postprocess_repair_enabled({})
    assert postprocess_repair_enabled({
        "text_transform_mode": "modernize",
        "profile_id": "quijote_mx_contemporary",
    })
    assert not postprocess_repair_enabled({
        "text_transform_mode": "modernize",
        "profile_id": "quijote_mx_contemporary",
        "postprocess_repair_enabled": False,
    })


def test_postprocess_repair_prompts_require_issue_local_edits():
    variants = postprocess_repair._build_prompt_variants(
        source_text="One correct sentence. One bad sentence.",
        current_text="Una oración correcta. <TRANSLATIONATION> Una oración mala.",
        context_before="",
        context_after="",
        target_language="Spanish",
        prompt_options={},
        issues=[
            postprocess_repair.RepairIssue(
                code="wrapper_artifact",
                severity="high",
                reason="Visible wrapper artifact.",
            )
        ],
        chunk_index=0,
    )

    assert variants
    for _prompt_id, _system, prompt in variants:
        normalized = re.sub(r"\s+", " ", prompt)
        assert "Issue-local repair contract" in prompt
        assert "Keep every unaffected sentence unchanged" in normalized
        assert "Do not reroll the whole chunk" in normalized


def test_detect_repair_issues_flags_profile_scoped_treatment_residue(tmp_path, monkeypatch):
    profile_id = _create_test_profile(tmp_path, monkeypatch)
    source = (
        "-Non fuyan las vuestras mercedes, ni teman desaguisado alguno, "
        "ca a la orden de caballería que profeso non toca ni atañe facerle "
        "a ninguno. " * 4
    )
    issues = detect_repair_issues(
        source,
        source,
        chunk_index=44,
        prompt_options={
            "text_transform_mode": "modernize",
            "profile_id": profile_id,
        },
    )

    codes = {issue.code for issue in issues}
    assert "near_identical_modernization" in codes
    assert "lexical_archaism" in codes
    assert "profile_treatment_residual" in codes


def test_detect_repair_issues_flags_colonial_spanish_not_modernized():
    source = (
        "SERÍA el gran Moctezuma de edad de hasta cuarenta años, e cenceño e "
        "pocas carnes, y la color no muy moreno. Señor Moctezuma, bien podéis "
        "creer que si os queréis ir a vuestros palacios, traíanle frutas y "
        "servíase con barro de Cholula. " * 3
    )

    issues = detect_repair_issues(
        source,
        source,
        chunk_index=2,
        prompt_options={
            "text_transform_mode": "modernize",
            "profile_id": "auto_historia_mx",
            "target_locale": "es-MX",
        },
    )

    by_code = {issue.code: issue for issue in issues}
    assert by_code["modernization_residue"].severity == "high"
    assert "near_identical_modernization" in by_code


@pytest.mark.asyncio
async def test_repair_flagged_chunks_tries_alternate_prompts_until_pass():
    source = (
        "-Non fuyan las vuestras mercedes, ni teman desaguisado alguno, "
        "ca a la orden de caballería que profeso non toca ni atañe facerle "
        "a ninguno. " * 4
    )
    bad = source
    good = " ".join([
        "—No huyan, señoras, ni teman agravio alguno, porque a la orden de "
        "caballería que profeso no le toca ni le corresponde hacer daño a nadie, "
        "mucho menos a doncellas tan altas como ustedes parecen."
        for _ in range(4)
    ])
    client = FakeClient([
        f"<TRANSLATION>{bad}</TRANSLATION>",
        f"<TRANSLATION>{good}</TRANSLATION>",
    ])
    saved = []

    result = await repair_flagged_chunks(
        refined_parts=[bad],
        structured_chunks=[{"main_content": source, "context_before": "", "context_after": ""}],
        target_language="Spanish",
        model_name="deepseek-v4-pro",
        api_endpoint="https://api.deepseek.com/chat/completions",
        llm_provider="deepseek",
        prompt_options={
            "text_transform_mode": "modernize",
            "profile_id": "quijote_mx_contemporary",
            "target_locale": "es-MX",
        },
        checkpoint_callback=lambda idx, src, out: saved.append((idx, src, out)),
        llm_client=client,
    )

    assert result.flagged_indices == [0]
    assert result.repaired_indices == [0]
    assert result.parts == [good]
    assert len(client.calls) == 2
    assert saved == [(0, source, good)]
    assert "Current flawed candidate" in client.calls[0]["prompt"]
    assert client.calls[0]["system_prompt"] != client.calls[1]["system_prompt"]
    assert "Cervantes" not in client.calls[0]["prompt"]
    assert "Cervantine" not in client.calls[0]["prompt"]


def test_postprocess_repair_has_no_book_specific_proper_nouns_hardcoded():
    banned = [
        "Don Quijote",
        "Sancho",
        "Dulcinea",
        "Rocinante",
        "Cervantes",
        "Montezuma",
        "Moctezuma",
    ]
    source = postprocess_repair.__file__
    text = open(source, encoding="utf-8").read()
    offenders = [term for term in banned if term in text]
    assert not offenders


@pytest.mark.asyncio
async def test_txt_refiner_runs_postprocess_repair_before_save(tmp_path, monkeypatch):
    output_path = tmp_path / "out.txt"
    seen = {}

    monkeypatch.setattr(
        txt_refiner,
        "split_text_into_chunks",
        lambda *_args, **_kwargs: [
            {"context_before": "", "main_content": "draft 0", "context_after": ""},
            {"context_before": "", "main_content": "draft 1", "context_after": ""},
        ],
    )

    async def fake_refine_chunks(**kwargs):
        if kwargs["stats_callback"]:
            kwargs["stats_callback"]({
                "total_chunks": 2,
                "completed_chunks": 2,
                "failed_chunks": 0,
            })
        return ["refinado 0", "refinado 1"]

    async def fake_repair_flagged_chunks(**kwargs):
        seen["parts"] = kwargs["refined_parts"]
        seen["chunks"] = kwargs["structured_chunks"]
        return postprocess_repair.PostprocessRepairResult(
            parts=["refinado 0", "reparado 1"],
            flagged_indices=[1],
            repaired_indices=[1],
            kept_indices=[],
            failed_indices=[],
        )

    monkeypatch.setattr(txt_refiner, "refine_chunks", fake_refine_chunks)
    monkeypatch.setattr(txt_refiner, "repair_flagged_chunks", fake_repair_flagged_chunks)

    ok = await txt_refiner.refine_text_content(
        translated_text="draft 0\n\ndraft 1",
        output_filepath=str(output_path),
        target_language="Spanish",
        prompt_options={
            "editorial_quality_report": False,
            "text_transform_mode": "modernize",
            "profile_id": "quijote_mx_contemporary",
        },
    )

    assert ok is True
    assert seen["parts"] == ["refinado 0", "refinado 1"]
    assert seen["chunks"][1]["main_content"] == "draft 1"
    assert output_path.read_text(encoding="utf-8") == "refinado 0\nreparado 1"
