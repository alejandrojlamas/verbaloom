from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from src.core.candidate_result import CandidateIssue
from src.core.quality_assurance.config import QualityAssuranceConfig
from src.core.quality_assurance.extractors import build_manifest
from src.core.quality_assurance.gates import GateStatus, evaluate_quality_gates
from src.core.quality_assurance.models import BookManifest, TranslationUnit, UnitStatus
from src.core.quality_assurance.reports import _finding_count, write_quality_reports
from src.core.quality_assurance.runner import run_quality_assurance
from src.core.quality_assurance import validators
from src.core.quality_assurance.validators import (
    validate_manifest,
    validate_translation_unit,
    validate_unit_entities,
    validate_unit_language,
    validate_unit_typography,
)
from src.persistence.checkpoint_manager import CheckpointManager
from tests.characterization import fixtures


def _unit(source: str, target: str, *, index: int = 0) -> TranslationUnit:
    return TranslationUnit.create(
        document_id="doc:test",
        order_index=index,
        source_text=source,
        source_language="English",
        target_language="Spanish",
        translated_text=target,
        final_text=target,
        structural_path=f"/paragraphs/{index}",
        status=UnitStatus.TRANSLATED,
    )


def _manifest(tmp_path, units) -> BookManifest:
    source = tmp_path / "source.txt"
    output = tmp_path / "output.txt"
    source.write_text("\n\n".join(unit.source_text for unit in units), encoding="utf-8")
    output.write_text("\n\n".join(unit.best_text for unit in units), encoding="utf-8")
    return BookManifest(
        document_id="doc:test",
        run_id="run-test",
        source_path=str(source),
        output_path=str(output),
        source_language="English",
        target_language="Spanish",
        target_locale="es-MX",
        source_format="txt",
        output_format="txt",
        units=list(units),
    )


def test_unit_identity_is_stable_and_independent_of_translation():
    first = _unit("The same source text.", "El primer candidato.")
    second = _unit("The same source text.", "Un candidato completamente distinto.")

    assert first.unit_id == second.unit_id
    assert first.checksum_source == second.checksum_source
    assert first.checksum_structure == second.checksum_structure


def test_missing_checkpoint_unit_blocks_full_coverage(tmp_path):
    source = tmp_path / "source.txt"
    output = tmp_path / "output.txt"
    source.write_text("One.\n\nTwo.", encoding="utf-8")
    output.write_text("Uno.", encoding="utf-8")
    checkpoint = {
        "job": {
            "file_type": "txt",
            "config": {"model": "test", "llm_provider": "fake"},
            "progress": {"total_chunks": 2},
        },
        "chunks": [
            {
                "chunk_index": 0,
                "original_text": "One.",
                "translated_text": "Uno.",
                "chunk_data": {},
                "status": "completed",
            }
        ],
    }
    manifest = build_manifest(
        source_path=source,
        output_path=output,
        source_language="English",
        target_language="Spanish",
        checkpoint_data=checkpoint,
    )
    bundle = validate_manifest(manifest, QualityAssuranceConfig())
    report = evaluate_quality_gates(manifest, bundle, QualityAssuranceConfig())

    assert report.status == GateStatus.BLOCKED
    assert any(issue.code == "missing_checkpoint_unit" for issue in report.issues)


def test_identical_untranslated_unit_is_blocking(tmp_path):
    unit = _unit(
        "The original paragraph remains completely unchanged in the final document.",
        "The original paragraph remains completely unchanged in the final document.",
    )
    manifest = _manifest(tmp_path, [unit])

    report = evaluate_quality_gates(
        manifest,
        validate_manifest(manifest, QualityAssuranceConfig()),
        QualityAssuranceConfig(),
    )

    assert report.status == GateStatus.BLOCKED
    assert any(issue.code == "untranslated_source" for issue in report.issues)


def test_changed_number_and_scientific_name_are_critical(tmp_path):
    unit = _unit(
        "In 1913 the study described Homo sapiens in detail.",
        "En 1931 el estudio describio Homo erectus en detalle.",
    )
    manifest = _manifest(tmp_path, [unit])
    report = evaluate_quality_gates(
        manifest,
        validate_manifest(manifest, QualityAssuranceConfig()),
        QualityAssuranceConfig(),
    )

    codes = {issue.code for issue in report.issues}
    assert "number_mismatch" in codes
    assert "scientific_name_mismatch" in codes
    assert report.status == GateStatus.BLOCKED


def test_ordinary_capitalized_prose_is_not_a_scientific_name():
    unit = _unit(
        "Sheila stood by the door while Houston had already left.",
        "Sheila estaba junto a la puerta cuando Houston ya se habia ido.",
    )

    result, detail = validate_unit_entities(unit, QualityAssuranceConfig())

    assert not any(issue.code == "scientific_name_mismatch" for issue in result.issues)
    assert detail["scientific_name"]["source"] == {}


def test_scientific_detector_rejects_copula_articles_and_prose_gerunds():
    unit = _unit(
        "El informe describe una especie con suficiente contexto narrativo.",
        "Este cuerpo es una especie de espacio en blanco. Gloria Swanson como "
        "una especie de sí misma. Esa fue una especie de traición. "
        "Son una especie resistente; Ferguson formando una especie distinta.",
    )

    result, detail = validate_unit_entities(unit, QualityAssuranceConfig())

    assert not any(issue.code == "scientific_name_mismatch" for issue in result.issues)
    assert detail["scientific_name"]["target"] == {}


def test_intentionally_sanitized_url_is_not_required_by_final_entity_gate():
    unit = _unit(
        "La biografía termina con www.example.com/author para conocer más.",
        "La biografía termina aquí para el lector.",
    )
    unit.source_reference["publication_audited"] = True
    unit.source_reference["intentional_exclusions"] = [
        "www.example.com/author",
    ]

    result, detail = validate_unit_entities(unit, QualityAssuranceConfig())

    assert not any(issue.code == "url_mismatch" for issue in result.issues)
    assert detail["url"]["intentionally_excluded"] == {
        "www.example.com/author": 1
    }


def test_unmarked_missing_url_remains_critical():
    unit = _unit(
        "La biografía termina con www.example.com/author para conocer más.",
        "La biografía termina aquí para el lector.",
    )
    unit.source_reference["publication_audited"] = True

    result, _ = validate_unit_entities(unit, QualityAssuranceConfig())

    issues = [issue for issue in result.issues if issue.code == "url_mismatch"]
    assert len(issues) == 1
    assert issues[0].severity == "critical"
    assert result.passed is False


def test_entity_numbers_and_metric_spacing_use_canonical_values():
    unit = _unit(
        "La colonia pone 40000 huevos y el ejemplar mide 2cm.",
        "La colonia pone 40 000 huevos y el ejemplar mide 2 cm.",
    )

    result, detail = validate_unit_entities(unit, QualityAssuranceConfig())

    assert not any(issue.code in {"number_mismatch", "measurement_mismatch"} for issue in result.issues)
    assert detail["number"]["source"] == detail["number"]["target"]
    assert detail["measurement"]["source"] == detail["measurement"]["target"]


def test_identifier_suffix_is_not_misread_as_single_letter_measurement():
    unit = _unit(
        "The archive entry was recorded without a public identifier.",
        "La entrada del archivo se registro con el identificador 1615A.",
    )

    result, _ = validate_unit_entities(unit, QualityAssuranceConfig())

    assert not any(issue.code == "measurement_mismatch" for issue in result.issues)


def test_toc_number_followed_by_english_article_is_not_measurement():
    unit = _unit(
        "Contents 1 Divine Right 2 A Civilizing Mission 18 A World War on Truth",
        "Contenido 1 Derecho divino 2 Una mision civilizatoria 18 Una guerra mundial contra la verdad",
    )

    result, detail = validate_unit_entities(unit, QualityAssuranceConfig())

    assert not any(issue.code == "measurement_mismatch" for issue in result.issues)
    assert detail["measurement"]["source"] == {}
    assert detail["measurement"]["target"] == {}


def test_spanish_range_is_not_misread_as_amperage():
    unit = _unit(
        "The useful range is from 2 to 3 chapters.",
        "El intervalo util va de 2 a 3 capitulos.",
    )

    result, detail = validate_unit_entities(unit, QualityAssuranceConfig())

    assert not any(issue.code == "measurement_mismatch" for issue in result.issues)
    assert detail["measurement"]["target"] == {}


def test_real_single_letter_measurement_remains_protected():
    unit = _unit(
        "The circuit carries 2 A current at 5 V.",
        "El circuito conduce 3 A de corriente a 5 V.",
    )

    result, detail = validate_unit_entities(unit, QualityAssuranceConfig())

    issues = [issue for issue in result.issues if issue.code == "measurement_mismatch"]
    assert len(issues) == 1
    assert issues[0].severity == "critical"
    assert detail["measurement"]["source"] == {"2A": 1, "5V": 1}
    assert detail["measurement"]["target"] == {"3A": 1, "5V": 1}


def test_bibliographic_year_followed_by_split_word_is_not_liters():
    unit = _unit(
        "IV. Of Studies by Sir Francis Bacon, 1597 S tudies serve for delight.",
        "IV. De los estudios por Sir Francis Bacon, 1597 L os estudios sirven para el deleite.",
    )

    result, detail = validate_unit_entities(unit, QualityAssuranceConfig())

    assert not any(issue.code == "measurement_mismatch" for issue in result.issues)
    assert detail["measurement"]["source"] == {}
    assert detail["measurement"]["target"] == {}


def test_large_liter_measurement_is_not_suppressed_by_year_guard():
    unit = _unit(
        "The tank stores 2000 L of water.",
        "El tanque almacena 2100 L de agua.",
    )

    result, detail = validate_unit_entities(unit, QualityAssuranceConfig())

    issues = [issue for issue in result.issues if issue.code == "measurement_mismatch"]
    assert len(issues) == 1
    assert detail["measurement"]["source"] == {"2000L": 1}
    assert detail["measurement"]["target"] == {"2100L": 1}


def test_sparse_numeric_diff_in_publication_audited_chapter_is_warning_only():
    unit = _unit(
        "The source chapter contains stable narrative prose. " * 120,
        ("El capitulo contiene prosa narrativa estable. " * 120) + "En 80, todo cambio.",
    )
    unit.source_reference["publication_audited"] = True

    result, _ = validate_unit_entities(unit, QualityAssuranceConfig())

    number_issues = [issue for issue in result.issues if issue.code == "number_mismatch"]
    assert len(number_issues) == 1
    assert number_issues[0].severity == "medium"
    assert result.passed is True


def test_publication_audited_aggregate_only_downgrades_numeric_differences():
    unit = _unit(
        ("In 1913 the study described Homo sapiens in detail. " * 100),
        ("En 1931 el estudio describió Homo erectus con detalle. " * 100),
    )
    unit.source_reference["publication_audited"] = True

    result, _ = validate_unit_entities(unit, QualityAssuranceConfig())

    issues = {issue.code: issue for issue in result.issues}
    assert issues["number_mismatch"].severity == "medium"
    assert issues["scientific_name_mismatch"].severity == "critical"
    assert result.passed is False


def test_publication_audited_chapter_skips_phrase_residual_but_keeps_language_detection(monkeypatch):
    unit = _unit(
        "The source chapter contains enough text to require translation.",
        "El capitulo esta traducido al espanol con claridad y conserva un nombre como Houston.",
    )
    unit.source_reference["publication_audited"] = True
    monkeypatch.setattr(
        validators,
        "target_language_gate_issues",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("phrase gate must be skipped")),
    )

    result = validate_unit_language(unit, QualityAssuranceConfig())

    assert result.passed is True


def test_publication_audited_chapter_still_blocks_source_dominant_output():
    source = "This complete English chapter remains in the source language and must be rejected. " * 12
    unit = _unit(source, source)
    unit.source_reference["publication_audited"] = True

    result = validate_unit_language(unit, QualityAssuranceConfig())

    assert result.passed is False
    assert any(issue.code == "source_language_unit" for issue in result.issues)


def test_output_guard_warning_does_not_fail_translation_coverage(monkeypatch):
    unit = _unit(
        "The source paragraph contains enough stable content for translation.",
        "El parrafo traducido contiene suficiente contenido estable.",
    )
    monkeypatch.setattr(
        validators,
        "guard_llm_output",
        lambda *args, **kwargs: SimpleNamespace(
            issues=[
                CandidateIssue(
                    code="non_idempotent_output_cleanup",
                    severity="warning",
                    message="Cleanup stabilized after a second pass.",
                )
            ]
        ),
    )

    result = validate_translation_unit(unit, QualityAssuranceConfig())

    assert result.passed is True
    assert result.issues[0].severity == "medium"


def test_camelcase_name_is_review_only_not_a_typography_blocker():
    unit = _unit(
        "PetroGlobe remained the registered company name.",
        "PetroGlobe siguio siendo el nombre registrado de la empresa.",
    )

    result = validate_unit_typography(unit, QualityAssuranceConfig())

    word_join = [issue for issue in result.issues if issue.code == "spacing_word_join"]
    assert len(word_join) == 1
    assert word_join[0].severity == "medium"
    assert result.passed is True


def test_lost_spacing_blocks_typography_gate(tmp_path):
    unit = _unit(
        "The first sentence ends. Another sentence starts.",
        "La primera oracion termina.Otra oracion comienza.",
    )
    manifest = _manifest(tmp_path, [unit])
    report = evaluate_quality_gates(
        manifest,
        validate_manifest(manifest, QualityAssuranceConfig()),
        QualityAssuranceConfig(),
    )

    assert any(issue.code == "spacing_period_join" for issue in report.issues)
    assert report.status == GateStatus.BLOCKED


def test_protected_entity_requires_an_approved_target(tmp_path):
    unit = _unit(
        "The book was published by Eichborn Verlag in 1995.",
        "El libro fue publicado por Editorial Equivocada en 1995.",
    )
    manifest = _manifest(tmp_path, [unit])
    bundle = validate_manifest(
        manifest,
        QualityAssuranceConfig(),
        protected_entities=[
            {
                "source": "Eichborn Verlag",
                "allowed_targets": ["Eichborn Verlag"],
                "type": "publisher",
                "locked": True,
            }
        ],
    )
    report = evaluate_quality_gates(manifest, bundle, QualityAssuranceConfig())

    assert any(issue.code == "protected_entity_mismatch" for issue in report.issues)
    assert report.status == GateStatus.BLOCKED


def test_clean_manifest_passes_and_writes_all_required_reports(tmp_path):
    unit = _unit(
        "The traveler crossed the bridge and returned to the old house.",
        "El viajero cruzo el puente y regreso a la casa antigua.",
    )
    manifest = _manifest(tmp_path, [unit])
    config = QualityAssuranceConfig()
    bundle = validate_manifest(manifest, config)
    report = evaluate_quality_gates(manifest, bundle, config)

    assert report.publishable is True
    paths = write_quality_reports(tmp_path / "reports", manifest=manifest, gate_report=report, bundle=bundle)
    assert paths.manifest.exists()
    assert paths.report_json.exists()
    assert paths.report_html.exists()
    assert paths.failed_units.exists()
    assert paths.entity_diff.exists()
    assert paths.language_report.exists()
    assert paths.structure_report.exists()
    assert paths.export_report.exists()
    payload = json.loads(paths.report_json.read_text(encoding="utf-8"))
    assert len(payload["quality_gates"]) == 10
    assert payload["coverage"]["translated"] == 1
    assert payload["coverage"]["reviewed"] == 1
    assert payload["coverage"]["audited"] == 1
    assert payload["coverage"]["approved"] == 1


def test_quality_run_marks_only_failed_checkpoint_unit_for_sparse_repair(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "source.txt"
    output = tmp_path / "output.txt"
    source.write_text("First source paragraph.\n\nSecond source paragraph that must change.", encoding="utf-8")
    output.write_text("Primer parrafo.\n\nSecond source paragraph that must change.", encoding="utf-8")
    manager = CheckpointManager(db_path=str(tmp_path / "jobs.db"))
    assert manager.start_job(
        "run-repair",
        "txt",
        {
            "source_language": "English",
            "target_language": "Spanish",
            "model": "fake",
            "llm_provider": "fake",
        },
    )
    manager.save_checkpoint(
        "run-repair", 0, "First source paragraph.", "Primer parrafo.", {},
        total_chunks=2, completed_chunks=1, failed_chunks=0,
    )
    manager.save_checkpoint(
        "run-repair", 1,
        "Second source paragraph that must change.",
        "Second source paragraph that must change.",
        {}, total_chunks=2, completed_chunks=2, failed_chunks=0,
    )
    checkpoint = manager.load_checkpoint("run-repair")

    result = run_quality_assurance(
        source_path=source,
        output_path=output,
        source_language="English",
        target_language="Spanish",
        run_id="run-repair",
        checkpoint_data=checkpoint,
        report_root=tmp_path / "quality",
        checkpoint_manager=manager,
    )

    assert result.publishable is False
    assert result.repair_checkpoint_indices == [1]
    repaired = manager.load_checkpoint("run-repair")
    assert repaired["failed_chunk_indices"] == [1]
    assert repaired["chunks"][0]["translated_text"] == "Primer parrafo."
    assert repaired["chunks"][1]["translated_text"] is None
    assert repaired["resume_from_index"] == 1
    assert repaired["chunks"][1]["chunk_data"]["rejected_translation"].startswith("Second source")


def test_manifest_detects_duplicate_identity_and_changed_order(tmp_path):
    first = _unit("First source paragraph.", "Primer parrafo.", index=0)
    second = _unit("Second source paragraph.", "Segundo parrafo.", index=1)
    manifest = _manifest(tmp_path, [second, first, first])

    issues = manifest.validate_identity()

    codes = {issue.code for issue in issues}
    assert "duplicate_unit_id" in codes
    assert "non_contiguous_order" in codes
    assert "unit_order_changed" in codes


def test_strict_checkpoint_waits_for_quality_gate_before_completed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manager = CheckpointManager(db_path=str(tmp_path / "jobs.db"))
    config = {
        "prompt_options": {"strict_quality_assurance": True},
        "quality_assurance": {"strict": True},
    }
    assert manager.start_job("strict-run", "txt", config)
    manager.save_checkpoint(
        "strict-run",
        0,
        "Source paragraph.",
        "Parrafo traducido.",
        {},
        total_chunks=1,
        completed_chunks=1,
        failed_chunks=0,
    )

    checkpoint = manager.load_checkpoint("strict-run")

    assert checkpoint["checkpoint_complete"] is True
    assert checkpoint["job"]["status"] == "validating"
    assert manager.mark_completed("strict-run") is True
    assert manager.get_job("strict-run")["status"] == "validating"
    assert manager.mark_completed("strict-run", quality_gate_passed=True) is True
    assert manager.get_job("strict-run")["status"] == "completed"


def test_source_script_returned_unchanged_is_blocked_for_language_without_spaces(tmp_path):
    unit = TranslationUnit.create(
        document_id="doc:cjk",
        order_index=0,
        source_text="这是一个完整的中文段落，它应该被翻译成西班牙语，而不是原样返回。",
        source_language="Chinese",
        target_language="Spanish",
        translated_text="这是一个完整的中文段落，它应该被翻译成西班牙语，而不是原样返回。",
        final_text="这是一个完整的中文段落，它应该被翻译成西班牙语，而不是原样返回。",
        status=UnitStatus.TRANSLATED,
    )
    manifest = _manifest(tmp_path, [unit])

    report = evaluate_quality_gates(
        manifest,
        validate_manifest(manifest, QualityAssuranceConfig()),
        QualityAssuranceConfig(),
    )

    assert report.status == GateStatus.BLOCKED
    assert any(issue.gate == "language" or "script" in issue.code for issue in report.issues)


def test_legitimate_third_language_quote_can_be_preserved(tmp_path):
    quote = (
        '"Il faut surtout pardonner a ces ames malheureuses qui regardent sans '
        'comprendre le profond desespoir des vaincus."'
    )
    unit = _unit(quote, quote)
    manifest = _manifest(tmp_path, [unit])

    report = evaluate_quality_gates(
        manifest,
        validate_manifest(manifest, QualityAssuranceConfig()),
        QualityAssuranceConfig(),
    )

    assert report.publishable is True


def test_semantic_omission_and_invalid_locale_block_publication(tmp_path):
    unit = _unit(
        "The source contains a long causal explanation with several important clauses, examples, and a conclusion that must all remain in the translated paragraph.",
        "Una explicacion breve.",
    )
    manifest = _manifest(tmp_path, [unit])
    manifest.target_locale = "sp"

    report = evaluate_quality_gates(
        manifest,
        validate_manifest(manifest, QualityAssuranceConfig()),
        QualityAssuranceConfig(),
    )

    codes = {issue.code for issue in report.issues}
    assert "probable_semantic_omission" in codes
    assert "invalid_target_language_code" in codes
    assert report.status == GateStatus.BLOCKED


def test_large_manifest_identity_validation_is_incremental_and_complete():
    units = [
        TranslationUnit.create(
            document_id="doc:large",
            order_index=index,
            source_text=f"Source unit {index} with enough stable content.",
            source_language="English",
            target_language="Spanish",
            translated_text=f"Unidad traducida {index} con contenido estable.",
            status=UnitStatus.TRANSLATED,
        )
        for index in range(5000)
    ]
    manifest = BookManifest(
        document_id="doc:large",
        source_path="large.txt",
        output_path="large-es.txt",
        source_language="English",
        target_language="Spanish",
        units=units,
    )

    assert manifest.validate_identity() == []
    assert manifest.counts()["translatable"] == 5000


def test_verifiable_dates_percentages_currency_measurements_and_identifiers_are_protected(tmp_path):
    unit = _unit(
        "On 2024-01-31 revenue was $20, growth was 50%, mass was 3 kg, and DOI: 10.1234/ABC.9 was cited.",
        "El 2024-02-01 los ingresos fueron $30, el crecimiento fue de 60%, la masa fue de 4 kg y se cito DOI: 10.1234/XYZ.1.",
    )
    manifest = _manifest(tmp_path, [unit])

    report = evaluate_quality_gates(
        manifest,
        validate_manifest(manifest, QualityAssuranceConfig()),
        QualityAssuranceConfig(),
    )

    codes = {issue.code for issue in report.issues}
    assert {"date_mismatch", "percentage_mismatch", "currency_mismatch", "measurement_mismatch", "identifier_mismatch"} <= codes
    assert report.status == GateStatus.BLOCKED


def test_duplicate_substantial_output_from_different_sources_is_blocked(tmp_path):
    repeated = (
        "Este es un parrafo traducido suficientemente largo para demostrar que dos unidades "
        "fuente distintas no deben producir exactamente la misma salida editorial sustancial."
    )
    first = _unit("The first source describes a mountain expedition in winter with several details.", repeated, index=0)
    second = _unit("The second source describes a summer voyage across the ocean with other details.", repeated, index=1)
    manifest = _manifest(tmp_path, [first, second])

    report = evaluate_quality_gates(
        manifest,
        validate_manifest(manifest, QualityAssuranceConfig()),
        QualityAssuranceConfig(),
    )

    assert any(issue.code == "duplicate_unrelated_output" for issue in report.issues)
    assert report.status == GateStatus.BLOCKED


def test_extra_artifact_block_is_not_silently_ignored(tmp_path):
    source = tmp_path / "source.txt"
    output = tmp_path / "output.txt"
    source.write_text("A single source paragraph that should be translated.", encoding="utf-8")
    output.write_text("Un solo parrafo traducido.\n\nUn bloque adicional que no existe en la fuente.", encoding="utf-8")
    manifest = build_manifest(
        source_path=source,
        output_path=output,
        source_language="English",
        target_language="Spanish",
    )

    report = evaluate_quality_gates(
        manifest,
        validate_manifest(manifest, QualityAssuranceConfig()),
        QualityAssuranceConfig(),
    )

    assert any(issue.code == "artifact_block_count_mismatch" for issue in report.issues)
    assert report.status == GateStatus.BLOCKED


def test_structure_checksum_detects_manifest_mutation():
    unit = _unit("A stable source paragraph.", "Un parrafo fuente estable.")
    unit.structural_path = "/mutated/path"
    manifest = BookManifest(
        document_id="doc:test",
        source_path="source.txt",
        output_path="output.txt",
        source_language="English",
        target_language="Spanish",
        units=[unit],
    )

    assert any(issue.code == "structure_checksum_mismatch" for issue in manifest.validate_identity())


def test_report_path_is_rewritten_after_quarantine(tmp_path):
    unit = _unit("The traveler returned safely to the village.", "El viajero regreso a salvo al pueblo.")
    manifest = _manifest(tmp_path, [unit])
    run = run_quality_assurance(
        source_path=manifest.source_path,
        output_path=manifest.output_path,
        source_language="English",
        target_language="Spanish",
        run_id="quarantine-path",
        report_root=tmp_path / "quality",
    )
    quarantined = tmp_path / "[partial] output.txt"
    Path(manifest.output_path).replace(quarantined)

    run.update_output_path(quarantined)

    payload = json.loads(run.paths.report_json.read_text(encoding="utf-8"))
    assert payload["output"]["path"] == str(quarantined)


def test_broken_epub_blocks_and_still_writes_diagnostic_reports(tmp_path):
    source = fixtures.build_epub(tmp_path)
    output = tmp_path / "broken.epub"
    output.write_bytes(b"not an epub package")

    run = run_quality_assurance(
        source_path=source,
        output_path=output,
        source_language="English",
        target_language="French",
        run_id="broken-epub",
        report_root=tmp_path / "quality",
    )

    codes = {issue.code for issue in run.report.issues}
    assert run.publishable is False
    assert "output_block_extraction_failed" in codes
    assert "invalid_epub" in codes
    assert run.paths.report_json.is_file()
    assert run.paths.structure_report.is_file()


def test_report_finding_count_accepts_epub_error_lists():
    assert _finding_count(["missing archive member", "broken link"]) == 2
    assert _finding_count(3) == 3
