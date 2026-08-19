"""Book editorial profile routes."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import math
import os
from pathlib import Path
import re
import shutil
import threading
import time
from typing import Any, Mapping
import uuid

from flask import Blueprint, current_app, jsonify, request
import yaml

from src.core.book_profiles.artifacts import editorial_artifact_counts
from src.core.book_profiles import (
    build_profile_glossary_block,
    build_profile_impact_preview,
    build_profile_knowledge_base,
    extract_profile_prep_text_from_bytes,
    load_book_profile,
    prepare_book_profile_from_text,
    profile_glossary_match_summary,
    resolve_profiles_root,
)
from src.core.book_profiles.glossary_editor import (
    ProfileGlossaryConflictError,
    ProfileGlossaryEditError,
    apply_profile_glossary_action,
    editable_profile_glossary_files,
    merge_profile_glossary_entries,
)
from src.core.book_profiles.profile_goals import resolve_profile_goal
from src.core.llm.factory import create_llm_provider
from src.core.output_formats import extract_readable_text
from src.core.usage import reset_usage_context, set_usage_context
import src.config as _config
from src.utils.language_detector import LanguageDetector
from src.utils.provider_security import (
    EndpointCredentialError,
    resolve_api_key_for_endpoint,
)


_MAX_PREP_UPLOAD_BYTES = 120 * 1024 * 1024
_PROFILE_PREP_JOBS: dict[str, dict[str, Any]] = {}
_PROFILE_PREP_LOCK = threading.Lock()
_PROFILE_PREP_JOB_TTL_SECONDS = 6 * 60 * 60
_SAFE_PROFILE_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_ACTIVE_PROFILE_STATUSES = {"queued", "running", "rate_limited", "resuming", "pausing"}
_MAX_PROFILE_PREP_LLM_CHUNKS = 600
_MAX_PROFILE_PREP_LOCAL_TERMS = 5000


class ProfilePrepRequestError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def create_profile_blueprint(state_manager=None):
    bp = Blueprint("book_profiles", __name__)

    @bp.route("/api/book-profiles", methods=["GET"])
    def list_book_profiles():
        root = resolve_profiles_root()
        profiles: list[dict[str, Any]] = []
        if root.exists():
            for config_path in sorted(root.glob("*/profile.yml")):
                profile_id = config_path.parent.name
                if profile_id.startswith("_") or profile_id == "common":
                    continue
                try:
                    profile = load_book_profile(profile_id, allow_missing=True)
                except Exception as exc:
                    profiles.append(_broken_profile_row(
                        config_path,
                        load_error=str(exc),
                        state_manager=state_manager,
                    ))
                    continue
                if profile is None:
                    continue
                in_use = _profile_in_active_job(profile.profile_id, state_manager)
                profiles.append({
                    "profile_id": profile.profile_id,
                    "name": profile.name,
                    "target_locale": profile.target_locale,
                    "approved_count": profile.approved_count,
                    "pending_count": profile.pending_count,
                    "profile_goal": _profile_goal_from_config(profile.raw_config),
                    "generated_profile": bool(profile.raw_config.get("generated_profile")),
                    "source_name": profile.raw_config.get("source_name") or "",
                    "preflight_model": profile.raw_config.get("preflight_model") or "",
                    "editorial_artifact_counts": editorial_artifact_counts(profile.editorial_artifacts),
                    "knowledge_summary": _profile_knowledge_summary(profile),
                    "deletable": _profile_can_be_deleted(profile.profile_id),
                    "in_use": in_use,
                })
        return jsonify({"profiles": profiles, "count": len(profiles)})

    @bp.route("/api/book-profiles/<profile_id>", methods=["DELETE"])
    def delete_book_profile(profile_id: str):
        profile_id = str(profile_id or "").strip()
        if not _safe_profile_id(profile_id):
            return jsonify({"error": "Invalid profile id."}), 400
        if not _profile_can_be_deleted(profile_id):
            return jsonify({"error": "This profile cannot be deleted."}), 403
        if _profile_in_active_job(profile_id, state_manager):
            return jsonify({"error": "This profile is being used by an active job."}), 409

        root = resolve_profiles_root()
        profile_dir = (root / profile_id).resolve()
        try:
            profile_dir.relative_to(root.resolve())
        except ValueError:
            return jsonify({"error": "Invalid profile path."}), 400
        if not (profile_dir / "profile.yml").exists():
            return jsonify({"error": "Book profile not found."}), 404

        try:
            shutil.rmtree(profile_dir)
        except Exception as exc:
            return jsonify({"error": f"Could not delete book profile: {exc}"}), 500
        return jsonify({"deleted": True, "profile_id": profile_id})

    @bp.route("/api/book-profiles/<profile_id>/glossary", methods=["GET"])
    def get_book_profile_glossary(profile_id: str):
        profile_id = str(profile_id or "").strip()
        if not _safe_profile_id(profile_id):
            return jsonify({"error": "Invalid profile id."}), 400
        try:
            profile = load_book_profile(profile_id, allow_missing=True)
        except Exception as exc:
            root = resolve_profiles_root()
            config_path = root / profile_id / "profile.yml"
            if not config_path.exists():
                return jsonify({"error": "Book profile not found."}), 404
            profile_payload = _broken_profile_row(
                config_path,
                load_error=str(exc),
                state_manager=state_manager,
            )
            return jsonify({
                "profile": profile_payload,
                "entries": [],
                "approved": [],
                "pending": [],
                "counts": {
                    "total": 0,
                    "approved": 0,
                    "pending": 0,
                },
                "load_error": str(exc),
            })
        if profile is None:
            return jsonify({"error": "Book profile not found."}), 404

        return jsonify(_book_profile_glossary_payload(profile))

    @bp.route("/api/book-profiles/<profile_id>/glossary/entry", methods=["PATCH"])
    def edit_book_profile_glossary_entry(profile_id: str):
        profile_id = str(profile_id or "").strip()
        if not _safe_profile_id(profile_id):
            return jsonify({"error": "Invalid profile id."}), 400
        payload = request.get_json(silent=True) or {}
        try:
            apply_profile_glossary_action(
                profile_id,
                source_file=str(payload.get("source_file") or ""),
                entry_index=int(payload.get("entry_index")),
                action=str(payload.get("action") or "update"),
                target=str(payload.get("target") or ""),
                rationale=str(payload.get("rationale") or ""),
                review_rationale=str(payload.get("review_rationale") or ""),
                updates=payload.get("updates") if isinstance(payload.get("updates"), Mapping) else {},
                expected_source=str(payload.get("expected_source") or ""),
            )
            profile = load_book_profile(profile_id, allow_missing=False)
        except (TypeError, ValueError):
            return jsonify({"error": "entry_index must be an integer."}), 400
        except ProfileGlossaryConflictError as exc:
            return jsonify({"error": str(exc), "code": "glossary_edit_conflict"}), 409
        except ProfileGlossaryEditError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            current_app.logger.exception("Could not edit profile glossary entry")
            return jsonify({"error": f"Could not edit profile glossary entry: {exc}"}), 500
        return jsonify(_book_profile_glossary_payload(profile))

    @bp.route("/api/book-profiles/<profile_id>/glossary/merge", methods=["POST"])
    def merge_book_profile_glossary_entries(profile_id: str):
        profile_id = str(profile_id or "").strip()
        if not _safe_profile_id(profile_id):
            return jsonify({"error": "Invalid profile id."}), 400
        payload = request.get_json(silent=True) or {}
        raw_indices = payload.get("entry_indices")
        if not isinstance(raw_indices, list):
            return jsonify({"error": "entry_indices must be a list."}), 400
        try:
            expected_sources = {
                int(item.get("entry_index")): str(item.get("source") or "")
                for item in (payload.get("expected_entries") or [])
                if isinstance(item, Mapping) and item.get("entry_index") is not None
            }
            merge_profile_glossary_entries(
                profile_id,
                source_file=str(payload.get("source_file") or ""),
                entry_indices=[int(index) for index in raw_indices],
                target=str(payload.get("target") or ""),
                rationale=str(payload.get("rationale") or ""),
                expected_sources=expected_sources,
            )
            profile = load_book_profile(profile_id, allow_missing=False)
        except (TypeError, ValueError):
            return jsonify({"error": "entry_indices must contain integers."}), 400
        except ProfileGlossaryConflictError as exc:
            return jsonify({"error": str(exc), "code": "glossary_edit_conflict"}), 409
        except ProfileGlossaryEditError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            current_app.logger.exception("Could not merge profile glossary entries")
            return jsonify({"error": f"Could not merge profile glossary entries: {exc}"}), 500
        return jsonify(_book_profile_glossary_payload(profile))

    @bp.route("/api/book-profiles/<profile_id>/glossary/preview-block", methods=["POST"])
    def preview_book_profile_glossary_block(profile_id: str):
        profile_id = str(profile_id or "").strip()
        if not _safe_profile_id(profile_id):
            return jsonify({"error": "Invalid profile id."}), 400
        payload = request.get_json(silent=True) or {}
        text = str(payload.get("text") or "")
        purpose = str(payload.get("purpose") or "translation").strip().lower() or "translation"
        try:
            profile = load_book_profile(profile_id, allow_missing=True)
        except Exception as exc:
            return jsonify({"error": f"Could not load book profile: {exc}"}), 500
        if profile is None:
            return jsonify({"error": "Book profile not found."}), 404

        block = build_profile_glossary_block(
            text,
            {"editorial_mode": "book_profile", "profile_id": profile.profile_id},
            purpose=purpose,
        )
        summary = profile_glossary_match_summary(
            text,
            {"editorial_mode": "book_profile", "profile_id": profile.profile_id},
            purpose=purpose,
        ) or {}
        return jsonify({
            "profile_id": profile.profile_id,
            "purpose": purpose,
            "block": block,
            "matched_count": summary.get("matched_terms", 0),
            "total_terms": summary.get("total_terms", profile.approved_count),
            "approved_count": profile.approved_count,
            "pending_count": profile.pending_count,
            "capped": bool(summary.get("capped")),
        })

    @bp.route("/api/book-profiles/<profile_id>/editorial-map", methods=["GET"])
    def get_book_profile_editorial_map(profile_id: str):
        profile_id = str(profile_id or "").strip()
        if not _safe_profile_id(profile_id):
            return jsonify({"error": "Invalid profile id."}), 400
        try:
            profile = load_book_profile(profile_id, allow_missing=True)
        except Exception as exc:
            return jsonify({"error": f"Could not load book profile: {exc}"}), 500
        if profile is None:
            return jsonify({"error": "Book profile not found."}), 404
        return jsonify({
            "profile": {
                "profile_id": profile.profile_id,
                "name": profile.name,
                "target_locale": profile.target_locale,
                "source_name": profile.raw_config.get("source_name") or "",
                "generated_profile": bool(profile.raw_config.get("generated_profile")),
                "preflight_model": profile.raw_config.get("preflight_model") or "",
            },
            "editorial_map": dict(profile.editorial_artifacts or {}),
            "counts": editorial_artifact_counts(profile.editorial_artifacts),
        })

    @bp.route("/api/book-profiles/<profile_id>/knowledge-base", methods=["GET"])
    def get_book_profile_knowledge_base(profile_id: str):
        profile_id = str(profile_id or "").strip()
        if not _safe_profile_id(profile_id):
            return jsonify({"error": "Invalid profile id."}), 400
        try:
            profile = load_book_profile(profile_id, allow_missing=True)
        except Exception as exc:
            return jsonify({"error": f"Could not load book profile: {exc}"}), 500
        if profile is None:
            return jsonify({"error": "Book profile not found."}), 404
        return jsonify({"knowledge_base": build_profile_knowledge_base(profile).to_dict()})

    @bp.route("/api/book-profiles/<profile_id>/impact-preview", methods=["POST"])
    def preview_book_profile_impact(profile_id: str):
        profile_id = str(profile_id or "").strip()
        if not _safe_profile_id(profile_id):
            return jsonify({"error": "Invalid profile id."}), 400
        try:
            profile = load_book_profile(profile_id, allow_missing=True)
        except Exception as exc:
            return jsonify({"error": f"Could not load book profile: {exc}"}), 500
        if profile is None:
            return jsonify({"error": "Book profile not found."}), 404
        try:
            text, source_name = _collect_profile_impact_text()
        except ProfilePrepRequestError as exc:
            return jsonify({"error": str(exc)}), exc.status_code
        payload = request.form if (request.content_type or "").lower().startswith("multipart/form-data") else (request.get_json(silent=True) or {})
        purpose = str(payload.get("purpose") or "translation").strip().lower() or "translation"
        preview = build_profile_impact_preview(profile, text, purpose=purpose)
        preview["source_name"] = source_name
        return jsonify(preview)

    @bp.route("/api/book-profiles/prepare", methods=["POST"])
    def prepare_book_profile():
        try:
            payload = _collect_profile_prep_payload()
            result = _run_profile_preparation(payload)
            return jsonify({
                "profile": result.to_dict(),
                "message": "Profile prepared.",
            }), 201
        except ProfilePrepRequestError as exc:
            return jsonify({"error": str(exc)}), exc.status_code
        except Exception as exc:
            current_app.logger.exception("Profile preparation request failed before job creation")
            return jsonify({"error": str(exc)}), 500

    @bp.route("/api/book-profiles/prepare-jobs", methods=["POST"])
    def start_prepare_book_profile_job():
        try:
            payload = _collect_profile_prep_payload()
        except ProfilePrepRequestError as exc:
            return jsonify({"error": str(exc)}), exc.status_code
        except Exception as exc:
            current_app.logger.exception("Profile preparation job request failed before job creation")
            return jsonify({"error": str(exc)}), 500

        request_fingerprint = _profile_prep_request_fingerprint(payload)
        with _PROFILE_PREP_LOCK:
            _cleanup_profile_prep_jobs_locked()
            existing = next(
                (
                    item for item in _PROFILE_PREP_JOBS.values()
                    if item.get("request_fingerprint") == request_fingerprint
                    and item.get("status") in {"queued", "running"}
                ),
                None,
            )
            if existing is not None:
                snapshot = _public_profile_prep_job_snapshot(existing)
                snapshot["status_url"] = f"/api/book-profiles/prepare-jobs/{snapshot['prep_id']}"
                snapshot["recovered_existing"] = True
                return jsonify(snapshot), 202

            prep_id = uuid.uuid4().hex
            job = _new_profile_prep_job(
                prep_id,
                payload,
                request_fingerprint=request_fingerprint,
            )
            _PROFILE_PREP_JOBS[prep_id] = job

        thread = threading.Thread(
            target=_run_profile_prep_job,
            args=(prep_id, payload),
            name=f"profile-prep-{prep_id[:8]}",
            daemon=True,
        )
        thread.start()

        snapshot = _profile_prep_job_snapshot(prep_id)
        snapshot["status_url"] = f"/api/book-profiles/prepare-jobs/{prep_id}"
        return jsonify(snapshot), 202

    @bp.route("/api/book-profiles/prepare-jobs", methods=["GET"])
    def list_prepare_book_profile_jobs():
        statuses = {
            item.strip().lower()
            for item in str(request.args.get("statuses") or "").split(",")
            if item.strip()
        }
        try:
            limit = max(1, min(int(request.args.get("limit") or 20), 100))
        except (TypeError, ValueError):
            return jsonify({"error": "Profile preparation job limit must be an integer."}), 400
        jobs = _profile_prep_job_list(statuses=statuses, limit=limit)
        return jsonify({"jobs": jobs})

    @bp.route("/api/book-profiles/prepare-jobs/<prep_id>", methods=["GET"])
    def get_prepare_book_profile_job(prep_id: str):
        snapshot = _profile_prep_job_snapshot(prep_id)
        if not snapshot:
            return jsonify({"error": "Profile preparation job not found."}), 404
        return jsonify(snapshot)

    return bp


def _safe_profile_id(profile_id: str) -> bool:
    return bool(profile_id and _SAFE_PROFILE_ID_RE.fullmatch(profile_id))


def _profile_can_be_deleted(profile_id: str) -> bool:
    if not _safe_profile_id(profile_id) or profile_id.startswith("_") or profile_id == "common":
        return False
    root = resolve_profiles_root()
    config = _read_yaml_mapping_safe(root / profile_id / "profile.yml")
    if bool(config.get("protected_profile")):
        return False
    return True


def _broken_profile_row(config_path, *, load_error: str, state_manager=None) -> dict[str, Any]:
    profile_id = config_path.parent.name
    config = _read_yaml_mapping_safe(config_path)
    resolved_id = str(config.get("profile_id") or profile_id).strip() or profile_id
    return {
        "profile_id": resolved_id,
        "name": str(config.get("name") or resolved_id),
        "target_locale": str(config.get("target_locale") or ""),
        "approved_count": 0,
        "pending_count": 0,
        "profile_goal": _profile_goal_from_config(config),
        "generated_profile": bool(config.get("generated_profile")),
        "source_name": config.get("source_name") or "",
        "preflight_model": config.get("preflight_model") or "",
        "editorial_artifact_counts": {},
        "deletable": _profile_can_be_deleted(resolved_id),
        "in_use": _profile_in_active_job(resolved_id, state_manager),
        "load_error": load_error,
    }


def _profile_knowledge_summary(profile) -> dict[str, Any]:
    fingerprint = _profile_summary_fingerprint(profile.root)
    return dict(_profile_knowledge_summary_cached(
        profile.profile_id,
        str(profile.root),
        fingerprint,
    ))


@lru_cache(maxsize=128)
def _profile_knowledge_summary_cached(
    profile_id: str,
    profile_root: str,
    fingerprint: tuple[tuple[str, int, int], ...],
) -> dict[str, Any]:
    del fingerprint
    try:
        root = Path(profile_root).parent
        profile = load_book_profile(
            profile_id,
            profiles_root=root,
            allow_missing=True,
        )
        if profile is None:
            return {}
        kb = build_profile_knowledge_base(profile)
    except Exception:
        return {}
    glossary = kb.glossary
    prompt = kb.prompt_readiness
    return {
        "translated_terms": glossary.get("translated_terms", 0),
        "preserve_terms": glossary.get("preserve_terms", 0),
        "source_equals_target_approved": glossary.get("source_equals_target_approved", 0),
        "saturated_buckets": kb.editorial_map.get("saturated_buckets", []),
        "signal_index_available": bool(kb.signal_index.get("available")),
        "signal_risk_flags": kb.signal_index.get("risk_flags", []),
        "ready": prompt.get("ready", False),
        "warnings": prompt.get("warnings", []),
        "recommended_action": prompt.get("recommended_action", ""),
    }


def _profile_summary_fingerprint(
    profile_root: Path,
) -> tuple[tuple[str, int, int], ...]:
    """Track only files that can change a profile's knowledge summary."""
    rows: list[tuple[str, int, int]] = []
    root = Path(profile_root)
    if not root.exists():
        return ()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.casefold() not in {
            ".json",
            ".md",
            ".txt",
            ".yaml",
            ".yml",
        }:
            continue
        try:
            stat_result = path.stat()
        except OSError:
            continue
        rows.append((
            str(path.relative_to(root)),
            int(stat_result.st_mtime_ns),
            int(stat_result.st_size),
        ))
    return tuple(rows)


def _read_yaml_mapping_safe(path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(text) if text.strip() else {}
    except Exception:
        return {}
    return dict(data) if isinstance(data, Mapping) else {}


def _profile_goal_from_config(config: Mapping[str, Any]) -> str:
    rules = config.get("business_rules")
    if isinstance(rules, Mapping):
        explicit = str(config.get("profile_goal") or rules.get("goal") or "").strip()
        if explicit:
            return explicit
    explicit = str(config.get("profile_goal") or config.get("goal") or "").strip()
    if explicit:
        return explicit
    if config.get("audiobook") or config.get("audio_sanitization"):
        return "audiobook"
    if bool(config.get("generated_profile")) or str(config.get("editorial_mode") or "") == "book_profile":
        return resolve_profile_goal(
            "",
            transform_mode=str(config.get("transform_mode") or ""),
            source_name=str(config.get("source_name") or ""),
            profile_config=config,
        ).key
    return ""


def _profile_in_active_job(profile_id: str, state_manager=None) -> bool:
    if state_manager is None:
        return False
    try:
        translations = state_manager.get_all_translations()
    except Exception:
        translations = getattr(state_manager, "_translations", {})
    if not isinstance(translations, dict):
        return False
    for data in translations.values():
        if not isinstance(data, dict):
            continue
        status = str(data.get("status") or "").strip().lower()
        if status not in _ACTIVE_PROFILE_STATUSES:
            continue
        if _config_mentions_profile(data.get("config") or {}, profile_id):
            return True
    return False


def _config_mentions_profile(value: Any, profile_id: str) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "profile_id" and str(item or "") == profile_id:
                return True
            if isinstance(item, (dict, list, tuple)) and _config_mentions_profile(item, profile_id):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_config_mentions_profile(item, profile_id) for item in value)
    return False


def _collect_profile_prep_payload() -> dict[str, Any]:
    content_type = (request.content_type or "").lower()
    is_multipart = content_type.startswith("multipart/form-data")
    form = request.form if is_multipart else (request.get_json(silent=True) or {})

    language = str(form.get("language") or form.get("target_language") or "Spanish")
    target_locale = str(form.get("target_locale") or ("es-MX" if language.casefold() == "spanish" else ""))
    transform_mode = str(form.get("transform_mode") or "modernize")
    profile_goal = str(form.get("profile_goal") or transform_mode)
    profile_id = str(form.get("profile_id") or "")
    profile_name = str(form.get("profile_name") or "")
    source_name = str(form.get("source_name") or "")

    try:
        requested_max_llm_chunks = _optional_int_form_value(form, "max_llm_chunks")
        requested_llm_chunk_chars = _optional_int_form_value(form, "llm_chunk_chars")
        requested_max_local_terms = _optional_int_form_value(form, "max_local_terms")
    except (TypeError, ValueError) as exc:
        raise ProfilePrepRequestError("Profile preparation limits must be integers.") from exc
    llm_full_coverage = _bool_form_value(form, "llm_full_coverage", True)

    texts: list[str] = []
    filenames: list[str] = []
    if is_multipart:
        uploads = request.files.getlist("files") or list(request.files.values())
        if not uploads:
            raise ProfilePrepRequestError("No files were uploaded.")
        for upload in uploads:
            if not upload or not upload.filename:
                continue
            raw = upload.read(_MAX_PREP_UPLOAD_BYTES + 1)
            if len(raw) > _MAX_PREP_UPLOAD_BYTES:
                raise ProfilePrepRequestError("File too large for profile preparation.", 413)
            extracted = extract_profile_prep_text_from_bytes(raw, upload.filename).strip()
            if extracted:
                texts.append(extracted)
                filenames.append(upload.filename)
    else:
        text = str(form.get("text") or "").strip()
        if text:
            texts.append(text)
            filenames.append(source_name or "pasted-text.txt")

    if not texts:
        raise ProfilePrepRequestError("Could not extract readable text for profile preparation.")

    if not source_name:
        source_name = filenames[0] if len(filenames) == 1 else "combined-book-profile.txt"

    provider_type = str(form.get("provider") or "deepseek").lower()
    model = str(form.get("model") or "deepseek-v4-flash")
    review_model = str(
        form.get("review_model")
        or form.get("term_review_model")
        or ("deepseek-v4-pro" if provider_type == "deepseek" else model)
    )
    default_endpoint = {
        "ollama": _config.API_ENDPOINT,
        "openai": _config.OPENAI_API_ENDPOINT,
        "deepseek": _config.DEEPSEEK_API_ENDPOINT,
        "mistral": _config.MISTRAL_API_ENDPOINT,
        "poe": _config.POE_API_ENDPOINT,
        "nim": _config.NIM_API_ENDPOINT,
    }.get(provider_type, "")
    api_endpoint = str(form.get("api_endpoint") or default_endpoint)
    env_var = f"{provider_type.upper()}_API_KEY"
    if provider_type in {"openai", "nim"}:
        try:
            api_key, _key_source = resolve_api_key_for_endpoint(
                form.get("api_key"),
                env_var,
                endpoint=api_endpoint,
                default_endpoint=default_endpoint,
            )
        except EndpointCredentialError as exc:
            raise ProfilePrepRequestError(str(exc)) from exc
    else:
        api_key = str(form.get("api_key") or "")
        if not api_key or api_key == "__USE_ENV__":
            api_key = os.getenv(env_var) or getattr(_config, env_var, "")

    combined_text = "\n\n".join(texts)
    source_language = str(form.get("source_language") or "").strip()
    source_language_confidence = 0.0
    if not source_language or source_language.casefold() in {"auto", "autodetect", "autodetectar"}:
        detection_sample = _distributed_text_sample(combined_text, 10_000)
        detected_language, source_language_confidence = LanguageDetector.detect_language_from_text(
            detection_sample,
            confidence_threshold=0.55,
        )
        source_language = detected_language or "Auto"
    limits = _profile_prep_limits(
        combined_text,
        profile_goal=profile_goal,
        max_llm_chunks=requested_max_llm_chunks,
        llm_chunk_chars=requested_llm_chunk_chars,
        max_local_terms=requested_max_local_terms,
    )
    goal_rules = resolve_profile_goal(
        profile_goal,
        transform_mode=transform_mode,
        source_name=source_name,
    )

    return {
        "combined_text": combined_text,
        "source_name": source_name,
        "source_language": source_language,
        "source_language_confidence": round(float(source_language_confidence), 4),
        "language": language,
        "target_locale": target_locale,
        "transform_mode": transform_mode,
        "profile_goal": goal_rules.key,
        "profile_goal_label": goal_rules.label,
        "profile_id": profile_id,
        "profile_name": profile_name,
        "provider_type": provider_type,
        "model": model,
        "review_model": review_model,
        "api_key": api_key,
        "api_endpoint": api_endpoint,
        "max_llm_chunks": limits["max_llm_chunks"],
        "llm_chunk_chars": limits["llm_chunk_chars"],
        "max_local_terms": limits["max_local_terms"],
        "llm_full_coverage": llm_full_coverage,
    }


def _distributed_text_sample(text: str, max_chars: int) -> str:
    """Return a bounded beginning/middle/end sample for cheap language detection."""
    value = str(text or "")
    limit = max(300, int(max_chars or 0))
    if len(value) <= limit:
        return value
    third = max(100, limit // 3)
    middle = len(value) // 2
    half = third // 2
    return "\n".join((
        value[:third],
        value[max(0, middle - half):middle + half],
        value[-third:],
    ))[:limit]


def _collect_profile_impact_text() -> tuple[str, str]:
    content_type = (request.content_type or "").lower()
    is_multipart = content_type.startswith("multipart/form-data")
    if is_multipart:
        uploads = request.files.getlist("files") or list(request.files.values())
        texts: list[str] = []
        names: list[str] = []
        for upload in uploads:
            if not upload or not upload.filename:
                continue
            raw = upload.read(_MAX_PREP_UPLOAD_BYTES + 1)
            if len(raw) > _MAX_PREP_UPLOAD_BYTES:
                raise ProfilePrepRequestError("File too large for profile impact preview.", 413)
            extracted = extract_profile_prep_text_from_bytes(raw, upload.filename).strip()
            if extracted:
                texts.append(extracted)
                names.append(upload.filename)
        if not texts:
            raise ProfilePrepRequestError("Could not extract readable text for profile impact preview.")
        return "\n\n".join(texts), names[0] if len(names) == 1 else "uploaded-files"

    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text") or "").strip()
    if text:
        return text, str(payload.get("source_name") or "pasted-text.txt")
    file_path = str(payload.get("file_path") or "").strip()
    if file_path:
        path = _resolve_managed_profile_impact_path(file_path)
        if not path.exists():
            raise ProfilePrepRequestError("File path for profile impact preview does not exist.", 404)
        try:
            return extract_readable_text(path), path.name
        except Exception as exc:
            raise ProfilePrepRequestError(f"Could not extract readable text for profile impact preview: {exc}") from exc
    raise ProfilePrepRequestError("No text or file was provided for profile impact preview.")


def _resolve_managed_profile_impact_path(file_path: str) -> Path:
    """Resolve JSON file paths only inside the app-managed output directory.

    Browser clients normally upload the selected file as multipart data. The
    JSON path variant exists for already uploaded/generated artifacts, but must
    not become a read primitive for arbitrary files on the host Mac.
    """
    path = Path(file_path).expanduser().resolve()
    output_root = Path(_config.OUTPUT_DIR).expanduser().resolve()
    try:
        path.relative_to(output_root)
    except ValueError as exc:
        raise ProfilePrepRequestError(
            "File path for profile impact preview is outside the managed output directory.",
            403,
        ) from exc
    return path


def _int_form_value(form: Any, key: str, default: int) -> int:
    value = form.get(key, None)
    if value is None or value == "":
        return default
    return int(value)


def _optional_int_form_value(form: Any, key: str) -> int | None:
    value = form.get(key, None)
    if value is None or value == "":
        return None
    return int(value)


def _bool_form_value(form: Any, key: str, default: bool) -> bool:
    value = form.get(key, None)
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "si", "sí"}


def _profile_prep_limits(
    text: str,
    *,
    profile_goal: str = "",
    max_llm_chunks: int | None,
    llm_chunk_chars: int | None,
    max_local_terms: int | None,
) -> dict[str, int]:
    chars = max(0, len(text or ""))
    goal_rules = resolve_profile_goal(profile_goal)
    auto_chunk_chars = 12000
    if chars >= 2_000_000:
        auto_chunk_chars = 18000
    elif chars >= 750_000:
        auto_chunk_chars = 16000
    elif chars >= 200_000:
        auto_chunk_chars = 14000

    resolved_chunk_chars = (
        max(2500, min(int(llm_chunk_chars), 30000))
        if llm_chunk_chars is not None
        else auto_chunk_chars
    )
    auto_chunks = max(1, math.ceil(max(1, chars) / resolved_chunk_chars))
    resolved_llm_chunks = (
        max(0, min(int(max_llm_chunks), _MAX_PROFILE_PREP_LLM_CHUNKS))
        if max_llm_chunks is not None
        else max(1, min(auto_chunks, _MAX_PROFILE_PREP_LLM_CHUNKS))
    )

    resolved_local_terms = (
        max(1, min(int(max_local_terms), _MAX_PROFILE_PREP_LOCAL_TERMS))
        if max_local_terms is not None
        else goal_rules.local_terms_limit(chars, hard_cap=_MAX_PROFILE_PREP_LOCAL_TERMS)
    )

    return {
        "max_llm_chunks": resolved_llm_chunks,
        "llm_chunk_chars": resolved_chunk_chars,
        "max_local_terms": resolved_local_terms,
    }


def _run_profile_preparation(
    payload: dict[str, Any],
    progress_callback=None,
):
    usage_token = set_usage_context(
        process_id=payload.get("profile_id"),
        process_type="profile_preparation",
        phase="profile_discovery",
        book_name=payload.get("source_name"),
        input_filename=payload.get("source_name"),
        output_filename=f"{payload.get('profile_id')}/profile.yml",
        provider=payload.get("provider_type"),
        model=payload.get("review_model") or payload.get("model"),
    )
    provider = None
    review_provider = None
    try:
        if payload["max_llm_chunks"] > 0:
            try:
                provider = create_llm_provider(
                    provider_type=payload["provider_type"],
                    model=payload["model"],
                    api_key=payload["api_key"],
                    api_endpoint=payload["api_endpoint"],
                    context_window=1_000_000 if payload["provider_type"] == "deepseek" else None,
                    deepseek_disable_thinking=True,
                )
            except Exception as exc:
                raise ProfilePrepRequestError(
                    f"Could not initialize {payload['provider_type']} profile discovery model: {exc}"
                ) from exc
            try:
                review_provider = create_llm_provider(
                    provider_type=payload["provider_type"],
                    model=payload["review_model"],
                    api_key=payload["api_key"],
                    api_endpoint=payload["api_endpoint"],
                    context_window=1_000_000 if payload["provider_type"] == "deepseek" else None,
                    deepseek_disable_thinking=True,
                )
            except Exception as exc:
                if provider is not None:
                    try:
                        _run_async(provider.close())
                    except Exception:
                        pass
                raise ProfilePrepRequestError(
                    f"Could not initialize {payload['provider_type']} profile term reviewer model: {exc}"
                ) from exc

        async def _prepare():
            try:
                return await prepare_book_profile_from_text(
                    payload["combined_text"],
                    source_name=payload["source_name"],
                    source_language=payload.get("source_language") or "Auto",
                    profile_id=payload["profile_id"],
                    profile_name=payload["profile_name"],
                    language=payload["language"],
                    target_locale=payload["target_locale"],
                    transform_mode=payload["transform_mode"],
                    profile_goal=payload.get("profile_goal") or payload["transform_mode"],
                    llm_provider=provider,
                    term_review_provider=review_provider,
                    provider_name=payload["provider_type"] if provider is not None else "",
                    model=payload["model"] if provider is not None else "",
                    term_review_provider_name=payload["provider_type"] if review_provider is not None else "",
                    term_review_model=payload["review_model"] if review_provider is not None else "",
                    max_local_terms=payload["max_local_terms"],
                    llm_chunk_chars=payload["llm_chunk_chars"],
                    max_llm_chunks=payload["max_llm_chunks"] if provider is not None else 0,
                    llm_full_coverage=bool(payload.get("llm_full_coverage")),
                    progress_callback=progress_callback,
                )
            finally:
                if provider is not None:
                    try:
                        await provider.close()
                    except Exception:
                        pass
                if review_provider is not None and review_provider is not provider:
                    try:
                        await review_provider.close()
                    except Exception:
                        pass

        return _run_async(_prepare())
    finally:
        reset_usage_context(usage_token)


def _new_profile_prep_job(
    prep_id: str,
    payload: dict[str, Any],
    *,
    request_fingerprint: str = "",
) -> dict[str, Any]:
    now = _utc_now_iso()
    return {
        "prep_id": prep_id,
        "request_fingerprint": request_fingerprint,
        "status": "queued",
        "progress": 0,
        "current_stage": "queued",
        "message": "Preparacion de perfil en cola.",
        "source_name": payload["source_name"],
        "provider": payload["provider_type"],
        "model": payload["model"],
        "review_model": payload.get("review_model") or "",
        "profile_goal": payload.get("profile_goal") or "",
        "profile_goal_label": payload.get("profile_goal_label") or "",
        "coverage_mode": "full" if payload.get("llm_full_coverage") else "sampled",
        "max_llm_chunks": payload["max_llm_chunks"],
        "llm_chunk_chars": payload["llm_chunk_chars"],
        "max_local_terms": payload["max_local_terms"],
        "created_at": now,
        "updated_at": now,
        "logs": [{
            "time": now,
            "stage": "queued",
            "message": "Preparacion de perfil en cola.",
            "progress": 0,
        }],
        "result": None,
        "error": "",
    }


def _run_profile_prep_job(prep_id: str, payload: dict[str, Any]) -> None:
    _update_profile_prep_job(
        prep_id,
        status="running",
        progress=1,
        current_stage="started",
        message="Extraccion terminada; DeepSeek Flash leera el libro completo para preparar el perfil.",
    )

    def progress_callback(event: dict[str, Any]) -> None:
        _update_profile_prep_job(
            prep_id,
            status="running",
            progress=event.get("progress"),
            current_stage=event.get("stage"),
            message=event.get("message"),
            event=event,
        )

    try:
        result = _run_profile_preparation(payload, progress_callback=progress_callback)
        _update_profile_prep_job(
            prep_id,
            status="completed",
            progress=100,
            current_stage="completed",
            message="Perfil editorial guardado automaticamente.",
            result={"profile": result.to_dict(), "message": "Profile prepared."},
        )
    except Exception as exc:
        _update_profile_prep_job(
            prep_id,
            status="failed",
            current_stage="failed",
            message=f"No se pudo preparar el perfil: {exc}",
            error=str(exc),
        )


def _update_profile_prep_job(prep_id: str, **updates: Any) -> None:
    now = _utc_now_iso()
    with _PROFILE_PREP_LOCK:
        job = _PROFILE_PREP_JOBS.get(prep_id)
        if not job:
            return
        event = updates.pop("event", None) or {}
        for key, value in updates.items():
            if value is not None:
                job[key] = value
        job["updated_at"] = now
        message = str(job.get("message") or "").strip()
        stage = str(job.get("current_stage") or "").strip()
        progress = int(job.get("progress") or 0)
        if message:
            logs = job.setdefault("logs", [])
            last = logs[-1] if logs else {}
            if last.get("message") != message or last.get("stage") != stage:
                log_item = {
                    "time": now,
                    "stage": stage,
                    "message": message,
                    "progress": progress,
                }
                for key in (
                    "chunk_index", "chunk_total", "llm_suggestions",
                    "local_candidates", "rejected_suggestions", "coverage_mode",
                    "editorial_artifacts",
                    "reviewed_terms", "auto_approved_preserve",
                    "auto_approved_translations", "pending_review",
                    "rejected_noise", "demoted_entries", "llm_calls",
                    "review_model", "term_batch_index", "term_batch_total",
                    "terms_reviewed", "terms_total", "review_decisions",
                    "review_warning", "review_batch_terms",
                ):
                    if key in event:
                        log_item[key] = event[key]
                logs.append(log_item)
                del logs[:-60]


def _profile_prep_job_snapshot(prep_id: str) -> dict[str, Any]:
    with _PROFILE_PREP_LOCK:
        job = _PROFILE_PREP_JOBS.get(prep_id)
        if not job:
            return {}
        return _public_profile_prep_job_snapshot(job)


def _public_profile_prep_job_snapshot(job: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = {
        key: value
        for key, value in job.items()
        if key != "request_fingerprint"
    }
    snapshot["logs"] = list(job.get("logs") or [])
    snapshot["result"] = dict(job.get("result") or {}) if job.get("result") else None
    return snapshot


def _profile_prep_job_list(
    *,
    statuses: set[str] | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    with _PROFILE_PREP_LOCK:
        _cleanup_profile_prep_jobs_locked()
        jobs = [
            _public_profile_prep_job_snapshot(job)
            for job in _PROFILE_PREP_JOBS.values()
            if not statuses or str(job.get("status") or "").lower() in statuses
        ]
    jobs.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
    return jobs[:limit]


def _profile_prep_request_fingerprint(payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for field in (
        "source_name",
        "language",
        "target_locale",
        "transform_mode",
        "profile_goal",
        "profile_id",
        "model",
        "review_model",
    ):
        digest.update(str(payload.get(field) or "").encode("utf-8", errors="replace"))
        digest.update(b"\0")
    digest.update(str(payload.get("combined_text") or "").encode("utf-8", errors="replace"))
    return digest.hexdigest()


def _cleanup_profile_prep_jobs_locked() -> None:
    now = time.time()
    expired: list[str] = []
    for prep_id, job in _PROFILE_PREP_JOBS.items():
        try:
            updated = datetime.fromisoformat(str(job.get("updated_at")).replace("Z", "+00:00")).timestamp()
        except Exception:
            updated = now
        if now - updated > _PROFILE_PREP_JOB_TTL_SECONDS:
            expired.append(prep_id)
    for prep_id in expired:
        _PROFILE_PREP_JOBS.pop(prep_id, None)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_async(coro):
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is None:
        return asyncio.run(coro)

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(coro)).result()


def _book_profile_glossary_payload(profile) -> dict[str, Any]:
    try:
        editable_files = set(editable_profile_glossary_files(profile.profile_id).keys())
    except Exception:
        editable_files = set()
    file_indices: dict[str, int] = {}
    entries: list[dict[str, Any]] = []
    for entry in profile.glossary_entries:
        source_file = entry.source_file or ""
        editable = source_file in editable_files
        entry_index = None
        if editable:
            entry_index = file_indices.get(source_file, 0)
            file_indices[source_file] = entry_index + 1
        entries.append(_profile_glossary_entry_to_api(
            entry,
            entry_index=entry_index,
            editable=editable,
        ))
    approved = [entry for entry in entries if entry.get("status") == "approved"]
    pending = [entry for entry in entries if entry.get("status") == "pending"]
    rejected = [entry for entry in entries if entry.get("status") in {"rejected", "superseded"}]
    counts = {
        "total": len(entries),
        "approved": len(approved),
        "pending": len(pending),
        "rejected": len(rejected),
        "editable": len([entry for entry in entries if entry.get("editable")]),
    }
    return {
        "profile": {
            "profile_id": profile.profile_id,
            "name": profile.name,
            "target_locale": profile.target_locale,
            "source_name": profile.raw_config.get("source_name") or "",
            "generated_profile": bool(profile.raw_config.get("generated_profile")),
            "preflight_model": profile.raw_config.get("preflight_model") or "",
            "editorial_artifact_counts": editorial_artifact_counts(profile.editorial_artifacts),
            "editable_glossary_files": sorted(editable_files),
        },
        "entries": entries,
        "approved": approved,
        "pending": pending,
        "rejected": rejected,
        "counts": counts,
        "total": counts["total"],
        "approved_count": counts["approved"],
        "pending_count": counts["pending"],
    }


def _profile_glossary_entry_to_api(
    entry,
    *,
    entry_index: int | None = None,
    editable: bool = False,
) -> dict[str, Any]:
    data = entry.to_dict()
    data["target"] = entry.target
    data["render_target"] = entry.render_target
    data["source_file"] = entry.source_file
    data["entry_index"] = entry_index
    data["editable"] = bool(editable)
    data["entry_key"] = (
        f"{entry.source_file}:{entry_index}"
        if entry.source_file and entry_index is not None
        else ""
    )
    return data
