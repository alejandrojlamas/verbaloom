"""Settings adapter for the core pipeline contract and non-billable preflight."""

import asyncio

from flask import jsonify, request

from src.core.llm.preflight import preflight_deepseek, preflight_openai
from src.core.pipeline_config import (
    MODELS,
    PRESETS,
    ROLES,
    configured_default_pipeline,
    pipeline_preset,
    validate_pipeline,
)
from src.utils.provider_security import resolve_pipeline_credentials


def register_pipeline_routes(bp):
    def reply(payload, status=200):
        response = jsonify(payload)
        response.headers["Cache-Control"] = "no-store"
        return response, status

    def body():
        value = request.get_json(silent=True)
        if not isinstance(value, dict):
            raise TypeError("Expected a JSON object")
        return value

    @bp.get("/api/pipeline/catalog")
    def get_pipeline_catalog():
        return reply({
            "version": 1, "roles": list(ROLES), "presets": list(PRESETS),
            "models": {key: sorted(value) for key, value in MODELS.items()},
            "default_pipeline": configured_default_pipeline(),
        })

    @bp.post("/api/pipeline/configure")
    def configure_pipeline():
        try:
            data = body()
            if "pipeline" in data:
                pipeline = validate_pipeline(data["pipeline"])
            else:
                baseline = data.get("baseline", {})
                if not isinstance(baseline, dict) or set(baseline) - {
                    "llm_provider", "model", "prompt_options",
                }:
                    raise ValueError("Invalid baseline")
                if "prompt_options" in baseline and not isinstance(baseline["prompt_options"], dict):
                    raise ValueError("Invalid baseline options")
                pipeline = pipeline_preset(data.get("preset"), baseline)
            return reply({"pipeline": pipeline})
        except (ValueError, TypeError, KeyError):
            return reply({"error": "invalid_pipeline"}, 400)

    @bp.post("/api/pipeline/preflight")
    def check_pipeline_access():
        try:
            data = body()
            pipeline = validate_pipeline(data.get("pipeline"))
            selected_roles = data.get("roles", list(ROLES))
            if (not isinstance(selected_roles, list) or not selected_roles
                    or any(not isinstance(role, str) or role not in ROLES for role in selected_roles)
                    or len(set(selected_roles)) != len(selected_roles)):
                raise ValueError("Invalid preflight role scope")
        except (ValueError, TypeError, KeyError):
            return reply({"error": "invalid_pipeline"}, 400)
        reports = []
        for provider in sorted({pipeline["roles"][name]["provider"] for name in selected_roles}):
            roles = [name for name in selected_roles if pipeline["roles"][name]["provider"] == provider]
            credentials = {"pipeline": pipeline}
            try:
                resolve_pipeline_credentials(credentials, data, roles=roles)
            except ValueError:
                return reply({"status": "missing_credentials", "provider": provider,
                              "generation_tested": False, "balance": None})
            models = sorted({pipeline["roles"][name]["model"] for name in roles})
            if provider == "openai":
                result = asyncio.run(preflight_openai(
                    api_key=credentials["openai_api_key"], models=models,
                ))
            else:
                result = asyncio.run(preflight_deepseek(
                    api_key=credentials["deepseek_api_key"], models=models,
                ))
            reports.append(result)
        return reply({
            "status": "models_visible" if all(item["status"] == "models_visible" for item in reports) else "not_ready",
            "providers": reports, "generation_tested": False, "balance": None,
            "roles": selected_roles,
        })
