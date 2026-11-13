"""Task-local execution scope for explicit roles and durable request budgets."""

import logging
from contextvars import ContextVar
from dataclasses import dataclass, field

from .pipeline_config import ENDPOINTS, ROLES, validate_pipeline

active_pipeline = ContextVar("active_pipeline", default=None)
request_role = ContextVar("pipeline_request_role", default=None)


def selected_role(role):
    """Effective request identity for reports and model helpers, without I/O."""
    runtime = active_pipeline.get()
    return dict(runtime.config["roles"][role]) if runtime is not None else {}


@dataclass
class PipelineRuntime:
    config: dict
    credentials: dict = field(repr=False)
    providers: dict = field(default_factory=dict, repr=False)
    budget_scope_ids: tuple = ()
    usage_store: object = field(default=None, repr=False)
    allowed_roles: tuple = ROLES

    def provider_for(self, role):
        from src.core.llm.exceptions import ProviderActionRequiredError
        from src.core.llm.factory import create_llm_provider

        if role not in self.allowed_roles:
            raise ProviderActionRequiredError(
                "An LLM request has no explicit pipeline role.",
                reason="pipeline_role_missing",
                provider="pipeline",
            )
        selected = self.config["roles"][role]
        key = tuple(sorted(selected.items()))
        if key not in self.providers:
            provider = selected["provider"]
            tracked = create_llm_provider(
                provider,
                model=selected["model"],
                api_key=self.credentials[provider],
                api_endpoint=ENDPOINTS[provider],
                reasoning_effort=selected["reasoning_effort"],
                max_output_tokens=selected["max_output_tokens"],
                deepseek_disable_thinking=True,
            )
            self.providers[key] = tracked._wrapped
        return self.providers[key], selected

    def client_for(self, role):
        from src.core.usage.tracking_provider import UsageTrackingProvider

        provider, selected = self.provider_for(role)
        return UsageTrackingProvider(provider, selected["provider"])

    async def close(self):
        providers, self.providers = self.providers, {}
        for provider in providers.values():
            try:
                await provider.close()
            except Exception:  # noqa: BLE001 - cleanup must not replace the job's outcome
                # Transport cleanup must not mask a completed job or its stop reason.
                logging.getLogger(__name__).warning(
                    "Could not close a pipeline provider transport"
                )


def activate_pipeline(
    pipeline,
    *,
    job_id,
    credentials,
    store=None,
    experiment_scope=None,
    roles=None,
    parent_scope=None,
):
    """Validate credentials before dispatch and reuse saved budget scope IDs."""
    if pipeline is None:
        return active_pipeline.set(None)
    config = validate_pipeline(pipeline)
    allowed_roles = ROLES if roles is None else tuple(roles)
    if not allowed_roles or any(role not in ROLES for role in allowed_roles):
        raise ValueError("Invalid execution roles")
    used = {config["roles"][role]["provider"] for role in allowed_roles}
    for provider in used:
        key = credentials.get(provider)
        if not isinstance(key, str) or not key.strip() or key == "__USE_ENV__":
            raise ValueError(f"Missing {provider} credentials for pipeline roles")
    scopes = []
    if config["budget"] is not None:
        from src.core.usage.budget import SpendBudget
        from src.core.usage.store import default_usage_store

        scope = f"job:{job_id}"
        SpendBudget(store or default_usage_store()).configure(scope, **config["budget"])
        scopes.append(scope)
    if experiment_scope:
        scopes.append(experiment_scope)
    if parent_scope:
        scopes.append(parent_scope)
    runtime = PipelineRuntime(
        config,
        {provider: credentials[provider] for provider in used},
        budget_scope_ids=tuple(scopes),
        usage_store=store,
        allowed_roles=allowed_roles,
    )
    return active_pipeline.set(runtime)


async def close_pipeline(token):
    runtime = active_pipeline.get()
    try:
        if runtime is not None:
            await runtime.close()
    finally:
        active_pipeline.reset(token)
