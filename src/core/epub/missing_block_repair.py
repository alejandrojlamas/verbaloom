"""Selective source-aware repair for meaningful EPUB blocks omitted in output."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Mapping

from src.core.fidelity_supervisor import target_language_gate_issues
from src.core.llm.factory import create_llm_provider
from src.core.llm.request_deadline import await_llm_call
from src.core.llm_output_guard import guard_llm_output

from .dom_boundaries import (
    MissingTextBlock,
    apply_epub_missing_text_replacements,
    find_epub_missing_text_blocks,
)


@dataclass
class MissingBlockRepairReport:
    found: int = 0
    repaired: int = 0
    rejected: int = 0
    remaining: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return self.remaining == 0 and not self.errors


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def _provider_from_config(config: Mapping[str, Any], log_callback=None):
    provider_name = str(config.get("llm_provider") or "ollama").lower()
    return create_llm_provider(
        provider_name,
        model=config.get("model"),
        api_endpoint=config.get("llm_api_endpoint"),
        openai_api_key=config.get("openai_api_key", ""),
        openrouter_api_key=config.get("openrouter_api_key", ""),
        gemini_api_key=config.get("gemini_api_key", ""),
        mistral_api_key=config.get("mistral_api_key", ""),
        deepseek_api_key=config.get("deepseek_api_key", ""),
        poe_api_key=config.get("poe_api_key", ""),
        nim_api_key=config.get("nim_api_key", ""),
        context_window=config.get("context_window", 4096),
        log_callback=log_callback,
    )


def _repair_prompt(
    finding: MissingTextBlock,
    *,
    source_language: str,
    target_language: str,
    target_locale: str,
) -> tuple[str, str]:
    locale_suffix = f" ({target_locale})" if target_locale else ""
    system = (
        "You repair one omitted block in an otherwise complete book translation. "
        f"Translate the SOURCE BLOCK from {source_language} into "
        f"{target_language}{locale_suffix}. "
        "Preserve every fact, name, number and nuance. Do not summarize, explain, "
        "merge with neighboring text, or return the source unchanged. If the source is "
        "grammatically damaged, render the smallest faithful intelligible equivalent. "
        "Return only the repaired target-language block, without labels or markup."
    )
    user = (
        "CONTEXT BEFORE (read only; do not translate):\n"
        f"{finding.context_before or '(none)'}\n\n"
        "SOURCE BLOCK TO REPAIR:\n"
        f"{finding.source_text}\n\n"
        "CONTEXT AFTER (read only; do not translate):\n"
        f"{finding.context_after or '(none)'}"
    )
    return system, user


async def repair_epub_missing_blocks_with_llm(
    source_epub: str,
    output_epub: str,
    *,
    config: Mapping[str, Any],
    log_callback=None,
    provider=None,
    max_blocks: int = 16,
    timeout: int = 180,
) -> MissingBlockRepairReport:
    """Repair only source-proven, meaningful blocks that are empty in output."""
    findings = find_epub_missing_text_blocks(
        source_epub,
        output_epub,
        limit=max(1, int(max_blocks)) + 1,
    )
    report = MissingBlockRepairReport(found=len(findings))
    if not findings:
        return report
    if len(findings) > max_blocks:
        report.errors.append(
            f"missing block repair cap exceeded ({len(findings)} > {max_blocks})"
        )
        findings = findings[:max_blocks]

    source_language = str(config.get("source_language") or "source language")
    target_language = str(config.get("target_language") or "target language")
    options = dict(config.get("prompt_options") or {})
    target_locale = str(options.get("target_locale") or "")
    llm = provider or _provider_from_config(config, log_callback=log_callback)
    owns_provider = provider is None
    replacements: dict[tuple[str, int], str] = {}
    try:
        for finding in findings:
            system_prompt, user_prompt = _repair_prompt(
                finding,
                source_language=source_language,
                target_language=target_language,
                target_locale=target_locale,
            )
            try:
                request_budget = max(30, min(int(timeout), 300))
                response = await await_llm_call(
                    llm.generate,
                    user_prompt,
                    provider=llm,
                    request_timeout=request_budget,
                    deadline=request_budget,
                    system_prompt=system_prompt,
                    temperature=0.1,
                )
                candidate = guard_llm_output(
                    response.content if response else "",
                    phase="epub_missing_block_repair",
                ).text.strip()
                gate_issues = target_language_gate_issues(
                    finding.source_text,
                    candidate,
                    source_language=source_language,
                    target_language=target_language,
                    phase="epub_missing_block_repair",
                    prompt_options={"target_language_gate": True},
                )
                source_words = re.findall(r"[^\W\d_]+", finding.source_text, re.UNICODE)
                exact_echo = (
                    source_language.casefold() != target_language.casefold()
                    and len(source_words) > 1
                    and _normalized_text(candidate) == _normalized_text(finding.source_text)
                )
                if not candidate or gate_issues or exact_echo:
                    report.rejected += 1
                    codes = [str(getattr(issue, "code", "language_gate")) for issue in gate_issues]
                    report.errors.append(
                        f"{finding.file_href} block {finding.block_index}: rejected "
                        f"({', '.join(codes) or 'empty_or_source_echo'})"
                    )
                    continue
                replacements[(finding.file_href, finding.block_index)] = candidate
            except Exception as exc:
                report.rejected += 1
                report.errors.append(
                    f"{finding.file_href} block {finding.block_index}: "
                    f"{type(exc).__name__}: {exc}"
                )
        report.repaired = apply_epub_missing_text_replacements(
            source_epub,
            output_epub,
            replacements,
        )
        report.remaining = len(
            find_epub_missing_text_blocks(source_epub, output_epub, limit=max_blocks + 1)
        )
        return report
    finally:
        if owns_provider:
            try:
                await llm.close()
            except Exception:
                pass
