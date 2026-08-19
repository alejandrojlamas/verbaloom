"""Final source-aware audit over bounded document samples.

This pass compares the assembled output against the original source at a few
relative positions. It is deliberately deterministic and non-blocking: it does
not rewrite the book, does not call an LLM, and does not replace chunk-level
fidelity supervision. Its job is to catch whole-artifact risks that only become
visible after assembly, such as prompt leaks near the end, missing numeric
facts, obvious source-language echoes, or severe style discontinuity.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import re
from pathlib import Path
from typing import Any

from src.core.fidelity_supervisor import target_language_gate_issues
from src.core.language_evidence import (
    has_target_language_contextual_evidence,
    looks_like_structured_language_metadata,
)
from src.core.llm_output_guard import guard_llm_output
from src.core.output_formats import extract_readable_text
from src.utils.language_detector import LanguageDetector
from src.utils.text_encoding import mojibake_score


_WORD_RE = re.compile(r"[\wÁÉÍÓÚÜÑáéíóúüñ'-]+", re.UNICODE)
_NUMBER_RE = re.compile(
    r"(?<![\w])(?:\d+(?:[.,]\d+)?(?:\s*(?:x|X|×|\*)\s*10\^?-?\d+)?|10\^?-?\d+)(?![\w])"
)
_PROPER_NOUN_RE = re.compile(
    r"\b[A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÜÑáéíóúüñ'’.-]{2,}(?:\s+"
    r"[A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÜÑáéíóúüñ'’.-]{2,}){0,3}\b"
)
_COMMON_CAPITALIZED = {
    "A",
    "An",
    "And",
    "But",
    "Chapter",
    "Contents",
    "El",
    "En",
    "He",
    "I",
    "It",
    "La",
    "Los",
    "She",
    "Table",
    "The",
    "They",
    "This",
    "We",
}


@dataclass(frozen=True)
class FinalSourceSampleIssue:
    code: str
    severity: str
    sample: str
    message: str
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        data = {
            "code": self.code,
            "severity": self.severity,
            "sample": self.sample,
            "message": self.message,
        }
        if self.detail:
            data["detail"] = self.detail
        return data


@dataclass
class FinalSourceSampleAuditReport:
    source_path: str
    output_path: str
    source_language: str = ""
    target_language: str = ""
    source_characters: int = 0
    output_characters: int = 0
    samples_checked: int = 0
    source_language_blocks: int = 0
    source_language_characters: int = 0
    source_language_ratio: float = 0.0
    issues: list[FinalSourceSampleIssue] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.issues

    @property
    def warning_count(self) -> int:
        return len([issue for issue in self.issues if issue.severity == "warning"])

    @property
    def error_count(self) -> int:
        return len([issue for issue in self.issues if issue.severity in {"error", "reject"}])

    def summary(self) -> str:
        if self.clean:
            return "no source-aware sample warnings"
        return f"{len(self.issues)} source-aware sample warning(s)"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "output_path": self.output_path,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "source_characters": self.source_characters,
            "output_characters": self.output_characters,
            "samples_checked": self.samples_checked,
            "source_language_blocks": self.source_language_blocks,
            "source_language_characters": self.source_language_characters,
            "source_language_ratio": self.source_language_ratio,
            "clean": self.clean,
            "warning_count": self.warning_count,
            "error_count": self.error_count,
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def to_markdown(self) -> str:
        lines = [
            "# Final Source-Aware Sample Audit",
            "",
            f"- Source: {Path(self.source_path).name}",
            f"- Output: {Path(self.output_path).name}",
            f"- Source language: {self.source_language or 'N/A'}",
            f"- Target language: {self.target_language or 'N/A'}",
            f"- Source characters: {self.source_characters}",
            f"- Output characters: {self.output_characters}",
            f"- Samples checked: {self.samples_checked}",
            f"- Source-language blocks in output: {self.source_language_blocks}",
            f"- Source-language characters in output: {self.source_language_characters}",
            f"- Source-language ratio: {self.source_language_ratio:.2%}",
            f"- Status: {'clean' if self.clean else 'warnings'}",
        ]
        if self.issues:
            lines.extend(["", "## Issues"])
            for issue in self.issues:
                detail = f" ({issue.detail})" if issue.detail else ""
                lines.append(
                    f"- [{issue.severity}] {issue.sample}: {issue.code} - {issue.message}{detail}"
                )
        return "\n".join(lines).strip() + "\n"

    def write(self, path: str | Path | None = None) -> Path:
        report_path = Path(path) if path else final_source_sample_audit_path(self.output_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(self.to_markdown(), encoding="utf-8")
        return report_path


@dataclass(frozen=True)
class _OutputLanguageBlock:
    text: str
    document_hint: str = ""


def final_source_sample_audit_path(output_filepath: str | Path) -> Path:
    output = Path(output_filepath)
    return output.with_name(f"{output.stem} - final source sample audit.md")


def audit_final_output_against_source_samples(
    source_filepath: str | Path,
    output_filepath: str | Path,
    *,
    source_language: str = "",
    target_language: str = "",
    write_report: bool = True,
    max_samples: int = 3,
    sample_chars: int = 2400,
) -> FinalSourceSampleAuditReport:
    """Compare source and assembled output at a few relative positions."""
    source_path = Path(source_filepath)
    output_path = Path(output_filepath)
    report = FinalSourceSampleAuditReport(
        source_path=str(source_path),
        output_path=str(output_path),
        source_language=source_language,
        target_language=target_language,
    )
    if not source_path.exists():
        report.issues.append(FinalSourceSampleIssue(
            "source_missing",
            "warning",
            "setup",
            "Source file does not exist; final source-aware audit could not run.",
        ))
        if write_report:
            report.write()
        return report
    if not output_path.exists():
        report.issues.append(FinalSourceSampleIssue(
            "output_missing",
            "warning",
            "setup",
            "Output file does not exist; final source-aware audit could not run.",
        ))
        if write_report:
            report.write()
        return report

    try:
        source_text = extract_readable_text(source_path)
        output_text = extract_readable_text(output_path)
    except Exception as exc:
        report.issues.append(FinalSourceSampleIssue(
            "text_extraction_failed",
            "warning",
            "setup",
            "Could not extract readable text for final source-aware audit.",
            str(exc),
        ))
        if write_report:
            report.write()
        return report

    source_text = source_text or ""
    output_text = output_text or ""
    report.source_characters = len(source_text)
    report.output_characters = len(output_text)
    _audit_full_output_language_coverage(
        report,
        output_text,
        output_path=output_path,
    )
    source_samples = _relative_samples(source_text, max_samples=max_samples, sample_chars=sample_chars)
    output_samples = _relative_samples(output_text, max_samples=max_samples, sample_chars=sample_chars)
    report.samples_checked = min(len(source_samples), len(output_samples))
    if not source_samples:
        report.issues.append(FinalSourceSampleIssue(
            "source_text_empty",
            "warning",
            "setup",
            "Source extraction produced no readable text; sample comparison could not run.",
        ))
    if not output_samples:
        report.issues.append(FinalSourceSampleIssue(
            "output_text_empty",
            "error",
            "setup",
            "Final output contains no readable text; sample comparison could not run.",
        ))
    output_full_folded = _fold(output_text)

    previous_output_sample = ""
    for index, ((label, source_sample), (_out_label, output_sample)) in enumerate(
        zip(source_samples, output_samples),
        start=1,
    ):
        sample_id = f"{index}:{label}"
        _audit_sample_pair(
            report,
            sample_id,
            source_sample,
            output_sample,
            output_full_folded=output_full_folded,
            previous_output_sample=previous_output_sample,
        )
        previous_output_sample = output_sample

    if write_report and report.issues:
        report.write()
    return report


def _audit_full_output_language_coverage(
    report: FinalSourceSampleAuditReport,
    output_text: str,
    *,
    output_path: Path | None = None,
) -> None:
    """Scan every assembled paragraph for substantial source-language residue."""
    source_key = _normalized_language(report.source_language)
    target_key = _normalized_language(report.target_language)
    if not source_key or not target_key or source_key == target_key:
        return

    blocks = _output_language_blocks(output_path, output_text)
    eligible = [item for item in blocks if len(item.text) >= 80]
    if not eligible:
        return

    source_blocks: list[tuple[str, float]] = []
    total_characters = sum(len(item.text) for item in eligible)
    for block in eligible:
        if looks_like_structured_language_metadata(
            block.text,
            document_hint=block.document_hint,
            target_language=target_key,
        ):
            continue
        detected, confidence = LanguageDetector.detect_language_from_text(
            block.text,
            confidence_threshold=0.90,
        )
        if (
            _normalized_language(detected or "") == source_key
            and not has_target_language_contextual_evidence(
                block.text,
                target_language=target_key,
            )
        ):
            source_blocks.append((block.text, float(confidence or 0.0)))

    source_characters = sum(len(block) for block, _confidence in source_blocks)
    ratio = source_characters / max(1, total_characters)
    report.source_language_blocks = len(source_blocks)
    report.source_language_characters = source_characters
    report.source_language_ratio = ratio
    if not source_blocks:
        return

    severe = (
        source_characters >= max(1200, int(total_characters * 0.015))
        or len(source_blocks) >= 3
        or max(len(block) for block, _confidence in source_blocks) >= 1200
    )
    examples = " | ".join(
        re.sub(r"\s+", " ", block)[:100]
        for block, _confidence in source_blocks[:3]
    )
    report.issues.append(FinalSourceSampleIssue(
        "source_language_residual_coverage",
        "error" if severe else "warning",
        "whole_output",
        "The assembled output still contains blocks dominated by the source language.",
        f"blocks={len(source_blocks)}; chars={source_characters}; ratio={ratio:.2%}; examples={examples}",
    ))


def _output_language_blocks(
    output_path: Path | None,
    output_text: str,
) -> list[_OutputLanguageBlock]:
    """Return structure-aware EPUB blocks, with a plain-text fallback."""
    if output_path is not None and output_path.suffix.lower() == ".epub":
        try:
            from src.core.epub.publication_gate import snapshot_epub

            snapshot = snapshot_epub(output_path, recover=True)
            blocks: list[_OutputLanguageBlock] = []
            seen: set[tuple[str, str]] = set()
            for unit in snapshot.units:
                text = re.sub(r"\s+", " ", unit.text or "").strip()
                key = (unit.file_href, unit.dom_path)
                if not text or key in seen:
                    continue
                seen.add(key)
                blocks.append(_OutputLanguageBlock(
                    text=text,
                    document_hint=unit.file_href,
                ))
            if blocks:
                return blocks
        except Exception:
            pass

    return [
        _OutputLanguageBlock(re.sub(r"\s+", " ", item or "").strip())
        for item in re.split(r"\n\s*\n+", output_text or "")
        if item.strip()
    ]


def _normalized_language(value: str) -> str:
    aliases = {
        "de": "german", "deutsch": "german", "alemán": "german", "aleman": "german",
        "es": "spanish", "español": "spanish", "espanol": "spanish",
        "en": "english", "fr": "french", "cs": "czech", "nl": "dutch",
        "it": "italian", "pt": "portuguese", "pl": "polish",
    }
    key = str(value or "").strip().casefold()
    return aliases.get(key, key)


def _audit_sample_pair(
    report: FinalSourceSampleAuditReport,
    sample_id: str,
    source_sample: str,
    output_sample: str,
    *,
    output_full_folded: str,
    previous_output_sample: str,
) -> None:
    source_words = _word_count(source_sample)
    output_words = _word_count(output_sample)
    if source_words >= 80 and output_words >= 10:
        ratio = output_words / max(1, source_words)
        if ratio < 0.35 or ratio > 2.75:
            report.issues.append(FinalSourceSampleIssue(
                "sample_length_ratio_outlier",
                "warning",
                sample_id,
                "Output sample length is far outside the source sample range.",
                f"source_words={source_words}; output_words={output_words}; ratio={ratio:.2f}",
            ))

    gate_issues = target_language_gate_issues(
        source_sample,
        output_sample,
        source_language=report.source_language,
        target_language=report.target_language,
        phase="final_source_sample",
        prompt_options={"target_language_gate": True},
    )
    for issue in gate_issues:
        report.issues.append(FinalSourceSampleIssue(
            issue.code,
            "warning",
            sample_id,
            issue.message,
            issue.detail,
        ))

    guard = guard_llm_output(
        output_sample,
        phase="final_source_sample",
        style_reference=previous_output_sample,
    )
    for issue in guard.issues:
        report.issues.append(FinalSourceSampleIssue(
            issue.code,
            "warning",
            sample_id,
            issue.message,
            issue.detail,
        ))

    sample_mojibake = mojibake_score(output_sample)
    if sample_mojibake:
        report.issues.append(FinalSourceSampleIssue(
            "mojibake_in_output_sample",
            "warning",
            sample_id,
            "Output sample contains mojibake markers.",
            f"score={sample_mojibake}",
        ))

    missing_numbers = _missing_numbers(source_sample, output_sample)
    if missing_numbers:
        report.issues.append(FinalSourceSampleIssue(
            "numbers_missing_in_output_sample",
            "warning",
            sample_id,
            "Some numeric facts from the source sample were not found in the output sample.",
            ", ".join(missing_numbers[:10]),
        ))

    absent_names = _globally_absent_repeated_names(source_sample, output_full_folded)
    if absent_names:
        report.issues.append(FinalSourceSampleIssue(
            "repeated_source_names_absent_globally",
            "warning",
            sample_id,
            "Repeated source names in this sample do not appear anywhere in the final output.",
            ", ".join(absent_names[:10]),
        ))


def _relative_samples(text: str, *, max_samples: int, sample_chars: int) -> list[tuple[str, str]]:
    compact = (text or "").strip()
    if not compact:
        return []
    if len(compact) <= sample_chars:
        return [("whole", compact)]
    if max_samples <= 3:
        anchors = [
            ("start", 0),
            ("middle", max(0, (len(compact) - sample_chars) // 2)),
            ("end", max(0, len(compact) - sample_chars)),
        ][:max_samples]
    else:
        anchors = [
            ("start", 0),
            ("early", len(compact) // 4),
            ("middle", len(compact) // 2),
            ("late", (len(compact) * 3) // 4),
            ("end", max(0, len(compact) - sample_chars)),
        ][:max_samples]
    samples: list[tuple[str, str]] = []
    seen: set[tuple[int, int]] = set()
    for label, anchor in anchors:
        start = max(0, min(anchor, len(compact) - sample_chars))
        end = min(len(compact), start + sample_chars)
        key = (start, end)
        if key in seen:
            continue
        seen.add(key)
        samples.append((label, compact[start:end]))
    return samples


def _missing_numbers(source_sample: str, output_sample: str) -> list[str]:
    source_numbers = Counter(_NUMBER_RE.findall(source_sample or ""))
    if len(source_numbers) < 2:
        return []
    output_numbers = Counter(_NUMBER_RE.findall(output_sample or ""))
    missing: list[str] = []
    for number, count in source_numbers.items():
        deficit = count - output_numbers.get(number, 0)
        if deficit > 0:
            missing.extend([number] * deficit)
    if len(missing) < 2 and len(missing) / max(1, sum(source_numbers.values())) < 0.30:
        return []
    return missing


def _globally_absent_repeated_names(source_sample: str, output_full_folded: str) -> list[str]:
    names = Counter(
        name.strip()
        for name in _PROPER_NOUN_RE.findall(source_sample or "")
        if _proper_name_candidate(name)
    )
    absent = []
    for name, count in names.items():
        if count < 2:
            continue
        if _fold(name) not in output_full_folded:
            absent.append(name)
    return absent


def _proper_name_candidate(name: str) -> bool:
    words = [word.strip(".,;:!?()[]{}") for word in (name or "").split()]
    words = [word for word in words if word]
    if not words or len(words) > 4:
        return False
    if all(word in _COMMON_CAPITALIZED for word in words):
        return False
    if any(any(char.isdigit() for char in word) for word in words):
        return False
    return True


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def _fold(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").casefold()
