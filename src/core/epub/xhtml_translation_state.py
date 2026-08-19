"""
XHTML Translation State Management

This module provides serializable state management for XHTML translation,
enabling interruption and resume at the chunk level.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from .unit_contract import (
    EPUB_PIPELINE_VERSION,
    EPUB_PROMPT_VERSION,
    TranslationUnitStatus,
    UnitStageStatus,
    invalidate_unit_stages,
    invalidated_stages,
    prompt_versions,
    stage_fingerprints,
    text_sha256,
    validate_resume_prefix,
)


@dataclass
class XHTMLTranslationState:
    """
    Serializable state for partial XHTML translation.

    This class captures the complete translation state at any point during
    XHTML processing, allowing for interruption and exact resume from the
    last translated chunk.
    """

    # Identification
    file_path: str
    translation_id: str
    file_href: str  # Relative path in EPUB (e.g., "OEBPS/chapter1.xhtml")

    # Translation Configuration
    source_language: str
    target_language: str
    model_name: str
    max_tokens_per_chunk: int
    max_retries: int

    # Chunking State
    chunks: List[Dict[str, Any]]  # Complete list of chunks
    # Each chunk contains:
    #   - text: str (with local placeholders)
    #   - local_tag_map: Dict[str, str]
    #   - global_indices: List[int]

    global_tag_map: Dict[str, str]  # Global placeholder → HTML tag mapping
    placeholder_format: Tuple[str, str]  # (prefix, suffix) e.g., ("[[", "]]")

    # Translation Progress
    translated_chunks: List[str]  # Already translated chunks (with global indices)
    current_chunk_index: int  # Next chunk to translate (0-based)

    # Original Document Metadata
    original_body_html: str  # Original body HTML (for reference)
    doc_metadata: Dict[str, Any]  # Namespaces, attributes, etc.

    # Statistics
    stats: Dict[str, Any]  # Serialized TranslationMetrics (file-local)

    # Timestamps
    created_at: str  # ISO 8601 format
    updated_at: str  # ISO 8601 format

    # Global Statistics (for EPUB with multiple XHTML files)
    global_stats: Optional[Dict[str, Any]] = None  # Global stats across all files

    # Options (with defaults - must come after non-default fields)
    prompt_options: Optional[Dict[str, Any]] = None
    literary_continuity_state: Optional[Dict[str, Any]] = None
    bilingual: bool = False
    original_chunks: Optional[List[Dict[str, Any]]] = None  # For bilingual mode

    # Technical Content Protection (always enabled)
    protect_technical: bool = True

    # Strict unit/cache contract. Legacy checkpoints omit these fields and are
    # never accepted as validated strict-state prefixes.
    strict_contract: bool = False
    pipeline_version: str = EPUB_PIPELINE_VERSION
    prompt_version: str = EPUB_PROMPT_VERSION
    config_fingerprint: str = ""
    source_document_hash: str = ""
    stage_fingerprints: Optional[Dict[str, str]] = None
    prompt_versions_by_stage: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize state to JSON-compatible dictionary.

        Returns:
            Dictionary containing all state information
        """
        return {
            'file_path': self.file_path,
            'translation_id': self.translation_id,
            'file_href': self.file_href,
            'source_language': self.source_language,
            'target_language': self.target_language,
            'model_name': self.model_name,
            'max_tokens_per_chunk': self.max_tokens_per_chunk,
            'max_retries': self.max_retries,
            'chunks': self.chunks,
            'global_tag_map': self.global_tag_map,
            'placeholder_format': list(self.placeholder_format),  # Convert tuple to list for JSON
            'translated_chunks': self.translated_chunks,
            'current_chunk_index': self.current_chunk_index,
            'original_body_html': self.original_body_html,
            'doc_metadata': self.doc_metadata,
            'stats': self.stats,
            'prompt_options': self.prompt_options,
            'literary_continuity_state': self.literary_continuity_state,
            'bilingual': self.bilingual,
            'original_chunks': self.original_chunks,
            'protect_technical': self.protect_technical,
            'strict_contract': self.strict_contract,
            'pipeline_version': self.pipeline_version,
            'prompt_version': self.prompt_version,
            'config_fingerprint': self.config_fingerprint,
            'source_document_hash': self.source_document_hash,
            'stage_fingerprints': dict(self.stage_fingerprints or {}),
            'prompt_versions_by_stage': dict(self.prompt_versions_by_stage or {}),
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'global_stats': self.global_stats,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'XHTMLTranslationState':
        """
        Deserialize state from dictionary.

        Args:
            data: Dictionary containing serialized state

        Returns:
            XHTMLTranslationState instance
        """
        return cls(
            file_path=data['file_path'],
            translation_id=data['translation_id'],
            file_href=data['file_href'],
            source_language=data['source_language'],
            target_language=data['target_language'],
            model_name=data['model_name'],
            max_tokens_per_chunk=data['max_tokens_per_chunk'],
            max_retries=data['max_retries'],
            chunks=data['chunks'],
            global_tag_map=data['global_tag_map'],
            placeholder_format=tuple(data['placeholder_format']),  # Convert list back to tuple
            translated_chunks=data['translated_chunks'],
            current_chunk_index=data['current_chunk_index'],
            original_body_html=data['original_body_html'],
            doc_metadata=data['doc_metadata'],
            stats=data['stats'],
            prompt_options=data.get('prompt_options'),
            literary_continuity_state=data.get('literary_continuity_state'),
            bilingual=data.get('bilingual', False),
            original_chunks=data.get('original_chunks'),
            protect_technical=data.get('protect_technical', True),
            strict_contract=data.get('strict_contract', False),
            pipeline_version=data.get('pipeline_version', 'legacy'),
            prompt_version=data.get('prompt_version', 'legacy'),
            config_fingerprint=data.get('config_fingerprint', ''),
            source_document_hash=data.get('source_document_hash', ''),
            stage_fingerprints=dict(data.get('stage_fingerprints') or {}),
            prompt_versions_by_stage=dict(data.get('prompt_versions_by_stage') or {}),
            created_at=data['created_at'],
            updated_at=data['updated_at'],
            global_stats=data.get('global_stats'),
        )

    def validate(self) -> bool:
        """
        Validate the consistency of the state.

        Returns:
            True if state is valid, False otherwise
        """
        # Check that current_chunk_index is within bounds
        if self.current_chunk_index < 0 or self.current_chunk_index > len(self.chunks):
            return False

        # Check that translated_chunks matches current_chunk_index
        if len(self.translated_chunks) != self.current_chunk_index:
            return False

        # Check that placeholder_format is valid
        if not isinstance(self.placeholder_format, tuple) or len(self.placeholder_format) != 2:
            return False

        # Check required fields are not empty
        if not self.file_path or not self.translation_id or not self.file_href:
            return False

        # Check chunks structure
        if not isinstance(self.chunks, list):
            return False

        for chunk in self.chunks:
            if not isinstance(chunk, dict):
                return False
            if 'text' not in chunk or 'local_tag_map' not in chunk or 'global_indices' not in chunk:
                return False

        if self.strict_contract:
            if self.pipeline_version != EPUB_PIPELINE_VERSION:
                return False
            if self.prompt_version != EPUB_PROMPT_VERSION:
                return False
            if not self.config_fingerprint or not self.source_document_hash:
                return False
            fingerprints = self.stage_fingerprints or {
                'translation': self.config_fingerprint,
            }
            if fingerprints.get('translation') != self.config_fingerprint:
                return False
            if self.prompt_versions_by_stage and not isinstance(self.prompt_versions_by_stage, dict):
                return False
            valid_prefix, _reason = validate_resume_prefix(
                self.chunks,
                self.translated_chunks,
                self.current_chunk_index,
            )
            if not valid_prefix:
                return False

        return True

    def reconcile_recoverable_candidate_mismatches(self) -> bool:
        """Repair a narrow interrupted-audit checkpoint inconsistency.

        An audit repair can replace the resumable candidate immediately before
        the audit rejects it. Older checkpoints persisted that latest candidate
        in ``translated_chunks`` but left the unit record pointing at the prior
        candidate. Only a failed review/audit unit is safe to reconcile this
        way; any mismatch in an otherwise successful unit remains invalid.
        """
        if self.current_chunk_index != len(self.translated_chunks):
            return False
        if self.current_chunk_index > len(self.chunks):
            return False

        recoverable: list[tuple[Dict[str, Any], str, str]] = []
        for index in range(self.current_chunk_index):
            chunk = self.chunks[index]
            record = chunk.get("unit") if isinstance(chunk, dict) else None
            if not isinstance(record, dict):
                return False
            if record.get("source_hash") != text_sha256(str(chunk.get("text") or "")):
                return False

            candidate = str(self.translated_chunks[index] or "")
            candidate_hash = text_sha256(candidate)
            if record.get("translation_hash") == candidate_hash:
                continue

            translation_completed = (
                record.get("translation_status") == UnitStageStatus.COMPLETED.value
            )
            unresolved_later_stage = (
                record.get("review_status")
                in {UnitStageStatus.IN_PROGRESS.value, UnitStageStatus.FAILED.value}
                or record.get("audit_status")
                in {UnitStageStatus.IN_PROGRESS.value, UnitStageStatus.FAILED.value}
            )
            if not candidate or not translation_completed or not unresolved_later_stage:
                return False
            recoverable.append((record, candidate, candidate_hash))

        if not recoverable:
            return False

        for record, candidate, candidate_hash in recoverable:
            prior_failure = {
                "failure_reason": record.get("failure_reason"),
                "translation_hash": record.get("translation_hash"),
                "review_status": record.get("review_status"),
                "audit_status": record.get("audit_status"),
            }
            record["translation"] = candidate
            record["translation_hash"] = candidate_hash
            record["translation_status"] = UnitStageStatus.COMPLETED.value
            # The newer candidate was never editorially reviewed, so only the
            # expensive translation stage may be reused.
            record["review"] = None
            record["review_status"] = UnitStageStatus.PENDING.value
            record["audit"] = None
            record["audit_status"] = UnitStageStatus.PENDING.value
            record["status"] = TranslationUnitStatus.TRANSLATED.value
            record["failure_reason"] = None
            record.setdefault("checkpoint_recoveries", []).append(prior_failure)
        return True

    def compatible_with(
        self,
        *,
        config_fingerprint: str,
        source_document_hash: str,
        stage_fingerprints: Optional[Dict[str, str]] = None,
    ) -> bool:
        """Return whether this checkpoint can safely resume the current source/config."""
        saved_fingerprints = self.stage_fingerprints or {
            'translation': self.config_fingerprint,
        }
        current_fingerprints = stage_fingerprints or {
            'translation': config_fingerprint,
        }
        return bool(
            self.strict_contract
            and self.pipeline_version == EPUB_PIPELINE_VERSION
            and self.prompt_version == EPUB_PROMPT_VERSION
            and self.config_fingerprint == config_fingerprint
            and saved_fingerprints.get('translation') == current_fingerprints.get('translation')
            and self.source_document_hash == source_document_hash
            and self.validate()
        )

    def migrate_legacy_automatic_recovery_metadata(
        self,
        *,
        prompt_options: Optional[Dict[str, Any]],
        current_stage_fingerprints: Dict[str, str],
        source_document_hash: str,
        current_max_retries: Optional[int] = None,
    ) -> bool:
        """Reuse a prefix invalidated only by the legacy retry metadata bug.

        Old automatic retries wrote a cycle counter and a generic recovery
        instruction into the editorial prompt.  The counter changed on every
        worker, so an otherwise identical resume looked stale.  This migration
        proves the saved fingerprint against its own options before replacing
        that runtime-only identity with the current stable fingerprints.
        """
        saved_options = dict(self.prompt_options or {})
        custom = str(saved_options.get("custom_instructions") or "")
        legacy_instruction_start = "AUTOMATIC FAILED-CHUNK RECOVERY"
        has_legacy_metadata = bool(
            saved_options.get("automatic_failure_recovery")
            or saved_options.get("automatic_failure_recovery_cycle")
            or legacy_instruction_start in custom
        )
        if not (
            has_legacy_metadata
            and self.strict_contract
            and self.source_document_hash == source_document_hash
            and self.validate()
        ):
            return False

        expected_saved = stage_fingerprints(
            source_language=self.source_language,
            target_language=self.target_language,
            model_name=self.model_name,
            max_tokens_per_chunk=self.max_tokens_per_chunk,
            max_retries=self.max_retries,
            prompt_options=saved_options,
        )
        sanitized_saved_options = dict(saved_options)
        sanitized_saved_options.pop("automatic_failure_recovery", None)
        sanitized_saved_options.pop("automatic_failure_recovery_cycle", None)
        sanitized_custom = str(
            sanitized_saved_options.get("custom_instructions") or ""
        )
        legacy_offset = sanitized_custom.find(legacy_instruction_start)
        if legacy_offset >= 0:
            sanitized_custom = sanitized_custom[:legacy_offset].rstrip()
            if sanitized_custom:
                sanitized_saved_options["custom_instructions"] = sanitized_custom
            else:
                sanitized_saved_options.pop("custom_instructions", None)

        current_options = dict(prompt_options or {})
        try:
            active_max_retries = max(
                1,
                int(
                    self.max_retries
                    if current_max_retries is None
                    else current_max_retries
                ),
            )
        except (TypeError, ValueError):
            return False
        expected_current = stage_fingerprints(
            source_language=self.source_language,
            target_language=self.target_language,
            model_name=self.model_name,
            max_tokens_per_chunk=self.max_tokens_per_chunk,
            max_retries=active_max_retries,
            prompt_options=sanitized_saved_options,
        )
        saved_fingerprints = self.stage_fingerprints or {
            "translation": self.config_fingerprint,
        }
        if not (
            current_options == sanitized_saved_options
            and expected_saved.get("translation") == self.config_fingerprint
            and saved_fingerprints.get("translation") == self.config_fingerprint
            and current_stage_fingerprints.get("translation")
            == expected_current.get("translation")
        ):
            return False

        active_prompt_versions = prompt_versions()
        for chunk in self.chunks:
            record = chunk.get("unit") if isinstance(chunk, dict) else None
            if not isinstance(record, dict):
                continue
            record["stage_fingerprints"] = dict(current_stage_fingerprints)
            record["prompt_versions"] = dict(active_prompt_versions)
            record["runtime_recovery_metadata_migrated"] = True

        self.prompt_options = current_options
        self.max_retries = active_max_retries
        self.config_fingerprint = current_stage_fingerprints["translation"]
        self.stage_fingerprints = dict(current_stage_fingerprints)
        self.prompt_versions_by_stage = dict(active_prompt_versions)
        if isinstance(self.stats, dict):
            self.stats["failed_chunks"] = 0
        if isinstance(self.global_stats, dict):
            self.global_stats["failed_chunks"] = 0
        if isinstance(self.doc_metadata, dict):
            self.doc_metadata["quality_stage"] = "runtime_recovery_metadata_migrated"

        return self.compatible_with(
            config_fingerprint=self.config_fingerprint,
            source_document_hash=source_document_hash,
            stage_fingerprints=current_stage_fingerprints,
        )

    def migrate_corrected_profile_scope(
        self,
        *,
        prompt_options: Optional[Dict[str, Any]],
        current_stage_fingerprints: Dict[str, str],
        source_document_hash: str,
    ) -> bool:
        """Reuse translated text after correcting a cross-book profile.

        The migration is intentionally narrow. It is accepted only when the
        saved translation fingerprint can be reproduced by replacing the
        current profile ID with the internally recorded previous ID. The
        translated text is retained, while review and audit are invalidated so
        the corrected profile must inspect every reused candidate.
        """
        options = dict(prompt_options or {})
        previous_profile_id = str(
            options.get("_profile_scope_corrected_from") or ""
        ).strip()
        current_profile_id = str(options.get("profile_id") or "").strip()
        correction = str(options.get("_profile_scope_correction") or "").strip()
        saved_profile_id = str(
            (self.prompt_options or {}).get("profile_id") or ""
        ).strip()

        if not (
            self.strict_contract
            and correction == "replaced"
            and previous_profile_id
            and current_profile_id
            and previous_profile_id != current_profile_id
            and saved_profile_id == previous_profile_id
            and self.source_document_hash == source_document_hash
            and self.validate()
        ):
            return False

        previous_options = dict(options)
        previous_options["profile_id"] = previous_profile_id
        expected_previous = stage_fingerprints(
            source_language=self.source_language,
            target_language=self.target_language,
            model_name=self.model_name,
            max_tokens_per_chunk=self.max_tokens_per_chunk,
            max_retries=self.max_retries,
            prompt_options=previous_options,
        )
        saved_fingerprints = self.stage_fingerprints or {
            "translation": self.config_fingerprint,
        }
        if not (
            expected_previous.get("translation") == self.config_fingerprint
            and saved_fingerprints.get("translation") == self.config_fingerprint
            and current_stage_fingerprints.get("translation")
        ):
            return False

        active_prompt_versions = prompt_versions()
        for chunk in self.chunks:
            record = chunk.get("unit") if isinstance(chunk, dict) else None
            if not isinstance(record, dict):
                continue
            invalidate_unit_stages(record, ("review", "audit"))
            record["stage_fingerprints"] = dict(current_stage_fingerprints)
            record["prompt_versions"] = dict(active_prompt_versions)
            record["profile_scope_migration"] = {
                "from": previous_profile_id,
                "to": current_profile_id,
                "translation_reused": True,
                "review_and_audit_invalidated": True,
            }

        self.prompt_options = options
        self.config_fingerprint = current_stage_fingerprints["translation"]
        self.stage_fingerprints = dict(current_stage_fingerprints)
        self.prompt_versions_by_stage = dict(active_prompt_versions)
        if isinstance(self.stats, dict):
            self.stats["failed_chunks"] = 0
        if isinstance(self.global_stats, dict):
            self.global_stats["failed_chunks"] = 0
        if isinstance(self.doc_metadata, dict):
            self.doc_metadata["quality_stage"] = "profile_scope_migrated"

        return self.compatible_with(
            config_fingerprint=self.config_fingerprint,
            source_document_hash=source_document_hash,
            stage_fingerprints=current_stage_fingerprints,
        )

    def stages_to_invalidate(self, current: Dict[str, str]) -> Tuple[str, ...]:
        """Return later stages that must be recomputed for this checkpoint."""
        saved = self.stage_fingerprints or {'translation': self.config_fingerprint}
        return invalidated_stages(saved, current)

    def get_progress_percentage(self) -> float:
        """
        Calculate translation progress as percentage.

        Returns:
            Progress percentage (0.0 to 100.0)
        """
        if not self.chunks:
            return 0.0
        return (self.current_chunk_index / len(self.chunks)) * 100.0

    def get_remaining_chunks(self) -> int:
        """
        Get number of remaining chunks to translate.

        Returns:
            Number of chunks remaining
        """
        return len(self.chunks) - self.current_chunk_index

    def __repr__(self) -> str:
        """String representation for debugging."""
        progress = self.get_progress_percentage()
        return (
            f"XHTMLTranslationState("
            f"file_href='{self.file_href}', "
            f"progress={progress:.1f}%, "
            f"chunks={self.current_chunk_index}/{len(self.chunks)}, "
            f"updated_at='{self.updated_at}'"
            f")"
        )
