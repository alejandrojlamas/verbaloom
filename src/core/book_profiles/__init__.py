"""Book-scoped editorial profiles and glossaries."""

from .artifacts import (
    build_editorial_signal_index,
    editorial_artifact_counts,
    editorial_artifact_saturation,
    load_editorial_artifacts,
    load_editorial_signal_index,
    render_editorial_brief,
    write_editorial_signal_index,
)
from .loader import (
    BookProfileError,
    create_profile,
    infer_profile_id_from_metadata,
    load_book_profile,
    profile_exists,
    profile_matches_source_metadata,
    resolve_profiles_root,
)
from .knowledge import build_profile_knowledge_base
from .impact import build_profile_impact_preview
from src.core.editorial_knowledge import build_editorial_knowledge_base
from src.core.editorial_knowledge_compiler import compile_profile_prompt_context
from .migration import migrate_generated_profile_glossaries
from .profile_goals import (
    ProfileGoalRules,
    profile_goal_options,
    resolve_profile_goal,
)
from .rendering import (
    apply_profile_exact_translation_corrections,
    apply_profile_glossary_corrections,
    build_profile_glossary_block,
    build_profile_glossary_context,
    build_profile_instruction_block,
    build_profile_report_summary,
    profile_glossary_match_summary,
    profile_enabled,
)
from .preparation import (
    PreparedProfileResult,
    distributed_discovery_chunks,
    extract_profile_prep_text_from_bytes,
    full_coverage_discovery_chunks,
    prepare_book_profile_from_text,
)

__all__ = [
    "BookProfileError",
    "apply_profile_exact_translation_corrections",
    "apply_profile_glossary_corrections",
    "build_profile_glossary_block",
    "build_profile_glossary_context",
    "build_profile_impact_preview",
    "build_profile_knowledge_base",
    "build_editorial_knowledge_base",
    "build_editorial_signal_index",
    "compile_profile_prompt_context",
    "build_profile_instruction_block",
    "build_profile_report_summary",
    "create_profile",
    "distributed_discovery_chunks",
    "editorial_artifact_counts",
    "editorial_artifact_saturation",
    "extract_profile_prep_text_from_bytes",
    "full_coverage_discovery_chunks",
    "infer_profile_id_from_metadata",
    "load_book_profile",
    "load_editorial_artifacts",
    "load_editorial_signal_index",
    "migrate_generated_profile_glossaries",
    "PreparedProfileResult",
    "ProfileGoalRules",
    "profile_goal_options",
    "prepare_book_profile_from_text",
    "profile_exists",
    "profile_matches_source_metadata",
    "profile_glossary_match_summary",
    "profile_enabled",
    "render_editorial_brief",
    "resolve_profile_goal",
    "resolve_profiles_root",
    "write_editorial_signal_index",
]
