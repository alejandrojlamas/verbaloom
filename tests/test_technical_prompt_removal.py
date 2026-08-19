"""
Test that technical content prompt section is NOT added when protect_technical=True.

With the new placeholder-based protection system, technical content (code, formulas,
HTML entities) is hidden from the LLM in placeholders. Therefore, the system prompt
section instructing the LLM to "not translate code" is obsolete and should not be added.
"""

import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prompts.prompts import (
    generate_translation_prompt,
    TECHNICAL_CONTENT_SECTION,
    _build_optional_prompt_sections
)


def test_technical_section_not_added_with_protection():
    """Test that technical content section is NOT added when preserve_technical_content=True."""

    # Build optional sections with technical protection enabled
    prompt_options = {'preserve_technical_content': True}
    optional_sections = _build_optional_prompt_sections(prompt_options)

    # The technical content section should NOT be in the prompt
    assert TECHNICAL_CONTENT_SECTION not in optional_sections, (
        "Technical content section should NOT be added - content is now protected via placeholders"
    )

    # The optional sections should be empty (no sections added)
    assert optional_sections == "", (
        "No optional sections should be added when only preserve_technical_content=True"
    )


def test_text_cleanup_still_works():
    """Test that text_cleanup section still works independently."""

    # Build optional sections with text cleanup enabled
    prompt_options = {'text_cleanup': True}
    optional_sections = _build_optional_prompt_sections(prompt_options)

    # Text cleanup section SHOULD be present
    assert "TEXT CLEANUP" in optional_sections
    assert "OCR errors" in optional_sections


def test_combined_options():
    """Test that preserve_technical_content doesn't interfere with other options."""

    # Build optional sections with both options
    prompt_options = {
        'preserve_technical_content': True,
        'text_cleanup': True
    }
    optional_sections = _build_optional_prompt_sections(prompt_options)

    # Only text cleanup should be present
    assert TECHNICAL_CONTENT_SECTION not in optional_sections
    assert "TEXT CLEANUP" in optional_sections


def test_full_prompt_generation():
    """Test full prompt generation with preserve_technical_content=True."""

    prompt_pair = generate_translation_prompt(
        main_content="Translate this text",
        context_before="",
        context_after="",
        previous_translation_context="",
        source_language="English",
        target_language="French",
        has_placeholders=True,
        prompt_options={'preserve_technical_content': True}
    )

    # Technical content section should NOT be in system prompt
    assert TECHNICAL_CONTENT_SECTION not in prompt_pair.system, (
        "Technical content section should NOT be in system prompt"
    )

    # But placeholder instructions SHOULD still be present
    assert "placeholder" in prompt_pair.system.lower(), (
        "Placeholder instructions should still be present"
    )


def test_backward_compatibility():
    """Test that the flag still exists and doesn't break anything."""

    # The flag should still be accepted (even if it does nothing now)
    prompt_options = {'preserve_technical_content': True}

    # Should not raise any errors
    optional_sections = _build_optional_prompt_sections(prompt_options)

    # Should return empty string (no sections added)
    assert isinstance(optional_sections, str)


def test_rationale_documentation():
    """Test that the code documents why the section is not added."""

    # Read the source code to verify documentation
    source_file = Path(__file__).parent.parent / "src" / "prompts" / "prompts.py"
    source_code = source_file.read_text(encoding='utf-8')

    # Verify that there's a comment explaining why the section is not added
    assert "placeholder system" in source_code.lower(), (
        "Code should document that placeholder system handles protection"
    )

    assert "hidden in placeholders" in source_code.lower(), (
        "Code should explain that content is hidden in placeholders"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
