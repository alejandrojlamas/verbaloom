"""
Unit tests for generate_ner_extraction_prompt.

Verifies the prompt builder embeds the source/target languages, declares the
required NER tags and category labels, and wraps the input text between the
configured SOURCE_TEXT delimiters without truncation.
"""
import pytest
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.prompts.prompts import generate_ner_extraction_prompt, PromptPair
from src.config import INPUT_TAG_IN, INPUT_TAG_OUT


class TestGenerateNerExtractionPrompt:
    """Tests for the NER extraction prompt builder."""

    def test_returns_prompt_pair_with_non_empty_fields(self):
        """Returns a PromptPair with non-empty system and user prompts."""
        prompt = generate_ner_extraction_prompt("Some sample text", "Chinese", "English")
        assert isinstance(prompt, PromptPair)
        assert prompt.system
        assert prompt.user

    def test_languages_appear_in_system_prompt(self):
        """Source and target language names appear verbatim in the system prompt."""
        prompt = generate_ner_extraction_prompt("Some sample text", "Chinese", "English")
        assert "Chinese" in prompt.system
        assert "English" in prompt.system

    def test_system_prompt_mentions_required_tags(self):
        """System prompt names the required <NER_JSON> and </NER_JSON> wrappers."""
        prompt = generate_ner_extraction_prompt("Some sample text", "Chinese", "English")
        assert "<NER_JSON>" in prompt.system
        assert "</NER_JSON>" in prompt.system

    def test_system_prompt_enumerates_all_categories(self):
        """System prompt lists all six allowed category labels."""
        prompt = generate_ner_extraction_prompt("Some sample text", "Chinese", "English")
        for category in ("character", "location", "organization", "item", "title", "other"):
            assert category in prompt.system, f"missing category '{category}' in system prompt"

    def test_user_prompt_embeds_input_between_source_text_tags(self):
        """User prompt embeds the input text between INPUT_TAG_IN and INPUT_TAG_OUT."""
        text = "The quick brown fox jumps over the lazy dog."
        prompt = generate_ner_extraction_prompt(text, "Chinese", "English")
        assert INPUT_TAG_IN in prompt.user
        assert INPUT_TAG_OUT in prompt.user
        # The input text must appear between the two tags.
        start = prompt.user.index(INPUT_TAG_IN) + len(INPUT_TAG_IN)
        end = prompt.user.index(INPUT_TAG_OUT)
        assert start < end
        assert text in prompt.user[start:end]

    def test_long_input_text_is_preserved_fully(self):
        """The prompt builder does not truncate long input text (truncation is upstream)."""
        long_text = "A" * 50000
        prompt = generate_ner_extraction_prompt(long_text, "Chinese", "English")
        assert long_text in prompt.user


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
