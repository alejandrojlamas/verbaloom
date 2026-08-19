"""
Translation extraction from LLM responses.

This module provides utilities for extracting translations from LLM responses,
handling various response formats including thinking blocks.
"""

import re
from typing import Optional

from src.utils.text_encoding import strip_translation_wrapper_artifacts


class TranslationExtractor:
    """
    Extracts translation text from LLM responses.

    Handles:
        - Extraction between custom tags (e.g., <Translated>...</Translated>)
        - Removal of <think>...</think> blocks
        - Fallback to raw response if tags not found
        - Various edge cases and malformed responses

    Example:
        >>> extractor = TranslationExtractor("<Translated>", "</Translated>")
        >>> response = "<think>reasoning</think><Translated>Hello</Translated>"
        >>> extractor.extract(response)
        'Hello'
    """

    def __init__(self, tag_in: str, tag_out: str):
        """
        Initialize the extractor with custom tags.

        Args:
            tag_in: Opening tag (e.g., "<Translated>")
            tag_out: Closing tag (e.g., "</Translated>")
        """
        self._tag_in = tag_in
        self._tag_out = tag_out
        # Pre-compile regex for efficiency
        self._compiled_regex = self._compile_extraction_regex()
        self._fuzzy_wrapper_regex = self._compile_fuzzy_wrapper_regex()

    def _compile_extraction_regex(self) -> re.Pattern:
        """
        Compile regex pattern for extraction.

        Returns:
            Compiled regex pattern
        """
        return re.compile(
            rf"{re.escape(self._tag_in)}(.*?){re.escape(self._tag_out)}",
            re.DOTALL
        )

    def _compile_fuzzy_wrapper_regex(self) -> re.Pattern:
        """
        Compile a conservative fallback for malformed wrapper tags.

        Some models duplicate the tail of the required TRANSLATION tag, yielding
        wrappers like ``<TRANSLATIONATION>``. This fallback only broadens the
        configured translation wrapper; it does not accept arbitrary XML-like
        tags, because technical books may legitimately contain code tags.
        """
        opening_tag_match = re.match(r'<(\w+)>', self._tag_in)
        closing_tag_match = re.match(r'</(\w+)>', self._tag_out)
        if not opening_tag_match or not closing_tag_match:
            return self._compiled_regex

        opening_name = opening_tag_match.group(1)
        closing_name = closing_tag_match.group(1)
        if opening_name.upper() != "TRANSLATION" or closing_name.upper() != "TRANSLATION":
            return self._compiled_regex

        fuzzy_name = r'TRANSLATION(?:ATION)*'
        return re.compile(
            rf'<\s*{fuzzy_name}\s*>(.*)</\s*{fuzzy_name}\s*>',
            re.DOTALL | re.IGNORECASE,
        )

    def _clean_extracted(self, content: str) -> str:
        """Clean protocol wrappers that survived inside extracted content."""
        return strip_translation_wrapper_artifacts(content).strip()

    def extract(self, response: str) -> Optional[str]:
        """
        Extract translation from response using configured tags with strict validation.

        Returns the content between tag_in and tag_out.
        Prefers responses where tags are at exact boundaries for better reliability.

        NOTE: This method completely ignores content within <think></think> tags,
        as these are used by certain LLMs for internal reasoning and should not
        be searched for translation tags.

        Args:
            response: Raw LLM response text

        Returns:
            Extracted translation, or None if extraction fails
        """
        if not response:
            return None

        # Trim whitespace from response
        response = response.strip()
        original_length = len(response)

        # Remove all <think>...</think> blocks completely
        response = self._remove_think_blocks(response)

        # Remove markdown code block wrappers (some providers like Gemini may wrap in ```)
        response = self._remove_markdown_code_blocks(response)

        response = response.strip()

        if len(response) < original_length:
            print(f"[DEBUG] Think blocks removed: {original_length} -> {len(response)} chars (-{original_length - len(response)})")
            print(f"[DEBUG] Response after think removal (first 200 chars): {response[:200]}")

        # STRICT VALIDATION: Check if response starts and ends with correct tags
        starts_correctly = response.startswith(self._tag_in)
        ends_correctly = response.endswith(self._tag_out)

        if starts_correctly and ends_correctly:
            # Perfect format - extract content between boundary tags
            content = response[len(self._tag_in):-len(self._tag_out)]
            return self._clean_extracted(content)

        # FALLBACK: Try regex search for tags anywhere in response (less strict)
        match = self._compiled_regex.search(response)
        if match:
            extracted = self._clean_extracted(match.group(1))

            # Warn if extraction was from middle of response (indicates LLM didn't follow instructions)
            if not starts_correctly or not ends_correctly:
                print(f"⚠️  Warning: Translation tags found but not at response boundaries.")
                print(f"   Response started with tags: {starts_correctly}")
                print(f"   Response ended with tags: {ends_correctly}")
                print(f"   This may indicate the LLM added extra text. Using extracted content anyway.")

            return extracted

        # FUZZY FALLBACK: Entire answer was wrapped in a malformed translation tag
        # such as <TRANSLATIONATION>...</TRANSLATIONATION>.
        fuzzy_wrapper_match = self._fuzzy_wrapper_regex.search(response)
        if fuzzy_wrapper_match:
            extracted = self._clean_extracted(fuzzy_wrapper_match.group(1))
            print("[WARN] Fuzzy tag match: translation wrapper tag was malformed.")
            return extracted

        # FUZZY FALLBACK: Opening tag found but closing tag is malformed
        # Some models (e.g. Gemini without thinking) write </TRANATION> instead of </TRANSLATION>
        if starts_correctly:
            # Extract tag name from closing tag (e.g. "TRANSLATION" from "</TRANSLATION>")
            closing_tag_match = re.match(r'</(\w+)>', self._tag_out)
            if closing_tag_match:
                tag_name = closing_tag_match.group(1)
                # Look for any closing tag that starts with the same prefix (at least 3 chars)
                prefix = tag_name[:3]
                fuzzy_pattern = re.compile(
                    rf'{re.escape(self._tag_in)}(.*?)</\w*{re.escape(prefix)}\w*>',
                    re.DOTALL
                )
                fuzzy_match = fuzzy_pattern.search(response)
                if fuzzy_match:
                    extracted = self._clean_extracted(fuzzy_match.group(1))
                    print(f"[WARN] Fuzzy tag match: closing tag was malformed, extracted content using prefix '</{prefix}...'")
                    return extracted

        # No tags found at all
        return None

    def _remove_markdown_code_blocks(self, response: str) -> str:
        """
        Remove markdown code block wrappers from response.

        Some providers (notably Gemini) may wrap responses in markdown code blocks
        like ```xml\\n...\\n``` which prevents tag extraction.

        Args:
            response: Text potentially wrapped in markdown code blocks

        Returns:
            Text with markdown code block wrappers removed
        """
        # Match ```lang\n...\n``` wrapping the entire response
        match = re.match(r'^```\w*\s*\n(.*?)\n```\s*$', response, re.DOTALL)
        if match:
            return match.group(1)
        return response

    def _remove_think_blocks(self, response: str) -> str:
        """
        Remove all <think>...</think> blocks from response.

        IMPORTANT: These blocks contain LLM's internal reasoning and should be completely ignored.
        Do NOT search for translation tags inside these blocks!

        Args:
            response: Text potentially containing think blocks

        Returns:
            Text with think blocks removed

        Note:
            Handles nested tags and various formatting styles
        """
        # Case 1: Complete <think>...</think> blocks
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)

        # Case 2: Orphan closing tag </think> (when Ollama truncates the opening tag)
        # Remove everything from the beginning up to and including </think>,
        # BUT only if the prefix does not contain the opening translation tag.
        before_orphan_removal = response
        think_match = re.search(r'</think>', response, flags=re.IGNORECASE)
        if think_match:
            prefix = response[:think_match.end()]
            if self._tag_in.lower() not in prefix.lower():
                response = response[think_match.end():].lstrip()

        if before_orphan_removal != response:
            removed_length = len(before_orphan_removal) - len(response)
            print(f"[DEBUG] Orphan </think> detected - removed {removed_length} characters from beginning")

        return response
