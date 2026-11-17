"""HTML tag classification and priority detection.

This module provides utilities for classifying HTML tags by type and
determining split priorities for HTML-aware chunking.
"""


import re


_TAG_NAME = re.compile(r"<(/?)(?:[\w.-]+:)?([\w-]+)(?=[\s/>])", re.IGNORECASE)


class TagClassifier:
    """Classifies HTML tags by type and determines split priorities.

    This class provides methods to identify block-level tags, determine
    split priorities for chunking, and detect chapter headings.
    """

    BLOCK_TAGS = {'p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                  'blockquote', 'section', 'article', 'li', 'tr', 'td', 'th',
                  'figure', 'figcaption', 'caption'}

    CHAPTER_HEADINGS = {'h1', 'h2', 'h3'}
    MAJOR_SECTIONS = {'h4', 'h5', 'h6', 'section', 'article'}
    PARAGRAPHS = {'p', 'div', 'blockquote'}

    def block_boundaries(self, group: str):
        for match in _TAG_NAME.finditer(group):
            name = match.group(2).lower()
            if name in self.BLOCK_TAGS:
                yield match.group(1), name

    def get_split_priority(self, tag: str) -> int:
        """Get priority for splitting at this tag.

        Lower number = higher priority (preferred split point).

        Priority levels:
        1: Chapter headings (h1, h2, h3)
        2: Major sections (h4, h5, h6, section, article)
        3: Paragraphs and divs (p, div, blockquote)
        4: Other blocks (li, tr, td, th)

        Args:
            tag: HTML tag string (e.g., "</p>", "<div>")

        Returns:
            Priority level (1-4)
        """
        closing = {name for slash, name in self.block_boundaries(tag) if slash}
        if closing & self.CHAPTER_HEADINGS:
            return 1
        if closing & self.MAJOR_SECTIONS:
            return 2
        if closing & self.PARAGRAPHS:
            return 3
        return 4

    def is_block_closing_tag(self, tag: str) -> bool:
        """Check if tag is a block closing tag.

        Examples: "</p>", "</div>", "</h1>"

        Args:
            tag: HTML tag string

        Returns:
            True if tag is a block closing tag
        """
        return any(slash for slash, _name in self.block_boundaries(tag))

    def is_block_opening_tag(self, tag: str) -> bool:
        """Check if tag is a block opening tag.

        Examples: "<p>", "<div>", "<h1 class='title'>"

        Args:
            tag: HTML tag string

        Returns:
            True if tag is a block opening tag
        """
        return any(not slash for slash, _name in self.block_boundaries(tag))

    def is_chapter_heading(self, tag: str) -> bool:
        """Check if tag is a chapter heading (h1-h3).

        Chapter headings are high-priority split points as they typically
        mark major structural boundaries in EPUB content.

        Args:
            tag: HTML tag string

        Returns:
            True if tag is a chapter heading
        """
        return any(slash and name in self.CHAPTER_HEADINGS for slash, name in self.block_boundaries(tag))
