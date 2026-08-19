"""
Core translation modules
"""

__all__ = [
    'split_text_into_chunks',
    'generate_translation_request',
    'translate_epub_file'
]


def __getattr__(name):
    if name == 'split_text_into_chunks':
        from .text_processor import split_text_into_chunks
        return split_text_into_chunks
    if name == 'generate_translation_request':
        from .translator import generate_translation_request
        return generate_translation_request
    if name == 'translate_epub_file':
        from .epub import translate_epub_file
        return translate_epub_file
    raise AttributeError(f"module 'src.core' has no attribute {name!r}")
