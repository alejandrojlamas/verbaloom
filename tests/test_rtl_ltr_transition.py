"""
Tests for RTL to LTR transition (e.g., Arabic -> French)

This tests the removal of RTL styles when translating from an RTL language
to an LTR language.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.epub.rtl_support import (
    is_rtl_language,
    remove_rtl_from_html,
    update_opf_for_ltr,
    apply_rtl_to_epub_directory,
)


class TestRTLToLTRTransition:
    """Test RTL to LTR transitions"""

    def test_detect_rtl_to_ltr_transition(self):
        """Should detect RTL -> LTR transition correctly"""
        # Arabic (RTL) -> French (LTR) = transition
        assert is_rtl_language('Arabic') is True
        assert is_rtl_language('French') is False
        
        # Hebrew (RTL) -> English (LTR) = transition
        assert is_rtl_language('Hebrew') is True
        assert is_rtl_language('English') is False
        
        # French (LTR) -> Arabic (RTL) = NOT this case
        assert is_rtl_language('French') is False
        assert is_rtl_language('Arabic') is True

    def test_remove_rtl_from_html_basic(self):
        """Should remove RTL styles and set LTR"""
        html_with_rtl = """<!DOCTYPE html>
<html dir="rtl" lang="ar">
<head>
    <title>Test</title>
    <style type="text/css">
/* RTL Support - Auto-generated for ar */
html, body { direction: rtl !important; text-align: right !important; }
    </style>
</head>
<body>
    <p style="text-align: right;">Arabic text</p>
</body>
</html>"""
        
        result = remove_rtl_from_html(html_with_rtl)
        
        # Should change dir to ltr
        assert 'dir="ltr"' in result
        assert 'dir="rtl"' not in result
        
        # Should have LTR reset CSS
        assert 'direction: ltr' in result
        assert 'text-align: left' in result

    def test_remove_rtl_multiple_styles(self):
        """Should remove multiple RTL style blocks"""
        html_with_multiple_rtl = """<!DOCTYPE html>
<html dir="rtl">
<head>
    <style>/* RTL Support */ body { direction: rtl; }</style>
    <link rel="stylesheet" href="style.css">
    <style>/* RTL Support */ p { text-align: right; }</style>
</head>
<body><p>Text</p></body>
</html>"""
        
        result = remove_rtl_from_html(html_with_multiple_rtl)
        
        # RTL styles should be removed
        assert result.count('direction: rtl') == 0
        
        # LTR should be set
        assert 'dir="ltr"' in result
        assert 'direction: ltr' in result

    def test_apply_rtl_to_epub_rtl_to_ltr(self, tmp_path):
        """Test full RTL->LTR transition workflow"""
        # Create mock EPUB structure with RTL content
        epub_dir = tmp_path / "epub"
        epub_dir.mkdir()
        oebps_dir = epub_dir / "OEBPS"
        oebps_dir.mkdir()
        
        # Create HTML file with RTL
        chapter = oebps_dir / "chapter.xhtml"
        chapter.write_text("""<?xml version="1.0"?>
<html dir="rtl" lang="ar">
<head>
    <style type="text/css">
/* RTL Support */
html { direction: rtl !important; }
    </style>
</head>
<body><p>Arabic text</p></body>
</html>""", encoding='utf-8')
        
        # Create OPF with RTL progression
        opf = oebps_dir / "content.opf"
        opf.write_text("""<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
    <metadata></metadata>
    <manifest></manifest>
    <spine page-progression-direction="rtl"></spine>
</package>""", encoding='utf-8')
        
        # Apply RTL->LTR transition (Arabic -> French)
        result = apply_rtl_to_epub_directory(str(epub_dir), 'French', 'Arabic')
        
        assert result['was_transition'] is True
        assert result['css_removed'] == 1
        assert result['opf_updated'] is True
        assert result['is_rtl'] is False
        
        # Check HTML was updated
        content = chapter.read_text(encoding='utf-8')
        assert 'dir="ltr"' in content
        assert 'direction: ltr' in content
        assert 'direction: rtl' not in content
        
        # Check OPF was updated
        opf_content = opf.read_text(encoding='utf-8')
        assert 'page-progression-direction="ltr"' in opf_content
        assert 'page-progression-direction="rtl"' not in opf_content

    def test_apply_rtl_to_epub_ltr_to_rtl(self, tmp_path):
        """Test LTR->RTL transition workflow"""
        epub_dir = tmp_path / "epub"
        epub_dir.mkdir()
        oebps_dir = epub_dir / "OEBPS"
        oebps_dir.mkdir()
        
        chapter = oebps_dir / "chapter.xhtml"
        chapter.write_text("""<?xml version="1.0"?>
<html>
<head><title>Test</title></head>
<body><p>French text</p></body>
</html>""", encoding='utf-8')
        
        opf = oebps_dir / "content.opf"
        opf.write_text("""<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
    <spine toc="ncx"></spine>
</package>""", encoding='utf-8')
        
        # Apply LTR->RTL transition (French -> Arabic)
        result = apply_rtl_to_epub_directory(str(epub_dir), 'Arabic', 'French')
        
        assert result['was_transition'] is False  # Not RTL->LTR
        assert result['is_rtl'] is True
        assert result['css_injected'] == 1
        assert result['opf_updated'] is True
        
        # Check RTL was applied
        content = chapter.read_text(encoding='utf-8')
        assert 'dir="rtl"' in content
        assert 'direction: rtl' in content

    def test_apply_rtl_to_epub_ltr_to_ltr(self, tmp_path):
        """Test LTR->LTR (no change needed)"""
        epub_dir = tmp_path / "epub"
        epub_dir.mkdir()
        oebps_dir = epub_dir / "OEBPS"
        oebps_dir.mkdir()
        
        chapter = oebps_dir / "chapter.xhtml"
        chapter.write_text("""<html><body><p>Text</p></body></html>""", encoding='utf-8')
        
        # Apply LTR->LTR (French -> English)
        result = apply_rtl_to_epub_directory(str(epub_dir), 'English', 'French')
        
        assert result['was_transition'] is False
        assert result['is_rtl'] is False
        assert result['css_injected'] == 0
        assert result['css_removed'] == 0

    def test_apply_rtl_to_epub_rtl_to_rtl(self, tmp_path):
        """Test RTL->RTL (apply/update RTL styles)"""
        epub_dir = tmp_path / "epub"
        epub_dir.mkdir()
        oebps_dir = epub_dir / "OEBPS"
        oebps_dir.mkdir()
        
        chapter = oebps_dir / "chapter.xhtml"
        chapter.write_text("""<html dir="rtl"><body><p>Text</p></body></html>""", encoding='utf-8')
        
        opf = oebps_dir / "content.opf"
        opf.write_text("""<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
    <spine page-progression-direction="rtl"></spine>
</package>""", encoding='utf-8')
        
        # Apply RTL->RTL (Arabic -> Hebrew)
        result = apply_rtl_to_epub_directory(str(epub_dir), 'Hebrew', 'Arabic')
        
        assert result['was_transition'] is False
        assert result['is_rtl'] is True
        # Should still inject/update CSS even if already RTL

    def test_update_opf_for_ltr(self, tmp_path):
        """Should update OPF from RTL to LTR progression"""
        opf_content = """<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
    <metadata></metadata>
    <manifest></manifest>
    <spine page-progression-direction="rtl" toc="ncx">
        <itemref idref="chapter1"/>
    </spine>
</package>"""
        
        opf_path = tmp_path / "content.opf"
        opf_path.write_text(opf_content, encoding='utf-8')
        
        result = update_opf_for_ltr(str(opf_path))
        
        assert result is True
        
        updated = opf_path.read_text(encoding='utf-8')
        assert 'page-progression-direction="ltr"' in updated
        assert 'page-progression-direction="rtl"' not in updated

    def test_update_opf_for_ltr_already_ltr(self, tmp_path):
        """Should handle OPF already in LTR"""
        opf_content = """<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
    <spine toc="ncx"></spine>
</package>"""
        
        opf_path = tmp_path / "content.opf"
        opf_path.write_text(opf_content, encoding='utf-8')
        
        result = update_opf_for_ltr(str(opf_path))
        
        # Should return False as no change was needed
        assert result is False


class TestEdgeCasesTransition:
    """Test edge cases for RTL/LTR transitions"""

    def test_remove_rtl_with_invalid_html(self):
        """Should handle invalid HTML gracefully"""
        invalid_html = "Not valid < html >"
        result = remove_rtl_from_html(invalid_html)
        # Should not crash and return something
        assert isinstance(result, str)
        assert len(result) > 0

    def test_remove_rtl_without_rtl_content(self):
        """Should handle HTML without RTL"""
        normal_html = """<html><head><title>Test</title></head>
<body><p>Normal text</p></body></html>"""
        
        result = remove_rtl_from_html(normal_html)
        # Should add LTR CSS anyway
        assert 'direction: ltr' in result

    def test_none_source_language(self, tmp_path):
        """Should handle None source language (defaults to no transition)"""
        epub_dir = tmp_path / "epub"
        epub_dir.mkdir()
        
        result = apply_rtl_to_epub_directory(str(epub_dir), 'Arabic', None)
        
        # Should apply RTL (target is RTL, source is None/unknown)
        assert result['is_rtl'] is True
        assert result['was_transition'] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
