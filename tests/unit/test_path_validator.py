"""
Unit tests for PathValidator class.

Tests filename validation to prevent directory traversal attacks
while allowing legitimate filenames with ellipsis and special characters.
"""
import pytest
import sys
import importlib.util
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import PathValidator directly from file to avoid importing flask dependencies
spec = importlib.util.spec_from_file_location(
    "path_validator",
    project_root / "src" / "api" / "services" / "path_validator.py"
)
path_validator_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(path_validator_module)
PathValidator = path_validator_module.PathValidator


class TestPathValidator:
    """Tests for PathValidator class."""

    def test_validate_normal_filename(self):
        """Test validation of normal filenames."""
        is_valid, error = PathValidator.validate_filename("normal_file.txt")
        assert is_valid is True
        assert error == ""

    def test_validate_filename_with_spaces(self):
        """Test validation of filenames with spaces."""
        is_valid, error = PathValidator.validate_filename("file with spaces.docx")
        assert is_valid is True
        assert error == ""

    def test_validate_filename_with_ellipsis(self):
        """Test validation of filenames with ellipsis (three dots)."""
        is_valid, error = PathValidator.validate_filename("file...with...dots.txt")
        assert is_valid is True
        assert error == ""

    def test_validate_filename_with_multiple_ellipsis(self):
        """Test validation of filenames with multiple ellipsis."""
        is_valid, error = PathValidator.validate_filename("voici un plan de cours complet, peux tu extraires... (French).docx")
        assert is_valid is True
        assert error == ""

    def test_validate_filename_with_two_dots(self):
        """Test validation of filenames with two consecutive dots (no separators)."""
        is_valid, error = PathValidator.validate_filename("file..txt")
        assert is_valid is True
        assert error == ""

    def test_validate_filename_with_four_dots(self):
        """Test validation of filenames with four consecutive dots."""
        is_valid, error = PathValidator.validate_filename("file....doc")
        assert is_valid is True
        assert error == ""

    def test_reject_directory_traversal_unix(self):
        """Test rejection of directory traversal attempts (Unix style)."""
        is_valid, error = PathValidator.validate_filename("../../../etc/passwd")
        assert is_valid is False
        assert "directory traversal" in error

    def test_reject_directory_traversal_windows(self):
        """Test rejection of directory traversal attempts (Windows style)."""
        is_valid, error = PathValidator.validate_filename("..\\..\\..\\windows\\system32")
        assert is_valid is False
        assert "directory traversal" in error or "path separators" in error

    def test_reject_embedded_directory_traversal(self):
        """Test rejection of embedded directory traversal."""
        is_valid, error = PathValidator.validate_filename("foo/../bar.txt")
        assert is_valid is False
        assert "directory traversal" in error or "path separators" in error

    def test_reject_absolute_path_unix(self):
        """Test rejection of absolute paths (Unix style)."""
        is_valid, error = PathValidator.validate_filename("/etc/passwd")
        assert is_valid is False
        assert "absolute path" in error or "path separators" in error

    def test_reject_absolute_path_windows(self):
        """Test rejection of absolute paths (Windows style)."""
        is_valid, error = PathValidator.validate_filename("C:\\Windows\\System32")
        assert is_valid is False
        assert "path" in error.lower()

    def test_reject_empty_filename(self):
        """Test rejection of empty filename."""
        is_valid, error = PathValidator.validate_filename("")
        assert is_valid is False
        assert "empty" in error.lower()

    def test_reject_path_separator_forward_slash(self):
        """Test rejection of filenames with forward slash separator."""
        is_valid, error = PathValidator.validate_filename("foo/bar.txt")
        assert is_valid is False
        assert "path separator" in error.lower()

    def test_reject_path_separator_backslash(self):
        """Test rejection of filenames with backslash separator."""
        is_valid, error = PathValidator.validate_filename("foo\\bar.txt")
        assert is_valid is False
        assert "path separator" in error.lower()

    def test_reject_filename_too_long(self):
        """Test rejection of filenames exceeding maximum length."""
        long_filename = "a" * 300
        is_valid, error = PathValidator.validate_filename(long_filename)
        assert is_valid is False
        assert "too long" in error.lower()

    def test_validate_filename_at_max_length(self):
        """Test validation of filename at maximum allowed length."""
        max_length_filename = "a" * 250 + ".txt"  # 254 chars total
        is_valid, error = PathValidator.validate_filename(max_length_filename)
        assert is_valid is True
        assert error == ""

    def test_validate_filename_with_special_characters(self):
        """Test validation of filenames with allowed special characters."""
        test_cases = [
            "file-name.txt",
            "file_name.txt",
            "file.name.txt",
            "file (copy).txt",
            "file [1].txt",
            "file {backup}.txt",
        ]
        for filename in test_cases:
            is_valid, error = PathValidator.validate_filename(filename)
            assert is_valid is True, f"Failed for: {filename}, error: {error}"

    def test_validate_filenames_list(self):
        """Test validation of a list of filenames."""
        filenames = [
            "file1.txt",
            "file2.docx",
            "file...with...dots.txt"
        ]
        is_valid, error = PathValidator.validate_filenames(filenames)
        assert is_valid is True
        assert error == ""

    def test_reject_filenames_list_with_invalid(self):
        """Test rejection of a list containing invalid filenames."""
        filenames = [
            "file1.txt",
            "../../../etc/passwd",
            "file3.docx"
        ]
        is_valid, error = PathValidator.validate_filenames(filenames)
        assert is_valid is False
        assert "passwd" in error

    def test_reject_filenames_not_list(self):
        """Test rejection when filenames parameter is not a list."""
        is_valid, error = PathValidator.validate_filenames("not_a_list")
        assert is_valid is False
        assert "must be a list" in error.lower()

    def test_reject_empty_filenames_list(self):
        """Test rejection of empty filenames list."""
        is_valid, error = PathValidator.validate_filenames([])
        assert is_valid is False
        assert "no filenames" in error.lower()

    def test_validate_filename_with_unicode(self):
        """Test validation of filenames with Unicode characters."""
        test_cases = [
            "fichier_français.txt",
            "文件.txt",
            "archivo_español.docx",
        ]
        for filename in test_cases:
            is_valid, error = PathValidator.validate_filename(filename)
            # Should be valid as long as no path separators or traversal
            assert is_valid is True, f"Failed for: {filename}, error: {error}"

    def test_reject_directory_traversal_at_start(self):
        """Test rejection of filenames starting with directory traversal."""
        is_valid, error = PathValidator.validate_filename("../file.txt")
        assert is_valid is False
        assert "directory traversal" in error or "absolute path" in error or "path separator" in error

    def test_reject_directory_traversal_at_end(self):
        """Test rejection of filenames ending with directory traversal."""
        is_valid, error = PathValidator.validate_filename("file/..")
        assert is_valid is False
        assert "directory traversal" in error or "path separator" in error

    def test_validate_filename_with_dot_prefix(self):
        """Test validation of hidden files (dot prefix)."""
        is_valid, error = PathValidator.validate_filename(".hidden_file.txt")
        assert is_valid is True
        assert error == ""

    def test_validate_filename_with_only_dots_in_basename(self):
        """Test validation of filenames with only dots in basename."""
        is_valid, error = PathValidator.validate_filename("....txt")
        assert is_valid is True
        assert error == ""


class TestPathValidatorEdgeCases:
    """Tests for edge cases and security-critical scenarios."""

    def test_null_byte_injection(self):
        """Test handling of null byte injection attempts."""
        filename_with_null = "file\x00.txt"
        is_valid, error = PathValidator.validate_filename(filename_with_null)
        assert is_valid is False
        assert "control characters" in error

    def test_very_long_extension(self):
        """Test handling of very long file extensions."""
        filename = "file." + "x" * 100
        is_valid, error = PathValidator.validate_filename(filename)
        # Should be valid if under max length
        assert is_valid is True

    def test_multiple_extensions(self):
        """Test handling of multiple file extensions."""
        is_valid, error = PathValidator.validate_filename("file.tar.gz")
        assert is_valid is True
        assert error == ""

    def test_no_extension(self):
        """Test handling of filenames without extensions."""
        is_valid, error = PathValidator.validate_filename("README")
        assert is_valid is True
        assert error == ""

    def test_directory_traversal_variations(self):
        """Test various directory traversal attack patterns."""
        attack_patterns = [
            "../..",
            "..\\..",
            "..\\../",
            "../.\\",
            "....//",
            "..\\\\..\\\\",
        ]
        for pattern in attack_patterns:
            is_valid, error = PathValidator.validate_filename(pattern)
            assert is_valid is False, f"Should reject: {pattern}"

    def test_whitespace_only_filename(self):
        """Test handling of whitespace-only filenames."""
        # Note: The current implementation treats whitespace as valid characters
        # This could be debated, but we test current behavior
        is_valid, error = PathValidator.validate_filename("   ")
        # After stripping happens in the actual usage, this would fail differently
        # We test the validator as-is
        assert is_valid is True  # Validator allows it, but usage should strip

    def test_real_world_problematic_filename(self):
        """Test the real-world case that triggered this fix."""
        filename = "voici un plan de cours complet, peux tu extraires... (French).docx"
        is_valid, error = PathValidator.validate_filename(filename)
        assert is_valid is True
        assert error == ""


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
