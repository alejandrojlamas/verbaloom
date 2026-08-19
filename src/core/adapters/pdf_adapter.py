"""PDF adapter for the generic translation system.

The adapter extracts readable PDF text and translates it as plain text. The
output is UTF-8 text, not a layout-preserving PDF.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from .format_adapter import FormatAdapter
from .translation_unit import TranslationUnit
from src.core.pdf import extract_pdf_text_with_structure
from src.utils.text_encoding import encode_utf8_text_download


class PdfAdapter(FormatAdapter):
    """Adapter for text-based PDF translation."""

    def __init__(self, input_file_path: str, output_file_path: str, config: Dict[str, Any]):
        super().__init__(input_file_path, output_file_path, config)
        self.text: str = ""
        self.chunks: List[Dict[str, str]] = []
        self.translated_chunks: List[Optional[str]] = []
        self.last_error: Optional[str] = None

    async def prepare_for_translation(self) -> bool:
        try:
            self.text, structure_report = extract_pdf_text_with_structure(self.input_file_path)
            if not self.text.strip():
                self.last_error = (
                    "PDF has no extractable text. Scanned/image-only PDFs need OCR first."
                )
                return False
            if structure_report.has_structured_blocks:
                prompt_options = self.config.get("prompt_options") or {}
                prompt_options["structured_layout_active"] = True
                prompt_options["structured_layout_version"] = structure_report.version
                prompt_options["structured_layout_summary"] = {
                    "tables_detected": structure_report.tables_detected,
                    "figure_text_blocks_detected": structure_report.figure_text_blocks_detected,
                    "repairs_applied": structure_report.repairs_applied,
                    "block_type_counts": structure_report.block_type_counts,
                    "policy_counts": structure_report.policy_counts,
                    "excluded_blocks": structure_report.excluded_blocks,
                    "cleaned_blocks": structure_report.cleaned_blocks,
                    "preserved_blocks": structure_report.preserved_blocks,
                    "reconstructed_blocks": structure_report.reconstructed_blocks,
                }
                self.config["prompt_options"] = prompt_options

            from src.core.text_processor import split_text_into_chunks

            self.chunks = split_text_into_chunks(
                text=self.text,
                max_tokens_per_chunk=self.config.get("max_tokens_per_chunk"),
                soft_limit_ratio=self.config.get("soft_limit_ratio"),
            )
            if not self.chunks:
                self.last_error = "PDF text did not produce any translation chunks."
                return False
            self.translated_chunks = [None] * len(self.chunks)
            return True
        except Exception as exc:
            self.last_error = f"Could not extract PDF text: {exc}"
            return False

    def get_translation_units(self) -> List[TranslationUnit]:
        units: List[TranslationUnit] = []
        for i, chunk in enumerate(self.chunks):
            units.append(
                TranslationUnit(
                    unit_id=f"chunk_{i}",
                    content=chunk["main_content"],
                    context_before=chunk.get("context_before", ""),
                    context_after=chunk.get("context_after", ""),
                    metadata={
                        "chunk_index": i,
                        "total_chunks": len(self.chunks),
                    },
                )
            )
        return units

    async def save_unit_translation(self, unit_id: str, translated_content: str) -> bool:
        try:
            chunk_index = int(unit_id.split("_")[1])
            if 0 <= chunk_index < len(self.translated_chunks):
                self.translated_chunks[chunk_index] = translated_content
                return True
            return False
        except Exception:
            return False

    async def reconstruct_output(self, bilingual: bool = False) -> bytes:
        text_chunks: List[str] = []
        separator = "-" * 40

        for i, translated_chunk in enumerate(self.translated_chunks):
            original = self.chunks[i]["main_content"].strip()
            translated = translated_chunk.strip() if translated_chunk else original
            if bilingual:
                text_chunks.append(f"{original}\n\n{translated}\n\n{separator}")
            else:
                text_chunks.append(translated)

        joiner = "\n\n" if bilingual else "\n"
        final_text = joiner.join(text_chunks)
        if bilingual and final_text.endswith(separator):
            final_text = final_text[:-len(separator)].rstrip()
        return encode_utf8_text_download(final_text)

    async def resume_from_checkpoint(self, checkpoint_data: Dict[str, Any]) -> int:
        try:
            for chunk_data in checkpoint_data.get("chunks", []):
                if chunk_data.get("status") != "completed":
                    continue
                metadata = chunk_data.get("chunk_data", {})
                chunk_index = metadata.get("chunk_index")
                translated_text = chunk_data.get("translated_text")
                if chunk_index is not None and translated_text is not None:
                    if 0 <= chunk_index < len(self.translated_chunks):
                        self.translated_chunks[chunk_index] = translated_text
            return checkpoint_data.get("resume_from_index", 0)
        except Exception:
            return 0

    async def cleanup(self):
        pass

    @property
    def format_name(self) -> str:
        return "pdf"

    def __repr__(self) -> str:
        return (
            f"PdfAdapter("
            f"input={self.input_file_path.name}, "
            f"output={self.output_file_path.name}, "
            f"chunks={len(self.chunks)})"
        )
