"""PDF refine-only mode.

PDF refinement is text-first: extract readable PDF text, normalize OCR-like
scan artifacts when detected, refine as plain text, and write text output.
"""

from typing import Optional, Callable, Dict, Any

from src.config import DEFAULT_MODEL, API_ENDPOINT
from src.core.pdf import extract_pdf_text
from .txt_refiner import refine_text_content


async def refine_pdf_file(
    input_filepath: str,
    output_filepath: str,
    target_language: str,
    model_name: str = DEFAULT_MODEL,
    cli_api_endpoint: str = API_ENDPOINT,
    log_callback: Optional[Callable] = None,
    stats_callback: Optional[Callable] = None,
    check_interruption_callback: Optional[Callable] = None,
    llm_provider: str = "ollama",
    gemini_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    openrouter_api_key: Optional[str] = None,
    mistral_api_key: Optional[str] = None,
    deepseek_api_key: Optional[str] = None,
    poe_api_key: Optional[str] = None,
    nim_api_key: Optional[str] = None,
    context_window: int = 2048,
    auto_adjust_context: bool = True,
    max_tokens_per_chunk: Optional[int] = None,
    soft_limit_ratio: Optional[float] = None,
    prompt_options: Optional[Dict[str, Any]] = None,
    checkpoint_manager: Any = None,
    translation_id: Optional[str] = None,
    resume_from_index: int = 0,
) -> bool:
    """Run a refinement-only pass on readable PDF text."""
    try:
        text = extract_pdf_text(input_filepath)
    except Exception as exc:
        if log_callback:
            log_callback("pdf_refine_extract_error", f"Could not extract PDF text: {exc}")
        return False

    if log_callback:
        log_callback(
            "pdf_refine_text_extracted",
            f"Extracted {len(text)} characters from PDF for refinement."
        )

    return await refine_text_content(
        translated_text=text,
        output_filepath=output_filepath,
        target_language=target_language,
        model_name=model_name,
        cli_api_endpoint=cli_api_endpoint,
        log_callback=log_callback,
        stats_callback=stats_callback,
        check_interruption_callback=check_interruption_callback,
        llm_provider=llm_provider,
        gemini_api_key=gemini_api_key,
        openai_api_key=openai_api_key,
        openrouter_api_key=openrouter_api_key,
        mistral_api_key=mistral_api_key,
        deepseek_api_key=deepseek_api_key,
        poe_api_key=poe_api_key,
        nim_api_key=nim_api_key,
        context_window=context_window,
        auto_adjust_context=auto_adjust_context,
        max_tokens_per_chunk=max_tokens_per_chunk,
        soft_limit_ratio=soft_limit_ratio,
        prompt_options=prompt_options,
        checkpoint_manager=checkpoint_manager,
        translation_id=translation_id,
        resume_from_index=resume_from_index,
    )
