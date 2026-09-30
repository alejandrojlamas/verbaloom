"""
Cost estimation routes.

Exposes:
- GET  /api/pricing/defaults : default pricing table per provider/model
- POST /api/cost/estimate    : estimate USD cost for a translation job
"""
import logging
from pathlib import Path

from flask import Blueprint, request, jsonify

from src.core.deepseek_pricing import (
    effective_estimate_tier,
    get_deepseek_pricing_status,
)
from src.core.pricing import (
    DEEPSEEK_PRICING_TIERS,
    DEFAULT_PRICING,
    LAST_UPDATED,
    get_default_pricing,
    CostEstimator,
)
from src.config import MAX_TOKENS_PER_CHUNK
from src.api.services.path_validator import PathValidator


logger = logging.getLogger('cost_routes')


LOCAL_PROVIDERS = {"ollama"}
PROVIDERS_WITH_API_PRICING = {"openrouter", "poe"}


def create_cost_blueprint(output_dir):
    """
    Create the cost estimation blueprint.

    Args:
        output_dir: base output directory (used to resolve uploaded files)
    """
    bp = Blueprint('cost', __name__)
    uploads_dir = Path(output_dir) / 'uploads'

    @bp.route('/api/pricing/defaults', methods=['GET'])
    def get_pricing_defaults():
        """Return the default pricing table and last-updated date."""
        pricing = {provider: dict(models) for provider, models in DEFAULT_PRICING.items()}
        deepseek_status = get_deepseek_pricing_status()
        deepseek_tier = effective_estimate_tier(deepseek_status)
        pricing["deepseek"] = DEEPSEEK_PRICING_TIERS[deepseek_tier]
        return jsonify({
            "pricing": pricing,
            "last_updated": LAST_UPDATED,
            "local_providers": sorted(LOCAL_PROVIDERS),
            "providers_with_api_pricing": sorted(PROVIDERS_WITH_API_PRICING),
            "pricing_context": {
                "deepseek": {
                    "current_tier": deepseek_status.pricing_tier,
                    "effective_estimate_tier": deepseek_tier,
                    "off_peak_guard_enabled": deepseek_status.enabled,
                    "source_url": deepseek_status.source_url,
                }
            },
        })

    @bp.route('/api/cost/estimate', methods=['POST'])
    def estimate_cost():
        """
        Estimate translation cost.

        Body:
            provider: str (required)
            model:    str (required)
            text:     str (optional) — direct text content
            file_path: str (optional) — path to an uploaded file (relative or
                       absolute under <output_dir>/uploads). One of text|file_path
                       is required.
            src_lang: str (optional)
            tgt_lang: str (optional)
            pricing:  {"input": float, "output": float} per 1M (optional, overrides defaults)
            options:  {"refine": bool, "text_cleanup": bool} (optional)
        """
        try:
            data = request.get_json(silent=True) or {}

            provider = (data.get('provider') or '').strip().lower()
            model = (data.get('model') or '').strip()

            if not provider:
                return jsonify({"error": "provider is required"}), 400
            if not model:
                return jsonify({"error": "model is required"}), 400

            if provider in LOCAL_PROVIDERS:
                return jsonify({
                    "free": True,
                    "provider": provider,
                    "model": model,
                    "message": "Local model — no API cost",
                })

            pricing_tier = None
            pricing_status = None
            if provider == "deepseek":
                pricing_status = get_deepseek_pricing_status()
                pricing_tier = effective_estimate_tier(pricing_status)

            pricing = _resolve_pricing(
                provider,
                model,
                data.get('pricing'),
                pricing_tier=pricing_tier,
            )
            if pricing is None:
                return jsonify({
                    "unknown": True,
                    "provider": provider,
                    "model": model,
                    "message": (
                        "Pricing not available for this model. "
                        "You can edit prices manually from the cost badge."
                    ),
                })

            text = _resolve_text_input(data, uploads_dir)
            if text is None:
                return jsonify({
                    "no_content": True,
                    "provider": provider,
                    "model": model,
                    "message": "No text or file provided yet",
                })

            options = data.get('options') or {}
            src_lang = data.get('src_lang') or ''
            tgt_lang = data.get('tgt_lang') or ''

            estimator = CostEstimator(
                provider=provider,
                model=model,
                pricing=pricing,
                max_tokens_per_chunk=MAX_TOKENS_PER_CHUNK,
            )
            result = estimator.estimate(
                text=text,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                options=options,
            )

            result["pricing_source"] = _pricing_source(
                provider,
                model,
                data.get('pricing'),
                pricing_tier=pricing_tier,
            )
            result["pricing_last_updated"] = LAST_UPDATED
            if pricing_status is not None:
                result["pricing_tier"] = pricing_tier
                result["pricing_current_tier"] = pricing_status.pricing_tier
                result["pricing_waits_for_off_peak"] = bool(
                    pricing_status.enabled and pricing_status.pricing_tier == "peak"
                )
                result["pricing_source_url"] = pricing_status.source_url
            return jsonify(result)

        except Exception as e:
            logger.exception("Error in cost estimation: %s", e)
            return jsonify({"error": f"Estimation failed: {e}"}), 500

    return bp


def _resolve_pricing(
    provider: str,
    model: str,
    override: dict | None,
    *,
    pricing_tier: str | None = None,
):
    if isinstance(override, dict) and 'input' in override and 'output' in override:
        try:
            pricing = {
                "input": float(override['input']),
                "output": float(override['output']),
            }
            for key in ("input_cache_hit", "input_cache_miss"):
                if key in override:
                    pricing[key] = float(override[key])
            return pricing
        except (TypeError, ValueError):
            pass
    return get_default_pricing(provider, model, pricing_tier=pricing_tier)


def _pricing_source(
    provider: str,
    model: str,
    override: dict | None,
    *,
    pricing_tier: str | None = None,
) -> str:
    if isinstance(override, dict) and 'input' in override and 'output' in override:
        return "user_override"
    if provider in PROVIDERS_WITH_API_PRICING:
        return "provider_api"
    if get_default_pricing(provider, model, pricing_tier=pricing_tier) is not None:
        return "default_table"
    return "unknown"


def _resolve_text_input(data: dict, uploads_dir: Path) -> str | None:
    """Return the text to estimate from, or None if no content provided."""
    text = data.get('text')
    if isinstance(text, str) and text.strip():
        return text

    file_path_raw = data.get('file_path')
    if not file_path_raw:
        return None

    file_path = Path(file_path_raw)
    if not file_path.is_absolute():
        file_path = uploads_dir / file_path

    try:
        file_path = PathValidator.resolve_managed_file(file_path, [uploads_dir])
    except (ValueError, FileNotFoundError, OSError):
        logger.warning("Refused estimation: file is outside managed uploads")
        return None

    return _extract_text_for_estimation(file_path)


def _extract_text_for_estimation(file_path: Path) -> str:
    """
    Extract translatable text from a file for token counting.

    Best-effort extraction — accuracy doesn't matter as much as speed here.
    """
    suffix = file_path.suffix.lower()

    if suffix in ('.txt', '.srt'):
        try:
            return file_path.read_text(encoding='utf-8', errors='replace')
        except OSError as e:
            logger.warning("Failed to read %s: %s", file_path, e)
            return ''

    if suffix == '.epub':
        return _extract_epub_text(file_path)

    if suffix == '.docx':
        return _extract_docx_text(file_path)

    if suffix == '.pdf':
        return _extract_pdf_text(file_path)

    try:
        return file_path.read_text(encoding='utf-8', errors='replace')
    except OSError:
        return ''


def _extract_epub_text(file_path: Path) -> str:
    """Extract visible EPUB text in spine order for token estimation."""
    try:
        from src.core.output_formats import extract_readable_text
        return extract_readable_text(file_path)
    except Exception as e:
        logger.warning("Failed to extract EPUB text from %s: %s", file_path, e)
        return ''


def _extract_docx_text(file_path: Path) -> str:
    """Extract text from DOCX paragraphs."""
    try:
        from docx import Document
        doc = Document(str(file_path))
        return '\n\n'.join(p.text for p in doc.paragraphs if p.text and p.text.strip())
    except Exception as e:
        logger.warning("Failed to extract DOCX text from %s: %s", file_path, e)
        return ''


def _extract_pdf_text(file_path: Path) -> str:
    """Extract readable text from a PDF for token counting."""
    try:
        from src.core.pdf import extract_pdf_text
        return extract_pdf_text(file_path)
    except Exception as e:
        logger.warning("Failed to extract PDF text from %s: %s", file_path, e)
        return ''
