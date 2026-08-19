"""
Configuration and health check routes
"""
import os
import sys
import logging
import re
import time
from flask import Blueprint, request, jsonify, render_template, make_response
from pathlib import Path

# UI locales served by /static/locales/<code>/*.json. Keep in sync with the
# SUPPORTED_LOCALES constant in src/web/static/js/i18n/i18n.js. The list is
# small enough that duplication beats a separate config file for now.
SUPPORTED_UI_LOCALES = ['en', 'fr', 'es', 'de', 'zh-CN', 'ja', 'ko']
UI_LOCALE_COOKIE = 'ui_locale'
UI_LOCALE_COOKIE_MAX_AGE = 60 * 60 * 24 * 365  # 1 year


def resolve_ui_locale(req):
    """Pick the UI locale to render the page with.

    Order: explicit cookie (user's last choice) → Accept-Language best match
    → 'en'. Returning the code is enough; i18next-http-backend loads the
    JSON files asynchronously from /static/locales/<code>/.
    """
    cookie_locale = req.cookies.get(UI_LOCALE_COOKIE)
    if cookie_locale in SUPPORTED_UI_LOCALES:
        return cookie_locale
    best = req.accept_languages.best_match(SUPPORTED_UI_LOCALES)
    return best or 'en'


def get_base_path():
    """Get base path for resources (templates, static files)"""
    # In PyInstaller bundle, use the temporary extraction directory
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    return os.getcwd()


def get_config_path():
    """Get base path for configuration files (.env)"""
    return os.getcwd()

import src.config as _config
from src.config import reload_config
from src import __version__
from src.core.llm.base import normalize_api_keys
from .mobile_access_routes import (
    local_tailnet_ip as _local_tailnet_ip,
    register_mobile_access_routes,
)
from .provider_model_routes import register_provider_model_routes

# Setup logger for this module
logger = logging.getLogger('config_routes')
if _config.DEBUG_MODE:
    logger.setLevel(logging.DEBUG)

SUPPORTED_DOCUMENT_FORMATS = ["txt", "epub", "srt", "docx", "pdf"]
SUPPORTED_OUTPUT_FORMATS = ["auto", "txt", "docx", "pdf", "epub"]


def create_config_blueprint(server_session_id=None):
    """Create and configure the config blueprint

    Args:
        server_session_id: Server session ID from state manager (optional, generates new if not provided)
    """
    bp = Blueprint('config', __name__)

    # Store server startup time/session ID to detect restarts
    # Use provided session_id from state_manager if available, otherwise generate new
    # Ensure it's an integer for consistency with health check response
    startup_time = int(server_session_id) if server_session_id else int(time.time())

    register_mobile_access_routes(
        bp,
        startup_time=startup_time,
        server_version=__version__,
        config_path_provider=get_config_path,
        tailnet_ip_provider=lambda: _local_tailnet_ip(),
        logger=logger,
    )
    register_provider_model_routes(bp, config=_config, logger=logger)

    @bp.route('/')
    @bp.route('/verbaloom')
    @bp.route('/verbaloom/')
    def serve_interface():
        """Serve the main translation interface.

        Switched from send_from_directory to render_template so Jinja can
        inject the initial UI locale into the HTML — that avoids the
        English flash on first paint when the user prefers another locale.
        """
        base_path = get_base_path()
        templates_dir = os.path.join(base_path, 'src', 'web', 'templates')
        interface_path = os.path.join(templates_dir, 'translation_interface.html')
        if not os.path.exists(interface_path):
            return f"<h1>Error: Interface not found</h1><p>Looked in: {interface_path}</p>", 404

        response = make_response(render_template(
            'translation_interface.html',
            initial_locale=resolve_ui_locale(request),
            supported_locales=SUPPORTED_UI_LOCALES,
            app_version=__version__,
            asset_version=f"{__version__}-{startup_time}",
        ))
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    @bp.route('/reset-sample')
    def reset_sample_state():
        """Clear stale Sample & Compare browser state and reload the UI.

        This is intentionally a tiny HTML response so it works from Android
        Chrome even when the main app's module graph is stuck in an old page.
        """
        target = f"/verbaloom?sample_reset={startup_time}"
        response = make_response(f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Resetting sample...</title>
  <style>
    body {{ font-family: system-ui, -apple-system, sans-serif; background:#07111d; color:#eef4ff; display:grid; min-height:100vh; place-items:center; margin:0; }}
    main {{ max-width:28rem; padding:2rem; line-height:1.45; }}
    strong {{ color:#bfff2f; }}
  </style>
</head>
<body>
  <main>
    <h1>Resetting sample...</h1>
    <p>Clearing local <strong>Sample &amp; Compare</strong> state and reloading the app.</p>
  </main>
  <script>
    (async () => {{
      try {{
        Object.keys(localStorage || {{}}).forEach((key) => {{
          if (key.startsWith('verbaloom.sample.') || key.startsWith('tbl.sample.')) {{
            localStorage.removeItem(key);
          }}
        }});
        if (window.caches && caches.keys) {{
          const keys = await caches.keys();
          await Promise.all(keys.map((key) => caches.delete(key)));
        }}
        if (navigator.serviceWorker && navigator.serviceWorker.getRegistrations) {{
          const regs = await navigator.serviceWorker.getRegistrations();
          await Promise.all(regs.map((reg) => reg.unregister()));
        }}
      }} catch (err) {{
        console.warn('sample reset cleanup failed', err);
      }}
      window.location.replace({target!r});
    }})();
  </script>
</body>
</html>""")
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    @bp.route('/api/ui-locale', methods=['POST'])
    def set_ui_locale():
        """Persist the user's UI locale choice in a long-lived cookie.

        The client also calls i18next.changeLanguage() to swap the running
        UI; this route exists so the next full page load renders the right
        locale server-side (no English flash).
        """
        data = request.get_json(silent=True) or {}
        locale = data.get('locale')
        if locale not in SUPPORTED_UI_LOCALES:
            return jsonify({"success": False, "error": f"Unsupported locale: {locale}"}), 400

        response = make_response(jsonify({"success": True, "locale": locale}))
        response.set_cookie(
            UI_LOCALE_COOKIE,
            locale,
            max_age=UI_LOCALE_COOKIE_MAX_AGE,
            samesite='Lax',
            httponly=True,
        )
        return response

    @bp.route('/api/ui-locale', methods=['GET'])
    def get_ui_locale():
        """Return the locale the server would pick for this request."""
        return jsonify({
            "locale": resolve_ui_locale(request),
            "supported": SUPPORTED_UI_LOCALES,
        })

    @bp.route('/api/health', methods=['GET'])
    def health_check():
        """API health check endpoint"""
        return jsonify({
            "status": "ok",
            "message": "Translation API is running",
            "translate_module": "loaded",
            "ollama_default_endpoint": _config.API_ENDPOINT,
            "supported_formats": SUPPORTED_DOCUMENT_FORMATS,
            "supported_output_formats": SUPPORTED_OUTPUT_FORMATS,
            "version": __version__,
            "startup_time": startup_time,  # Used to detect server restarts
            "session_id": startup_time  # Alias for compatibility with LifecycleManager
        })

    @bp.route('/api/config', methods=['GET'])
    def get_default_config():
        """Get default configuration values"""
        # For API keys, send a masked indicator if configured, empty string if not.
        # Also expose the pool size so the UI can signal active multi-key rotation.
        def mask_api_key(raw):
            """Return (masked_last_key, key_count). Empty/0 means not configured."""
            keys = normalize_api_keys(raw)
            if not keys:
                return "", 0
            last = keys[-1]
            masked = "***" + last[-4:] if len(last) > 4 else "***"
            return masked, len(keys)

        gemini_mask, gemini_count = mask_api_key(_config.GEMINI_API_KEY)
        openai_mask, openai_count = mask_api_key(_config.OPENAI_API_KEY)
        openrouter_mask, openrouter_count = mask_api_key(_config.OPENROUTER_API_KEY)
        mistral_mask, mistral_count = mask_api_key(_config.MISTRAL_API_KEY)
        deepseek_mask, deepseek_count = mask_api_key(_config.DEEPSEEK_API_KEY)
        poe_mask, poe_count = mask_api_key(_config.POE_API_KEY)
        nim_mask, nim_count = mask_api_key(_config.NIM_API_KEY)

        provider_default_models = {
            "gemini": _config.GEMINI_MODEL,
            "openrouter": _config.OPENROUTER_MODEL,
            "mistral": _config.MISTRAL_MODEL,
            "deepseek": _config.DEEPSEEK_MODEL,
            "poe": _config.POE_MODEL,
            "nim": _config.NIM_MODEL,
        }
        effective_default_model = (
            provider_default_models.get((_config.LLM_PROVIDER or "").lower())
            or _config.DEFAULT_MODEL
        )

        config_response = {
            "api_endpoint": _config.API_ENDPOINT,
            "ollama_api_endpoint": _config.OLLAMA_API_ENDPOINT,
            "openai_api_endpoint": _config.OPENAI_API_ENDPOINT,
            "llm_provider": _config.LLM_PROVIDER,
            "default_model": effective_default_model,
            "default_source_language": _config.DEFAULT_SOURCE_LANGUAGE,
            "default_target_language": _config.DEFAULT_TARGET_LANGUAGE,
            "timeout": _config.REQUEST_TIMEOUT,
            "context_window": _config.OLLAMA_NUM_CTX,
            "max_attempts": _config.MAX_TRANSLATION_ATTEMPTS,
            "retry_delay": 2,
            "supported_formats": SUPPORTED_DOCUMENT_FORMATS,
            "supported_output_formats": SUPPORTED_OUTPUT_FORMATS,
            "gemini_api_key": gemini_mask,
            "openai_api_key": openai_mask,
            "openrouter_api_key": openrouter_mask,
            "mistral_api_key": mistral_mask,
            "deepseek_api_key": deepseek_mask,
            "poe_api_key": poe_mask,
            "nim_api_key": nim_mask,
            "gemini_api_key_count": gemini_count,
            "openai_api_key_count": openai_count,
            "openrouter_api_key_count": openrouter_count,
            "mistral_api_key_count": mistral_count,
            "deepseek_api_key_count": deepseek_count,
            "poe_api_key_count": poe_count,
            "nim_api_key_count": nim_count,
            "gemini_api_key_configured": gemini_count > 0,
            "openai_api_key_configured": openai_count > 0,
            "openrouter_api_key_configured": openrouter_count > 0,
            "mistral_api_key_configured": mistral_count > 0,
            "deepseek_api_key_configured": deepseek_count > 0,
            "poe_api_key_configured": poe_count > 0,
            "nim_api_key_configured": nim_count > 0,
            "output_filename_pattern": _config.OUTPUT_FILENAME_PATTERN,
            "max_tokens_per_chunk": int(_config.MAX_TOKENS_PER_CHUNK),
            "disable_auto_pause": str(_config.DISABLE_AUTO_PAUSE).strip().lower() == 'true',
            # Webhook notifications — returned as-is for editing. URLs and tokens
            # only ever travel between this server and the same-origin browser
            # session that already controls the .env on disk.
            "notify_webhook_url": _config.NOTIFY_WEBHOOK_URL,
            "notify_webhook_method": _config.NOTIFY_WEBHOOK_METHOD,
            "notify_webhook_headers": _config.NOTIFY_WEBHOOK_HEADERS,
            "notify_webhook_payload": _config.NOTIFY_WEBHOOK_PAYLOAD,
            "notify_on_success": bool(_config.NOTIFY_ON_SUCCESS),
            "notify_on_failure": bool(_config.NOTIFY_ON_FAILURE),
            "notify_on_interruption": bool(_config.NOTIFY_ON_INTERRUPTION),
            "notify_timeout_seconds": int(_config.NOTIFY_TIMEOUT_SECONDS),
            "notify_configured": bool(_config.NOTIFY_WEBHOOK_URL)
        }

        return jsonify(config_response)

    @bp.route('/api/config/max-tokens', methods=['GET'])
    def get_max_tokens():
        """Get MAX_TOKENS_PER_CHUNK configuration value for UI preview height adjustment"""
        provider = (request.args.get('provider') or '').strip() or None
        return jsonify({
            "max_tokens_per_chunk": _config.max_tokens_per_chunk_for_provider(provider),
            "provider": provider or _config.LLM_PROVIDER,
        })

    @bp.route('/api/model/warning', methods=['GET'])
    def get_model_warning():
        """
        Get thinking model warning for a specific model (instant lookup).

        This endpoint checks if a model is an uncontrollable thinking model
        and returns an appropriate warning message for the UI.

        Query params:
            model: Model name (e.g., "qwen3:30b")
            endpoint: Optional API endpoint (for cache differentiation)

        Returns:
            JSON with warning message if applicable, or null if no warning
        """
        model = request.args.get('model', '')
        endpoint = request.args.get('endpoint', '')

        if not model:
            return jsonify({"warning": None, "behavior": None})

        try:
            from src.core.llm import (
                get_model_warning_message,
                get_thinking_behavior_sync,
                ThinkingBehavior
            )

            warning = get_model_warning_message(model, endpoint)
            behavior = get_thinking_behavior_sync(model, endpoint)

            return jsonify({
                "warning": warning,
                "behavior": behavior.value if behavior else None,
                "is_uncontrollable": behavior == ThinkingBehavior.UNCONTROLLABLE if behavior else False,
                "is_thinking_model": behavior in [ThinkingBehavior.CONTROLLABLE, ThinkingBehavior.UNCONTROLLABLE] if behavior else False
            })

        except Exception as e:
            return jsonify({"warning": None, "behavior": None, "error": str(e)})

    @bp.route('/api/custom-instructions', methods=['GET'])
    def get_custom_instructions():
        """List available custom instruction files from Custom_Instructions/ folder.

        Each entry carries `has_translation` / `has_refinement` so the UI can
        filter presets per phase. `.txt` files (legacy) apply to both phases;
        `.yaml`/`.yml` files report the phases actually present in the file.
        """
        from src.utils.custom_instructions import list_custom_instructions

        try:
            project_root = Path(get_config_path())
            custom_instructions_dir = project_root / 'Custom_Instructions'

            if not custom_instructions_dir.exists():
                return jsonify({"files": [], "count": 0, "status": "folder_not_found"})

            files = list_custom_instructions(custom_instructions_dir)
            return jsonify({"files": files, "count": len(files), "status": "ok"})

        except Exception as e:
            logger.error(f"Error listing custom instructions: {e}")
            return jsonify({"files": [], "count": 0, "status": "error", "error": str(e)})

    @bp.route('/api/custom-instructions/open-folder', methods=['POST'])
    def open_custom_instructions_folder():
        """Open the Custom_Instructions folder in the system file explorer"""
        import subprocess
        import platform

        try:
            project_root = Path(get_config_path())
            custom_instructions_dir = project_root / 'Custom_Instructions'

            # Create folder if it doesn't exist
            if not custom_instructions_dir.exists():
                custom_instructions_dir.mkdir(parents=True, exist_ok=True)

            abs_path = str(custom_instructions_dir.resolve())
            system = platform.system()

            if system == 'Windows':
                os.startfile(abs_path)
            elif system == 'Darwin':  # macOS
                subprocess.run(['open', abs_path], check=True)
            else:  # Linux and others
                subprocess.run(['xdg-open', abs_path], check=True)

            return jsonify({"success": True, "path": "Custom_Instructions"})

        except Exception as e:
            logger.error(f"Error opening custom instructions folder: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    def _get_env_file_path():
        """Get the path to the .env file"""
        config_path = get_config_path()
        return Path(config_path) / '.env'

    # Keys whose values may contain spaces, '#', or JSON braces. python-dotenv
    # parses unquoted values up to a '#' (treated as inline comment), so a raw
    # JSON payload like {"text":"hi #1"} would be silently truncated. Wrap in
    # single quotes (JSON never produces single quotes, so no escape needed).
    _QUOTED_ENV_KEYS = {
        'NOTIFY_WEBHOOK_URL',
        'NOTIFY_WEBHOOK_HEADERS',
        'NOTIFY_WEBHOOK_PAYLOAD',
        'OUTPUT_FILENAME_PATTERN',
    }

    def _format_env_value(key: str, value: str) -> str:
        if not value:
            return ''
        if key not in _QUOTED_ENV_KEYS:
            return value
        if "'" not in value:
            return f"'{value}'"
        # Fallback if the user did inject single quotes — use double quotes
        # with the minimal escaping python-dotenv understands.
        escaped = value.replace('\\', '\\\\').replace('"', '\\"')
        return f'"{escaped}"'

    def _update_env_file(updates: dict) -> bool:
        """
        Update specific keys in the .env file.
        Creates the file if it doesn't exist.

        Args:
            updates: Dictionary of key-value pairs to update

        Returns:
            True if successful, False otherwise
        """
        env_path = _get_env_file_path()

        # Read existing content or start fresh
        existing_lines = []
        file_is_new = not env_path.exists()

        if env_path.exists():
            with open(env_path, 'r', encoding='utf-8') as f:
                existing_lines = f.readlines()
        else:
            # Create file with header if it doesn't exist
            existing_lines = [
                "# Translation API Configuration\n",
                "# This file was automatically created by the web interface\n",
                "# You can edit these values manually or via the web UI\n",
                "\n"
            ]

        # Track which keys we've updated
        updated_keys = set()
        new_lines = []

        for line in existing_lines:
            stripped = line.strip()

            # Skip empty lines and comments, keep them as-is
            if not stripped or stripped.startswith('#'):
                new_lines.append(line)
                continue

            # Check if this line has a key we want to update
            match = re.match(r'^([A-Z_][A-Z0-9_]*)=', stripped)
            if match:
                key = match.group(1)
                if key in updates:
                    formatted = _format_env_value(key, updates[key])
                    new_lines.append(f"{key}={formatted}\n")
                    updated_keys.add(key)
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)

        # Add any keys that weren't in the file
        for key, value in updates.items():
            if key not in updated_keys:
                formatted = _format_env_value(key, value)
                new_lines.append(f"{key}={formatted}\n")

        # Write back
        with open(env_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)

        return True

    @bp.route('/api/settings', methods=['POST'])
    def save_settings():
        """
        Save user settings to .env file.

        Accepts JSON with settings to save. Only specific keys are allowed
        for security reasons.
        """
        allowed_keys = {
            'GEMINI_API_KEY',
            'GEMINI_MODEL',
            'OPENAI_API_KEY',
            'OPENROUTER_API_KEY',
            'OPENROUTER_MODEL',
            'MISTRAL_API_KEY',
            'MISTRAL_MODEL',
            'DEEPSEEK_API_KEY',
            'DEEPSEEK_MODEL',
            'POE_API_KEY',
            'POE_MODEL',
            'NIM_API_KEY',
            'NIM_MODEL',
            'DEFAULT_MODEL',
            'LLM_PROVIDER',
            'OLLAMA_API_ENDPOINT',
            'OPENAI_API_ENDPOINT',
            'OUTPUT_FILENAME_PATTERN',
            'MAX_TOKENS_PER_CHUNK',
            'DISABLE_AUTO_PAUSE',
            'NOTIFY_WEBHOOK_URL',
            'NOTIFY_WEBHOOK_METHOD',
            'NOTIFY_WEBHOOK_HEADERS',
            'NOTIFY_WEBHOOK_PAYLOAD',
            'NOTIFY_ON_SUCCESS',
            'NOTIFY_ON_FAILURE',
            'NOTIFY_ON_INTERRUPTION',
            'NOTIFY_TIMEOUT_SECONDS'
        }

        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No data provided"}), 400

            # Filter to only allowed keys
            updates = {}
            for key, value in data.items():
                if key in allowed_keys:
                    # Sanitize value - remove newlines and dangerous characters
                    safe_value = str(value).replace('\n', '').replace('\r', '')
                    # Clamp MAX_TOKENS_PER_CHUNK to the same range the UI enforces
                    # so a hand-crafted POST can't break the chunker.
                    if key == 'MAX_TOKENS_PER_CHUNK':
                        try:
                            n = int(safe_value)
                        except (TypeError, ValueError):
                            continue
                        safe_value = str(max(50, min(1000, n)))
                    updates[key] = safe_value

            if not updates:
                return jsonify({"error": "No valid settings to save"}), 400

            # Update the .env file
            _update_env_file(updates)

            # Refresh module-level config so subsequent reads see the new values
            # without requiring a server restart.
            reload_config()

            logger.info(f"Settings saved and reloaded: {list(updates.keys())}")

            return jsonify({
                "success": True,
                "message": f"Saved {len(updates)} setting(s)",
                "saved_keys": list(updates.keys())
            })

        except Exception as e:
            logger.error(f"Error saving settings: {e}")
            return jsonify({"error": f"Failed to save settings: {str(e)}"}), 500

    @bp.route('/api/notifications/test', methods=['POST'])
    def test_notification():
        """Send a test webhook using the current saved configuration.

        The notifier reads from src.config (post-reload), so the test always
        reflects what's actually on disk. Returns success/failure with a hint
        when the URL is empty or the event flag is off.
        """
        from src.utils import notifier

        if not _config.NOTIFY_WEBHOOK_URL:
            return jsonify({
                "success": False,
                "error": "NOTIFY_WEBHOOK_URL is empty. Set a URL and save before testing."
            }), 400

        data = request.get_json(silent=True) or {}
        event = data.get('event', notifier.EVENT_SUCCESS)
        if event not in notifier.known_events():
            return jsonify({
                "success": False,
                "error": f"Unknown event '{event}'. Use one of: {', '.join(notifier.known_events())}"
            }), 400

        flag_map = {
            notifier.EVENT_SUCCESS: 'NOTIFY_ON_SUCCESS',
            notifier.EVENT_FAILURE: 'NOTIFY_ON_FAILURE',
            notifier.EVENT_INTERRUPTION: 'NOTIFY_ON_INTERRUPTION',
        }
        if not bool(getattr(_config, flag_map[event], False)):
            return jsonify({
                "success": False,
                "error": f"Event '{event}' is disabled. Enable it and save before testing."
            }), 400

        ctx = {
            "file": "test-file.epub",
            "output": "test-file (French).epub",
            "duration_seconds": 12.3,
            "provider": _config.LLM_PROVIDER,
            "model": _config.DEFAULT_MODEL or "test-model",
            "source_lang": "English",
            "target_lang": "French",
            "error": "Sample error for failure event" if event == notifier.EVENT_FAILURE else None,
            "translation_id": "test-job-id",
        }

        try:
            sent = notifier.notify(event, ctx)
        except Exception as exc:
            logger.exception("Test webhook raised unexpectedly")
            return jsonify({
                "success": False,
                "error": f"Unexpected error: {exc}"
            }), 500

        if sent:
            return jsonify({
                "success": True,
                "message": f"Test {event} notification sent successfully."
            })
        return jsonify({
            "success": False,
            "error": "Webhook call failed. Check the server logs (enable DEBUG_MODE for details), the URL, headers and payload format."
        }), 502

    @bp.route('/api/settings', methods=['GET'])
    def get_settings():
        """
        Get current settings that can be modified via the UI.
        Returns only the keys that are user-configurable.
        API keys are masked for security - only indicates if configured.
        """
        return jsonify({
            "gemini_api_key_configured": bool(_config.GEMINI_API_KEY),
            "openai_api_key_configured": bool(_config.OPENAI_API_KEY),
            "openrouter_api_key_configured": bool(_config.OPENROUTER_API_KEY),
            "mistral_api_key_configured": bool(_config.MISTRAL_API_KEY),
            "deepseek_api_key_configured": bool(_config.DEEPSEEK_API_KEY),
            "poe_api_key_configured": bool(_config.POE_API_KEY),
            "nim_api_key_configured": bool(_config.NIM_API_KEY),
            "default_model": _config.DEFAULT_MODEL or "",
            "llm_provider": _config.LLM_PROVIDER,
            "api_endpoint": _config.API_ENDPOINT or "",
            "ollama_api_endpoint": _config.OLLAMA_API_ENDPOINT or "",
            "openai_api_endpoint": _config.OPENAI_API_ENDPOINT or ""
        })

    return bp
