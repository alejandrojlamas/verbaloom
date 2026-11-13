"""Isolated browser QA server: real routes/UI, synthetic workers, no provider IO.

Run from the checkout with the project venv. All mutable files live in a fresh
temporary directory. This is not a translation-quality or hardware benchmark.
"""

import os
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from unittest.mock import patch

    import httpx
    import requests
    from flask import Flask
    from flask_socketio import SocketIO

    with tempfile.TemporaryDirectory(prefix="verbaloom-settings-qa-") as folder:
        os.chdir(folder)
        os.environ.update(OUTPUT_DIR=str(Path(folder) / "outputs"),
                          BOOK_PROFILES_DIR=str(Path(folder) / "profiles"),
                          VERBALOOM_DATA_DIR=str(Path(folder) / "data"),
                          TRANSLATION_MEMORY_DB=str(Path(folder) / "data" / "translation_memory.db"),
                          OPENAI_API_KEY="YOUR_API_KEY_HERE", DEEPSEEK_API_KEY="YOUR_API_KEY_HERE")
        from src.api import routes
        from src.api.blueprints import (
            config_routes,
            pipeline_routes,
            provider_model_routes,
            translation_routes,
        )
        from src.api.blueprints.provider_model_routes import ProviderModelCatalog
        from src.api.translation_state import TranslationStateManager
        from src.core.pipeline_config import MODELS
        from src.persistence.checkpoint_manager import CheckpointManager

        config_routes.get_base_path = lambda: str(ROOT)
        config_routes._config.LLM_PROVIDER = "deepseek"
        config_routes._config.DEFAULT_MODEL = "deepseek-v4-pro"
        for provider in ("OPENAI", "DEEPSEEK", "GEMINI", "POE", "MISTRAL", "NIM", "OPENROUTER"):
            setattr(config_routes._config, f"{provider}_API_KEY", "YOUR_API_KEY_HERE" if provider in {"OPENAI", "DEEPSEEK"} else "")
        config_routes._config.DEEPSEEK_MODEL = "deepseek-v4-pro"
        translation_routes.get_deepseek_pricing_status = lambda: SimpleNamespace(disabled=False)
        provider_model_routes.get_deepseek_pricing_status = lambda: SimpleNamespace(
            to_dict=lambda: {'disabled': False, 'enabled': True, 'pricing_tier': 'qa',
                             'display_timezone': 'America/Mexico_City'},
        )

        def models(self, provider, **kwargs):
            choices = sorted(MODELS.get(provider, set()))
            return {"models": [{"id": name, "name": name} for name in choices],
                    "model_names": choices, "default": "deepseek-v4-pro" if provider == "deepseek" else "gpt-6-luna",
                    "count": len(choices), "status": f"{provider}_connected"}

        async def preflight(**kwargs):
            return {"status": "models_visible", "models": [], "generation_tested": False, "balance": None}

        ProviderModelCatalog.list_models = models
        pipeline_routes.preflight_openai = preflight
        pipeline_routes.preflight_deepseek = preflight
        manager = CheckpointManager(str(Path(folder) / "jobs.db"))
        state = TranslationStateManager(manager)
        app = Flask(__name__, template_folder=str(ROOT / "src/web/templates"),
                    static_folder=str(ROOT / "src/web/static"), static_url_path="/static")
        app.config["TEMPLATES_AUTO_RELOAD"] = True

        @app.after_request
        def isolated_qa_header(response):
            response.headers["X-Verbaloom-QA"] = "isolated"
            return response

        socketio = SocketIO(app, async_mode="threading")
        stop = threading.Event()

        def worker(run_id, config, **kwargs):
            source = Path(config.get("file_path") or Path(folder) / f"{run_id}.txt")
            if not source.exists():
                source.write_text(config.get("text") or "A fixture paragraph.")
            if not manager.get_job(run_id):
                manager.start_job(run_id, "txt", config, str(source))
            if not config.get('is_resume'):
                manager.save_checkpoint(run_id, 0, source.read_text(), "Fixture translation", total_chunks=2, completed_chunks=1)
            manager.mark_running(run_id)
            state.set_interrupted(run_id, False)
            state.set_translation_field(run_id, "status", "running")
            state.update_stats(run_id, {
                "total_chunks": 2, "completed_chunks": 1, "failed_chunks": 0,
                "stage_progress": {"stage": "audit", "done": 3, "total": 8, "section": 2, "sections": 4},
            })

            def monitor():
                while not stop.wait(.1):
                    if state.get_translation_field(run_id, "interrupted"):
                        manager.mark_interrupted(run_id)
                        state.set_translation_field(run_id, "status", "interrupted")
                        socketio.emit("translation_update", {"translation_id": run_id, "status": "interrupted"})
                        return

            threading.Thread(target=monitor, daemon=True).start()
            return True

        routes.configure_routes(app, state, os.environ["OUTPUT_DIR"], worker, socketio)
        from src.api.websocket import configure_websocket_handlers
        configure_websocket_handlers(socketio, state)
        app.config["TESTING"] = True
        with patch.object(httpx.AsyncClient, "send", side_effect=httpx.ConnectError("Provider network disabled in QA")), \
             patch.object(requests.Session, "request", side_effect=requests.ConnectionError("Provider network disabled in QA")):
            print(f"QA_STATE={folder}", flush=True)
            try:
                socketio.run(app, host="127.0.0.1", port=5099, allow_unsafe_werkzeug=True, use_reloader=False)
            finally:
                stop.set()
                manager.close()


if __name__ == "__main__":
    main()
