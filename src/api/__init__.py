"""
API modules for Flask web server
"""
from .routes import configure_routes
from .websocket import configure_websocket_handlers, emit_update
from .handlers import start_translation_job

__all__ = [
    'configure_routes',
    'configure_websocket_handlers',
    'emit_update',
    'start_translation_job'
]