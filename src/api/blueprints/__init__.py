"""
API Routes
"""
from .config_routes import create_config_blueprint
from .translation_routes import create_translation_blueprint
from .file_routes import create_file_blueprint
from .security_routes import create_security_blueprint
from .tts_routes import create_tts_blueprint
from .glossary_routes import create_glossary_blueprint
from .cost_routes import create_cost_blueprint
from .usage_routes import create_usage_blueprint
from .version_routes import create_version_blueprint
from .sample_routes import create_sample_blueprint
from .profile_routes import create_profile_blueprint

__all__ = [
    'create_config_blueprint',
    'create_translation_blueprint',
    'create_file_blueprint',
    'create_security_blueprint',
    'create_tts_blueprint',
    'create_glossary_blueprint',
    'create_cost_blueprint',
    'create_usage_blueprint',
    'create_version_blueprint',
    'create_sample_blueprint',
    'create_profile_blueprint',
]
