"""
Test for GitHub Issue #108: DEFAULT_TARGET_LANGUAGE not respected in UI

This test verifies that:
1. DEFAULT_TARGET_LANGUAGE is exposed via /api/config endpoint
2. The config response includes the server-side default target language

Regression test for: https://github.com/hydropix/TranslateBooksWithLLMs/issues/108
"""

import pytest
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestDefaultTargetLanguageBug:
    """
    Tests for DEFAULT_TARGET_LANGUAGE configuration exposure in API.
    
    GitHub Issue #108: When DEFAULT_TARGET_LANGUAGE is configured in the
    environment (e.g., Kubernetes deployment), the UI was ignoring it and
    always using browser language detection instead.
    
    Root cause: The /api/config endpoint did not expose DEFAULT_TARGET_LANGUAGE
    to the frontend, so the JavaScript code always fell back to browser detection.
    
    Fix:
    1. Backend (config_routes.py): Add default_target_language to /api/config response
    2. Frontend (form-manager.js): Use server-provided default when available,
       only fall back to browser detection when server value is empty
    """

    @pytest.fixture
    def app(self):
        """Create a Flask app for testing with isolated config."""
        # Set test values BEFORE importing Flask app components
        os.environ['DEFAULT_TARGET_LANGUAGE'] = 'Italian'
        os.environ['DEFAULT_SOURCE_LANGUAGE'] = 'Spanish'
        
        # Import and reload config to pick up test values
        import importlib
        from src import config
        importlib.reload(config)
        
        # Now import and create Flask app (it will use the reloaded config)
        from flask import Flask
        from src.api.blueprints.config_routes import create_config_blueprint
        
        app = Flask(__name__)
        bp = create_config_blueprint(server_session_id=12345)
        app.register_blueprint(bp)
        
        yield app
        
        # Cleanup: restore original values after test
        del os.environ['DEFAULT_TARGET_LANGUAGE']
        del os.environ['DEFAULT_SOURCE_LANGUAGE']

    def test_api_config_response_contains_target_language(self, app):
        """
        Test that /api/config response includes default_target_language field.
        
        This is the main regression test for GitHub issue #108.
        Before the fix, the response did NOT include this field.
        """
        with app.test_client() as client:
            response = client.get('/api/config')
            
            assert response.status_code == 200, \
                f"Expected 200 but got {response.status_code}"
            
            data = response.get_json()
            assert data is not None, "Response should be valid JSON"
            
            # KEY ASSERTION: default_target_language must be present
            assert 'default_target_language' in data, \
                "BUG REGRESSION: 'default_target_language' missing from /api/config. " \
                f"Got keys: {list(data.keys())}"
            
            # It should have the value from environment
            assert data['default_target_language'] == 'Italian', \
                f"Expected 'Italian' but got '{data.get('default_target_language')}'"

    def test_api_config_response_contains_source_language(self, app):
        """
        Test that /api/config response also includes default_source_language field.
        
        For consistency, we expose both source and target language defaults.
        """
        with app.test_client() as client:
            response = client.get('/api/config')
            data = response.get_json()
            
            assert 'default_source_language' in data, \
                "'default_source_language' should be in /api/config response"
            
            assert data['default_source_language'] == 'Spanish', \
                f"Expected 'Spanish' but got '{data.get('default_source_language')}'"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
