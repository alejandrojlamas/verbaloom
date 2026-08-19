"""
Test for GitHub Issue #108 (Part 2): Frontend initialization sequence

This test verifies the JavaScript initialization sequence is correct by:
1. Checking that the initialization order in index.js is correct
2. Verifying that ProviderManager waits for defaultConfigLoaded event
3. Ensuring SettingsManager doesn't trigger model loading prematurely

Regression test for: https://github.com/hydropix/TranslateBooksWithLLMs/issues/108
"""

import pytest
import re
from pathlib import Path


class TestFrontendInitializationSequence:
    """
    Tests to verify the frontend initialization sequence prevents the bug.
    
    The bug occurred because:
    1. SettingsManager.initialize() triggered model loading with localStorage endpoint
    2. This happened BEFORE FormManager.loadDefaultConfig() received server config
    
    The fix ensures:
    1. SettingsManager doesn't trigger model loading during initialization
    2. ProviderManager waits for 'defaultConfigLoaded' event before loading models
    3. FormManager.loadDefaultConfig() updates endpoints from server first
    """

    @pytest.fixture
    def js_files(self):
        """Get paths to relevant JavaScript files."""
        base_path = Path(__file__).parent.parent.parent / "src" / "web" / "static" / "js"
        return {
            "index_js": base_path / "index.js",
            "settings_manager": base_path / "core" / "settings-manager.js",
            "provider_manager": base_path / "providers" / "provider-manager.js",
            "form_manager": base_path / "ui" / "form-manager.js",
            "lifecycle_manager": base_path / "utils" / "lifecycle-manager.js",
            "glossary_manager": base_path / "glossary" / "glossary-manager.js",
        }

    def test_index_js_initialization_order(self, js_files):
        """
        Verify that index.js initializes modules in the correct order.
        
        Expected order:
        1. SettingsManager.initialize()
        2. FormManager.initialize()
        3. ProviderManager.initialize()
        
        This order is crucial because:
        - SettingsManager restores preferences (but shouldn't trigger model loading)
        - FormManager starts loading server config
        - ProviderManager sets up listeners and waits for config
        """
        index_js = js_files["index_js"]
        assert index_js.exists(), f"index.js not found at {index_js}"
        
        content = index_js.read_text(encoding='utf-8')
        
        # Find the initialization calls
        settings_match = re.search(r'SettingsManager\.initialize\(\)', content)
        form_match = re.search(r'FormManager\.initialize\(\)', content)
        provider_match = re.search(r'ProviderManager\.initialize\(\)', content)
        
        assert settings_match, "SettingsManager.initialize() not found in index.js"
        assert form_match, "FormManager.initialize() not found in index.js"
        assert provider_match, "ProviderManager.initialize() not found in index.js"
        
        # Verify order: SettingsManager → FormManager → ProviderManager
        settings_pos = settings_match.start()
        form_pos = form_match.start()
        provider_pos = provider_match.start()
        
        assert settings_pos < form_pos < provider_pos, \
            f"Initialization order is wrong. Expected: SettingsManager → FormManager → ProviderManager. " \
            f"Positions: SettingsManager={settings_pos}, FormManager={form_pos}, ProviderManager={provider_pos}"

    def test_settings_manager_does_not_trigger_change_event(self, js_files):
        """
        Verify that SettingsManager.loadLocalPreferences() does NOT trigger
        the 'change' event when restoring the provider.
        
        This was the root cause of the bug - triggering 'change' caused
        immediate model loading with the wrong endpoint.
        """
        settings_manager = js_files["settings_manager"]
        assert settings_manager.exists(), f"settings-manager.js not found at {settings_manager}"
        
        content = settings_manager.read_text(encoding='utf-8')
        
        # Find the loadLocalPreferences method and the provider restoration code
        # Look for the pattern where lastProvider is restored
        provider_restore_pattern = r'if\s*\(\s*prefs\.lastProvider\s*\)\s*\{[^}]+providerSelect\.value\s*=\s*prefs\.lastProvider;'
        match = re.search(provider_restore_pattern, content, re.DOTALL)
        
        assert match, "Could not find lastProvider restoration code in SettingsManager"
        
        # Get the code block after providerSelect.value = prefs.lastProvider
        start_pos = match.end()
        # Find the closing brace of this if block
        brace_count = 1
        pos = start_pos
        while brace_count > 0 and pos < len(content):
            if content[pos] == '{':
                brace_count += 1
            elif content[pos] == '}':
                brace_count -= 1
            pos += 1
        
        provider_restore_block = content[start_pos:pos]
        
        # The fix: should NOT have dispatchEvent(new Event('change'))
        # It should have a comment explaining why we don't trigger the event
        has_comment_explaining_fix = (
            "don't trigger" in provider_restore_block.lower() or
            "don't trigger" in provider_restore_block.lower() or
            "wait for" in provider_restore_block.lower() or
            "defaultConfigLoaded" in provider_restore_block
        )
        
        has_dispatch_event = "dispatchEvent" in provider_restore_block
        
        assert not has_dispatch_event, \
            "BUG REGRESSION: SettingsManager still triggers 'change' event when restoring provider. " \
            "This causes model loading with wrong endpoint before server config is loaded."
        
        assert has_comment_explaining_fix, \
            "SettingsManager should have a comment explaining why it doesn't trigger 'change' event, " \
            "and mention the fix for GitHub issue #108."

    def test_provider_manager_waits_for_default_config_loaded(self, js_files):
        """
        Verify that ProviderManager.initialize() waits for the 'defaultConfigLoaded'
        event before loading models.
        
        This ensures models are loaded with the correct endpoint from server config.
        """
        provider_manager = js_files["provider_manager"]
        assert provider_manager.exists(), f"provider-manager.js not found at {provider_manager}"
        
        content = provider_manager.read_text(encoding='utf-8')
        
        # Check that initialize() adds an event listener for 'defaultConfigLoaded'
        has_event_listener = "defaultConfigLoaded" in content
        
        assert has_event_listener, \
            "BUG: ProviderManager does not listen for 'defaultConfigLoaded' event. " \
            "It should wait for server config before loading models."
        
        # Check that toggleProviderSettings is called with false initially
        # (to show UI without loading models)
        has_initial_call = "toggleProviderSettings(false)" in content
        
        assert has_initial_call, \
            "ProviderManager.initialize() should call toggleProviderSettings(false) initially " \
            "to show UI without loading models, then wait for defaultConfigLoaded event."
        
        # Should have { once: true } to ensure the event listener only fires once
        has_once_option = "once: true" in content or '{once:true}' in content.replace(' ', '')
        
        assert has_once_option, \
            "The 'defaultConfigLoaded' event listener should have { once: true } option " \
            "to ensure it only fires during initialization."

    def test_form_manager_dispatches_default_config_loaded_event(self, js_files):
        """
        Verify that FormManager.loadDefaultConfig() dispatches the 'defaultConfigLoaded'
        event after receiving server config.
        """
        form_manager = js_files["form_manager"]
        assert form_manager.exists(), f"form-manager.js not found at {form_manager}"
        
        content = form_manager.read_text(encoding='utf-8')
        
        # Check that loadDefaultConfig dispatches the event
        has_dispatch = "defaultConfigLoaded" in content
        
        assert has_dispatch, \
            "FormManager should dispatch 'defaultConfigLoaded' event after loading server config."

    def test_comments_reference_github_issue(self, js_files):
        """
        Verify that the fix includes comments referencing GitHub issue #108.
        
        This serves as documentation for future developers.
        """
        provider_manager = js_files["provider_manager"]
        settings_manager = js_files["settings_manager"]
        
        provider_content = provider_manager.read_text(encoding='utf-8')
        settings_content = settings_manager.read_text(encoding='utf-8')
        
        # Check for GitHub issue reference in at least one of the files
        has_issue_reference = (
            "#108" in provider_content or
            "#108" in settings_content or
            "issue #108" in provider_content.lower() or
            "issue #108" in settings_content.lower()
        )
        
        assert has_issue_reference, \
            "The fix should include comments referencing GitHub issue #108 for documentation."

    def test_websocket_connection_has_one_initialization_owner(self, js_files):
        """index.js must be the only startup owner of the WebSocket connection."""
        index_content = js_files["index_js"].read_text(encoding="utf-8")
        lifecycle_content = js_files["lifecycle_manager"].read_text(encoding="utf-8")

        assert index_content.count("WebSocketManager.connect();") == 1
        assert "WebSocketManager.connect();" not in lifecycle_content

        wiring_position = index_content.index("wireModuleEvents();")
        connection_position = index_content.index("WebSocketManager.connect();")
        assert wiring_position < connection_position

    def test_profile_impact_preview_uses_the_active_flow_file_and_purpose(self, js_files):
        content = js_files["glossary_manager"].read_text(encoding="utf-8")

        assert "selectId === 'transformGlossarySelect'" in content
        assert "isTransformation ? 'transformFileInput' : 'fileInput'" in content
        assert "isTransformation ? 'transformation' : 'translation'" in content
        assert "expected_source: entry.source" in content
        assert "function profileEntrySuggestedTarget(entry)" in content
        assert "rendered.includes('|') ? '' : rendered" in content


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
