"""Regression tests for the frontend design-system split.

The app still uses legacy IDs/classes for JavaScript compatibility, but the
template should load shared component CSS instead of carrying a large inline
style block. These tests guard the load order and the component contracts that
new UI work should reuse.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "src" / "web" / "templates" / "translation_interface.html"
DESIGN_SYSTEM_CSS = ROOT / "src" / "web" / "static" / "css" / "design-system.css"
APP_COMPONENTS_CSS = ROOT / "src" / "web" / "static" / "css" / "app-components.css"
THEME_MANAGER_JS = ROOT / "src" / "web" / "static" / "js" / "utils" / "theme-manager.js"


def test_template_uses_external_component_stylesheets():
    template = TEMPLATE.read_text(encoding="utf-8")

    assert "<style>" not in template
    assert "</style>" not in template
    assert "/static/style.css" in template
    assert "/static/css/design-system.css" in template
    assert "/static/css/app-components.css" in template

    legacy_pos = template.index("/static/style.css")
    ds_pos = template.index("/static/css/design-system.css")
    app_pos = template.index("/static/css/app-components.css")
    assert legacy_pos < ds_pos < app_pos


def test_template_declares_mobile_browser_chrome_colors():
    template = TEMPLATE.read_text(encoding="utf-8")

    assert 'viewport-fit=cover' in template
    assert 'name="color-scheme"' in template
    assert 'id="themeColorMeta"' in template
    assert '#080c12' in template
    assert 'applyInitialTheme' in template
    assert "tbl-theme-preference" in template


def test_design_system_exposes_required_components():
    css = DESIGN_SYSTEM_CSS.read_text(encoding="utf-8")

    required_contracts = [
        "--ds-surface",
        "--ds-primary",
        "--ds-radius-md",
        ".ds-button",
        ".ds-button--primary",
        ".ds-tabs",
        ".ds-tab",
        ".ds-card",
        ".ds-progress",
        ".ds-table",
        ".ds-modal",
        ".ds-mobile-sheet",
    ]
    for contract in required_contracts:
        assert contract in css

    assert "min-height: 100dvh" in css
    assert "background-color: var(--ds-bg)" in css


def test_design_system_bridges_legacy_classes_without_feature_ids():
    css = DESIGN_SYSTEM_CSS.read_text(encoding="utf-8")

    for legacy_class in [
        ":where(.btn)",
        ":where(.tab-nav",
        ":where(.main-card",
        ":where(.progress-bar-container",
        ":where(.file-table",
        ":where(.modal-overlay",
    ]:
        assert legacy_class in css

    assert "#glossary" not in css.lower()
    assert "#profilePrep" not in css
    assert "#topTabNav" not in css


def test_app_component_styles_preserve_template_specific_rules():
    css = APP_COMPONENTS_CSS.read_text(encoding="utf-8")

    required_selectors = [
        "#topTabNav",
        ".transform-grid",
        ".header .status-section",
        ".profile-prep-card",
        ".glossary-toolbar-btn",
        "#profileGlossaryListTable",
        "#glossaryTermsTable",
        "#ner-modal",
    ]
    for selector in required_selectors:
        assert selector in css

    assert "<style>" not in css
    assert "</style>" not in css


def test_theme_manager_syncs_browser_chrome_color():
    js = THEME_MANAGER_JS.read_text(encoding="utf-8")

    assert "themeColorMeta" in js
    assert "syncBrowserChrome" in js
    assert "#080c12" in js
    assert "#f5f7fb" in js
    assert "document.body.style.backgroundColor" in js
