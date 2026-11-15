from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
API_CLIENT_JS = ROOT / "src/web/static/js/core/api-client.js"
PROFILE_PREP_JS = ROOT / "src/web/static/js/translation/profile-prep-manager.js"
INDEX_JS = ROOT / "src/web/static/js/index.js"
TEMPLATE = ROOT / "src/web/templates/translation_interface.html"


def test_api_client_marks_network_fetch_errors():
    js = API_CLIENT_JS.read_text()

    assert "function createNetworkError" in js
    assert "error.network = true" in js
    assert js.count("throw createNetworkError(error)") >= 3


def test_profile_prep_poll_retries_transient_network_errors():
    js = PROFILE_PREP_JS.read_text()

    assert "MAX_POLL_FAILURE_MS" in js
    assert "isTransientPollError" in js
    assert "showTransientPollIssue" in js
    assert "profileJobFailed" in js
    assert "profile_prep_reconnecting" in js
    assert "profile_prep_reconnect_failed" in js
    assert "ApiClient.getBookProfilePreparation" in js
    assert "Failed to fetch".lower() in js.lower()
    assert "this.updateFromJob(job)" in js


def test_profile_prep_tracking_survives_page_reload():
    js = PROFILE_PREP_JS.read_text()
    api_js = API_CLIENT_JS.read_text()

    assert "tbl.activeProfilePreparationId" in js
    assert "restoreTrackedJob" in js
    assert "persistTrackedJob(started.prep_id)" in js
    assert "statuses: 'queued,running'" in js
    assert "getBookProfilePreparations" in api_js
    assert "/api/book-profiles/prepare-jobs?" in api_js


def test_profile_prep_import_cache_busted():
    js = INDEX_JS.read_text()

    assert "profile-prep-manager.js?v=20261114-openai-primary" in js


def test_profile_prep_explains_term_batches_vs_book_chunks():
    js = PROFILE_PREP_JS.read_text()
    template = TEMPLATE.read_text()

    assert "profilePrepSuggestionsLabel" in template
    assert "profile_prep_status_term_review" in js
    assert "profile_prep_status_book_reading" in js
    assert "profile_prep_stat_term_batches" in js
    assert "profile_prep_stat_book_chunks" in js
    assert "stageLabel(stage)" in js


def test_profile_prep_follows_the_primary_provider_instead_of_fixed_deepseek():
    js = PROFILE_PREP_JS.read_text()

    # No request is built with a fixed DeepSeek provider or key any more.
    assert "formData.append('provider', 'deepseek')" not in js
    assert "ApiKeyUtils.getValue('deepseekApiKey')" not in js
    assert "formData.append('provider', models.provider)" in js
    assert "formData.append('review_model', models.reviewModel)" in js
    assert "ApiKeyUtils.getValue(models.keyField)" in js
    # Official OpenAI only: a compatible local server keeps the previous models.
    assert "isOfficialOpenAIEndpoint(endpoint)" in js
    assert "url.hostname === 'api.openai.com'" in js
    assert "model: 'gpt-6-luna'" in js
    assert "reviewModel: 'gpt-6.1-terra'" in js
    assert "keyField: 'openaiApiKey'" in js
    assert "keyField: 'deepseekApiKey'" in js


def test_profile_prep_labels_name_the_model_that_is_used():
    js = PROFILE_PREP_JS.read_text()
    template = TEMPLATE.read_text()
    locales = ROOT / "src/web/static/locales"

    assert "refreshModelLabels()" in js
    for event in ("defaultConfigLoaded", "providerSettingsChanged", "pipelineChanged"):
        assert event in js
    for key in (
        "transform:profile_prep_button",
        "transform:profile_prep_help",
        "transform:profile_prep_model_default",
    ):
        marker = f'data-i18n="{key}" data-i18n-params='
        assert marker in template, key
    for locale in ("de", "en", "es", "fr", "ja", "ko", "zh-CN"):
        transform = (locales / locale / "transform.json").read_text(encoding="utf-8")
        settings = (locales / locale / "settings.json").read_text(encoding="utf-8")
        assert "DeepSeek Flash" not in transform, locale
        assert transform.count("{{model}}") == 4, locale
        strict = next(
            line for line in settings.splitlines() if '"profile_strength_strict"' in line
        )
        assert "{{model}}" not in strict and "DeepSeek" not in strict, locale
