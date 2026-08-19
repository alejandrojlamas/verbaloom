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

    assert "profile-prep-manager.js?v=20260710-profile-recovery" in js


def test_profile_prep_explains_term_batches_vs_book_chunks():
    js = PROFILE_PREP_JS.read_text()
    template = TEMPLATE.read_text()

    assert "profilePrepSuggestionsLabel" in template
    assert "profile_prep_status_term_review" in js
    assert "profile_prep_status_book_reading" in js
    assert "profile_prep_stat_term_batches" in js
    assert "profile_prep_stat_book_chunks" in js
    assert "stageLabel(stage)" in js
