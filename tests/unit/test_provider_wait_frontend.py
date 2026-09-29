from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_provider_wait_is_active_and_not_terminal_in_translation_tracker():
    source = (ROOT / "src/web/static/js/translation/translation-tracker.js").read_text(
        encoding="utf-8"
    )

    active_line = next(line for line in source.splitlines() if "ACTIVE_STATUSES" in line)
    terminal_line = next(line for line in source.splitlines() if "TERMINAL_STATUSES" in line)
    assert "provider_wait" in active_line
    assert "provider_wait" not in terminal_line
    assert "data.status === 'provider_wait'" in source


def test_global_interrupt_finds_provider_wait_job():
    source = (ROOT / "src/web/static/js/index.js").read_text(encoding="utf-8")
    assert "'provider_wait'" in source
