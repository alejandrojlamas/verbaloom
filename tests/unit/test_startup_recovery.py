from copy import deepcopy

from src.api.blueprints.translation_routes import (
    _checkpoint_has_resumable_work,
    _persist_manual_pause_request,
)
from src.api.startup_recovery import (
    build_startup_resume_config,
    restore_jobs_after_restart,
)


class _CheckpointManager:
    def __init__(self, checkpoints, stale_ids):
        self.checkpoints = deepcopy(checkpoints)
        self.stale_ids = list(stale_ids)
        self.statuses = {}
        self.config_updates = {}

    def get_stale_active_job_ids(self):
        return list(self.stale_ids)

    def reset_running_jobs_on_startup(self):
        return len(self.stale_ids)

    def load_checkpoint(self, translation_id):
        return deepcopy(self.checkpoints.get(translation_id))

    def get_preserved_input_path(self, translation_id):
        checkpoint = self.checkpoints.get(translation_id) or {}
        return ((checkpoint.get("job") or {}).get("config") or {}).get(
            "preserved_input_path"
        )

    def mark_running(self, translation_id):
        self.statuses[translation_id] = "running"
        return True

    def mark_paused(self, translation_id):
        self.statuses[translation_id] = "paused"
        return True

    def update_job_config(self, translation_id, config):
        self.config_updates[translation_id] = deepcopy(config)
        return True


class _StateManager:
    def __init__(self, checkpoint_manager, resumable_jobs):
        self.checkpoint_manager = checkpoint_manager
        self.resumable_jobs = deepcopy(resumable_jobs)
        self.states = {}

    def get_resumable_jobs(self):
        return deepcopy(self.resumable_jobs)

    def restore_job_from_checkpoint(self, translation_id):
        checkpoint = self.checkpoint_manager.load_checkpoint(translation_id)
        if not checkpoint:
            return False
        self.states[translation_id] = {
            "config": deepcopy(checkpoint["job"]["config"]),
            "status": "paused",
            "interrupted": False,
        }
        return True

    def set_translation_field(self, translation_id, field, value):
        self.states.setdefault(translation_id, {})[field] = deepcopy(value)
        return True

    def get_translation(self, translation_id):
        return deepcopy(self.states.get(translation_id))


def _checkpoint(source_path, *, status="running", manual=False, resume_index=4):
    return {
        "resume_from_index": resume_index,
        "checkpoint_complete": False,
        "job": {
            "translation_id": "job-1",
            "status": status,
            "config": {
                "file_type": "epub",
                "preserved_input_path": str(source_path),
                "_manual_pause_requested": manual,
            },
            "progress": {"total_chunks": 10, "completed_chunks": resume_index},
        },
    }


def test_startup_auto_resumes_one_genuinely_active_job(tmp_path):
    source = tmp_path / "book.epub"
    source.write_bytes(b"book")
    checkpoint = _checkpoint(source)
    manager = _CheckpointManager({"job-1": checkpoint}, ["job-1"])
    state = _StateManager(
        manager,
        [{"translation_id": "job-1", "file_type": "epub", "progress": {}}],
    )
    starts = []

    report, _ = restore_jobs_after_restart(
        state,
        lambda translation_id, config: starts.append((translation_id, config)),
    )

    assert report.resumed_job_ids == ["job-1"]
    assert starts[0][0] == "job-1"
    assert starts[0][1]["resume_from_index"] == 4
    assert starts[0][1]["file_path"] == str(source)
    assert state.states["job-1"]["status"] == "running"


def test_startup_preserves_durable_provider_schedule_for_worker_preflight(tmp_path):
    source = tmp_path / "book.epub"
    source.write_bytes(b"book")
    checkpoint = _checkpoint(source)
    checkpoint["job"]["config"].update({
        "_scheduled_resume_at_epoch": 2_000_000_000.0,
        "_scheduled_resume_at_utc": "2033-05-18T03:33:20+00:00",
        "_scheduled_resume_reason": "deepseek_peak_pricing",
        "_scheduled_resume_status": "pricing_wait",
    })
    manager = _CheckpointManager({"job-1": checkpoint}, ["job-1"])
    state = _StateManager(
        manager,
        [{"translation_id": "job-1", "file_type": "epub", "progress": {}}],
    )
    starts = []

    report, _ = restore_jobs_after_restart(
        state,
        lambda translation_id, config: starts.append((translation_id, config)),
    )

    assert report.resumed_job_ids == ["job-1"]
    assert starts[0][1]["_scheduled_resume_at_epoch"] == 2_000_000_000.0
    assert starts[0][1]["_scheduled_resume_status"] == "pricing_wait"


def test_startup_never_revives_a_manual_pause(tmp_path):
    source = tmp_path / "book.epub"
    source.write_bytes(b"book")
    checkpoint = _checkpoint(source, manual=True)
    manager = _CheckpointManager({"job-1": checkpoint}, ["job-1"])
    state = _StateManager(
        manager,
        [{"translation_id": "job-1", "file_type": "epub", "progress": {}}],
    )
    starts = []

    report, _ = restore_jobs_after_restart(
        state,
        lambda translation_id, config: starts.append((translation_id, config)),
    )

    assert starts == []
    assert report.skipped["job-1"] == "manual pause requested"


def test_startup_requires_preserved_source(tmp_path):
    missing = tmp_path / "missing.epub"
    checkpoint = _checkpoint(missing)
    manager = _CheckpointManager({"job-1": checkpoint}, ["job-1"])

    config, reason = build_startup_resume_config(checkpoint, manager)

    assert config is None
    assert reason == "preserved source missing"


def test_startup_rejects_preserved_source_outside_checkpoint_storage(tmp_path):
    source = tmp_path / "outside" / "book.epub"
    source.parent.mkdir()
    source.write_bytes(b"book")
    checkpoint = _checkpoint(source)
    manager = _CheckpointManager({"job-1": checkpoint}, ["job-1"])
    manager.uploads_dir = tmp_path / "managed"

    config, reason = build_startup_resume_config(checkpoint, manager)

    assert config is None
    assert reason == "preserved source outside managed checkpoint storage"


def test_startup_strips_legacy_key_from_custom_endpoint(tmp_path):
    source = tmp_path / "book.epub"
    source.write_bytes(b"book")
    checkpoint = _checkpoint(source)
    checkpoint["job"]["config"].update({
        "llm_provider": "openai",
        "llm_api_endpoint": "https://gateway.example.test/v1/chat/completions",
        "openai_api_key": "legacy-saved-secret",
    })
    manager = _CheckpointManager({"job-1": checkpoint}, ["job-1"])

    config, reason = build_startup_resume_config(checkpoint, manager)

    assert reason == ""
    assert config["openai_api_key"] == ""
    assert config["_credential_sources"]["openai"] == "none"


def test_startup_limits_concurrent_auto_resumes(tmp_path):
    checkpoints = {}
    resumable = []
    for translation_id in ("job-1", "job-2"):
        source = tmp_path / f"{translation_id}.epub"
        source.write_bytes(b"book")
        checkpoint = _checkpoint(source)
        checkpoint["job"]["translation_id"] = translation_id
        checkpoints[translation_id] = checkpoint
        resumable.append(
            {"translation_id": translation_id, "file_type": "epub", "progress": {}}
        )
    manager = _CheckpointManager(checkpoints, ["job-1", "job-2"])
    state = _StateManager(manager, resumable)
    starts = []

    report, _ = restore_jobs_after_restart(
        state,
        lambda translation_id, config: starts.append(translation_id),
        max_auto_resumes=1,
    )

    assert starts == ["job-1"]
    assert report.skipped["job-2"] == "another stale job is already recovering"
    assert state.states["job-2"]["status"] == "paused"


def test_checkpoint_complete_can_resume_finalization_but_completed_cannot():
    assert _checkpoint_has_resumable_work({
        "checkpoint_complete": True,
        "job": {"status": "validating"},
    })
    assert not _checkpoint_has_resumable_work({
        "checkpoint_complete": True,
        "job": {"status": "completed"},
    })


def test_manual_pause_is_persisted_immediately():
    manager = _CheckpointManager({}, [])
    state = _StateManager(manager, [])
    state.states["job-1"] = {"config": {"model": "deepseek-v4-pro"}}

    _persist_manual_pause_request(state, "job-1")

    config = state.states["job-1"]["config"]
    assert config["_manual_pause_requested"] is True
    assert state.states["job-1"]["pause_reason"] == "manual"
    assert manager.config_updates["job-1"]["_manual_pause_requested"] is True
    assert manager.statuses["job-1"] == "paused"
