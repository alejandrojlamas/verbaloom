"""Accepted alignment work survives reconstruction without skipping validation."""

import asyncio
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from src.core.epub import paragraph_alignment as alignment
from src.core.epub.dom_boundaries import audit_epub_dom_boundaries
from src.persistence.checkpoint_manager import CheckpointManager
from tests.unit.epub.test_paragraph_alignment import Provider, SOURCE, TARGET, _book


ANCHORS = '["Compra lo que quieras.", "Esperaron el tren nocturno."]'
GOOD = (
    "<p>Asintio. Tendre que comprar algunas cosas antes de irme de esta ciudad.</p>"
    "<p>Compra lo que quieras. Cargalo a la habitacion y traeme el recibo.</p>"
    "<p>Esperaron el tren nocturno. Volvia a llegar tarde.</p>"
)


def _chapters(path, paragraphs):
    with ZipFile(path, "w") as book:
        for index, body in enumerate(paragraphs):
            book.writestr(f"chapter-{index}.xhtml", "<html><body>" + body + "</body></html>")
        book.writestr("image.jpg", b"opaque image")
    return path


@pytest.mark.asyncio
async def test_checkpoint_rebuild_reuses_validated_alignment_without_second_request(tmp_path):
    source, target = _book(tmp_path / "source.epub", SOURCE), _book(tmp_path / "target.epub", TARGET)
    manager = CheckpointManager(str(tmp_path / "jobs.db"), uploads_dir=tmp_path / "uploads")
    cache = manager.uploads_dir / "job" / "paragraph_alignment.json"
    provider = Provider(ANCHORS)
    try:
        with ZipFile(target) as book:
            original = book.read("chapter.xhtml")
        assert manager.save_epub_file("job", "chapter.xhtml", original)
        assert await alignment.repair_epub_paragraph_alignment(
            source, target, config={}, provider=provider, cache_path=cache,
        ) == 1
        assert provider.calls == 1
        restored = tmp_path / "restored"
        assert manager.restore_epub_files("job", restored)
        assert (restored / "chapter.xhtml").read_bytes() == original
        with ZipFile(target, "w") as book:
            book.writestr("chapter.xhtml", original)
            book.writestr("image.jpg", b"opaque image")
        assert not audit_epub_dom_boundaries(source, target).clean
        assert await alignment.repair_epub_paragraph_alignment(
            source, target, config={}, provider=provider, cache_path=cache,
        ) == 1
        assert provider.calls == 1
        assert audit_epub_dom_boundaries(source, target).clean
        with ZipFile(target) as book:
            assert book.read("image.jpg") == b"opaque image"
        assert (restored / "chapter.xhtml").read_bytes() == original
    finally:
        manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cached_first", [True, False])
async def test_replay_does_not_consume_paid_cap_even_after_three_new_requests(tmp_path, cached_first):
    source = _chapters(tmp_path / "source.epub", [SOURCE] * 4)
    target = _chapters(tmp_path / "target.epub", [TARGET] * 4 if cached_first else [GOOD] * 3 + [TARGET])
    cache = tmp_path / "alignment.json"
    provider = Provider(ANCHORS)
    assert await alignment.repair_epub_paragraph_alignment(
        source, target, config={}, provider=provider, cache_path=cache,
    ) == (3 if cached_first else 1)
    assert provider.calls == (3 if cached_first else 1)
    _chapters(target, [TARGET] * 4)
    assert await alignment.repair_epub_paragraph_alignment(
        source, target, config={}, provider=provider, cache_path=cache,
    ) == 4
    assert provider.calls == 4
    assert audit_epub_dom_boundaries(source, target).clean


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", ["source", "target", "markup", "schema"])
async def test_changed_inputs_or_schema_do_not_reuse_old_anchors(tmp_path, changed):
    source, target = _book(tmp_path / "source.epub", SOURCE), _book(tmp_path / "target.epub", TARGET)
    cache = tmp_path / "alignment.json"
    provider = Provider(ANCHORS)
    assert await alignment.repair_epub_paragraph_alignment(
        source, target, config={}, provider=provider, cache_path=cache,
    ) == 1
    new_target = TARGET
    if changed == "source":
        _book(source, SOURCE.replace("They waited", "They lingered"))
    elif changed == "target":
        new_target = TARGET.replace("Asintio.", "Asintió.")
    elif changed == "markup":
        new_target = TARGET.replace("<p>", '<p class="prose">')
    else:
        data = json.loads(cache.read_text())
        data["schema"] = "paragraph-alignment/future-version"
        cache.write_text(json.dumps(data))
    _book(target, new_target)
    assert await alignment.repair_epub_paragraph_alignment(
        source, target, config={}, provider=provider, cache_path=cache,
    ) == 1
    assert provider.calls == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("corruption", ["invalid_json", "invalid_anchors", "result_hash"])
async def test_corrupt_cache_cannot_change_text_or_replace_normal_repair(tmp_path, corruption):
    source, target = _book(tmp_path / "source.epub", SOURCE), _book(tmp_path / "target.epub", TARGET)
    cache = tmp_path / "alignment.json"
    assert await alignment.repair_epub_paragraph_alignment(
        source, target, config={}, provider=Provider(ANCHORS), cache_path=cache,
    ) == 1
    if corruption == "invalid_json":
        cache.write_text("{broken")
    else:
        data = json.loads(cache.read_text())
        entry = next(iter(data["entries"].values()))
        if corruption == "invalid_anchors":
            entry["anchors"] = ["Fabricated start.", "Not in the translation."]
        else:
            entry["result_sha256"] = "0" * 64
        cache.write_text(json.dumps(data))
    _book(target, TARGET)
    before = target.read_bytes()
    provider = Provider("[]")
    assert await alignment.repair_epub_paragraph_alignment(
        source, target, config={}, provider=provider, cache_path=cache,
    ) == 0
    assert provider.calls == 2
    assert target.read_bytes() == before


@pytest.mark.asyncio
async def test_failed_alignment_is_never_cached(tmp_path):
    source, target = _book(tmp_path / "source.epub", SOURCE), _book(tmp_path / "target.epub", TARGET)
    cache = tmp_path / "alignment.json"
    before = target.read_bytes()
    assert await alignment.repair_epub_paragraph_alignment(
        source, target, config={}, provider=Provider("[]"), cache_path=cache,
    ) == 0
    assert not cache.exists()
    assert target.read_bytes() == before


@pytest.mark.asyncio
@pytest.mark.parametrize("exception", [RuntimeError, asyncio.CancelledError])
async def test_accepted_region_survives_a_later_failure_or_cancellation(tmp_path, exception):
    source = _chapters(tmp_path / "source.epub", [SOURCE] * 2)
    target = _chapters(tmp_path / "target.epub", [TARGET] * 2)
    cache = tmp_path / "alignment.json"
    before = target.read_bytes()

    class InterruptedProvider(Provider):
        async def generate(self, *args, **kwargs):
            if self.calls:
                raise exception("Later request interrupted")
            return await super().generate(*args, **kwargs)

    with pytest.raises(exception):
        await alignment.repair_epub_paragraph_alignment(
            source, target, config={}, provider=InterruptedProvider(ANCHORS), cache_path=cache,
        )
    assert len(json.loads(cache.read_text())["entries"]) == 1
    assert target.read_bytes() == before
    provider = Provider(ANCHORS)
    assert await alignment.repair_epub_paragraph_alignment(
        source, target, config={}, provider=provider, cache_path=cache,
    ) == 2
    assert provider.calls == 1
    assert audit_epub_dom_boundaries(source, target).clean


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["write", "replace_and_cleanup"])
async def test_cache_io_failure_does_not_discard_successful_repair(tmp_path, monkeypatch, failure):
    source, target = _book(tmp_path / "source.epub", SOURCE), _book(tmp_path / "target.epub", TARGET)
    cache = tmp_path / "alignment.json"
    messages = []
    if failure == "write":
        def fail_write(*args, **kwargs):
            raise OSError("Cache storage unavailable")
        monkeypatch.setattr(alignment.tempfile, "NamedTemporaryFile", fail_write)
    else:
        original_replace, original_unlink = Path.replace, Path.unlink

        def fail_replace(path, destination):
            if Path(destination) == cache:
                raise OSError("Cache replace unavailable")
            return original_replace(path, destination)

        def fail_cleanup(path, *args, **kwargs):
            if path.parent == tmp_path and path.name.startswith(".paragraph-alignment-"):
                raise OSError("Cache cleanup unavailable")
            return original_unlink(path, *args, **kwargs)

        monkeypatch.setattr(Path, "replace", fail_replace)
        monkeypatch.setattr(Path, "unlink", fail_cleanup)
    assert await alignment.repair_epub_paragraph_alignment(
        source, target, config={}, provider=Provider(ANCHORS), cache_path=cache,
        log_callback=lambda event, message: messages.append(event),
    ) == 1
    assert audit_epub_dom_boundaries(source, target).clean
    assert "paragraph_alignment_cache_warning" in messages
    assert not cache.exists()
