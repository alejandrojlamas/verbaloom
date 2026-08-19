from scripts.code_stats import collect_stats


def test_code_stats_excludes_generated_and_local_artifact_directories(tmp_path):
    included = {
        tmp_path / "src" / "app.py": "print('app')\n",
        tmp_path / "tests" / "test_app.py": "def test_app(): pass\n",
    }
    excluded = {
        tmp_path / "output" / "report.json": "{}\n",
        tmp_path / "test-results" / "results.json": "{}\n",
        tmp_path / ".playwright-cli" / "session.json": "{}\n",
        tmp_path / "profiles" / "book" / "profile.yml": "title: Book\n",
        tmp_path / "profiles_backup_20260709" / "profile.yml": "title: Backup\n",
    }

    for path, content in {**included, **excluded}.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    stats = collect_stats(tmp_path)
    scanned_paths = {path.relative_to(tmp_path).as_posix() for path, _ in stats}

    assert scanned_paths == {"src/app.py", "tests/test_app.py"}
