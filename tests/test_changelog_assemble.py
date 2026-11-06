import os
import stat
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "changelog_assemble.py"
CHANGELOG = (
    "# Changelog\n\nIntro text.\n\n## Unreleased\n\n- Fix: existing entry.\n  continued.\n\n## v1.0.0\n\n- New: old.\n"
)


def make_repo(tmp_path, fragments=None, changelog=CHANGELOG):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "CHANGELOG.md").write_text(changelog, encoding="utf-8")
    fragment_dir = tmp_path / "changelog.d"
    fragment_dir.mkdir()
    (fragment_dir / "README.md").write_text("# Fragments\n\nNot a fragment.\n", encoding="utf-8")
    for name, text in (fragments or {}).items():
        (fragment_dir / name).write_text(text, encoding="utf-8")
    return tmp_path


def run(repo, *args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--repo", str(repo), *args], capture_output=True, text=True, check=False
    )


def changelog(repo):
    return (repo / "CHANGELOG.md").read_bytes()


def fragment_names(repo):
    return sorted(p.name for p in (repo / "changelog.d").iterdir())


def test_fragments_land_in_file_name_order_and_are_deleted(tmp_path):
    repo = make_repo(tmp_path, {"20-b.md": "- Fix: second.\n", "10-a.md": "- New: first.\n  more detail.\n"})
    result = run(repo)
    assert result.returncode == 0, result.stderr
    assert changelog(repo).decode() == CHANGELOG.replace(
        "## Unreleased\n\n", "## Unreleased\n\n- New: first.\n  more detail.\n- Fix: second.\n"
    )
    assert fragment_names(repo) == ["README.md"]


def test_keep_leaves_fragment_files(tmp_path):
    repo = make_repo(tmp_path, {"1-a.md": "- New: kept.\n"})
    result = run(repo, "--keep")
    assert result.returncode == 0, result.stderr
    assert b"- New: kept.\n- Fix: existing entry." in changelog(repo)
    assert fragment_names(repo) == ["1-a.md", "README.md"]


def test_check_changes_nothing(tmp_path):
    repo = make_repo(tmp_path, {"1-a.md": "- New: preview.\n"})
    before = changelog(repo)
    result = run(repo, "--check")
    assert result.returncode == 0, result.stderr
    assert "- New: preview." in result.stdout
    assert changelog(repo) == before
    assert fragment_names(repo) == ["1-a.md", "README.md"]


def test_heading_or_non_bullet_line_exits_1_and_changes_nothing(tmp_path):
    for bad in ("## Heading\n- New: x.\n", "- New: x.\nnot a bullet\n"):
        repo = make_repo(tmp_path / str(len(bad)), {"1-good.md": "- New: fine.\n", "2-bad.md": bad})
        before = changelog(repo)
        result = run(repo)
        assert result.returncode == 1
        assert "2-bad.md" in result.stderr
        assert changelog(repo) == before
        assert fragment_names(repo) == ["1-good.md", "2-bad.md", "README.md"]


def test_empty_fragment_exits_1(tmp_path):
    repo = make_repo(tmp_path, {"1-empty.md": "\n  \n"})
    before = changelog(repo)
    assert run(repo).returncode == 1
    assert changelog(repo) == before


def test_no_fragments_is_a_byte_identical_no_op(tmp_path):
    repo = make_repo(tmp_path)
    before = changelog(repo)
    result = run(repo)
    assert result.returncode == 0
    assert changelog(repo) == before


def test_readme_is_ignored(tmp_path):
    repo = make_repo(tmp_path, {"1-a.md": "- New: x.\n"})
    assert run(repo).returncode == 0
    assert b"Not a fragment" not in changelog(repo)
    assert fragment_names(repo) == ["README.md"]


def test_missing_unreleased_heading_exits_2(tmp_path):
    repo = make_repo(tmp_path, {"1-a.md": "- New: x.\n"}, changelog="# Changelog\n\n## v1.0.0\n\n- New: old.\n")
    before = changelog(repo)
    assert run(repo).returncode == 2
    assert changelog(repo) == before
    assert fragment_names(repo) == ["1-a.md", "README.md"]


def test_second_run_is_idempotent(tmp_path):
    repo = make_repo(tmp_path, {"1-a.md": "- New: once.\n"})
    assert run(repo).returncode == 0
    after_first = changelog(repo)
    assert run(repo).returncode == 0
    assert changelog(repo) == after_first
    assert after_first.count(b"- New: once.") == 1


def test_empty_unreleased_followed_by_heading(tmp_path):
    repo = make_repo(tmp_path, {"1-a.md": "- New: x.\n"}, changelog="# Changelog\n\n## Unreleased\n\n## v1.0.0\n\n- New: old.\n")
    assert run(repo).returncode == 0
    assert changelog(repo) == b"# Changelog\n\n## Unreleased\n\n- New: x.\n\n## v1.0.0\n\n- New: old.\n"


def test_fragment_without_trailing_newline(tmp_path):
    repo = make_repo(tmp_path, {"1-a.md": "- New: x.\n  more."})
    assert run(repo).returncode == 0
    assert changelog(repo) == CHANGELOG.replace("## Unreleased\n\n", "## Unreleased\n\n- New: x.\n  more.\n").encode()


def test_bad_file_name_exits_1_and_changes_nothing(tmp_path):
    for bad in ("Upper-Case.md", "1-a.markdown", "-lead.md", "under_score.md"):
        repo = make_repo(tmp_path / bad, {"1-good.md": "- New: fine.\n", bad: "- New: x.\n"})
        before = changelog(repo)
        result = run(repo)
        assert result.returncode == 1
        assert bad in result.stderr
        assert changelog(repo) == before
        assert fragment_names(repo) == sorted(["1-good.md", bad, "README.md"])


def test_file_mode_preserved_and_no_temp_file_left(tmp_path):
    repo = make_repo(tmp_path, {"1-a.md": "- New: x.\n"})
    os.chmod(repo / "CHANGELOG.md", 0o640)
    assert run(repo).returncode == 0
    assert stat.S_IMODE(os.stat(repo / "CHANGELOG.md").st_mode) == 0o640
    assert sorted(p.name for p in repo.iterdir()) == ["CHANGELOG.md", "changelog.d"]
