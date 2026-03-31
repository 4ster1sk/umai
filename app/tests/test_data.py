from pathlib import Path
from unittest.mock import patch

import pytest

from app.data import load_data


def _make_repo(tmp_path: Path) -> Path:
    """テスト用リポジトリ構造を作る。"""
    (tmp_path / "おやつ").mkdir()
    (tmp_path / "おやつ" / "2025-10-16.txt").write_text("赤福\n", encoding="utf-8")
    (tmp_path / "おやつ" / "2025-11-22.txt").write_text("福砂屋 カステラ", encoding="utf-8")
    (tmp_path / "ごはん").mkdir()
    (tmp_path / "ごはん" / "2025-12-01.txt").write_text("炊き込みご飯", encoding="utf-8")
    # 除外対象
    (tmp_path / ".git").mkdir()
    (tmp_path / "app").mkdir()
    (tmp_path / "__pycache__").mkdir()
    # 空ディレクトリ（結果に含まれないはず）
    (tmp_path / "空").mkdir()
    # 空ファイル（スキップされるはず）
    (tmp_path / "おやつ" / "2025-01-01.txt").write_text("", encoding="utf-8")
    return tmp_path


@pytest.fixture
def fake_repo(tmp_path):
    return _make_repo(tmp_path)


def test_categories_detected(fake_repo):
    with patch("app.data.REPO_ROOT", fake_repo):
        data = load_data()
    assert set(data.keys()) == {"おやつ", "ごはん"}


def test_excluded_dirs_not_in_result(fake_repo):
    with patch("app.data.REPO_ROOT", fake_repo):
        data = load_data()
    for excluded in (".git", "app", "__pycache__", "空"):
        assert excluded not in data


def test_items_parsed_correctly(fake_repo):
    with patch("app.data.REPO_ROOT", fake_repo):
        data = load_data()
    items = {e["item"] for e in data["おやつ"]}
    assert "赤福" in items
    assert "福砂屋 カステラ" in items


def test_empty_file_skipped(fake_repo):
    with patch("app.data.REPO_ROOT", fake_repo):
        data = load_data()
    items = [e["item"] for e in data["おやつ"]]
    assert "" not in items
    assert len(items) == 2


def test_empty_dir_excluded(fake_repo):
    with patch("app.data.REPO_ROOT", fake_repo):
        data = load_data()
    assert "空" not in data


def test_date_is_file_stem(fake_repo):
    with patch("app.data.REPO_ROOT", fake_repo):
        data = load_data()
    dates = {e["date"] for e in data["おやつ"]}
    assert "2025-10-16" in dates
    assert "2025-11-22" in dates
