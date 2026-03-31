import os
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
EXCLUDE_DIRS = {".git", "app", "__pycache__", ".github"}


def load_data() -> dict[str, list[dict[str, str]]]:
    """リポジトリルート配下のカテゴリディレクトリからデータを読み込む。

    Returns:
        {カテゴリ名: [{"date": "YYYY-MM-DD", "item": "食べたもの"}, ...]}
    """
    result: dict[str, list[dict[str, str]]] = {}

    for entry in sorted(REPO_ROOT.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name in EXCLUDE_DIRS or entry.name.startswith("."):
            continue

        items = []
        for txt in sorted(entry.glob("*.txt")):
            content = txt.read_text(encoding="utf-8").strip()
            if content:
                items.append({"date": txt.stem, "item": content})

        if items:
            result[entry.name] = items

    return result
