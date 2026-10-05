"""Cau hinh rieng may qua .env (src/settings.py) + chan ghi cung thong tin nhay cam trong code (An 2026-10-05)."""
import importlib
import os
import re
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _reload(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("DATA_ROOT", raising=False)
    else:
        monkeypatch.setenv("DATA_ROOT", value)
    import settings
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)  # khong doc .env that trong test
    return importlib.reload(settings)


def test_data_path_theo_bien_moi_truong(monkeypatch):
    s = _reload(monkeypatch, "X:/du_lieu/")
    assert s.data_path("features", "unified") == "X:/du_lieu/features/unified"
    assert s.data_path("labels/points_reference.csv") == "X:/du_lieu/labels/points_reference.csv"
    assert s.require_data_root() == "X:/du_lieu"


def test_thieu_data_root_bao_ro(monkeypatch):
    s = _reload(monkeypatch, None)
    assert s.UNSET in s.data_path("features")
    with pytest.raises(RuntimeError, match="DATA_ROOT"):
        s.require_data_root()
    importlib.reload(s)  # tra lai trang thai (bien moi truong that) cho test khac


# Mau KHONG duoc xuat hien trong code commit: duong dan o dia cu the, ID project GEE, email, khoa.
FORBIDDEN = [
    (r"\b[A-Z]:[/\\]Dataset_NCKH", "duong dan du lieu ghi cung - dung settings.data_path / .env DATA_ROOT"),
    (r"project-[0-9a-f]{8}-[0-9a-f]{4}", "ID project GEE ghi cung - dung .env EE_PROJECT"),
    (r"[\w.]+@(gmail|sv\.sgu)\.\w+", "email ca nhan"),
    (r"(?i)(api[_-]?key|secret|password|passwd)\s*=\s*['\"][^'\"]+['\"]", "khoa/mat khau ghi cung"),
    (r"C:[/\\]Users[/\\]", "duong dan thu muc nguoi dung"),
]


def test_khong_ghi_cung_thong_tin_nhay_cam():
    bad = []
    for sub in ("src", "scripts", "tests", "main.py", "component", "webapp/backend"):
        base = os.path.join(ROOT, sub)
        files = [base] if base.endswith(".py") else [os.path.join(d, f) for d, _, fs in os.walk(base) for f in fs
                                                     if f.endswith(".py") and "__pycache__" not in d]
        for f in files:
            if not os.path.exists(f) or os.path.abspath(f) == os.path.abspath(__file__):
                continue
            text = open(f, encoding="utf-8", errors="replace").read()
            for pat, why in FORBIDDEN:
                for m in re.finditer(pat, text):
                    bad.append(f"{os.path.relpath(f, ROOT)}: '{m.group(0)}' ({why})")
    assert not bad, "\n".join(bad)
