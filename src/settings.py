"""Cau hinh rieng tung may: doc file `.env` o goc repo (KHONG commit - da co trong .gitignore).

Moi thong tin nhay cam / phu thuoc may (duong dan du lieu, ID project GEE, thu muc Drive, khoa API...) dat trong
`.env`; code chi doc qua module nay hoac os.getenv. Mau cac bien: `.env.example` (co commit, khong chua gia tri that).
Bien moi truong co san (dat ngoai .env) duoc UU TIEN hon .env (load_dotenv override=False).

  DATA_ROOT         thu muc du lieu goc (vd D:/du_lieu/mekong)
  EE_PROJECT        Cloud project bat Earth Engine (scripts/gee_fetch.py ...)
  GEE_DRIVE_FOLDER  ten thu muc Google Drive nhan file xuat tu GEE
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = REPO_ROOT / ".env"

try:  # python-dotenv co trong requirements; thieu thi chi dung bien moi truong san co
    from dotenv import load_dotenv

    load_dotenv(ENV_FILE, override=False)
except ImportError:  # pragma: no cover
    pass

DATA_ROOT = os.getenv("DATA_ROOT", "").strip().rstrip("/\\")
UNSET = "<DATA_ROOT chua dat trong .env>"


def data_path(*parts: str) -> str:
    """Duong dan trong DATA_ROOT, dau '/' (vd data_path("features", "unified")).

    DATA_ROOT chua dat -> tra chuoi co ghi chu UNSET de loi "khong tim thay file" chi ro nguyen nhan; dung
    `require_data_root()` khi can bao loi ngay.
    """
    base = DATA_ROOT or UNSET
    return "/".join([base, *[p.strip("/\\") for p in parts if p]])


def require_data_root() -> str:
    if not DATA_ROOT:
        raise RuntimeError(f"Chua dat DATA_ROOT: tao {ENV_FILE} tu .env.example va dien DATA_ROOT=<thu muc du lieu>.")
    return DATA_ROOT
