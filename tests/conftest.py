"""Cho phep chay toan bo test (pipeline + webapp) trong mot phien pytest.

src/config.py va webapp/backend/config.py trung ten module `config`. Module
nao duoc import truoc se chiem sys.modules["config"] va lam hong phia con lai.
Tep nay dang ky mot module `config` hop nhat: lay src/config.py lam goc, bo
sung cac thuoc tinh chi co trong webapp/backend/config.py. Khong sua file
nguon nao; chi co tac dung khi chay pytest.
"""
import importlib.util
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
WEBAPP_DIR = os.path.join(REPO_ROOT, "webapp", "backend")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if WEBAPP_DIR not in sys.path:
    sys.path.append(WEBAPP_DIR)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


config_mod = _load("config", os.path.join(SRC_DIR, "config.py"))

web_config_path = os.path.join(WEBAPP_DIR, "config.py")
if os.path.exists(web_config_path):
    web_config = _load("webapp_backend_config", web_config_path)
    for attr in dir(web_config):
        if not attr.startswith("__") and not hasattr(config_mod, attr):
            setattr(config_mod, attr, getattr(web_config, attr))

sys.modules["config"] = config_mod
