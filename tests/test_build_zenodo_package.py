"""Goi Zenodo nhap: loc cot MRC, chuan hoa duong dan, quet nhay cam (du lieu gia)."""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import build_zenodo_package as bz  # noqa: E402

ROOTS = [("Q:/du_lieu", "<DATA_ROOT>"), ("Q:/repo", "<REPO>")]


def test_drop_columns_giu_nguyen_chuoi_va_bo_tch_tram(tmp_path):
    src, dst = tmp_path / "a.csv", tmp_path / "b.csv"
    src.write_text("cell_id,tch_wl_p20c,zos_mouth_p90,tram_x,rain_mm\nA,1.10,0.5000,9,3e-05\nB,,0.6,,4\n", encoding="utf-8")
    dropped = bz.drop_columns_csv(str(src), str(dst))
    assert dropped == ["tch_wl_p20c", "tram_x"]
    assert dst.read_text(encoding="utf-8") == "cell_id,zos_mouth_p90,rain_mm\nA,0.5000,3e-05\nB,0.6,4\n"  # zos_ giu, so khong doi dang


def test_normalize_duong_dan_kieu_windows_va_con_lai():
    t = "in Q:/du_lieu/features\\x.csv and Q:\\repo\\scripts and Z:/khac/thu/f.tif ok"
    assert bz.normalize_text(t, ROOTS) == "in <DATA_ROOT>/features/x.csv and <REPO>/scripts and <LOCAL>/f.tif ok"
    assert bz.normalize_json({"p": "Q:/du_lieu/a", "n": [1, "Q:/repo"]}, ROOTS) == {"p": "<DATA_ROOT>/a", "n": [1, "<REPO>"]}


def test_materialize_csvdrop_ghi_sidecar_va_dau_vet(tmp_path):
    src = tmp_path / "in" / "f.csv"
    src.parent.mkdir()
    src.write_text("cell_id,tch_wl_ok,v\nA,True,1\n", encoding="utf-8")
    json.dump({"script": "s.py", "path": "Q:/du_lieu/f.csv"}, open(str(src) + ".provenance.json", "w"))
    out = tmp_path / "out"
    r = bz.materialize(bz.Item(str(src), "features/x/f.csv", mode="csvdrop"), str(out))
    assert r["dropped"] == ["tch_wl_ok"]
    assert (out / "features/x/f.csv").read_text(encoding="utf-8") == "cell_id,v\nA,1\n"
    sc = json.load(open(out / "features/x/f.csv.provenance.json"))
    assert sc["zenodo_package"]["dropped_columns"] == ["tch_wl_ok"]
    assert sc["zenodo_package"]["source_sha256_before_filter"] == bz.file_sha256(str(src))
    assert bz.materialize(bz.Item(str(src), "features/x/f.csv", mode="csvdrop"), str(out)) == {"dropped": ["tch_wl_ok"]}  # chay lai: bo qua, van bao cot bo


def test_audit_bat_email_env_cot_mrc_va_duong_dan(tmp_path):
    (tmp_path / "features").mkdir()
    (tmp_path / "features" / "t.csv").write_text("cell_id,tch_wl_p20c\nA,1\n", encoding="utf-8")
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "n.md").write_text("lien he: nguoi.dung" + "@" + "example.org, file C:/may/rieng/x.csv", encoding="utf-8")
    (tmp_path / ".env").write_text("X=1", encoding="utf-8")
    errs, _ = bz.audit(str(tmp_path))
    kinds = {e[0] for e in errs}
    assert {"mrc_column", "email", "abs_path", "env_file"} <= kinds


def test_audit_sach(tmp_path):
    (tmp_path / "features").mkdir()
    (tmp_path / "features" / "t.csv").write_text("cell_id,zos_mouth_p90\nA,1\n", encoding="utf-8")
    assert bz.audit(str(tmp_path)) == ([], [])
