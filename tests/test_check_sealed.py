"""check_sealed.py: sha256 trung -> DAT; khac -> so tung o theo khoa (dung sai so, chuoi bang nhau); khong in gia tri."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import check_sealed  # noqa: E402

SECRET = 0.123456789  # gia tri dac biet: khong duoc xuat hien trong output


def _df():
    return pd.DataFrame({
        "family": ["F1", "F1", "F1", "phu"],
        "cmp_id": ["a-b", "a-c", "b-c", "a-b"],
        "delta_hat": [SECRET, -0.5, 0.25, 1.0],
        "p_holm": [0.01, np.nan, 1.0, 0.5],
        "label": ["cho_giu_rieng", "mo_ta", "tuong_duong", "chua_phan_dinh"],
        "seed_deltas": ["0.1;0.2;0.3", "1;2;3", "-1;-2;-3", "0;0;0"],
        "kiem_dinh": [True, False, True, True],
    })


def _write(df, path):
    df.to_csv(path, index=False)
    return str(path)


def _run(capsys, *args):
    code = check_sealed.main([str(a) for a in args])
    return code, capsys.readouterr().out


def test_sha_trung_dat(tmp_path, capsys):
    a = _write(_df(), tmp_path / "a.csv")
    b = _write(_df(), tmp_path / "b.csv")
    code, out = _run(capsys, a, b)
    assert code == 0 and "sha256 trung" in out


def test_khac_byte_trong_dung_sai_dat(tmp_path, capsys):
    d = _df()
    a = _write(d, tmp_path / "a.csv")
    d2 = d.iloc[::-1].copy()                      # dao thu tu dong
    d2["delta_hat"] = d2["delta_hat"] + 1e-12     # lech duoi dung sai
    b = _write(d2, tmp_path / "b.csv")
    code, out = _run(capsys, a, b)
    assert code == 0 and "sha256 khac" in out and "thu tu dong khac" in out and "DAT: moi o" in out
    assert str(SECRET)[:8] not in out


@pytest.mark.parametrize("col,val", [("delta_hat", SECRET + 1e-6), ("p_holm", 0.02), ("label", "xac_nhan"),
                                     ("seed_deltas", "0.1;0.2;0.31"), ("kiem_dinh", False)])
def test_lech_vuot_ma_2_khong_in_gia_tri(tmp_path, capsys, col, val):
    d = _df()
    a = _write(d, tmp_path / "a.csv")
    d2 = d.copy()
    d2.loc[0, col] = val
    b = _write(d2, tmp_path / "b.csv")
    code, out = _run(capsys, a, b)
    assert code == 2 and f"cot {col}: 1 o vuot" in out and f"cot lech ['{col}']" in out
    for s in (str(SECRET)[:8], "0.31", "xac_nhan", "cho_giu_rieng"):
        assert s not in out


def test_nan_chi_bang_nan(tmp_path, capsys):
    d = _df()
    a = _write(d, tmp_path / "a.csv")
    d2 = d.copy()
    d2.loc[1, "p_holm"] = 0.0
    code, out = _run(capsys, a, _write(d2, tmp_path / "b.csv"))
    assert code == 2 and "cot p_holm: 1 o" in out


def test_cols_chi_so_tap_cot(tmp_path, capsys):
    d = _df()
    a = _write(d, tmp_path / "a.csv")
    d2 = d.copy()
    d2["p_holm"] = d2["p_holm"] * 2
    d2["label"] = "mo_ta"
    b = _write(d2, tmp_path / "b.csv")
    assert _run(capsys, a, b, "--cols", "delta_hat", "seed_deltas")[0] == 0
    assert _run(capsys, a, b)[0] == 2
    code, out = _run(capsys, a, b, "--cols", "delta_hat", "khong_co")
    assert code == 2 and "thieu cot can so ['khong_co']" in out


def test_cau_truc_khac_ma_2(tmp_path, capsys):
    d = _df()
    a = _write(d, tmp_path / "a.csv")
    code, out = _run(capsys, a, _write(d.iloc[:3], tmp_path / "thieu_dong.csv"))
    assert code == 2 and "khoa chi o file moi: 1, chi o ban niem phong: 0" in out
    code, out = _run(capsys, a, _write(pd.concat([d, d.iloc[[0]]]), tmp_path / "trung.csv"))
    assert code == 2 and "trung khoa" in out
    code, out = _run(capsys, a, _write(d.drop(columns="label"), tmp_path / "thieu_cot.csv"))
    assert code == 2 and "chi o file moi ['label']" in out
    code, out = _run(capsys, a, _write(d.drop(columns="label"), tmp_path / "thieu_cot2.csv"), "--exclude", "label")
    assert code == 0 and "bo qua cot: ['label']" in out


def test_thieu_file_ma_1(tmp_path, capsys):
    a = _write(_df(), tmp_path / "a.csv")
    assert _run(capsys, a, tmp_path / "khong_co.csv")[0] == 1
    with pytest.raises(SystemExit) as e:
        check_sealed.main([a])
    assert e.value.code == 1
