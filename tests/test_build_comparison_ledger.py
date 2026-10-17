"""Test build_comparison_ledger: gom dong cap khung cung muc x bien tu bang tong hop gia lap."""
import pandas as pd

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from build_comparison_ledger import decide, ledger_for_target  # noqa: E402


def _kd(label, a, b):
    return dict(family="F1_chinh", grid_a=a, grid_b=b, delta_ref_level=5, area_ratio=1.0, G=19,
                n_common_point_seasons=10, delta_hat=1.0, delta_ref=10.0, delta_min_thr=0.5, ci_tost_low=0, ci_tost_high=2,
                ci_low=-1, ci_high=3, p_value=0.01, p_holm=0.02, seeds_same_dir=True, label=label,
                cong_hoc_duoc=True, muc_kiem_dinh=True)


def test_decision_hai_cap():
    kd = pd.DataFrame([_kd("cho_giu_rieng", "h3_res_5", "latlon_0.1552deg"), _kd("tuong_duong", "h3_res_5", "square_utm_17087m")])
    cap = pd.DataFrame([dict(grid_a="h3_res_5", grid_b="latlon_0.1552deg", label="x", label_cap="khong_tai_lap"),
                        dict(grid_a="h3_res_5", grid_b="square_utm_17087m", label="x", label_cap="tuong_duong")])
    area = {g: 1.0 for g in ["h3_res_5", "latlon_0.1552deg", "square_utm_17087m"]}
    r = ledger_for_target("v", kd, cap, None, area)
    assert [x["decision"] for x in r] == ["Inconclusive", "Equivalent"]
    assert r[0]["delta_rel"] == 0.1 and r[0]["level"] == "coarse"


def test_khong_co_luot_final_cho_giu_rieng_la_pending():
    kd = pd.DataFrame([_kd("cho_giu_rieng", "h3_res_5", "latlon_0.1552deg"), _kd("tuong_duong", "h3_res_5", "square_utm_17087m")])
    area = {g: 1.0 for g in ["h3_res_5", "latlon_0.1552deg", "square_utm_17087m"]}
    r = ledger_for_target("v", kd, None, None, area)
    assert [x["decision"] for x in r] == ["PendingFinal", "Equivalent"]
    assert {x["holdout_confirmed"] for x in r} == {"khong_co_luot_final"}


def test_decide():
    assert decide("cho_giu_rieng", "cho_giu_rieng") == "Different"
    assert decide("mo_ta", None) == "Descriptive"
    assert decide("chua_phan_dinh", "chua_phan_dinh") == "Inconclusive"
