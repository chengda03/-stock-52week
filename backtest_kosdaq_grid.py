# -*- coding: utf-8 -*-
"""
backtest_kosdaq_grid.py
=======================
[그리드] 코스닥 지수 조건을 여러 방식으로 설계해 시장필터에 넣어 비교.
기준선(60일선삭제+90일선이격도2.5%+ROE≥15%+거래대금30억+저평가+20일모멘텀+20슬롯+무제한불타기
+부분익절8%@50%+하드손절-12%)은 동일, 오직 '시장필터(is_bull)'만 변형.

비교 7종(동일 DataStore·동일 날 → 드리프트 0):
  · base            : KOSPI 단독(>200일선)
  · dualindex       : KOSPI>200 AND KOSDAQ>200
  · kosdaq_ma120    : KOSPI>200 AND KOSDAQ>120일선
  · kosdaq_ma150    : KOSPI>200 AND KOSDAQ>150일선
  · kosdaq_slope    : KOSPI>200 AND (KOSDAQ>200 OR KOSDAQ 20일수익률>0)
  · kosdaq_crashonly: KOSPI>200 AND KOSDAQ 60일수익률 > -15%
  · kosdaq_reentry  : KOSPI>200 AND (KOSDAQ>200 OR 200선 상향돌파 후 20일 이내)

엔진(run_engine)·비용·규칙은 backtest_dualindex.py 것을 그대로 재사용(시장필터만 교체).
정직성: 단일 경로(약 6.5년) 그리드서치 → 과최적화 위험 큼.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import data_layer
from backtest_roe_eps_event import load_index
from backtest_dualindex import run_engine, INITIAL_CASH
from strategy_core import MARKET_FILTER_MA_DAYS

import strategy_core_kosdaq_ma120 as v120
import strategy_core_kosdaq_ma150 as v150
import strategy_core_kosdaq_slope as vslope
import strategy_core_kosdaq_crashonly as vcrash
import strategy_core_kosdaq_reentry as vre

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

WINDOWS = [
    ("2020", "2020-01-01", "2020-12-31"),
    ("2021", "2021-01-01", "2021-12-31"),
    ("2022(하락장)", "2022-01-01", "2022-12-31"),
    ("2023", "2023-01-01", "2023-12-31"),
    ("2024", "2024-01-01", "2024-12-31"),
    ("2025", "2025-01-01", "2025-12-31"),
    ("2026(~7/14)", "2026-01-01", "2026-07-14"),
]


def _series(store, name):
    return load_index(name).reindex(store.trading_days).ffill().bfill()


def build_arrays(store):
    td = store.trading_days
    kc = store.kospi_close_v
    km = store.kospi_ma200_v
    kospi_bull = kc > km

    kq = _series(store, "kosdaq_index")
    qc = kq.to_numpy(dtype=float)
    q200 = kq.rolling(MARKET_FILTER_MA_DAYS, min_periods=1).mean().to_numpy(dtype=float)
    q120 = kq.rolling(120, min_periods=1).mean().to_numpy(dtype=float)
    q150 = kq.rolling(150, min_periods=1).mean().to_numpy(dtype=float)
    mom20 = (kq / kq.shift(20) - 1.0).to_numpy(dtype=float)
    ret60 = (kq / kq.shift(60) - 1.0).to_numpy(dtype=float)

    above200 = qc > q200
    # 200선 상향돌파 후 REENTRY_WINDOW 거래일 이내 여부
    n = len(td)
    in_reentry = np.zeros(n, dtype=bool)
    last_cross = -10 ** 9
    prev = False
    for di in range(n):
        cur = bool(above200[di])
        if cur and not prev:
            last_cross = di
        in_reentry[di] = (di - last_cross) < vre.REENTRY_WINDOW
        prev = cur

    arrays = {"kc": kc, "km": km, "kospi_bull": kospi_bull, "qc": qc, "q200": q200,
              "q120": q120, "q150": q150, "mom20": mom20, "ret60": ret60,
              "above200": above200, "in_reentry": in_reentry}
    return arrays


def build_is_bull(store, tag, A):
    """각 변형의 is_bull 함수(변형 파일)를 하루씩 호출해 per-day bool 배열 생성."""
    n = len(store.trading_days)
    kc, km = A["kc"], A["km"]
    out = np.zeros(n, dtype=bool)
    if tag == "base":
        return A["kospi_bull"].copy()
    if tag == "dualindex":
        return A["kospi_bull"] & A["above200"]
    for di in range(n):
        if tag == "kosdaq_ma120":
            out[di] = v120.is_bull_market(kc[di], km[di], A["qc"][di], A["q120"][di])
        elif tag == "kosdaq_ma150":
            out[di] = v150.is_bull_market(kc[di], km[di], A["qc"][di], A["q150"][di])
        elif tag == "kosdaq_slope":
            out[di] = vslope.is_bull_market(kc[di], km[di], A["qc"][di], A["q200"][di],
                                            A["mom20"][di] if np.isfinite(A["mom20"][di]) else -1.0)
        elif tag == "kosdaq_crashonly":
            r = A["ret60"][di]
            out[di] = vcrash.is_bull_market(kc[di], km[di], r if np.isfinite(r) else -1.0)
        elif tag == "kosdaq_reentry":
            out[di] = vre.is_bull_market(kc[di], km[di], bool(A["above200"][di]),
                                         bool(A["in_reentry"][di]))
    return out


TAGS = ["base", "dualindex", "kosdaq_ma120", "kosdaq_ma150",
        "kosdaq_slope", "kosdaq_crashonly", "kosdaq_reentry"]


def main():
    store = data_layer.get_store()
    A = build_arrays(store)
    td = store.trading_days
    bull = {tag: build_is_bull(store, tag, A) for tag in TAGS}

    # (1) 전체기간
    full = range(store.start_di, len(td))
    fullm = {tag: run_engine(store, bull[tag], full) for tag in TAGS}
    rows = []
    for tag in TAGS:
        m = fullm[tag]
        rows.append({"버전": tag, "CAGR%": round(m["CAGR"]*100, 2), "MDD%": round(m["MDD"]*100, 2),
                     "Sharpe": round(m["Sharpe"], 2), "Calmar": round(m["Calmar"], 2),
                     "손익비": round(m["PL"], 2), "승률%": round(m["win_rate"]*100, 1),
                     "총거래": m["n_trades"], "신규매수": m["n_buy"], "불타기": m["n_add"]})
    dff = pd.DataFrame(rows)
    print("\n" + "=" * 104)
    print(f" (1) 전체기간 비교  ·  {fullm['base']['start']} ~ {fullm['base']['end']}  ·  1억 · 거래비용 반영")
    print("=" * 104)
    print(dff.to_string(index=False))
    dff.to_csv(os.path.join(RESULT_DIR, "kosdaq_grid_full.csv"), index=False, encoding="utf-8-sig")

    # (2) 연도별 독립
    yearly = {tag: {} for tag in TAGS}
    for wtag, ws, we in WINDOWS:
        wsd = pd.Timestamp(ws); wed = pd.Timestamp(we)
        di_list = [di for di in range(len(td)) if wsd <= td[di] <= wed]
        for tag in TAGS:
            yearly[tag][wtag] = run_engine(store, bull[tag], di_list)["period_ret"] * 100

    yr_rows = []
    for wtag, _, _ in WINDOWS:
        row = {"구간": wtag}
        for tag in TAGS:
            row[tag] = round(yearly[tag][wtag], 2)
        yr_rows.append(row)
    dfy = pd.DataFrame(yr_rows)
    print("\n" + "=" * 104)
    print(" (2) 연도별 독립 백테스트 — 기간수익률%  (매년 1억 리셋)")
    print("=" * 104)
    print(dfy.to_string(index=False))
    dfy.to_csv(os.path.join(RESULT_DIR, "kosdaq_grid_yearly.csv"), index=False, encoding="utf-8-sig")

    # (3) 2024 방어 vs 2023 희생 (base 대비)
    b2024 = yearly["base"]["2024"]; b2023 = yearly["base"]["2023"]; b2025 = yearly["base"]["2025"]
    print("\n" + "=" * 104)
    print(" (3) base 대비 '2024 방어' vs '2023 희생'  (방어=Δ2024>0 클수록 좋음, 희생=Δ2023<0 작을수록 좋음)")
    print("=" * 104)
    trows = []
    for tag in TAGS:
        d24 = yearly[tag]["2024"] - b2024
        d23 = yearly[tag]["2023"] - b2023
        d25 = yearly[tag]["2025"] - b2025
        if tag in ("base",):
            ratio = "-"
        elif d23 >= 0:
            ratio = "∞(희생없음)"
        else:
            ratio = round(d24 / abs(d23), 2)
        trows.append({"버전": tag,
                      "2024수익%": round(yearly[tag]["2024"], 2), "Δ2024(방어)%p": round(d24, 2),
                      "2023수익%": round(yearly[tag]["2023"], 2), "Δ2023(희생)%p": round(d23, 2),
                      "Δ2025%p": round(d25, 2),
                      "방어/희생비율": ratio,
                      "전체CAGR%": round(fullm[tag]["CAGR"]*100, 2)})
    dft = pd.DataFrame(trows)
    print(dft.to_string(index=False))
    dft.to_csv(os.path.join(RESULT_DIR, "kosdaq_grid_tradeoff.csv"), index=False, encoding="utf-8-sig")

    print("\n" + "=" * 104)
    print(" ⚠ 정직성: 단일 경로(약 6.5년) 그리드서치. 여러 코스닥 조건 중 좋아 보이는 것은 과최적화일 수 있음.")
    print("=" * 104)


if __name__ == "__main__":
    main()
