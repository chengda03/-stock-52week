# -*- coding: utf-8 -*-
"""
backtest_kospi_slope.py
=======================
[실험] 코스닥에서 검증된 '위치+방향' 로직(>200선 OR 20일모멘텀>0)을 KOSPI 시장필터에도 적용.

비교 4종(동일 DataStore·동일 날 → 드리프트 0, 차이는 오직 시장필터):
  · base         : KOSPI>200선 (위치만)
  · kosdaq_slope : KOSPI>200선 AND (KOSDAQ>200선 OR KOSDAQ20일모멘텀>0)   ← 앞 실험 최선후보
  · kospi_slope  : (KOSPI>200선 OR KOSPI20일모멘텀>0)                     ← 코스닥조건 없음
  · both_slope   : (KOSPI>200 OR KOSPI모멘텀>0) AND (KOSDAQ>200 OR KOSDAQ모멘텀>0)

핵심 사전확인: 2022년(명확한 하락장·base 완전방어)에 'KOSPI<200선 AND KOSPI20일모멘텀>0'
(=하락장 중 반등시도)인 날이 며칠? 많으면 KOSPI 방향로직이 2022 방어를 깰 위험.

엔진·비용·규칙은 backtest_dualindex.run_engine 재사용(시장필터만 교체).
정직성: 단일 경로(약 6.5년). 특히 2022 완전방어를 깨는 변경은 신중히. 과최적화 위험 존재.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import data_layer
from backtest_roe_eps_event import load_index
from backtest_dualindex import run_engine
from strategy_core import MARKET_FILTER_MA_DAYS

import strategy_core_kospi_slope as vks
import strategy_core_both_slope as vboth
import strategy_core_kosdaq_slope as vkq

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

TAGS = ["base", "kosdaq_slope", "kospi_slope", "both_slope"]


def build_arrays(store):
    td = store.trading_days
    kc = store.kospi_close_v
    km = store.kospi_ma200_v
    ks_ser = pd.Series(kc, index=td)
    kospi_mom20 = (ks_ser / ks_ser.shift(20) - 1.0).to_numpy(dtype=float)

    kq = load_index("kosdaq_index").reindex(td).ffill().bfill()
    qc = kq.to_numpy(dtype=float)
    q200 = kq.rolling(MARKET_FILTER_MA_DAYS, min_periods=1).mean().to_numpy(dtype=float)
    kosdaq_mom20 = (kq / kq.shift(20) - 1.0).to_numpy(dtype=float)
    return {"kc": kc, "km": km, "kospi_mom20": kospi_mom20,
            "qc": qc, "q200": q200, "kosdaq_mom20": kosdaq_mom20}


def build_is_bull(store, tag, A):
    n = len(store.trading_days)
    kc, km, kmom = A["kc"], A["km"], A["kospi_mom20"]
    qc, q200, qmom = A["qc"], A["q200"], A["kosdaq_mom20"]
    kospi_bull = kc > km
    if tag == "base":
        return kospi_bull.copy()
    out = np.zeros(n, dtype=bool)
    for di in range(n):
        km20 = kmom[di] if np.isfinite(kmom[di]) else -1.0
        qm20 = qmom[di] if np.isfinite(qmom[di]) else -1.0
        if tag == "kosdaq_slope":
            if kc[di] > km[di]:
                out[di] = vkq.is_bull_market(kc[di], km[di], qc[di], q200[di], qm20)
            else:
                out[di] = False
        elif tag == "kospi_slope":
            out[di] = vks.is_bull_market(kc[di], km[di], km20)
        elif tag == "both_slope":
            out[di] = vboth.is_bull_market(kc[di], km[di], km20, qc[di], q200[di], qm20)
    return out


def precheck_2022(store, A):
    td = store.trading_days
    mask = (td >= pd.Timestamp("2022-01-01")) & (td <= pd.Timestamp("2022-12-31"))
    kc, km, kmom = A["kc"], A["km"], A["kospi_mom20"]
    ndays = int(mask.sum())
    below = (kc < km) & mask
    below_up = below & (kmom > 0.0)     # 하락장 중 20일 반등
    nbelow = int(below.sum()); nbu = int(below_up.sum())
    print("\n" + "=" * 92)
    print(" (사전확인) 2022년: KOSPI<200선(하락장) 중 20일모멘텀>0(반등시도)인 날")
    print("=" * 92)
    print(f"  2022 거래일 {ndays}일 · KOSPI<200선 {nbelow}일 · 그 중 20일모멘텀>0 : {nbu}일 "
          f"({nbu/ndays*100:.1f}% of 거래일, {nbu/nbelow*100:.1f}% of 하락일)")
    print("  → 이 날들에 kospi_slope/both_slope는 '강세장'으로 판정해 신규매수를 허용(=2022 방어 약화 위험)")
    # 전체기간에도 같은 패턴
    for tag, ws, we in [("전체", "2020-01-01", "2026-07-14")]:
        m2 = (td >= pd.Timestamp(ws)) & (td <= pd.Timestamp(we))
        bl = int(((kc < km) & m2).sum()); blu = int(((kc < km) & (kmom > 0) & m2).sum())
        print(f"  [전체] KOSPI<200선 {bl}일 · 그 중 20일모멘텀>0 {blu}일")


def main():
    store = data_layer.get_store()
    A = build_arrays(store)
    td = store.trading_days

    precheck_2022(store, A)

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
    print("\n" + "=" * 92)
    print(f" (1) 전체기간 비교  ·  {fullm['base']['start']} ~ {fullm['base']['end']}  ·  1억 · 거래비용 반영")
    print("=" * 92)
    print(dff.to_string(index=False))
    dff.to_csv(os.path.join(RESULT_DIR, "kospi_slope_full.csv"), index=False, encoding="utf-8-sig")

    # (2) 연도별
    yr_rows = []
    ymdd_rows = []
    for wtag, ws, we in WINDOWS:
        wsd = pd.Timestamp(ws); wed = pd.Timestamp(we)
        di_list = [di for di in range(len(td)) if wsd <= td[di] <= wed]
        r = {"구간": wtag}; rm = {"구간": wtag}
        for tag in TAGS:
            m = run_engine(store, bull[tag], di_list)
            r[tag] = round(m["period_ret"]*100, 2)
            rm[tag] = round(m["MDD"]*100, 2)
        yr_rows.append(r); ymdd_rows.append(rm)
    dfy = pd.DataFrame(yr_rows); dfm = pd.DataFrame(ymdd_rows)
    print("\n" + "=" * 92)
    print(" (2) 연도별 독립 백테스트 — 기간수익률%  (매년 1억 리셋)")
    print("=" * 92)
    print(dfy.to_string(index=False))
    print("\n [참고] 연도별 MDD%")
    print(dfm.to_string(index=False))
    dfy.to_csv(os.path.join(RESULT_DIR, "kospi_slope_yearly.csv"), index=False, encoding="utf-8-sig")
    dfm.to_csv(os.path.join(RESULT_DIR, "kospi_slope_yearly_mdd.csv"), index=False, encoding="utf-8-sig")

    print("\n" + "=" * 92)
    print(" ⚠ 정직성: 단일 경로(약 6.5년). 2022 완전방어를 깨는 변경은 특히 신중히. 과최적화 위험 존재.")
    print("=" * 92)


if __name__ == "__main__":
    main()
