# -*- coding: utf-8 -*-
"""
backtest_kosdaqslope_robust.py
==============================
[견고성검증] kosdaq_slope 로직(KOSPI>200 AND (KOSDAQ>200 OR KOSDAQ N일모멘텀>0))이
'20일'이라는 특정 lookback에 우연히 맞은 과최적화인지, 아니면 lookback 전 구간에서
완만하게 유지되는 진짜 신호인지 판정.

비교:
  · base            : KOSPI 단독(코스닥 조건 없음)
  · kosdaq_strict   : KOSPI>200 AND KOSDAQ>200 (방향로직 없음 = dualindex 동일, 안전 대안)
  · kosdaqslope_N   : N = 10/15/20/25/30/40일

판정 기준:
  - 10~40 전 구간에서 '2023 훼손 거의 없음 + 2024 상당히 방어'가 완만히 유지 → 로버스트(채택 검토)
  - 20 근처만 특별히 좋고 다른 값에서 확 나빠짐(뾰족/비단조) → 과최적화 → strict로 회귀

엔진·비용·규칙은 backtest_dualindex.run_engine 재사용(시장필터만 교체).
정직성: 단일 경로(약 6.5년). 하락/디커플링 표본이 얇아(2022·2024 각 1회) 근본적 과최적화 위험 상존.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import data_layer
from backtest_roe_eps_event import load_index
from backtest_dualindex import run_engine
from strategy_core import MARKET_FILTER_MA_DAYS

import strategy_core_kosdaqslope_10 as s10
import strategy_core_kosdaqslope_15 as s15
import strategy_core_kosdaqslope_20 as s20
import strategy_core_kosdaqslope_25 as s25
import strategy_core_kosdaqslope_30 as s30
import strategy_core_kosdaqslope_40 as s40
import strategy_core_kosdaq_strict as sstrict

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

SLOPE_MODS = [(10, s10), (15, s15), (20, s20), (25, s25), (30, s30), (40, s40)]
TAGS = ["base", "kosdaq_strict"] + [f"slope{n}" for n, _ in SLOPE_MODS]


def build_arrays(store):
    td = store.trading_days
    kc = store.kospi_close_v
    km = store.kospi_ma200_v
    kospi_bull = kc > km
    kq = load_index("kosdaq_index").reindex(td).ffill().bfill()
    qc = kq.to_numpy(dtype=float)
    q200 = kq.rolling(MARKET_FILTER_MA_DAYS, min_periods=1).mean().to_numpy(dtype=float)
    above200 = qc > q200
    mom = {n: (kq / kq.shift(n) - 1.0).to_numpy(dtype=float) for n, _ in SLOPE_MODS}
    return {"kc": kc, "km": km, "kospi_bull": kospi_bull, "qc": qc, "q200": q200,
            "above200": above200, "mom": mom}


def build_is_bull(store, tag, A):
    n = len(store.trading_days)
    kc, km = A["kc"], A["km"]
    if tag == "base":
        return A["kospi_bull"].copy()
    if tag == "kosdaq_strict":
        out = np.zeros(n, dtype=bool)
        for di in range(n):
            out[di] = sstrict.is_bull_market(kc[di], km[di], A["qc"][di], A["q200"][di])
        return out
    ndays = int(tag[5:])
    mod = dict(SLOPE_MODS)[ndays]
    momarr = A["mom"][ndays]
    out = np.zeros(n, dtype=bool)
    for di in range(n):
        m = momarr[di] if np.isfinite(momarr[di]) else -1.0
        out[di] = mod.is_bull_market(kc[di], km[di], A["qc"][di], A["q200"][di], m)
    return out


def main():
    store = data_layer.get_store()
    A = build_arrays(store)
    td = store.trading_days
    bull = {tag: build_is_bull(store, tag, A) for tag in TAGS}

    # 전체기간
    full = range(store.start_di, len(td))
    fullm = {tag: run_engine(store, bull[tag], full) for tag in TAGS}

    # 연도별
    yearly = {tag: {} for tag in TAGS}
    for wtag, ws, we in WINDOWS:
        wsd = pd.Timestamp(ws); wed = pd.Timestamp(we)
        di_list = [di for di in range(len(td)) if wsd <= td[di] <= wed]
        for tag in TAGS:
            yearly[tag][wtag] = run_engine(store, bull[tag], di_list)["period_ret"] * 100

    # (1) 전체기간 표
    rows = []
    for tag in TAGS:
        m = fullm[tag]
        rows.append({"버전": tag, "CAGR%": round(m["CAGR"]*100, 2), "MDD%": round(m["MDD"]*100, 2),
                     "Sharpe": round(m["Sharpe"], 2), "Calmar": round(m["Calmar"], 2),
                     "손익비": round(m["PL"], 2), "승률%": round(m["win_rate"]*100, 1),
                     "총거래": m["n_trades"], "신규매수": m["n_buy"]})
    dff = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print(f" (1) 전체기간 비교  ·  {fullm['base']['start']} ~ {fullm['base']['end']}  ·  1억 · 거래비용 반영")
    print("=" * 100)
    print(dff.to_string(index=False))
    dff.to_csv(os.path.join(RESULT_DIR, "kosdaqslope_robust_full.csv"), index=False, encoding="utf-8-sig")

    # (2) 연도별 표
    yr_rows = []
    for wtag, _, _ in WINDOWS:
        r = {"구간": wtag}
        for tag in TAGS:
            r[tag] = round(yearly[tag][wtag], 2)
        yr_rows.append(r)
    dfy = pd.DataFrame(yr_rows)
    print("\n" + "=" * 100)
    print(" (2) 연도별 독립 백테스트 — 기간수익률%  (매년 1억 리셋)")
    print("=" * 100)
    print(dfy.to_string(index=False))
    dfy.to_csv(os.path.join(RESULT_DIR, "kosdaqslope_robust_yearly.csv"), index=False, encoding="utf-8-sig")

    # (3) 견고성 요약: lookback별 2023훼손·2024방어 (base 대비)
    b23 = yearly["base"]["2023"]; b24 = yearly["base"]["2024"]
    print("\n" + "=" * 100)
    print(" (3) 견고성 요약 — base 대비 2023 훼손 / 2024 방어  (CAGR·MDD 병기)")
    print("=" * 100)
    srows = []
    for tag in TAGS:
        d23 = yearly[tag]["2023"] - b23
        d24 = yearly[tag]["2024"] - b24
        srows.append({"버전": tag,
                      "CAGR%": round(fullm[tag]["CAGR"]*100, 2),
                      "MDD%": round(fullm[tag]["MDD"]*100, 2),
                      "2023수익%": round(yearly[tag]["2023"], 2), "Δ2023(훼손)%p": round(d23, 2),
                      "2024수익%": round(yearly[tag]["2024"], 2), "Δ2024(방어)%p": round(d24, 2)})
    dfs = pd.DataFrame(srows)
    print(dfs.to_string(index=False))
    dfs.to_csv(os.path.join(RESULT_DIR, "kosdaqslope_robust_summary.csv"), index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print(" ⚠ 정직성: 단일 경로(약 6.5년). 디커플링/하락 표본이 얇아 lookback이 완만해도 근본적 과최적화 위험 상존.")
    print("=" * 100)


if __name__ == "__main__":
    main()
