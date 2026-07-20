# -*- coding: utf-8 -*-
"""
backtest_continuous_dualindex.py
================================
[연속 실전 시뮬] 1억을 2020-01-01~2026-07-14까지 리셋 없이 이어서 굴린 실제 자산 흐름을
base / slope30 / dualindex(단순 200일선, 방향로직 없음) 세 가지로 비교.
strategy_core.py 미수정. strategy_core_kosdaqslope_30 / strategy_core_kosdaq_strict 재사용.
엔진은 backtest_continuous_slope30.run_continuous 재사용(동일 규칙·비용).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import data_layer
from strategy_core import MARKET_FILTER_MA_DAYS
from backtest_roe_eps_event import load_index
from backtest_continuous_slope30 import run_continuous, year_end_snapshots, INITIAL_CASH
import strategy_core_kosdaqslope_30 as s30
import strategy_core_kosdaq_strict as sstrict

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)


def main():
    store = data_layer.get_store()
    td = store.trading_days
    kc = store.kospi_close_v; km = store.kospi_ma200_v
    kospi_bull = kc > km

    kq = load_index("kosdaq_index").reindex(td).ffill().bfill()
    qc = kq.to_numpy(dtype=float)
    q200 = kq.rolling(MARKET_FILTER_MA_DAYS, min_periods=1).mean().to_numpy(dtype=float)
    mom30 = (kq / kq.shift(s30.KOSDAQ_MOM_DAYS) - 1.0).to_numpy(dtype=float)

    slope_bull = np.zeros(len(td), dtype=bool)
    dual_bull = np.zeros(len(td), dtype=bool)
    for di in range(len(td)):
        m = mom30[di] if np.isfinite(mom30[di]) else -1.0
        slope_bull[di] = s30.is_bull_market(kc[di], km[di], qc[di], q200[di], m)
        dual_bull[di] = sstrict.is_bull_market(kc[di], km[di], qc[di], q200[di])

    dt = {
        "base": run_continuous(store, kospi_bull),
        "slope30": run_continuous(store, slope_bull),
        "dualindex": run_continuous(store, dual_bull),
    }
    snap = {k: year_end_snapshots(td, v) for k, v in dt.items()}
    finals = {k: v[-1] for k, v in dt.items()}
    last_date = str(td[-1].date())

    print("\n" + "=" * 100)
    print(f" 연속(리셋없음) 백테스트  ·  2020-01-01 ~ {last_date}  ·  초기 1억 · 거래비용 반영")
    print("=" * 100)
    for k in ("base", "slope30", "dualindex"):
        f = finals[k]
        print(f"  [{k:<9s}] 최종자산 {f:,.0f}원 ({f/1e8:,.4f}억)  · 누적수익 {f-INITIAL_CASH:,.0f}원")
    print(f"\n  base - dualindex   = {finals['base']-finals['dualindex']:,.0f}원 "
          f"({(finals['base']-finals['dualindex'])/1e8:,.4f}억)")
    print(f"  slope30 - dualindex= {finals['slope30']-finals['dualindex']:,.0f}원 "
          f"({(finals['slope30']-finals['dualindex'])/1e8:,.4f}억)")

    # 연도별 표
    rows = []
    prev = {k: INITIAL_CASH for k in dt}
    for yr in range(2020, 2027):
        if yr not in snap["base"]:
            continue
        d = snap["base"][yr][0]
        row = {"연도": yr, "기준일": d}
        for k in ("base", "slope30", "dualindex"):
            end = snap[k][yr][1]
            row[f"{k}_말(억)"] = round(end / 1e8, 4)
            row[f"{k}_그해수익(원)"] = int(round(end - prev[k]))
            row[f"{k}_%"] = round(end / prev[k] * 100 - 100, 2)
            prev[k] = end
        rows.append(row)
    df = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print(" 연도별 자산 흐름 (말자산=억원, 그해수익=원)")
    print("=" * 100)
    # 보기 좋게 두 블록으로 출력
    cols_asset = ["연도", "기준일", "base_말(억)", "slope30_말(억)", "dualindex_말(억)"]
    cols_gain = ["연도", "base_그해수익(원)", "base_%", "slope30_그해수익(원)", "slope30_%",
                 "dualindex_그해수익(원)", "dualindex_%"]
    print(df[cols_asset].to_string(index=False))
    print()
    print(df[cols_gain].to_string(index=False))
    df.to_csv(os.path.join(RESULT_DIR, "continuous_three_compare.csv"), index=False, encoding="utf-8-sig")

    r24 = next((r for r in rows if r["연도"] == 2024), None)
    if r24:
        print("\n" + "-" * 100)
        print(" [2024 연속투자 실제 손익금액]")
        print(f"   base      {r24['base_그해수익(원)']:,}원 ({r24['base_%']}%)")
        print(f"   slope30   {r24['slope30_그해수익(원)']:,}원 ({r24['slope30_%']}%)")
        print(f"   dualindex {r24['dualindex_그해수익(원)']:,}원 ({r24['dualindex_%']}%)")
    print("=" * 100)
    print(" ⚠ 정직성: 6.4년 단일경로 결과. 매년리셋 방어력과 연속투자 최종자산은 서로 반대 방향일 수 있음.")
    print("=" * 100)


if __name__ == "__main__":
    main()
