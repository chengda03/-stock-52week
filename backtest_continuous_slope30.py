# -*- coding: utf-8 -*-
"""
backtest_continuous_slope30.py
==============================
[연속 실전 시뮬] 1억원을 2020-01-01~2026-07-14까지 '리셋 없이' 계속 굴렸을 때의 실제 자산 흐름.
시장필터를 slope30(KOSPI>200 AND (KOSDAQ>200 OR KOSDAQ 30일모멘텀>0))으로 적용한 경우와
base(코스닥조건 없음)의 경우를 나란히 비교. strategy_core.py 미수정, strategy_core_kosdaqslope_30 재사용.

초기 1억 · 매수 0.115% / 매도 0.295% · 규칙은 backtest_v3와 동일(전량매도→부분익절→불타기→신규매수).
연말(마지막 거래일) 자산 스냅샷으로 연도별 시작/말/손익 정리.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import data_layer
import strategy_core as sc_base
from strategy_core import StockSnapshot, Position, MARKET_FILTER_MA_DAYS
from backtest_roe_eps_event import load_index
import strategy_core_kosdaqslope_30 as s30

INITIAL_CASH = 100_000_000
BUY_COST = 0.00115
SELL_COST = 0.00295
TRADE_START = pd.Timestamp("2020-01-01")

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)


def _build_snap(store, col, di) -> StockSnapshot:
    yr = int(store.day_fin_year[di])
    roe_a = store.roe_by_year.get(yr)
    eps_a = store.eps_by_year.get(yr)
    roe = float(roe_a[col]) if roe_a is not None else float("nan")
    eps = float(eps_a[col]) if eps_a is not None else float("nan")
    return StockSnapshot(
        ticker=store.valid_codes[col], date=store.trading_days[di].date(),
        price=float(store.close_v[di, col]), ma_buy=float(store.ma_buy_v[di, col]),
        ma_sell=float(store.ma_sell_v[di, col]), roe_pct=roe, eps=eps,
        momentum_20d=float(store.mom_v[di, col]),
        trading_value_20d_avg=float(store.tv_v[di, col]))


def run_continuous(store, is_bull_arr):
    """리셋 없이 전체 기간 연속. 일별 총자산 배열(모든 거래일 길이) 반환."""
    td = store.trading_days
    close_v = store.close_v; close_ff = store.close_ff
    c2c = store.code_to_col
    start_di = store.start_di

    cash = float(INITIAL_CASH)
    positions = {}; cost = {}
    daily_total = np.full(len(td), float(INITIAL_CASH))

    for di in range(len(td)):
        if di < start_di:
            daily_total[di] = INITIAL_CASH
            continue
        today = td[di].date()
        is_bull = bool(is_bull_arr[di])
        sold_today = set()

        for t in list(positions.keys()):
            col = c2c[t]; px = close_v[di, col]
            if not np.isfinite(px):
                continue
            pos = positions[t]
            s, _ = sc_base.check_sell_condition(pos, _build_snap(store, col, di))
            if s:
                cash += px * pos.shares * (1.0 - SELL_COST)
                del positions[t]; del cost[t]; sold_today.add(t)

        for t in list(positions.keys()):
            col = c2c[t]; px = close_v[di, col]
            if not np.isfinite(px) or px <= 0:
                continue
            pos = positions[t]
            if sc_base.check_partial_exit(pos, float(px)):
                before = pos.shares
                sh = sc_base.apply_partial_exit(pos, float(px))
                cash += px * sh * (1.0 - SELL_COST)
                cost[t] -= cost[t] * (sh / before)

        if is_bull and positions:
            def cur_ret(t):
                col = c2c[t]; px = close_v[di, col]
                return px / positions[t].avg_price if np.isfinite(px) else -1.0
            for t in sorted(positions.keys(), key=cur_ret, reverse=True):
                col = c2c[t]; px = close_v[di, col]
                if not np.isfinite(px) or px <= 0:
                    continue
                pos = positions[t]
                if sc_base.check_pyramid(pos, float(px), today, is_bull, cash):
                    spent = sc_base.PYRAMID_ADD_AMOUNT_WON * (1.0 + BUY_COST)
                    if cash < spent:
                        continue
                    sc_base.apply_pyramid(pos, float(px), today)
                    cash -= spent; cost[t] += spent

        free = sc_base.NUM_SLOTS - len(positions)
        if is_bull and free > 0:
            cands = []
            for t in store.get_universe(td[di]):
                if t in positions or t in sold_today:
                    continue
                col = c2c[t]; px = close_v[di, col]
                if not np.isfinite(px) or px <= 0:
                    continue
                s = _build_snap(store, col, di)
                ok, _ = sc_base.check_buy_conditions(s, True)
                if ok:
                    cands.append(s)
            for s in sc_base.rank_by_momentum(cands):
                if free <= 0:
                    break
                px = s.price; sh = int(sc_base.SLOT_AMOUNT_WON // px)
                if sh <= 0:
                    continue
                spent = sh * px * (1.0 + BUY_COST)
                if spent > cash:
                    continue
                positions[s.ticker] = Position(ticker=s.ticker, entry_price=px, avg_price=px,
                                               shares=float(sh), entry_date=today)
                cost[s.ticker] = spent; cash -= spent; free -= 1

        hv = 0.0
        for t, pos in positions.items():
            col = c2c[t]; p = close_v[di, col]
            if not np.isfinite(p):
                p = close_ff[di, col]
                if not np.isfinite(p):
                    p = pos.avg_price
            hv += p * pos.shares
        daily_total[di] = cash + hv

    return daily_total


def year_end_snapshots(td, daily_total):
    """각 연도 마지막 거래일 인덱스와 그날 총자산."""
    s = pd.Series(daily_total, index=td)
    out = {}
    for yr in range(2020, 2027):
        sub = s[(s.index >= pd.Timestamp(f"{yr}-01-01")) & (s.index <= pd.Timestamp(f"{yr}-12-31"))]
        if len(sub):
            out[yr] = (str(sub.index[-1].date()), float(sub.iloc[-1]))
    return out


def eok(x):
    return f"{x/1e8:,.4f}억"


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
    for di in range(len(td)):
        m = mom30[di] if np.isfinite(mom30[di]) else -1.0
        slope_bull[di] = s30.is_bull_market(kc[di], km[di], qc[di], q200[di], m)

    dt_base = run_continuous(store, kospi_bull)
    dt_slope = run_continuous(store, slope_bull)

    snap_b = year_end_snapshots(td, dt_base)
    snap_s = year_end_snapshots(td, dt_slope)

    final_b = dt_base[-1]; final_s = dt_slope[-1]
    last_date = str(td[-1].date())

    print("\n" + "=" * 96)
    print(f" 연속(리셋없음) 백테스트  ·  2020-01-01 ~ {last_date}  ·  초기 1억 · 거래비용 반영")
    print("=" * 96)
    print(f"  [base]    최종자산 {final_b:,.0f}원 ({eok(final_b)})  · 누적수익 {final_b-INITIAL_CASH:,.0f}원")
    print(f"  [slope30] 최종자산 {final_s:,.0f}원 ({eok(final_s)})  · 누적수익 {final_s-INITIAL_CASH:,.0f}원")
    print(f"  최종 금액 차이(slope30 - base) = {final_s-final_b:,.0f}원 ({(final_s-final_b)/1e8:,.4f}억)")

    # 연도별 표
    rows = []
    prev_b = INITIAL_CASH; prev_s = INITIAL_CASH
    for yr in range(2020, 2027):
        if yr not in snap_b:
            continue
        d_b, end_b = snap_b[yr]
        _, end_s = snap_s[yr]
        gain_b = end_b - prev_b; gain_s = end_s - prev_s
        ret_b = end_b / prev_b - 1.0; ret_s = end_s / prev_s - 1.0
        rows.append({
            "연도": yr, "기준일": d_b,
            "base_시작": round(prev_b/1e8, 4), "base_말": round(end_b/1e8, 4),
            "base_그해수익(원)": int(round(gain_b)), "base_수익%": round(ret_b*100, 2),
            "slope30_시작": round(prev_s/1e8, 4), "slope30_말": round(end_s/1e8, 4),
            "slope30_그해수익(원)": int(round(gain_s)), "slope30_수익%": round(ret_s*100, 2),
        })
        prev_b = end_b; prev_s = end_s
    df = pd.DataFrame(rows)
    print("\n" + "=" * 96)
    print(" 연도별 자산 흐름 (억원 단위 자산 + 그해 손익금액 원)")
    print("=" * 96)
    print(df.to_string(index=False))
    df.to_csv(os.path.join(RESULT_DIR, "continuous_slope30_vs_base.csv"), index=False, encoding="utf-8-sig")

    # 2024 손실 금액 강조
    r24 = next((r for r in rows if r["연도"] == 2024), None)
    if r24:
        print("\n" + "-" * 96)
        print(f" [2024 손실 금액] base {r24['base_그해수익(원)']:,}원 ({r24['base_수익%']}%)  vs  "
              f"slope30 {r24['slope30_그해수익(원)']:,}원 ({r24['slope30_수익%']}%)")
        print(f"   → 2024 한 해 금액 차이(slope30이 덜 잃은 금액) = "
              f"{r24['slope30_그해수익(원)']-r24['base_그해수익(원)']:,}원")
    print("=" * 96)


if __name__ == "__main__":
    main()
