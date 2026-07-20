# -*- coding: utf-8 -*-
"""
backtest_slope_compare.py
=========================
[실험] 확정 조건 위에서 시장필터에 '200일선 기울기(20일)' 조건을 추가.
2024년 손실의 추정 원인 "종가는 200일선 위지만 200일선이 평평한 가짜 강세장"을 걸러내는지 검증.

  기울기(%) = (오늘 200일선 - 20거래일전 200일선) / 20거래일전 200일선 × 100
  시장필터 = 종가>200일선 AND 기울기 > X%   (X = 0 / 1 / 2)

  · base    : 기울기 조건 없음 (원본 strategy_core, 종가>200일선만)
  · slope0/1/2 : 기울기 > 0/1/2%

먼저: [진단] 연도별로 '종가>200일선'인 날들의 기울기 분포(≤0 / 0~1 / 1~3 / ≥3%) →
      2024가 실제로 더 평평했는지(2023/2025/2026 대비) 확인 → 필터의 논리적 타당성 근거.
그다음: 전체기간 지표 + 연도별 독립(매년 1억 리셋) 비교.

판정/엔진은 strategy_core 함수 그대로. 시장필터(is_bull)만 기울기 조건 포함으로 교체
(신규매수·불타기 모두 이 필터로 게이팅 — 원본과 동일 구조).
정직성: 단일 경로 백테스트. 2024 완화 vs 대박구간 보존을 그대로 보고. 과최적화 주의.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

from strategy_core import (
    StockSnapshot, Position,
    check_buy_conditions, rank_by_momentum,
    check_sell_condition, check_pyramid, apply_pyramid,
    check_partial_exit, apply_partial_exit,
    NUM_SLOTS, SLOT_AMOUNT_WON, PYRAMID_ADD_AMOUNT_WON,
)
import data_layer

INITIAL_CASH = 100_000_000
BUY_COST = 0.00115
SELL_COST = 0.00295
SLOPE_LOOKBACK = 20

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

WINDOWS = [
    ("전체", "2020-01-01", "2026-07-14"),
    ("2020", "2020-01-01", "2020-12-31"),
    ("2021", "2021-01-01", "2021-12-31"),
    ("2022", "2022-01-01", "2022-12-31"),
    ("2023", "2023-01-01", "2023-12-31"),
    ("2024", "2024-01-01", "2024-12-31"),
    ("2025", "2025-01-01", "2025-12-31"),
    ("2026", "2026-01-01", "2026-07-14"),
]


def slope_panel(store) -> np.ndarray:
    """KOSPI 200일선의 20일 기울기(%) 패널."""
    ma = store.kospi_ma200_v.astype(float)
    prev = np.empty_like(ma); prev[:] = np.nan
    prev[SLOPE_LOOKBACK:] = ma[:-SLOPE_LOOKBACK]
    with np.errstate(invalid="ignore", divide="ignore"):
        return (ma - prev) / prev * 100.0


def _build_snap(store, col, di) -> StockSnapshot:
    yr = int(store.day_fin_year[di])
    roe_a = store.roe_by_year.get(yr)
    eps_a = store.eps_by_year.get(yr)
    roe = float(roe_a[col]) if roe_a is not None else float("nan")
    eps = float(eps_a[col]) if eps_a is not None else float("nan")
    return StockSnapshot(
        ticker=store.valid_codes[col],
        date=store.trading_days[di].date(),
        price=float(store.close_v[di, col]),
        ma_buy=float(store.ma_buy_v[di, col]),
        ma_sell=float(store.ma_sell_v[di, col]),
        roe_pct=roe, eps=eps,
        momentum_20d=float(store.mom_v[di, col]),
        trading_value_20d_avg=float(store.tv_v[di, col]),
    )


def run_window(store, win_start, win_end, slope_v, slope_min) -> dict:
    """slope_min=None → 기울기 조건 없음(base). 아니면 종가>200선 AND 기울기>slope_min."""
    td = store.trading_days
    close_v = store.close_v
    close_ff = store.close_ff
    c2c = store.code_to_col
    kc_v = store.kospi_close_v
    km_v = store.kospi_ma200_v

    ws = pd.Timestamp(win_start); we = pd.Timestamp(win_end)
    di_list = [di for di in range(len(td)) if ws <= td[di] <= we]

    cash = float(INITIAL_CASH)
    positions: dict[str, Position] = {}
    cost: dict[str, float] = {}
    equity: list[float] = []; days: list[pd.Timestamp] = []

    n_buy = n_add = n_partial = n_final = 0
    wins = 0
    gross_w = gross_l = 0.0
    bull_days = 0

    for di in di_list:
        today = td[di].date()
        pos_bull = kc_v[di] > km_v[di]
        if slope_min is None:
            is_bull = pos_bull
        else:
            sl = slope_v[di]
            is_bull = bool(pos_bull and np.isfinite(sl) and sl > slope_min)
        if is_bull:
            bull_days += 1
        sold_today: set[str] = set()

        for ticker in list(positions.keys()):
            col = c2c[ticker]; px = close_v[di, col]
            if not np.isfinite(px):
                continue
            pos = positions[ticker]; snap = _build_snap(store, col, di)
            should_sell, _ = check_sell_condition(pos, snap)
            if should_sell:
                proceeds = px * pos.shares * (1.0 - SELL_COST)
                cash += proceeds
                pnl = proceeds - cost[ticker]
                if pnl > 0: wins += 1; gross_w += pnl
                else: gross_l += pnl
                n_final += 1
                del positions[ticker]; del cost[ticker]; sold_today.add(ticker)

        for ticker in list(positions.keys()):
            col = c2c[ticker]; px = close_v[di, col]
            if not np.isfinite(px) or px <= 0:
                continue
            pos = positions[ticker]
            if check_partial_exit(pos, float(px)):
                shares_before = pos.shares
                sell_sh = apply_partial_exit(pos, float(px))
                ratio = sell_sh / shares_before
                proceeds = px * sell_sh * (1.0 - SELL_COST)
                cost_sold = cost[ticker] * ratio
                pnl = proceeds - cost_sold
                cash += proceeds; cost[ticker] -= cost_sold
                n_partial += 1
                if pnl > 0: wins += 1; gross_w += pnl
                else: gross_l += pnl

        if is_bull and positions:
            def cur_ret(t):
                col = c2c[t]; px = close_v[di, col]
                return px / positions[t].avg_price if np.isfinite(px) else -1.0
            for ticker in sorted(positions.keys(), key=cur_ret, reverse=True):
                col = c2c[ticker]; px = close_v[di, col]
                if not np.isfinite(px) or px <= 0:
                    continue
                pos = positions[ticker]
                if check_pyramid(pos, float(px), today, is_bull, cash):
                    spent = PYRAMID_ADD_AMOUNT_WON * (1.0 + BUY_COST)
                    if cash < spent:
                        continue
                    apply_pyramid(pos, float(px), today)
                    cash -= spent; cost[ticker] += spent; n_add += 1

        free = NUM_SLOTS - len(positions)
        if is_bull and free > 0:
            cands: list[StockSnapshot] = []
            for t in store.get_universe(td[di]):
                if t in positions or t in sold_today:
                    continue
                col = c2c[t]; px = close_v[di, col]
                if not np.isfinite(px) or px <= 0:
                    continue
                snap = _build_snap(store, col, di)
                ok, _ = check_buy_conditions(snap, is_bull)
                if ok:
                    cands.append(snap)
            for snap in rank_by_momentum(cands):
                if free <= 0:
                    break
                px = snap.price
                sh = int(SLOT_AMOUNT_WON // px)
                if sh <= 0:
                    continue
                spent = sh * px * (1.0 + BUY_COST)
                if spent > cash:
                    continue
                positions[snap.ticker] = Position(ticker=snap.ticker, entry_price=px,
                                                  avg_price=px, shares=float(sh), entry_date=today)
                cost[snap.ticker] = spent
                cash -= spent; n_buy += 1; free -= 1

        hv = 0.0
        for t, pos in positions.items():
            col = c2c[t]; p = close_v[di, col]
            if not np.isfinite(p):
                p = close_ff[di, col]
                if not np.isfinite(p):
                    p = pos.avg_price
            hv += p * pos.shares
        equity.append(cash + hv); days.append(td[di])

    s = pd.Series(equity, index=pd.DatetimeIndex(days))
    final = float(s.iloc[-1])
    years = (s.index[-1] - s.index[0]).days / 365.25
    period_ret = final / INITIAL_CASH - 1.0
    cagr = (final / INITIAL_CASH) ** (1.0 / max(years, 1e-9)) - 1.0
    peak = s.cummax()
    mdd = float(((s - peak) / peak).min())
    rets = s.pct_change().fillna(0)
    sharpe = float((rets.mean() / (rets.std() + 1e-12)) * math.sqrt(252))
    calmar = cagr / abs(mdd) if mdd != 0 else float("inf")
    n_sell = n_partial + n_final
    return {
        "period_ret": period_ret, "CAGR": cagr, "MDD": mdd, "Sharpe": sharpe, "Calmar": calmar,
        "PL": abs(gross_w / gross_l) if gross_l != 0 else float("inf"),
        "win_rate": wins / n_sell if n_sell else 0.0,
        "n_trades": n_buy + n_add + n_partial + n_final, "n_buy": n_buy,
        "n_days": len(di_list), "bull_days": bull_days,
    }


def diagnose_slope(store, slope_v):
    """연도별: 종가>200선인 날들의 기울기 분포(≤0 / 0~1 / 1~3 / ≥3%)."""
    td = store.trading_days
    kc_v = store.kospi_close_v; km_v = store.kospi_ma200_v
    print("\n" + "=" * 100)
    print(" [진단] 연도별 '종가>200일선'인 날의 200일선 20일기울기 분포 (2024가 실제로 더 평평했나?)")
    print("=" * 100)
    print(f"  {'연도':<6s}{'종가>200선 일수':>14s}{'기울기≤0%':>12s}{'0~1%':>10s}{'1~3%':>10s}{'≥3%':>10s}{'평균기울기':>12s}")
    for yr in [2023, 2024, 2025, 2026]:
        idx = [di for di in range(len(td)) if td[di].year == yr and kc_v[di] > km_v[di]]
        if not idx:
            print(f"  {yr:<6d}{'(강세장일 0)':>14s}")
            continue
        sl = np.array([slope_v[di] for di in idx])
        sl = sl[np.isfinite(sl)]
        n = len(sl)
        b_neg = int((sl <= 0).sum())
        b_01 = int(((sl > 0) & (sl <= 1)).sum())
        b_13 = int(((sl > 1) & (sl <= 3)).sum())
        b_3 = int((sl > 3).sum())
        print(f"  {yr:<6d}{n:>14d}{b_neg:>12d}{b_01:>10d}{b_13:>10d}{b_3:>10d}{sl.mean():>11.2f}%")


def main():
    store = data_layer.get_store()
    slope_v = slope_panel(store)

    diagnose_slope(store, slope_v)

    variants = [("base", None), ("slope0", 0.0), ("slope1", 1.0), ("slope2", 2.0)]
    res = {name: {} for name, _ in variants}
    for name, smin in variants:
        for tag, ws, we in WINDOWS:
            res[name][tag] = run_window(store, ws, we, slope_v, smin)

    print("\n" + "=" * 104)
    print(" [전체기간] 200일선 기울기 필터 비교  ·  2020-01-02~2026-07-14  ·  1억 · 거래비용 반영")
    print("=" * 104)
    rows = []
    for name, _ in variants:
        m = res[name]["전체"]
        rows.append({"버전": name, "CAGR%": round(m["CAGR"]*100, 2), "MDD%": round(m["MDD"]*100, 2),
                     "Sharpe": round(m["Sharpe"], 2), "Calmar": round(m["Calmar"], 2),
                     "손익비": round(m["PL"], 2), "승률%": round(m["win_rate"]*100, 1),
                     "총거래": m["n_trades"]})
    print(pd.DataFrame(rows).to_string(index=False))
    pd.DataFrame(rows).to_csv(os.path.join(RESULT_DIR, "slope_full_compare.csv"), index=False, encoding="utf-8-sig")

    print("\n" + "=" * 104)
    print(" [연도별 독립] 매년 1억 리셋 · 기간수익률%  ·  base vs slope0/1/2")
    print("=" * 104)
    yrows = []
    for tag, ws, we in WINDOWS:
        if tag == "전체":
            continue
        row = {"연도": tag}
        for name, _ in variants:
            row[f"{name}%"] = round(res[name][tag]["period_ret"]*100, 2)
        yrows.append(row)
    print(pd.DataFrame(yrows).to_string(index=False))
    pd.DataFrame(yrows).to_csv(os.path.join(RESULT_DIR, "slope_yearly_compare.csv"), index=False, encoding="utf-8-sig")

    print("\n  [2024년 상세] 수익률 / MDD / 신규매수 / 강세장일수")
    for name, _ in variants:
        m = res[name]["2024"]
        print(f"   {name:<8s} 수익 {m['period_ret']*100:+7.2f}%  MDD {m['MDD']*100:7.2f}%  "
              f"신규매수 {m['n_buy']:>3d}건  강세장일 {m['bull_days']}/{m['n_days']}")
    print("\n  [대박구간 강세장일수 변화] (base 대비 기울기필터가 얼마나 진입일을 줄이나)")
    for tag in ["2023", "2025", "2026"]:
        b = res["base"][tag]
        line = f"   {tag}: base {b['bull_days']}/{b['n_days']}"
        for name in ["slope0", "slope1", "slope2"]:
            line += f" · {name} {res[name][tag]['bull_days']}"
        print(line)

    print("\n" + "=" * 104)
    print(" ⚠ 정직성: 단일 경로 백테스트. 2024 완화가 대박구간 훼손을 넘어서는지 판단. 과최적화 주의.")
    print("=" * 104)
    return res


if __name__ == "__main__":
    main()
