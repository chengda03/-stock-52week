# -*- coding: utf-8 -*-
"""
backtest_ma240_compare.py
=========================
[실험] 확정 조건 위에서 시장필터 이동평균만 200일 → 240일로 바꾸면 어떻게 되나.
특히 연도별 분석에서 드러난 '2024년 -42.5% (횡보/휩쏘 장세)'를 240일선이 더 잘 걸러내는지 검증.

  · base(200일) : 원본 strategy_core (store.kospi_ma200_v)
  · ma240(240일): KOSPI 240일선으로 시장필터 교체 (나머지 조건 동일)

판정/엔진은 strategy_core 함수 그대로. 시장필터에 넣는 KOSPI MA 패널만 200↔240으로 교체.
전체기간 비교 + 연도별 독립 백테스트(매년 1억 리셋) + 연도별 '매수허용일(강세장일)' 진단.

정직성: 단일 경로 백테스트. 2024형 위험을 실제로 줄이는지, 대박구간 수익을 깎지 않는지 그대로 보고.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

from strategy_core import (
    StockSnapshot, Position,
    is_bull_market, check_buy_conditions, rank_by_momentum,
    check_sell_condition, check_pyramid, apply_pyramid,
    check_partial_exit, apply_partial_exit,
    NUM_SLOTS, SLOT_AMOUNT_WON, PYRAMID_ADD_AMOUNT_WON,
)
import data_layer

INITIAL_CASH = 100_000_000
BUY_COST = 0.00115
SELL_COST = 0.00295

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


def kospi_ma(store, days: int) -> np.ndarray:
    """data_layer와 동일 방식(rolling, min_periods=1)으로 KOSPI N일선 패널 계산."""
    return pd.Series(store.kospi_close_v).rolling(days, min_periods=1).mean().to_numpy(dtype=float)


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


def run_window(store, win_start, win_end, kospi_ma_v) -> dict:
    td = store.trading_days
    close_v = store.close_v
    close_ff = store.close_ff
    c2c = store.code_to_col
    kc_v = store.kospi_close_v

    ws = pd.Timestamp(win_start); we = pd.Timestamp(win_end)
    di_list = [di for di in range(len(td)) if ws <= td[di] <= we]

    cash = float(INITIAL_CASH)
    positions: dict[str, Position] = {}
    cost: dict[str, float] = {}
    equity: list[float] = []
    days: list[pd.Timestamp] = []

    n_buy = n_add = n_partial = n_final = 0
    wins = 0
    gross_w = gross_l = 0.0
    bull_days = 0

    for di in di_list:
        today = td[di].date()
        is_bull = is_bull_market(kc_v[di], kospi_ma_v[di])
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
        equity.append(cash + hv)
        days.append(td[di])

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


def main():
    store = data_layer.get_store()
    ma200 = store.kospi_ma200_v
    ma240 = kospi_ma(store, 240)

    res = {}
    for tag, ws, we in WINDOWS:
        res[tag] = {
            "200": run_window(store, ws, we, ma200),
            "240": run_window(store, ws, we, ma240),
        }

    # ---- 전체기간 비교표 ----
    print("\n" + "=" * 96)
    print(" [전체기간] 시장필터 200일선 vs 240일선  ·  2020-01-02~2026-07-14  ·  1억 · 거래비용 반영")
    print("=" * 96)
    full = pd.DataFrame([
        {"버전": "base(200일)", **{k: round(res["전체"]["200"][kk], v) for k, kk, v in
            [("CAGR%","CAGR",4),("MDD%","MDD",4),("Sharpe","Sharpe",2),("Calmar","Calmar",2),
             ("손익비","PL",2),("승률%","win_rate",4),("총거래","n_trades",0)]}},
        {"버전": "ma240(240일)", **{k: round(res["전체"]["240"][kk], v) for k, kk, v in
            [("CAGR%","CAGR",4),("MDD%","MDD",4),("Sharpe","Sharpe",2),("Calmar","Calmar",2),
             ("손익비","PL",2),("승률%","win_rate",4),("총거래","n_trades",0)]}},
    ])
    for c in ["CAGR%", "MDD%", "승률%"]:
        full[c] = (full[c] * 100).round(2)
    print(full.to_string(index=False))

    # ---- 연도별 비교표 ----
    print("\n" + "=" * 116)
    print(" [연도별 독립] 매년 1억 리셋  ·  200일 vs 240일  (기간수익률 / MDD / 신규매수 / 총거래 / 강세장일수)")
    print("=" * 116)
    yrows = []
    for tag, ws, we in WINDOWS:
        if tag == "전체":
            continue
        a = res[tag]["200"]; b = res[tag]["240"]
        yrows.append({
            "연도": tag,
            "수익_200%": round(a["period_ret"] * 100, 2),
            "수익_240%": round(b["period_ret"] * 100, 2),
            "MDD_200%": round(a["MDD"] * 100, 2),
            "MDD_240%": round(b["MDD"] * 100, 2),
            "신규_200": a["n_buy"],
            "신규_240": b["n_buy"],
            "거래_200": a["n_trades"],
            "거래_240": b["n_trades"],
            "강세장일_200": f"{a['bull_days']}/{a['n_days']}",
            "강세장일_240": f"{b['bull_days']}/{b['n_days']}",
        })
    ydf = pd.DataFrame(yrows)
    print(ydf.to_string(index=False))
    ydf.to_csv(os.path.join(RESULT_DIR, "ma240_yearly_compare.csv"), index=False, encoding="utf-8-sig")

    print("\n" + "=" * 116)
    print(" ⚠ 정직성: 단일 경로 백테스트. 240일선이 2024형 위험을 실제로 줄이는지 강세장일수·손실 변화로 판단. 과최적화 주의.")
    print("=" * 116)
    return res


if __name__ == "__main__":
    main()
