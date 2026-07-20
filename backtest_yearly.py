# -*- coding: utf-8 -*-
"""
backtest_yearly.py
==================
[연도별 독립 백테스트] strategy_core.py를 '그대로'(수정 없이) 사용하고, backtest_v3.py와
'완전히 동일한' 엔진 루프를 그대로 옮겨, 실행 기간만 연도별로 잘라 각각 1억원으로 리셋해 돌린다.
strategy_core.py / backtest_v3.py 원본은 건드리지 않는다(이 파일은 별도 러너).

목적: "매년 꾸준히 버는 전략인가, 특정 장세에 몰아서 버는 전략인가"를 정직하게 본다.
각 구간은 그 해 첫 거래일에 자산을 1억으로 리셋 → '그 해 홀로 있었다면' 독립적으로 계산
(이전 해 수익 이월 없음).

구간:
  · 전체   2020-01-01~2026-07-14 (참고)
  · 각 연도 2020/2021/2022(하락장)/2023/2024/2025/2026(~07-14)

★ 판정은 전부 strategy_core 함수만 사용(backtest_v3와 동일 규칙): check_buy_conditions,
  rank_by_momentum, check_sell_condition, check_pyramid/apply_pyramid, check_partial_exit/
  apply_partial_exit, is_bull_market, NUM_SLOTS/SLOT_AMOUNT_WON/PYRAMID_ADD_AMOUNT_WON.
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
CSV_PATH = os.path.join(RESULT_DIR, "yearly_independent.csv")

WINDOWS = [
    ("전체(참고)", "2020-01-01", "2026-07-14"),
    ("2020",       "2020-01-01", "2020-12-31"),
    ("2021",       "2021-01-01", "2021-12-31"),
    ("2022(하락장)", "2022-01-01", "2022-12-31"),
    ("2023",       "2023-01-01", "2023-12-31"),
    ("2024",       "2024-01-01", "2024-12-31"),
    ("2025",       "2025-01-01", "2025-12-31"),
    ("2026(~7/14)", "2026-01-01", "2026-07-14"),
]


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


def run_window(store, win_start, win_end) -> dict:
    """backtest_v3의 일별 루프를 그대로 사용하되, [win_start,win_end] 구간만 1억으로 새로 시작."""
    td = store.trading_days
    close_v = store.close_v
    close_ff = store.close_ff
    c2c = store.code_to_col

    ws = pd.Timestamp(win_start); we = pd.Timestamp(win_end)
    di_list = [di for di in range(len(td)) if ws <= td[di] <= we]
    if not di_list:
        raise ValueError(f"empty window {win_start}~{win_end}")

    cash = float(INITIAL_CASH)
    positions: dict[str, Position] = {}
    cost: dict[str, float] = {}
    equity: list[float] = []
    days: list[pd.Timestamp] = []

    n_buy = n_add = n_partial = n_final = 0
    wins = 0
    gross_w = gross_l = 0.0

    for di in di_list:
        today = td[di].date()
        is_bull = is_bull_market(store.kospi_close_v[di], store.kospi_ma200_v[di])
        sold_today: set[str] = set()

        # 1) 전량매도
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

        # 1-2) 부분익절
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

        # 2) 불타기
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

        # 3) 신규매수
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

        # 4) 일별 평가
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
    n_sell = n_partial + n_final
    return {
        "period_ret": period_ret, "CAGR": cagr, "MDD": mdd, "Sharpe": sharpe,
        "PL": abs(gross_w / gross_l) if gross_l != 0 else float("inf"),
        "win_rate": wins / n_sell if n_sell else 0.0,
        "n_trades": n_buy + n_add + n_partial + n_final, "n_buy": n_buy,
        "start": str(s.index[0].date()), "end": str(s.index[-1].date()), "years": years,
    }


def main():
    store = data_layer.get_store()
    rows = []
    for tag, ws, we in WINDOWS:
        m = run_window(store, ws, we)
        rows.append({
            "구간": tag,
            "기간": f"{m['start']}~{m['end']}",
            "기간수익률%": round(m["period_ret"] * 100, 2),
            "CAGR%": round(m["CAGR"] * 100, 2),
            "MDD%": round(m["MDD"] * 100, 2),
            "Sharpe": round(m["Sharpe"], 2),
            "손익비": round(m["PL"], 2),
            "승률%": round(m["win_rate"] * 100, 1),
            "총거래": m["n_trades"],
            "신규매수": m["n_buy"],
        })
    df = pd.DataFrame(rows)
    print("\n" + "=" * 116)
    print(" 연도별 독립 백테스트 (매년 1억 리셋 · strategy_core 그대로 · 거래비용 반영 · 같은 데이터)")
    print("=" * 116)
    print(df.to_string(index=False))
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"\n  CSV 저장: {CSV_PATH}")
    print("  ※ 2026은 반년(약 0.53년) 구간이라 CAGR은 연율 환산값(기간수익률보다 확대되어 보임). 기간수익률% 병기.")

    yearly = [r for r in rows if r["구간"] not in ("전체(참고)",)]
    best = max(yearly, key=lambda r: r["기간수익률%"])
    worst = min(yearly, key=lambda r: r["기간수익률%"])
    print("\n" + "-" * 116)
    print(f"  최고의 해 : {best['구간']} {best['기간수익률%']:+.2f}%   ·   최악의 해 : {worst['구간']} {worst['기간수익률%']:+.2f}%"
          f"   ·   변동폭 {best['기간수익률%']-worst['기간수익률%']:.1f}%p")
    print("=" * 116)
    return rows


if __name__ == "__main__":
    main()
