# -*- coding: utf-8 -*-
"""
backtest_exposure_compare.py
============================
[실험] 2024 손실이 '타이밍'이 아니라 '노출 폭(breadth)' 문제라는 진단에 따라, 타이밍이 아니라
'노출 방식'을 바꾸는 두 접근을 각각·결합 테스트.

  A) dailycap  : 하루 신규매수(첫 진입) 최대 N종목으로 제한(나머지 이월, 우선순위=20일모멘텀)
  B) splitentry: 신규진입시 250만만 투입 → 5거래일 생존시 6일째 나머지 250만 추가(불타기와 별개)
  C) combined  : A(=3) + B 동시

  · base / dailycap2 / dailycap3 / dailycap5 / splitentry / combined_ab

판정/엔진은 strategy_core 함수 그대로. A는 신규매수 루프에 하루 카운트 상한, B는 포지션별 2차투입
상태를 드라이버에서 관리. 전체기간 지표 + 연도별 독립(매년 1억 리셋) 비교. 특히 2024 완화 vs 대박구간 보존.
정직성: 단일 경로 백테스트. 실패시 원인 규명. 과최적화 주의.
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
SECOND_TRANCHE_DAYS = 5
SLOT_HALF = SLOT_AMOUNT_WON / 2.0   # 250만

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


def run_window(store, win_start, win_end, daily_cap=None, split_entry=False) -> dict:
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
    entry_di: dict[str, int] = {}
    second_added: dict[str, bool] = {}
    equity: list[float] = []; days: list[pd.Timestamp] = []

    n_buy = n_add = n_partial = n_final = n_second = 0
    wins = 0
    gross_w = gross_l = 0.0

    slot_first = SLOT_HALF if split_entry else float(SLOT_AMOUNT_WON)

    for di in di_list:
        today = td[di].date()
        is_bull = is_bull_market(kc_v[di], km_v[di])
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
                del positions[ticker]; del cost[ticker]
                entry_di.pop(ticker, None); second_added.pop(ticker, None)
                sold_today.add(ticker)

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

        # 2) 불타기 (무제한, 원본)
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

        # 2-B) 분할진입 2차 투입 (생존 5거래일 → 나머지 250만, 조건 없이)
        if split_entry:
            for ticker in list(positions.keys()):
                if ticker in sold_today or second_added.get(ticker):
                    continue
                if di - entry_di.get(ticker, di) < SECOND_TRANCHE_DAYS:
                    continue
                col = c2c[ticker]; px = close_v[di, col]
                if not np.isfinite(px) or px <= 0:
                    continue
                spent = SLOT_HALF * (1.0 + BUY_COST)
                if cash < spent:
                    continue   # 현금 부족 → 다음날 재시도
                pos = positions[ticker]
                add_shares = SLOT_HALF / px
                old_cost = pos.avg_price * pos.shares
                pos.avg_price = (old_cost + SLOT_HALF) / (pos.shares + add_shares)
                pos.shares += add_shares
                cost[ticker] += spent; cash -= spent
                second_added[ticker] = True; n_second += 1

        # 3) 신규매수 (하루 상한 daily_cap, 우선순위=모멘텀)
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
            new_today = 0
            for snap in rank_by_momentum(cands):
                if free <= 0:
                    break
                if daily_cap is not None and new_today >= daily_cap:
                    break
                px = snap.price
                sh = int(slot_first // px)
                if sh <= 0:
                    continue
                spent = sh * px * (1.0 + BUY_COST)
                if spent > cash:
                    continue
                positions[snap.ticker] = Position(ticker=snap.ticker, entry_price=px,
                                                  avg_price=px, shares=float(sh), entry_date=today)
                cost[snap.ticker] = spent
                entry_di[snap.ticker] = di
                second_added[snap.ticker] = False
                cash -= spent; n_buy += 1; free -= 1; new_today += 1

        # 4) 일별 평가
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
        "n_trades": n_buy + n_add + n_partial + n_final + n_second, "n_buy": n_buy,
        "n_second": n_second,
    }


def main():
    store = data_layer.get_store()
    variants = [
        ("base", dict()),
        ("dailycap2", dict(daily_cap=2)),
        ("dailycap3", dict(daily_cap=3)),
        ("dailycap5", dict(daily_cap=5)),
        ("splitentry", dict(split_entry=True)),
        ("combined_ab", dict(daily_cap=3, split_entry=True)),
    ]
    res = {name: {} for name, _ in variants}
    for name, kw in variants:
        for tag, ws, we in WINDOWS:
            res[name][tag] = run_window(store, ws, we, **kw)

    print("\n" + "=" * 110)
    print(" [전체기간] 노출 방식 실험 (A 하루제한 / B 분할진입 / C 결합)  ·  2020-01-02~2026-07-14  ·  1억 · 거래비용")
    print("=" * 110)
    rows = []
    for name, _ in variants:
        m = res[name]["전체"]
        rows.append({"버전": name, "CAGR%": round(m["CAGR"]*100, 2), "MDD%": round(m["MDD"]*100, 2),
                     "Sharpe": round(m["Sharpe"], 2), "Calmar": round(m["Calmar"], 2),
                     "손익비": round(m["PL"], 2), "승률%": round(m["win_rate"]*100, 1),
                     "총거래": m["n_trades"]})
    print(pd.DataFrame(rows).to_string(index=False))
    pd.DataFrame(rows).to_csv(os.path.join(RESULT_DIR, "exposure_full_compare.csv"), index=False, encoding="utf-8-sig")

    print("\n" + "=" * 110)
    print(" [연도별 독립] 매년 1억 리셋 · 기간수익률%  ·  base vs A/B/C")
    print("=" * 110)
    yrows = []
    for tag, ws, we in WINDOWS:
        if tag == "전체":
            continue
        row = {"연도": tag}
        for name, _ in variants:
            row[name] = round(res[name][tag]["period_ret"]*100, 2)
        yrows.append(row)
    print(pd.DataFrame(yrows).to_string(index=False))
    pd.DataFrame(yrows).to_csv(os.path.join(RESULT_DIR, "exposure_yearly_compare.csv"), index=False, encoding="utf-8-sig")

    print("\n  [2024년 상세] 수익률 / MDD / 신규매수건수")
    for name, _ in variants:
        m = res[name]["2024"]
        print(f"   {name:<12s} 수익 {m['period_ret']*100:+7.2f}%  MDD {m['MDD']*100:7.2f}%  신규매수 {m['n_buy']:>3d}건")

    print("\n" + "=" * 110)
    print(" ⚠ 정직성: 단일 경로 백테스트. 2024 완화가 대박구간 훼손을 넘어서는지 판단. 과최적화 주의.")
    print("=" * 110)
    return res


if __name__ == "__main__":
    main()
