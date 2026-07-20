# -*- coding: utf-8 -*-
"""
backtest_cooldown_compare.py
============================
[실험] 확정 조건 위에 '재진입 쿨다운'(전량매도 후 N거래일 동안 같은 종목 재매수 금지)을 추가.
2024년 -42.5% 손실의 추정 원인인 "매수→손절→금방 재매수→다시 손절" 반복을 끊는지 검증.

  · base       : 쿨다운 없음 (원본 strategy_core)
  · cooldown5  : 전량매도 후 5거래일 재매수 금지
  · cooldown10 : 10거래일
  · cooldown20 : 20거래일
(부분익절 50%만 매도된 경우는 쿨다운 대상 아님 — 전량매도만.)

먼저: [백테스트 전 진단] 2024년(독립창) base 기준으로 '전량매도 후 5/10/20거래일 내 같은 종목
재매수' 사례가 몇 건인지 카운트 → 이 패턴이 2024 손실에서 얼마나 큰 비중인지 가늠.
그다음: 전체기간 지표 + 연도별 독립(매년 1억 리셋) 비교.

판정/엔진은 strategy_core 함수 그대로. 쿨다운만 드라이버 신규매수 루프에서 적용.
정직성: 단일 경로 백테스트. 2024 손실이 재진입 패턴이 아니면 쿨다운은 안 들음 — 그대로 보고. 과최적화 주의.
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


def run_window(store, win_start, win_end, cooldown_days: int, collect_reentry: bool = False) -> dict:
    """cooldown_days=0 → 쿨다운 없음(base). collect_reentry=True → 전량매도후 재매수 gap 히스토그램 수집."""
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
    last_full_sell_di: dict[str, int] = {}   # 종목별 마지막 전량매도 di
    equity: list[float] = []
    days: list[pd.Timestamp] = []

    n_buy = n_add = n_partial = n_final = 0
    n_blocked = 0                              # 쿨다운으로 매수 차단된 신규매수 시도 횟수
    wins = 0
    gross_w = gross_l = 0.0
    reentry_gaps: list[int] = []               # (collect_reentry) 전량매도 후 재매수까지 거래일 gap

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
                last_full_sell_di[ticker] = di
                del positions[ticker]; del cost[ticker]; sold_today.add(ticker)

        # 1-2) 부분익절 (쿨다운 대상 아님)
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

        # 3) 신규매수 (+ 쿨다운/재진입 진단)
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
                t = snap.ticker
                # 쿨다운 차단 (전량매도 후 cooldown_days 이내 재매수 금지)
                if cooldown_days > 0 and t in last_full_sell_di \
                        and (di - last_full_sell_di[t]) <= cooldown_days:
                    n_blocked += 1
                    continue
                px = snap.price
                sh = int(SLOT_AMOUNT_WON // px)
                if sh <= 0:
                    continue
                spent = sh * px * (1.0 + BUY_COST)
                if spent > cash:
                    continue
                # 재진입 gap 진단: '실제로 매수된' 종목이 과거 전량매도 이력이 있으면 gap 기록
                if collect_reentry and t in last_full_sell_di:
                    reentry_gaps.append(di - last_full_sell_di[t])
                positions[t] = Position(ticker=t, entry_price=px, avg_price=px,
                                        shares=float(sh), entry_date=today)
                cost[t] = spent
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
    calmar = cagr / abs(mdd) if mdd != 0 else float("inf")
    n_sell = n_partial + n_final
    return {
        "period_ret": period_ret, "CAGR": cagr, "MDD": mdd, "Sharpe": sharpe, "Calmar": calmar,
        "PL": abs(gross_w / gross_l) if gross_l != 0 else float("inf"),
        "win_rate": wins / n_sell if n_sell else 0.0,
        "n_trades": n_buy + n_add + n_partial + n_final, "n_buy": n_buy,
        "n_blocked": n_blocked, "reentry_gaps": reentry_gaps,
    }


def main():
    store = data_layer.get_store()
    cds = [("base", 0), ("cooldown5", 5), ("cooldown10", 10), ("cooldown20", 20)]

    # ===== 백테스트 전 진단: 2024 독립창, base 기준 재진입 gap =====
    diag = run_window(store, "2024-01-01", "2024-12-31", 0, collect_reentry=True)
    gaps = np.array(diag["reentry_gaps"])
    print("\n" + "=" * 100)
    print(" [백테스트 전 진단] 2024년(독립창, base=쿨다운없음) — 전량매도된 종목이 다시 '매수조건 통과+매수'된 사례")
    print("=" * 100)
    print(f"  2024년 신규매수 총 {diag['n_buy']}건 중, '과거 전량매도 이력이 있는 종목의 재매수'는 {len(gaps)}건")
    if len(gaps):
        print(f"    · 전량매도 후 5거래일 이내 재매수 : {int((gaps<=5).sum())}건")
        print(f"    · 전량매도 후 10거래일 이내 재매수: {int((gaps<=10).sum())}건")
        print(f"    · 전량매도 후 20거래일 이내 재매수: {int((gaps<=20).sum())}건")
        print(f"    · 재매수 gap 중앙값 {np.median(gaps):.0f}거래일 · 평균 {gaps.mean():.1f}거래일")

    # ===== 전체기간 + 연도별 실행 =====
    res = {name: {} for name, _ in cds}
    for name, n in cds:
        for tag, ws, we in WINDOWS:
            res[name][tag] = run_window(store, ws, we, n)

    # 전체기간 비교표
    print("\n" + "=" * 104)
    print(" [전체기간] 재진입 쿨다운 비교  ·  2020-01-02~2026-07-14  ·  1억 · 거래비용 반영")
    print("=" * 104)
    rows = []
    for name, _ in cds:
        m = res[name]["전체"]
        rows.append({"버전": name,
                     "CAGR%": round(m["CAGR"]*100, 2), "MDD%": round(m["MDD"]*100, 2),
                     "Sharpe": round(m["Sharpe"], 2), "Calmar": round(m["Calmar"], 2),
                     "손익비": round(m["PL"], 2), "승률%": round(m["win_rate"]*100, 1),
                     "총거래": m["n_trades"], "쿨다운차단": m["n_blocked"]})
    fdf = pd.DataFrame(rows)
    print(fdf.to_string(index=False))

    # 연도별 비교표 (기간수익률 + MDD)
    print("\n" + "=" * 104)
    print(" [연도별 독립] 매년 1억 리셋 · 기간수익률%(상)/MDD%(하)  ·  base vs cd5/10/20")
    print("=" * 104)
    yrows = []
    for tag, ws, we in WINDOWS:
        if tag == "전체":
            continue
        row = {"연도": tag}
        for name, _ in cds:
            row[f"{name}_수익%"] = round(res[name][tag]["period_ret"]*100, 2)
        yrows.append(row)
    ydf = pd.DataFrame(yrows)
    print(ydf.to_string(index=False))

    print("\n  [2024년 상세] 수익률 / MDD / 신규매수 / 쿨다운차단")
    for name, _ in cds:
        m = res[name]["2024"]
        print(f"   {name:<11s} 수익 {m['period_ret']*100:+7.2f}%  MDD {m['MDD']*100:7.2f}%  "
              f"신규매수 {m['n_buy']:>3d}건  쿨다운차단 {m['n_blocked']:>3d}건")

    fdf.to_csv(os.path.join(RESULT_DIR, "cooldown_full_compare.csv"), index=False, encoding="utf-8-sig")
    ydf.to_csv(os.path.join(RESULT_DIR, "cooldown_yearly_compare.csv"), index=False, encoding="utf-8-sig")

    print("\n" + "=" * 104)
    print(" ⚠ 정직성: 단일 경로 백테스트. 2024 손실이 재진입 패턴이 아니면 쿨다운은 안 들음. 과최적화 주의.")
    print("=" * 104)
    return res


if __name__ == "__main__":
    main()
