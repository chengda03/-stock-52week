# -*- coding: utf-8 -*-
"""
backtest_s1_stagedexit.py — S1 "단계적 이탈/손절" 실험 (strategy_core 미변경)
============================================================================
현재 확정 매도: 90일선 이탈=전량, 평단가 -12%=전량.
제안(사용자): 이동평균·손절을 '두 단계'로 나눠 부분매도.
  · 60일선 이탈  → 잔량의 50% 매도   / 90일선 이탈 → 전량매도
  · 평단가 -10% → 잔량의 50% 매도   / 평단가 -12% → 전량매도
익절(+8%에서 50%)은 확정 규칙이므로 기본 유지(옵션으로 끄고도 비교).

전량매도(90일선/-12%)가 우선. 아니면 부분(60일선/-10%) → 그다음 +8% 익절 순.
strategy_core는 손대지 않고, 이 스크립트 안에서만 단계 로직을 구현해 비교한다.
비교기준: 확정 S1(+8%@50% 익절·전량손절) = CAGR +78.0% / MDD -27.6% / 승률 33.5%.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

from strategy_core import (
    Position, is_bull_market, check_buy_conditions, rank_by_momentum,
    check_pyramid, apply_pyramid,
    NUM_SLOTS, SLOT_AMOUNT_WON, PYRAMID_ADD_AMOUNT_WON,
    HARD_STOP_LOSS_PCT,
)
import data_layer
from backtest_v3 import _build_snap
from backtest_roe_eps_event import fpct, log

INITIAL_CASH = 100_000_000
BUY_COST = 0.00115
SELL_COST = 0.00295
TRADE_START = pd.Timestamp("2020-01-01")
TP_TRIGGER = 0.08          # 익절 트리거(진입가 대비)
TP_RATIO = 0.50            # 익절 매도비율
STOP10 = 0.10              # 평단가 -10% 부분손절
STOP12 = HARD_STOP_LOSS_PCT  # 평단가 -12% 전량손절
PART_RATIO = 0.50          # 단계 부분매도 비율

RESULT_DIR = os.path.join("data", "cache_s1_stagedexit", "results")
os.makedirs(RESULT_DIR, exist_ok=True)


def run(store, name, use_tp, staged_ma, staged_stop):
    """
    use_tp     : +8%에서 50% 익절 사용 여부
    staged_ma  : True면 60일선50%+90일선전량, False면 90일선 전량만
    staged_stop: True면 -10%50%+-12%전량, False면 -12% 전량만
    """
    td = store.trading_days
    close_v = store.close_v
    close_ff = store.close_ff
    c2c = store.code_to_col
    start_di = store.start_di

    cash = float(INITIAL_CASH)
    positions: dict[str, Position] = {}
    cost: dict[str, float] = {}
    tp_done: dict[str, bool] = {}
    ma60_done: dict[str, bool] = {}
    stop10_done: dict[str, bool] = {}
    daily = np.empty(len(td))

    n_buy = n_add = n_tp = n_part_stop = n_final = 0
    wins = 0
    gross_w = gross_l = 0.0

    def sell_part(ticker, px, frac):
        """잔량의 frac 매도 → (win?, pnl) 반영. 반환: 전량됐는지."""
        nonlocal cash, wins, gross_w, gross_l
        pos = positions[ticker]
        sell_sh = pos.shares * frac
        proceeds = px * sell_sh * (1.0 - SELL_COST)
        cost_sold = cost[ticker] * frac
        pnl = proceeds - cost_sold
        cash += proceeds
        pos.shares -= sell_sh
        cost[ticker] -= cost_sold
        if pnl > 0:
            wins += 1; gross_w += pnl
        else:
            gross_l += pnl
        return sell_sh

    def sell_all(ticker, px):
        nonlocal cash, wins, gross_w, gross_l
        pos = positions[ticker]
        proceeds = px * pos.shares * (1.0 - SELL_COST)
        pnl = proceeds - cost[ticker]
        cash += proceeds
        if pnl > 0:
            wins += 1; gross_w += pnl
        else:
            gross_l += pnl
        del positions[ticker]; del cost[ticker]
        tp_done.pop(ticker, None); ma60_done.pop(ticker, None); stop10_done.pop(ticker, None)

    for di in range(len(td)):
        today = td[di].date()
        if di < start_di:
            daily[di] = INITIAL_CASH
            continue
        is_bull = is_bull_market(store.kospi_close_v[di], store.kospi_ma200_v[di])
        sold_today: set[str] = set()

        # ---------- 1) 매도(전량 → 부분손절 → 익절) ----------
        for ticker in list(positions.keys()):
            col = c2c[ticker]
            px = close_v[di, col]
            if not np.isfinite(px) or px <= 0:
                continue
            pos = positions[ticker]
            snap = _build_snap(store, col, di)
            ma60 = snap.ma_buy    # 60일선
            ma90 = snap.ma_sell   # 90일선
            avg = pos.avg_price

            # (a) 전량매도: 90일선 이탈 or 평단가 -12%
            if px < ma90 or px <= avg * (1 - STOP12):
                n_final += 1
                sell_all(ticker, px)
                sold_today.add(ticker)
                continue

            # (b) 부분손절
            if staged_stop and not stop10_done.get(ticker, False) and px <= avg * (1 - STOP10):
                sell_part(ticker, px, PART_RATIO); stop10_done[ticker] = True; n_part_stop += 1
                if ticker not in positions:
                    continue
            if staged_ma and not ma60_done.get(ticker, False) and px < ma60:
                sell_part(ticker, px, PART_RATIO); ma60_done[ticker] = True; n_part_stop += 1
                if ticker not in positions:
                    continue

            # (c) +8% 부분익절
            if use_tp and not tp_done.get(ticker, False) and px >= pos.entry_price * (1 + TP_TRIGGER):
                sell_part(ticker, px, TP_RATIO); tp_done[ticker] = True; n_tp += 1

        # ---------- 2) 불타기 ----------
        if is_bull and positions:
            def cur_ret(t):
                col = c2c[t]; p = close_v[di, col]
                return p / positions[t].avg_price if np.isfinite(p) else -1.0
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

        # ---------- 3) 신규매수 ----------
        free = NUM_SLOTS - len(positions)
        if is_bull and free > 0:
            cands = []
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
                positions[snap.ticker] = Position(
                    ticker=snap.ticker, entry_price=px, avg_price=px,
                    shares=float(sh), entry_date=today)
                cost[snap.ticker] = spent
                tp_done[snap.ticker] = False
                ma60_done[snap.ticker] = False
                stop10_done[snap.ticker] = False
                cash -= spent; n_buy += 1; free -= 1

        # ---------- 4) 평가 ----------
        hv = 0.0
        for t, pos in positions.items():
            col = c2c[t]; p = close_v[di, col]
            if not np.isfinite(p):
                p = close_ff[di, col]
                if not np.isfinite(p):
                    p = pos.avg_price
            hv += p * pos.shares
        daily[di] = cash + hv

    dt = pd.Series(daily, index=td)
    dt = dt[dt.index >= TRADE_START]
    final = float(dt.iloc[-1])
    years = (dt.index[-1] - dt.index[0]).days / 365.25
    cagr = (final / INITIAL_CASH) ** (1.0 / max(years, 1e-9)) - 1.0
    peak = dt.cummax()
    mdd = float(((dt - peak) / peak).min())
    rets = dt.pct_change().fillna(0)
    sharpe = float((rets.mean() / (rets.std() + 1e-12)) * math.sqrt(252))
    calmar = cagr / abs(mdd) if mdd != 0 else float("inf")
    n_sell = n_tp + n_part_stop + n_final
    win_rate = wins / n_sell if n_sell else 0.0
    pl = abs(gross_w / gross_l) if gross_l != 0 else float("inf")

    return {"name": name, "final": final, "total_ret": final / INITIAL_CASH - 1,
            "CAGR": cagr, "MDD": mdd, "Sharpe": sharpe, "Calmar": calmar, "PL": pl,
            "win_rate": win_rate, "n_buy": n_buy, "n_add": n_add, "n_tp": n_tp,
            "n_part_stop": n_part_stop, "n_final": n_final, "n_sell": n_sell,
            "equity": dt}


def main():
    log("===== S1 단계적 이탈/손절 실험 =====")
    store = data_layer.get_store()

    variants = [
        # name, use_tp, staged_ma, staged_stop
        ("확정 S1(익절+8%@50%·전량손절)", True, False, False),
        ("확정익절 + 단계이탈(60/90·-10/-12)", True, True, True),
        ("익절없음 + 단계이탈(60/90·-10/-12)", False, True, True),
        ("원조 S1(익절없음·전량손절)", False, False, False),
    ]
    results = [run(store, n, tp, ma, sp) for n, tp, ma, sp in variants]

    print()
    print("=" * 116)
    print(" S1 단계적 이탈/손절 비교 (매수·불타기·시장필터 동일)")
    print("=" * 116)
    print(f"  {'변형':>34s} {'CAGR':>8s} {'MDD':>8s} {'승률':>7s} {'손익비':>6s} {'Sharpe':>7s} {'Calmar':>7s}"
          f" | {'신규':>4s} {'불타기':>5s} {'익절':>4s} {'부분손절':>6s} {'전량매도':>6s}")
    for r in results:
        print(f"  {r['name']:>34s} {fpct(r['CAGR']):>8s} {fpct(r['MDD']):>8s} {r['win_rate']*100:>6.1f}%"
              f" {r['PL']:>6.2f} {r['Sharpe']:>7.2f} {r['Calmar']:>7.2f} | {r['n_buy']:>4d} {r['n_add']:>5d}"
              f" {r['n_tp']:>4d} {r['n_part_stop']:>6d} {r['n_final']:>6d}")

    base = results[0]
    print()
    print("=" * 116)
    print(" 확정 S1 대비 변화")
    print("=" * 116)
    for r in results[1:]:
        print(f"  {r['name']:>34s} : CAGR {fpct(base['CAGR'])}→{fpct(r['CAGR'])} ({(r['CAGR']-base['CAGR'])*100:+.1f}%p)"
              f"   MDD {fpct(base['MDD'])}→{fpct(r['MDD'])}"
              f"   승률 {base['win_rate']*100:.1f}%→{r['win_rate']*100:.1f}%")

    _save(results)


def _save(results):
    rows = [{k: r[k] for k in ("name", "total_ret", "CAGR", "MDD", "Sharpe", "Calmar",
                               "PL", "win_rate", "n_buy", "n_add", "n_tp",
                               "n_part_stop", "n_final", "n_sell")} for r in results]
    pd.DataFrame(rows).to_csv(os.path.join(RESULT_DIR, "stagedexit_results.csv"),
                              index=False, encoding="utf-8-sig")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for f in ["Malgun Gothic", "NanumGothic", "AppleGothic", "Gulim"]:
            try:
                matplotlib.rcParams["font.family"] = f
                matplotlib.rcParams["axes.unicode_minus"] = False
                break
            except Exception:
                continue
        fig, ax = plt.subplots(figsize=(13, 7))
        for r in results:
            ax.plot(r["equity"].index, r["equity"].values, lw=1.5, label=r["name"])
        ax.set_yscale("log")
        ax.set_title("S1 단계적 이탈/손절 실험 — 자산곡선(로그)")
        ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, "stagedexit_curves.png"), dpi=140)
        plt.close(fig)
    except Exception as e:
        print(f"  (그래프 생략: {e})")
    print(f"\n  결과 저장 : {RESULT_DIR}")


if __name__ == "__main__":
    main()
