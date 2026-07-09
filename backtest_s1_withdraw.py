# -*- coding: utf-8 -*-
"""
backtest_s1_withdraw.py — S1 v2 + "수익 인출(인출가능금)" 변형
=============================================================
확정판 S1 v2(+8%@50% 부분익절 · 90일선/-12% 손절 · 불타기)를 그대로 두고,
현금을 두 갈래로 나눈다.
  · 운용현금(cash)      : 불타기/신규매수에 계속 쓰는 돈 (기존과 동일)
  · 인출가능금(withdrawn): 시스템 밖으로 빼둔 돈. 이후 절대 재투입 안 함.
                           (백테스트에선 그냥 누적만; 총자산에는 포함해 수익률 계산)

[규칙 A] 부분익절(+8%@50%) 회수금을 전부 인출가능금으로 이동(재투입 X).
[규칙 B] 총자산이 '직전 트리거 기준' 대비 X% 늘 때마다, 늘어난 금액의 Y%를
         운용현금 → 인출가능금으로 이동(운용현금 한도 내). 기준을 현재 총자산으로 갱신.

strategy_core.py / backtest_v3.py 는 수정하지 않음. 변형 1(규칙 없음)은
backtest_v3 확정판과 동일 로직 → baseline 재현 확인용.
"""
from __future__ import annotations

import os
import math
import numpy as np
import pandas as pd

from strategy_core import (
    Position, is_bull_market, check_buy_conditions, rank_by_momentum,
    check_sell_condition, check_pyramid, apply_pyramid,
    NUM_SLOTS, SLOT_AMOUNT_WON, PYRAMID_ADD_AMOUNT_WON,
    PARTIAL_EXIT_TRIGGER_PCT, PARTIAL_EXIT_RATIO,
)
import data_layer
from backtest_v3 import _build_snap, INITIAL_CASH, BUY_COST, SELL_COST, TRADE_START
from backtest_roe_eps_event import load_index, fpct, log

RESULT_DIR = os.path.join("data", "cache_s1_withdraw", "results")
os.makedirs(RESULT_DIR, exist_ok=True)


def run(store, name, rule_a, rule_b, X=0.5, Y=0.3):
    td = store.trading_days
    close_v = store.close_v
    close_ff = store.close_ff
    c2c = store.code_to_col
    start_di = store.start_di

    cash = float(INITIAL_CASH)   # 운용현금
    withdrawn = 0.0              # 인출가능금(누적, 재투입 안 함)
    positions: dict[str, Position] = {}
    cost: dict[str, float] = {}
    part_done: dict[str, bool] = {}

    daily_total = np.empty(len(td))
    ref_total = float(INITIAL_CASH)   # 규칙 B 트리거 기준 총자산
    max_weight = 0.0

    n_buy = n_add = n_partial = n_final = 0
    wins = 0
    gross_w = gross_l = 0.0

    for di in range(len(td)):
        today = td[di].date()
        if di < start_di:
            daily_total[di] = INITIAL_CASH
            continue
        is_bull = is_bull_market(store.kospi_close_v[di], store.kospi_ma200_v[di])
        sold_today: set[str] = set()

        # 1) 전량매도 (90일선/-12%) → 운용현금 회수(기존과 동일)
        for ticker in list(positions.keys()):
            col = c2c[ticker]; px = close_v[di, col]
            if not np.isfinite(px):
                continue
            pos = positions[ticker]
            snap = _build_snap(store, col, di)
            should_sell, _ = check_sell_condition(pos, snap)
            if should_sell:
                proceeds = px * pos.shares * (1.0 - SELL_COST)
                pnl = proceeds - cost[ticker]
                cash += proceeds
                if pnl > 0:
                    wins += 1; gross_w += pnl
                else:
                    gross_l += pnl
                n_final += 1
                del positions[ticker]; del cost[ticker]; part_done.pop(ticker, None)
                sold_today.add(ticker)

        # 1-2) 부분익절(+8%@50%). 규칙 A면 회수금 → 인출가능금, 아니면 운용현금
        for ticker in list(positions.keys()):
            col = c2c[ticker]; px = close_v[di, col]
            if not np.isfinite(px) or px <= 0:
                continue
            pos = positions[ticker]
            if not part_done.get(ticker, False) and px >= pos.entry_price * (1 + PARTIAL_EXIT_TRIGGER_PCT):
                sell_sh = pos.shares * PARTIAL_EXIT_RATIO
                ratio = sell_sh / pos.shares
                proceeds = px * sell_sh * (1.0 - SELL_COST)
                cost_sold = cost[ticker] * ratio
                pnl = proceeds - cost_sold
                pos.shares -= sell_sh
                cost[ticker] -= cost_sold
                if rule_a:
                    withdrawn += proceeds     # 인출가능금으로 이동(재투입 X)
                else:
                    cash += proceeds
                part_done[ticker] = True
                n_partial += 1
                if pnl > 0:
                    wins += 1; gross_w += pnl
                else:
                    gross_l += pnl

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
                cost[snap.ticker] = spent; part_done[snap.ticker] = False
                cash -= spent; n_buy += 1; free -= 1

        # 4) 평가
        hv = 0.0
        top_val = 0.0
        for t, pos in positions.items():
            col = c2c[t]; p = close_v[di, col]
            if not np.isfinite(p):
                p = close_ff[di, col]
                if not np.isfinite(p):
                    p = pos.avg_price
            v = p * pos.shares
            hv += v
            if v > top_val:
                top_val = v
        total = cash + hv + withdrawn
        daily_total[di] = total
        if total > 0:
            max_weight = max(max_weight, top_val / total)

        # 4-2) 규칙 B: 총자산이 기준 대비 +X% → 늘어난 금액의 Y%를 운용현금→인출가능금
        if rule_b and total >= ref_total * (1.0 + X):
            increase = total - ref_total
            move = min(Y * increase, cash)   # 운용현금 한도 내에서만 이동
            if move > 0:
                cash -= move
                withdrawn += move
            ref_total = total   # 다음 트리거 기준 갱신

    # ---------- 지표 (총자산 기준) ----------
    dt = pd.Series(daily_total, index=td)
    dt = dt[dt.index >= TRADE_START]
    final = float(dt.iloc[-1])
    years = (dt.index[-1] - dt.index[0]).days / 365.25
    cagr = (final / INITIAL_CASH) ** (1.0 / max(years, 1e-9)) - 1.0
    peak = dt.cummax()
    mdd = float(((dt - peak) / peak).min())
    rets = dt.pct_change().fillna(0)
    sharpe = float((rets.mean() / (rets.std() + 1e-12)) * math.sqrt(252))
    calmar = cagr / abs(mdd) if mdd != 0 else float("inf")
    n_sell = n_partial + n_final
    win_rate = wins / n_sell if n_sell else 0.0
    pl = abs(gross_w / gross_l) if gross_l != 0 else float("inf")
    operating = final - withdrawn   # 운용중(현금+평가액)

    return {"name": name, "equity": dt, "final": final, "CAGR": cagr, "MDD": mdd,
            "Sharpe": sharpe, "Calmar": calmar, "PL": pl, "win_rate": win_rate,
            "n_buy": n_buy, "n_add": n_add, "n_partial": n_partial, "n_final": n_final,
            "withdrawn": withdrawn, "operating": operating, "max_weight": max_weight}


def _yearly_ret(ser):
    return ser.groupby(ser.index.year).apply(lambda g: g.iloc[-1] / g.iloc[0] - 1)


def main():
    log("===== S1 v2 + 수익 인출(인출가능금) =====")
    store = data_layer.get_store()

    variants = [
        ("1.기존S1v2(인출없음)", False, False, 0, 0),
        ("2.A:부분익절인출",      True,  False, 0, 0),
        ("3.A+B X50 Y30",        True,  True,  0.5, 0.3),
        ("4.A+B X50 Y50",        True,  True,  0.5, 0.5),
        ("5.A+B X100 Y30",       True,  True,  1.0, 0.3),
        ("6.A+B X100 Y50",       True,  True,  1.0, 0.5),
        # 참고: 규칙 B만(A 없이) — 부분익절은 기존대로 재투입, 총자산 성장분만 인출
        ("7.B만 X50 Y30",        False, True,  0.5, 0.3),
        ("8.B만 X50 Y50",        False, True,  0.5, 0.5),
        ("9.B만 X100 Y30",       False, True,  1.0, 0.3),
        ("10.B만 X100 Y50",      False, True,  1.0, 0.5),
    ]
    results = [run(store, n, a, b, x, y) for n, a, b, x, y in variants]
    kospi = load_index("kospi_index").reindex(results[0]["equity"].index).ffill().bfill()

    print()
    print("=" * 118)
    print(" [요약] S1 v2 + 수익 인출  (지표는 모두 '총자산=운용+인출가능금' 기준)")
    print("=" * 118)
    print(f"  {'변형':>20s} {'CAGR':>8s} {'MDD':>8s} {'Sharpe':>7s} {'Calmar':>7s} {'손익비':>6s} {'승률':>6s}"
          f" {'최대비중':>7s} | {'총자산':>8s} {'운용중':>8s} {'인출가능금':>9s} {'인출%':>6s}")
    for r in results:
        wshare = r["withdrawn"] / r["final"] if r["final"] else 0
        print(f"  {r['name']:>20s} {fpct(r['CAGR']):>8s} {fpct(r['MDD']):>8s} {r['Sharpe']:>7.2f}"
              f" {r['Calmar']:>7.2f} {r['PL']:>6.2f} {r['win_rate']*100:>5.1f}% {r['max_weight']*100:>6.1f}%"
              f" | {r['final']/1e8:>6.2f}억 {r['operating']/1e8:>6.2f}억 {r['withdrawn']/1e8:>7.2f}억"
              f" {wshare*100:>5.1f}%")

    # 연도별 수익률(총자산 기준)
    ytab = pd.DataFrame({r["name"]: _yearly_ret(r["equity"]) for r in results})
    ytab["KOSPI"] = _yearly_ret(kospi)
    print()
    print("=" * 118)
    print(" [연도별 수익률] (총자산 기준)")
    print("=" * 118)
    print(ytab.map(lambda v: fpct(v) if pd.notna(v) else "-").to_string())

    print()
    print("=" * 100)
    print(" [6.4년 후 최종] 총자산 / 그중 인출가능금(이미 손에 쥘 수 있는 확정금액)")
    print("=" * 100)
    for r in results:
        wshare = r["withdrawn"] / r["final"] if r["final"] else 0
        print(f"  {r['name']:>20s} : 총자산 {r['final']/1e8:>6.2f}억  |  인출가능금 {r['withdrawn']/1e8:>6.2f}억"
              f" ({wshare*100:.1f}%)  |  운용중 {r['operating']/1e8:>6.2f}억")

    _save_chart(results, kospi, ytab)


def _save_chart(results, kospi, ytab):
    rows = [{k: r[k] for k in ("name", "CAGR", "MDD", "Sharpe", "Calmar", "PL",
                               "win_rate", "final", "operating", "withdrawn", "max_weight")}
            for r in results]
    pd.DataFrame(rows).to_csv(os.path.join(RESULT_DIR, "withdraw_summary.csv"),
                              index=False, encoding="utf-8-sig")
    ytab.to_csv(os.path.join(RESULT_DIR, "withdraw_yearly.csv"), encoding="utf-8-sig")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        for f in ["Malgun Gothic", "NanumGothic", "AppleGothic", "Gulim"]:
            try:
                matplotlib.rcParams["font.family"] = f
                matplotlib.rcParams["axes.unicode_minus"] = False
                break
            except Exception:
                continue
        fig, ax = plt.subplots(figsize=(14, 7))
        for r in results:
            ax.plot(r["equity"].index, r["equity"].values, lw=1.4, label=r["name"])
        kn = kospi / kospi.iloc[0] * INITIAL_CASH
        ax.plot(kn.index, kn.values, color="gray", lw=1.0, ls="--", label="KOSPI")
        ax.set_yscale("log")
        ax.set_title("S1 v2 + 수익 인출 : 총자산 곡선 (로그스케일)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e8:.1f}억"))
        ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3, which="both")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, "withdraw_curves.png"), dpi=140)
        plt.close(fig)
    except Exception as e:
        print(f"  (그래프 생략: {e})")
    print(f"\n  결과 저장 : {RESULT_DIR}")


if __name__ == "__main__":
    main()
