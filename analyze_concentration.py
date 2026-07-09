# -*- coding: utf-8 -*-
"""
analyze_concentration.py — backtest_v3 포지션 쏠림(단일종목 최대비중) 분석
=========================================================================
backtest_v3.py 와 '동일한' 매매 루프(strategy_core 관통)를 재생하되,
매 거래일마다 "포트폴리오 전체 자산 대비 가장 비중이 큰 단일종목의 비율"을
기록한다. (backtest_v3.py 자체는 수정하지 않음)

전체자산 = 현금 + Σ(보유종목 평가액),  단일비중 = 해당종목 평가액 / 전체자산.

[출력]
 · 전체 기간 최대 단일종목 비중 (날짜/종목명/비중)
 · 20% 이상 / 30% 이상이었던 일수와 비율
 · 상위 5개 쏠림 사례 (날짜/종목명/비중/그때의 불타기 단계)
 · 최대쏠림 구간 자산곡선 차트
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from strategy_core import (
    Position, is_bull_market, check_buy_conditions, rank_by_momentum,
    check_sell_condition, check_pyramid, apply_pyramid,
    check_partial_exit, apply_partial_exit,
    NUM_SLOTS, SLOT_AMOUNT_WON, PYRAMID_ADD_AMOUNT_WON,
)
import data_layer
from backtest_v3 import _build_snap, INITIAL_CASH, BUY_COST, SELL_COST, TRADE_START
from backtest_roe_eps_event import fpct, log

RESULT_DIR = os.path.join("data", "cache_v3", "results")
os.makedirs(RESULT_DIR, exist_ok=True)


def run():
    log("===== 단일종목 쏠림 분석 (backtest_v3 루프 재생) =====")
    store = data_layer.get_store()
    td = store.trading_days
    close_v = store.close_v
    close_ff = store.close_ff
    c2c = store.code_to_col
    start_di = store.start_di

    cash = float(INITIAL_CASH)
    positions: dict[str, Position] = {}
    cost: dict[str, float] = {}

    daily_total = np.empty(len(td))
    # 일별 쏠림 기록
    rec_date, rec_total, rec_top_w = [], [], []
    rec_top_ticker, rec_top_name, rec_top_pyr = [], [], []

    for di in range(len(td)):
        today = td[di].date()
        if di < start_di:
            daily_total[di] = INITIAL_CASH
            continue

        is_bull = is_bull_market(store.kospi_close_v[di], store.kospi_ma200_v[di])
        sold_today: set[str] = set()

        # 1) 전량매도
        for ticker in list(positions.keys()):
            col = c2c[ticker]
            px = close_v[di, col]
            if not np.isfinite(px):
                continue
            pos = positions[ticker]
            snap = _build_snap(store, col, di)
            should_sell, _ = check_sell_condition(pos, snap)
            if should_sell:
                cash += px * pos.shares * (1.0 - SELL_COST)
                del positions[ticker]; del cost[ticker]
                sold_today.add(ticker)

        # 1-2) 부분익절
        for ticker in list(positions.keys()):
            col = c2c[ticker]
            px = close_v[di, col]
            if not np.isfinite(px) or px <= 0:
                continue
            pos = positions[ticker]
            if check_partial_exit(pos, float(px)):
                shares_before = pos.shares
                sell_sh = apply_partial_exit(pos, float(px))
                ratio = sell_sh / shares_before
                cash += px * sell_sh * (1.0 - SELL_COST)
                cost[ticker] -= cost[ticker] * ratio

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
                    cash -= spent; cost[ticker] += spent

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
                cost[snap.ticker] = spent
                cash -= spent; free -= 1

        # 4) 일별 평가 + 쏠림 기록
        hv = 0.0
        hold_vals: dict[str, float] = {}
        for t, pos in positions.items():
            col = c2c[t]; p = close_v[di, col]
            if not np.isfinite(p):
                p = close_ff[di, col]
                if not np.isfinite(p):
                    p = pos.avg_price
            v = p * pos.shares
            hold_vals[t] = v
            hv += v
        total = cash + hv
        daily_total[di] = total

        if hold_vals and total > 0:
            top_t = max(hold_vals, key=hold_vals.get)
            rec_date.append(td[di])
            rec_total.append(total)
            rec_top_w.append(hold_vals[top_t] / total)
            rec_top_ticker.append(top_t)
            rec_top_name.append(store.name_map.get(top_t, top_t))
            rec_top_pyr.append(positions[top_t].pyramid_count)

    # ---------- 집계 ----------
    conc = pd.DataFrame({
        "date": rec_date, "total": rec_total, "top_weight": rec_top_w,
        "ticker": rec_top_ticker, "name": rec_top_name, "pyramid": rec_top_pyr,
    })
    conc = conc[conc["date"] >= TRADE_START].reset_index(drop=True)

    dt = pd.Series(daily_total, index=td)
    dt = dt[dt.index >= TRADE_START]

    n_days = len(conc)
    n20 = int((conc["top_weight"] >= 0.20).sum())
    n30 = int((conc["top_weight"] >= 0.30).sum())
    imax = conc["top_weight"].idxmax()
    row = conc.loc[imax]

    print()
    print("=" * 92)
    print(f" 단일종목 쏠림 분석  ·  보유일수 {n_days}일  ({conc['date'].iloc[0].date()} ~ {conc['date'].iloc[-1].date()})")
    print("=" * 92)
    print(f"  전체 기간 최대 단일종목 비중 : {row['top_weight']*100:.1f}%"
          f"  ({row['date'].date()} · {row['name']}({row['ticker']}) · 불타기 {int(row['pyramid'])}단계)")
    print(f"  평균 최대비중 : {conc['top_weight'].mean()*100:.1f}%   중앙값 : {conc['top_weight'].median()*100:.1f}%")
    print()
    print(f"  단일종목 비중 ≥ 20% : {n20:4d}일  ({n20/n_days*100:.1f}%)")
    print(f"  단일종목 비중 ≥ 30% : {n30:4d}일  ({n30/n_days*100:.1f}%)")

    print("\n  [ 상위 5개 최대 쏠림 사례 ]")
    print(f"    {'날짜':>12s} {'종목명':>12s} {'비중':>7s} {'불타기단계':>8s}")
    top5 = conc.sort_values("top_weight", ascending=False).head(5)
    for _, r in top5.iterrows():
        print(f"    {str(r['date'].date()):>12s} {r['name']:>12s} {r['top_weight']*100:>6.1f}% {int(r['pyramid']):>7d}단계")

    conc.to_csv(os.path.join(RESULT_DIR, "concentration_daily.csv"),
                index=False, encoding="utf-8-sig")
    _chart(conc, dt, row)
    print(f"\n  결과 저장 : {RESULT_DIR}\\concentration_daily.csv, concentration_peak.png")
    return conc


def _chart(conc, dt, peak_row):
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

        # 최대쏠림 날짜 전후 ±60거래일 구간
        peak_date = peak_row["date"]
        idx = conc.index[conc["date"] == peak_date][0]
        lo = conc["date"].iloc[max(0, idx - 60)]
        hi = conc["date"].iloc[min(len(conc) - 1, idx + 60)]
        seg = dt[(dt.index >= lo) & (dt.index <= hi)]
        seg_c = conc[(conc["date"] >= lo) & (conc["date"] <= hi)]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True,
                                       gridspec_kw={"height_ratios": [2, 1]})
        ax1.plot(seg.index, seg.values, color="#3182F6", lw=1.6, label="전체 자산")
        ax1.axvline(peak_date, color="crimson", ls="--", lw=1.2)
        ax1.set_title(f"최대쏠림 구간 자산곡선  ·  피크 {peak_date.date()} "
                      f"{peak_row['name']} {peak_row['top_weight']*100:.1f}%")
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e8:.1f}억"))
        ax1.legend(loc="upper left"); ax1.grid(alpha=0.3)

        ax2.plot(seg_c["date"], seg_c["top_weight"] * 100, color="#E8590C", lw=1.4,
                 label="최대 단일종목 비중")
        ax2.axhline(20, color="gray", ls=":", lw=1)
        ax2.axhline(30, color="gray", ls=":", lw=1)
        ax2.axvline(peak_date, color="crimson", ls="--", lw=1.2)
        ax2.set_ylabel("비중(%)"); ax2.legend(loc="upper left"); ax2.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, "concentration_peak.png"), dpi=140)
        plt.close(fig)
    except Exception as e:
        print(f"  (그래프 생략: {e})")


if __name__ == "__main__":
    run()
