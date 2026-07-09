# -*- coding: utf-8 -*-
"""
backtest_s1_cap.py — S1 v2 + "종목당 비중 상한(초과분만 덜어내기)" 변형
=====================================================================
확정판 S1 v2(1회성 +8%@50% 익절 · 90일선/-12% 손절 · 불타기)를 그대로 두고,
매일 평가 시 특정 종목의 비중이 상한(cap)을 넘으면 '초과분만' 매도해 cap으로
되돌린다. (사다리식처럼 전 물량을 반복 매도하지 않으므로 대박종목의 코어는 유지)

strategy_core.py 는 수정하지 않음. cap=None 은 baseline(=현재 확정판 S1 v2).
불타기(피라미딩)로 승자가 커져 cap을 넘기면 그 초과분만 잘라 쏠림만 억제.

[비교] baseline(무제한) vs cap 25% / 30% / 40%
[출력] CAGR·MDD·Sharpe·Calmar·손익비·승률, 쏠림지표(최대비중·30/40/50% 초과일),
       효성티앤씨 2021-08 최대비중, 1억 투자 최종금액, 자산곡선 차트.
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

HYOSUNG = "298020"
RESULT_DIR = os.path.join("data", "cache_s1_cap", "results")
os.makedirs(RESULT_DIR, exist_ok=True)


def run(store, cap):
    td = store.trading_days
    close_v = store.close_v
    close_ff = store.close_ff
    c2c = store.code_to_col
    start_di = store.start_di

    cash = float(INITIAL_CASH)
    positions: dict[str, Position] = {}
    cost: dict[str, float] = {}
    part_done: dict[str, bool] = {}

    daily_total = np.empty(len(td))
    rec_date, rec_total, rec_topw, rec_topn, rec_toppyr = [], [], [], [], []
    hyo_date, hyo_w = [], []

    n_buy = n_add = n_partial = n_final = n_cap = 0
    wins = 0
    gross_w = gross_l = 0.0
    # 카테고리 분리 집계
    clean_wins = 0                 # 진짜 청산(부분익절+최종매도) 중 이익 건수
    cap_wins = 0                   # 상한 트리밍 중 이익 건수
    cap_ret_sum = 0.0              # 상한 트리밍 수익률 합(평균용, px/평단가-1)

    def sell_shares(ticker, px, sell_sh, cat):
        """지정 수량 매도 → 손익 집계. cat='clean'|'cap'. 반환 pnl."""
        nonlocal cash, wins, gross_w, gross_l, clean_wins, cap_wins, cap_ret_sum
        pos = positions[ticker]
        ratio = sell_sh / pos.shares
        proceeds = px * sell_sh * (1.0 - SELL_COST)
        cost_sold = cost[ticker] * ratio
        pnl = proceeds - cost_sold
        cash += proceeds
        pos.shares -= sell_sh
        cost[ticker] -= cost_sold
        if pnl > 0:
            wins += 1; gross_w += pnl
        else:
            gross_l += pnl
        if cat == "clean":
            if pnl > 0:
                clean_wins += 1
        else:  # cap
            if pnl > 0:
                cap_wins += 1
            cap_ret_sum += (px / pos.avg_price - 1.0)
        return pnl

    for di in range(len(td)):
        today = td[di].date()
        if di < start_di:
            daily_total[di] = INITIAL_CASH
            continue
        is_bull = is_bull_market(store.kospi_close_v[di], store.kospi_ma200_v[di])
        sold_today: set[str] = set()

        # 1) 전량매도
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
                    wins += 1; gross_w += pnl; clean_wins += 1
                else:
                    gross_l += pnl
                n_final += 1
                del positions[ticker]; del cost[ticker]; part_done.pop(ticker, None)
                sold_today.add(ticker)

        # 1-2) 부분익절 (+8% 1회성)
        for ticker in list(positions.keys()):
            col = c2c[ticker]; px = close_v[di, col]
            if not np.isfinite(px) or px <= 0:
                continue
            pos = positions[ticker]
            if not part_done.get(ticker, False) and px >= pos.entry_price * (1 + PARTIAL_EXIT_TRIGGER_PCT):
                sell_shares(ticker, px, pos.shares * PARTIAL_EXIT_RATIO, "clean")
                part_done[ticker] = True
                n_partial += 1

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

        # 3-2) 비중 상한: 초과분만 덜어내기 (cap 지정 시)
        if cap is not None and positions:
            for _ in range(len(positions)):  # 몇 번 반복해 안정화
                # 현재 총자산·보유가치 계산
                vals = {}
                for t, pos in positions.items():
                    col = c2c[t]; p = close_v[di, col]
                    if not np.isfinite(p) or p <= 0:
                        continue
                    vals[t] = p * pos.shares
                total_now = cash + sum(vals.values())
                if total_now <= 0:
                    break
                # cap 초과 종목 중 비중 최대부터
                over = [(t, v) for t, v in vals.items() if v > cap * total_now + 1.0]
                if not over:
                    break
                t, v = max(over, key=lambda x: x[1])
                col = c2c[t]; px = close_v[di, col]
                target_v = cap * total_now
                sell_sh = (v - target_v) / px
                pos = positions[t]
                sell_sh = min(sell_sh, pos.shares * 0.999999)
                if sell_sh <= 0:
                    break
                sell_shares(t, px, sell_sh, "cap")
                n_cap += 1

        # 4) 평가 + 쏠림 기록
        hv = 0.0
        hold_vals: dict[str, float] = {}
        for t, pos in positions.items():
            col = c2c[t]; p = close_v[di, col]
            if not np.isfinite(p):
                p = close_ff[di, col]
                if not np.isfinite(p):
                    p = pos.avg_price
            v = p * pos.shares
            hold_vals[t] = v; hv += v
        total = cash + hv
        daily_total[di] = total
        if hold_vals and total > 0:
            top_t = max(hold_vals, key=hold_vals.get)
            rec_date.append(td[di]); rec_total.append(total)
            rec_topw.append(hold_vals[top_t] / total)
            rec_topn.append(store.name_map.get(top_t, top_t))
            rec_toppyr.append(positions[top_t].pyramid_count)
            if HYOSUNG in hold_vals:
                hyo_date.append(td[di]); hyo_w.append(hold_vals[HYOSUNG] / total)

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
    n_sell = n_partial + n_final + n_cap
    win_rate = wins / n_sell if n_sell else 0.0
    pl = abs(gross_w / gross_l) if gross_l != 0 else float("inf")
    # 카테고리별
    n_clean = n_partial + n_final
    win_clean = clean_wins / n_clean if n_clean else 0.0
    cap_win_rate = cap_wins / n_cap if n_cap else 0.0
    cap_avg_ret = cap_ret_sum / n_cap if n_cap else 0.0

    conc = pd.DataFrame({"date": rec_date, "total": rec_total, "top_weight": rec_topw,
                         "name": rec_topn, "pyramid": rec_toppyr})
    conc = conc[conc["date"] >= TRADE_START].reset_index(drop=True)
    hyo = pd.Series(hyo_w, index=pd.DatetimeIndex(hyo_date))

    return {"cap": cap, "equity": dt, "final": final, "CAGR": cagr, "MDD": mdd,
            "Sharpe": sharpe, "Calmar": calmar, "PL": pl, "win_rate": win_rate,
            "n_buy": n_buy, "n_add": n_add, "n_partial": n_partial,
            "n_final": n_final, "n_cap": n_cap, "conc": conc, "hyo": hyo,
            "n_clean": n_clean, "win_clean": win_clean,
            "cap_win_rate": cap_win_rate, "cap_avg_ret": cap_avg_ret}


def _yearly_ret(ser):
    return ser.groupby(ser.index.year).apply(lambda g: g.iloc[-1] / g.iloc[0] - 1)


def _invest_table(dt):
    ye = dt.groupby(dt.index.year).last()
    rows = []; prev = float(INITIAL_CASH)
    for yr, bal in ye.items():
        bal = float(bal)
        rows.append({"연도": yr, "연말잔고": bal,
                     "누적수익금": bal - INITIAL_CASH,
                     "누적수익률": bal / INITIAL_CASH - 1,
                     "연간수익금": bal - prev,
                     "연간수익률": bal / prev - 1})
        prev = bal
    return pd.DataFrame(rows).set_index("연도")


def _conc_stats(conc):
    imax = conc["top_weight"].idxmax(); r = conc.loc[imax]
    return {"max_w": r["top_weight"], "max_date": r["date"], "max_name": r["name"],
            "max_pyr": int(r["pyramid"]),
            "p30": (conc["top_weight"] >= 0.30).mean(),
            "p40": (conc["top_weight"] >= 0.40).mean(),
            "p50": (conc["top_weight"] >= 0.50).mean()}


def _hyo_aug(r):
    h = r["hyo"]; seg = h[(h.index >= "2021-08-01") & (h.index <= "2021-08-31")]
    return seg.max() if len(seg) else float("nan")


def main():
    log("===== S1 v2 + 종목당 비중상한(초과분 매도) =====")
    store = data_layer.get_store()
    caps = [None, 0.50, 0.40, 0.30]
    results = [run(store, c) for c in caps]
    base = results[0]
    kospi = load_index("kospi_index").reindex(base["equity"].index).ffill().bfill()

    def lbl(c):
        return "무제한(S1v2)" if c is None else f"상한 {int(c*100)}%"

    print()
    print("=" * 108)
    print(" [요약] S1 v2 + 종목당 비중상한 (초과분만 덜어내기)")
    print("=" * 108)
    print(f"  {'구분':>12s} {'CAGR':>8s} {'MDD':>8s} {'Sharpe':>7s} {'Calmar':>7s} {'손익비':>6s} {'최종금액':>9s}"
          f" | {'전체승률':>7s} {'청산승률':>7s} {'불타기':>5s} {'부분익절':>6s} {'상한절삭':>6s} {'최종매도':>6s}")
    for r in results:
        print(f"  {lbl(r['cap']):>12s} {fpct(r['CAGR']):>8s} {fpct(r['MDD']):>8s} {r['Sharpe']:>7.2f}"
              f" {r['Calmar']:>7.2f} {r['PL']:>6.2f} {r['final']/1e8:>7.2f}억"
              f" | {r['win_rate']*100:>6.1f}% {r['win_clean']*100:>6.1f}% {r['n_add']:>5d} {r['n_partial']:>6d}"
              f" {r['n_cap']:>6d} {r['n_final']:>6d}")

    # ---------- 매도 2종류 분리 승률 ----------
    print()
    print("=" * 108)
    print(" [매도 2종류 분리]  ①진짜 청산(부분익절+90일선+하드손절) 승률  vs  ②상한 트리밍(별도 집계)")
    print("=" * 108)
    print(f"  {'구분':>12s} | {'①청산 건수':>9s} {'①청산 승률':>9s} || {'②트리밍 건수':>10s} {'②트리밍 평균수익률':>14s} {'②트리밍 승률':>10s}")
    for r in results:
        print(f"  {lbl(r['cap']):>12s} | {r['n_clean']:>9d} {r['win_clean']*100:>8.1f}%"
              f" || {r['n_cap']:>10d} {fpct(r['cap_avg_ret']):>14s} {r['cap_win_rate']*100:>9.1f}%")

    print()
    print("=" * 108)
    print(" [쏠림] 최대 단일종목 비중 & 초과일 비율 & 효성티앤씨 2021-08 최대비중")
    print("=" * 108)
    print(f"  {'구분':>12s} {'최대비중':>8s} {'(날짜/종목/불타기)':>24s} {'≥30%':>6s} {'≥40%':>6s} {'≥50%':>6s} {'효성08':>7s}")
    for r in results:
        s = _conc_stats(r["conc"])
        tag = f"{s['max_date'].date()} {s['max_name']} {s['max_pyr']}단"
        print(f"  {lbl(r['cap']):>12s} {s['max_w']*100:>7.1f}% {tag:>24s} {s['p30']*100:>5.1f}% {s['p40']*100:>5.1f}%"
              f" {s['p50']*100:>5.1f}% {_hyo_aug(r)*100:>6.1f}%")

    # 연도별 수익률
    ytab = pd.DataFrame({lbl(r["cap"]): _yearly_ret(r["equity"]) for r in results})
    ytab["KOSPI"] = _yearly_ret(kospi)
    print()
    print("=" * 108)
    print(" [연도별 수익률]")
    print("=" * 108)
    print(ytab.map(lambda v: fpct(v) if pd.notna(v) else "-").to_string())

    print()
    print("=" * 108)
    print(" [최종 자산 비교] (6.4년 후, 1억 시작)")
    print("=" * 108)
    for r in results[1:]:
        d = base["final"] - r["final"]
        print(f"  {lbl(r['cap']):>12s} : {r['final']/1e8:>6.2f}억  (무제한 대비 {(-d)/1e8:+.2f}억,"
              f" {r['final']/base['final']*100:.1f}% 수준)")

    # ---------- 상한 50% 상세 ----------
    cap50 = next(r for r in results if r["cap"] == 0.50)
    print()
    print("=" * 92)
    print(" [상한 50% 상세]  연도별 수익률 (vs 무제한·KOSPI·KOSDAQ)")
    print("=" * 92)
    kosdaq = load_index("kosdaq_index").reindex(base["equity"].index).ffill().bfill()
    y50 = pd.DataFrame({
        "상한50%": _yearly_ret(cap50["equity"]),
        "무제한(S1v2)": _yearly_ret(base["equity"]),
        "KOSPI": _yearly_ret(kospi),
        "KOSDAQ": _yearly_ret(kosdaq),
    })
    print(y50.map(lambda v: fpct(v) if pd.notna(v) else "-").to_string())

    inv = _invest_table(cap50["equity"])
    print()
    print("=" * 100)
    print(" [상한 50%] 1억원 투자 시 연도별 결과")
    print("=" * 100)
    print(f"  {'연도':>10s} {'연말잔고':>16s} {'누적수익금':>16s} {'누적수익률':>10s} {'연간수익금':>16s} {'연간수익률':>10s}")
    print(f"  {'시작(2020-01)':>10s} {INITIAL_CASH:>16,.0f} {'—':>16s} {'—':>10s} {'—':>16s} {'—':>10s}")
    for yr, r in inv.iterrows():
        print(f"  {yr:>10d} {r['연말잔고']:>16,.0f} {r['누적수익금']:>+16,.0f} {fpct(r['누적수익률']):>10s}"
              f" {r['연간수익금']:>+16,.0f} {fpct(r['연간수익률']):>10s}")
    inv.to_csv(os.path.join(RESULT_DIR, "invest_cap50.csv"), encoding="utf-8-sig")

    _chart(results, kospi, lbl)


def _chart(results, kospi, lbl):
    for r in results:
        tag = "base" if r["cap"] is None else f"cap{int(r['cap']*100)}"
        r["conc"].to_csv(os.path.join(RESULT_DIR, f"conc_{tag}.csv"), index=False, encoding="utf-8-sig")
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
        colors = ["#3182F6", "#12B886", "#E8590C", "#9C36B5"]
        fig, ax = plt.subplots(figsize=(14, 7))
        for r, cl in zip(results, colors):
            ax.plot(r["equity"].index, r["equity"].values, lw=1.6, color=cl, label=lbl(r["cap"]))
        kn = kospi / kospi.iloc[0] * INITIAL_CASH
        ax.plot(kn.index, kn.values, color="gray", lw=1.0, ls="--", label="KOSPI")
        ax.set_yscale("log")
        ax.set_title("S1 v2 + 종목당 비중상한 : 무제한 vs 40/30/25% vs KOSPI (로그)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e8:.1f}억"))
        ax.legend(loc="upper left"); ax.grid(alpha=0.3, which="both")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, "cap_curves.png"), dpi=140)
        plt.close(fig)
    except Exception as e:
        print(f"  (그래프 생략: {e})")
    print(f"\n  결과 저장 : {RESULT_DIR}")


if __name__ == "__main__":
    main()
