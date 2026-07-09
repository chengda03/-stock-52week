# -*- coding: utf-8 -*-
"""
backtest_s1_ladder.py — S1 v2 부분익절: "1회성" vs "반복 사다리식" 비교
=======================================================================
strategy_core.py 는 건드리지 않고(=규칙 원본 보존), 이 스크립트 안에서만
두 가지 부분익절 방식을 구현해 비교한다. 매수·불타기·손절(90일선/-12%)은 동일.

[모드]
 once   : 진입가 +8% 도달 시 보유수량의 50% 매도 (1회성) = 현재 확정판 S1 v2
 ladder : 진입가 +8%, +16%, +24%, +32% ...(8%씩 단리)
          각 단계 최초 도달 시 '남은 보유수량'의 50% 매도 (반복).
          하루 1회 실행(불타기와 동일한 확인형) → 하루에 여러 단계를 건너뛰면
          다음 날부터 한 단계씩 이어 실행. 불타기로 붙은 물량도 전체수량 기준 적용.

집중도(쏠림)도 함께 측정: 일별 최대 단일종목 비중, 30/40/50% 초과 일수,
그리고 효성티앤씨(298020) 2021-08 구간 최대비중 재계산.
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

LADDER_STEP = 0.08          # 사다리 간격(+8%씩 단리)
HYOSUNG = "298020"          # 효성티앤씨
RESULT_DIR = os.path.join("data", "cache_s1_ladder", "results")
os.makedirs(RESULT_DIR, exist_ok=True)


def run(store, mode):
    td = store.trading_days
    close_v = store.close_v
    close_ff = store.close_ff
    c2c = store.code_to_col
    start_di = store.start_di

    cash = float(INITIAL_CASH)
    positions: dict[str, Position] = {}
    cost: dict[str, float] = {}
    part_done: dict[str, bool] = {}   # once 모드용
    ladder_step: dict[str, int] = {}  # ladder 모드용: 이미 실행한 최고 단계

    daily_total = np.empty(len(td))
    rec_date, rec_total, rec_topw, rec_topt, rec_topn, rec_toppyr = [], [], [], [], [], []
    hyo_date, hyo_w = [], []

    n_buy = n_add = n_partial = n_final = 0
    wins = 0
    gross_w = gross_l = 0.0

    def do_partial_sell(ticker, px, frac, di_today):
        nonlocal cash, wins, gross_w, gross_l, n_partial
        pos = positions[ticker]
        sell_sh = pos.shares * frac
        proceeds = px * sell_sh * (1.0 - SELL_COST)
        ratio = sell_sh / pos.shares
        cost_sold = cost[ticker] * ratio
        pnl = proceeds - cost_sold
        cash += proceeds
        pos.shares -= sell_sh
        cost[ticker] -= cost_sold
        n_partial += 1
        if pnl > 0:
            wins += 1; gross_w += pnl
        else:
            gross_l += pnl

    for di in range(len(td)):
        today = td[di].date()
        if di < start_di:
            daily_total[di] = INITIAL_CASH
            continue
        is_bull = is_bull_market(store.kospi_close_v[di], store.kospi_ma200_v[di])
        sold_today: set[str] = set()

        # 1) 전량매도 (90일선/-12%)
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
                del positions[ticker]; del cost[ticker]
                part_done.pop(ticker, None); ladder_step.pop(ticker, None)
                sold_today.add(ticker)

        # 1-2) 부분익절
        for ticker in list(positions.keys()):
            col = c2c[ticker]; px = close_v[di, col]
            if not np.isfinite(px) or px <= 0:
                continue
            pos = positions[ticker]
            if mode == "once":
                if not part_done.get(ticker, False) and px >= pos.entry_price * (1 + PARTIAL_EXIT_TRIGGER_PCT):
                    do_partial_sell(ticker, px, PARTIAL_EXIT_RATIO, di)
                    part_done[ticker] = True
            else:  # ladder — 하루 1단계
                step = ladder_step.get(ticker, 0)
                next_trigger = pos.entry_price * (1 + LADDER_STEP * (step + 1))
                if px >= next_trigger:
                    do_partial_sell(ticker, px, PARTIAL_EXIT_RATIO, di)
                    ladder_step[ticker] = step + 1

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
                cost[snap.ticker] = spent
                part_done[snap.ticker] = False
                ladder_step[snap.ticker] = 0
                cash -= spent; n_buy += 1; free -= 1

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
            rec_topt.append(top_t); rec_topn.append(store.name_map.get(top_t, top_t))
            rec_toppyr.append(positions[top_t].pyramid_count)
            if HYOSUNG in hold_vals:
                hyo_date.append(td[di]); hyo_w.append(hold_vals[HYOSUNG] / total)

    # ---------- 지표 ----------
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

    conc = pd.DataFrame({"date": rec_date, "total": rec_total, "top_weight": rec_topw,
                         "ticker": rec_topt, "name": rec_topn, "pyramid": rec_toppyr})
    conc = conc[conc["date"] >= TRADE_START].reset_index(drop=True)
    hyo = pd.Series(hyo_w, index=pd.DatetimeIndex(hyo_date))

    return {"mode": mode, "equity": dt, "final": final, "CAGR": cagr, "MDD": mdd,
            "Sharpe": sharpe, "Calmar": calmar, "PL": pl, "win_rate": win_rate,
            "n_buy": n_buy, "n_add": n_add, "n_partial": n_partial, "n_final": n_final,
            "conc": conc, "hyo": hyo}


def _yearly_ret(ser):
    return ser.groupby(ser.index.year).apply(lambda g: g.iloc[-1] / g.iloc[0] - 1)


def _invest_table(dt):
    """연말잔고/누적수익금/누적수익률/연간수익금/연간수익률."""
    ye = dt.groupby(dt.index.year).last()
    rows = []
    prev = float(INITIAL_CASH)
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
    n = len(conc)
    imax = conc["top_weight"].idxmax(); r = conc.loc[imax]
    return {"n": n, "max_w": r["top_weight"], "max_date": r["date"], "max_name": r["name"],
            "max_pyr": int(r["pyramid"]),
            "p30": (conc["top_weight"] >= 0.30).mean(),
            "p40": (conc["top_weight"] >= 0.40).mean(),
            "p50": (conc["top_weight"] >= 0.50).mean()}


def main():
    log("===== S1 부분익절: 1회성 vs 사다리식 =====")
    store = data_layer.get_store()
    once = run(store, "once")
    lad = run(store, "ladder")

    kospi = load_index("kospi_index").reindex(once["equity"].index).ffill().bfill()
    kosdaq = load_index("kosdaq_index").reindex(once["equity"].index).ffill().bfill()

    # ---------- 요약 지표 ----------
    print()
    print("=" * 100)
    print(" [요약] 1회성(S1 v2) vs 사다리식 부분익절")
    print("=" * 100)
    hdr = f"  {'구분':>10s} {'CAGR':>8s} {'MDD':>8s} {'Sharpe':>7s} {'Calmar':>7s} {'손익비':>6s} {'승률':>6s} | {'신규':>4s} {'불타기':>5s} {'부분익절':>6s} {'최종매도':>6s}"
    print(hdr)
    for r, nm in [(once, "1회성"), (lad, "사다리식")]:
        print(f"  {nm:>10s} {fpct(r['CAGR']):>8s} {fpct(r['MDD']):>8s} {r['Sharpe']:>7.2f}"
              f" {r['Calmar']:>7.2f} {r['PL']:>6.2f} {r['win_rate']*100:>5.1f}%"
              f" | {r['n_buy']:>4d} {r['n_add']:>5d} {r['n_partial']:>6d} {r['n_final']:>6d}")

    # ---------- 쏠림 지표 ----------
    so, sl = _conc_stats(once["conc"]), _conc_stats(lad["conc"])
    print()
    print("=" * 100)
    print(" [쏠림] 최대 단일종목 비중 & 초과 일수 비율")
    print("=" * 100)
    print(f"  {'구분':>10s} {'최대비중':>8s} {'(날짜/종목/불타기)':>24s} {'≥30%':>7s} {'≥40%':>7s} {'≥50%':>7s}")
    for s, nm in [(so, "1회성"), (sl, "사다리식")]:
        tag = f"{s['max_date'].date()} {s['max_name']} {s['max_pyr']}단계"
        print(f"  {nm:>10s} {s['max_w']*100:>7.1f}% {tag:>24s} {s['p30']*100:>6.1f}% {s['p40']*100:>6.1f}% {s['p50']*100:>6.1f}%")

    # 효성티앤씨 2021-08 구간
    def hyo_aug(r):
        h = r["hyo"]
        seg = h[(h.index >= "2021-08-01") & (h.index <= "2021-08-31")]
        return seg.max() if len(seg) else float("nan")
    print()
    print(f"  효성티앤씨 2021-08 최대비중 :  1회성 {hyo_aug(once)*100:.1f}%  →  사다리식 {hyo_aug(lad)*100:.1f}%")

    # ---------- 연도별 수익률 ----------
    ytab = pd.DataFrame({
        "1회성(S1v2)": _yearly_ret(once["equity"]),
        "사다리식": _yearly_ret(lad["equity"]),
        "KOSPI": _yearly_ret(kospi),
        "KOSDAQ": _yearly_ret(kosdaq),
    })
    print()
    print("=" * 100)
    print(" [연도별 수익률]")
    print("=" * 100)
    print(ytab.map(lambda v: fpct(v) if pd.notna(v) else "-").to_string())

    # ---------- 1억 투자 연도별 상세 (나란히) ----------
    to = _invest_table(once["equity"]); tl = _invest_table(lad["equity"])
    print()
    print("=" * 116)
    print(" [1억원 투자 연도별]  좌=1회성(S1 v2)  /  우=사다리식")
    print("=" * 116)
    print(f"  {'연도':>5s} | {'연말잔고(1회성)':>16s} {'누적수익률':>9s} {'연간수익률':>9s}"
          f" || {'연말잔고(사다리)':>16s} {'누적수익률':>9s} {'연간수익률':>9s}")
    for yr in to.index:
        o = to.loc[yr]; l = tl.loc[yr]
        print(f"  {yr:>5d} | {o['연말잔고']:>16,.0f} {fpct(o['누적수익률']):>9s} {fpct(o['연간수익률']):>9s}"
              f" || {l['연말잔고']:>16,.0f} {fpct(l['누적수익률']):>9s} {fpct(l['연간수익률']):>9s}")

    print()
    print(f"  누적수익금(1회성)  : {to['누적수익금'].iloc[-1]:>16,.0f} 원")
    print(f"  누적수익금(사다리) : {tl['누적수익금'].iloc[-1]:>16,.0f} 원")

    # ---------- 최종금액 비교 ----------
    diff = once["final"] - lad["final"]
    print()
    print("=" * 100)
    print(" [최종 자산 비교] (6.4년 후)")
    print("=" * 100)
    print(f"  1회성(S1 v2) : {once['final']:>16,.0f} 원  ({once['final']/1e8:.2f}억)")
    print(f"  사다리식     : {lad['final']:>16,.0f} 원  ({lad['final']/1e8:.2f}억)")
    print(f"  차이(1회성-사다리) : {diff:>+,.0f} 원  ({diff/1e8:+.2f}억)  "
          f"→ 사다리식은 1회성의 {lad['final']/once['final']*100:.1f}% 수준")

    _save_and_chart(once, lad, kospi, ytab, to, tl)


def _save_and_chart(once, lad, kospi, ytab, to, tl):
    once["conc"].to_csv(os.path.join(RESULT_DIR, "conc_once.csv"), index=False, encoding="utf-8-sig")
    lad["conc"].to_csv(os.path.join(RESULT_DIR, "conc_ladder.csv"), index=False, encoding="utf-8-sig")
    ytab.to_csv(os.path.join(RESULT_DIR, "yearly.csv"), encoding="utf-8-sig")
    to.to_csv(os.path.join(RESULT_DIR, "invest_once.csv"), encoding="utf-8-sig")
    tl.to_csv(os.path.join(RESULT_DIR, "invest_ladder.csv"), encoding="utf-8-sig")
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
        ax.plot(once["equity"].index, once["equity"].values, color="#3182F6", lw=1.7, label="1회성 (S1 v2)")
        ax.plot(lad["equity"].index, lad["equity"].values, color="#E8590C", lw=1.7, label="사다리식")
        kn = kospi / kospi.iloc[0] * INITIAL_CASH
        ax.plot(kn.index, kn.values, color="gray", lw=1.0, ls="--", label="KOSPI")
        ax.set_yscale("log")
        ax.set_title("S1 부분익절: 1회성 vs 사다리식 vs KOSPI (로그스케일)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e8:.1f}억"))
        ax.legend(loc="upper left"); ax.grid(alpha=0.3, which="both")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, "ladder_curves.png"), dpi=140)
        plt.close(fig)
    except Exception as e:
        print(f"  (그래프 생략: {e})")
    print(f"\n  결과 저장 : {RESULT_DIR}")


if __name__ == "__main__":
    main()
