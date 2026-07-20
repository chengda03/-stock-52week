# -*- coding: utf-8 -*-
"""
backtest_dualindex.py
=====================
[실험] 시장필터를 "KOSPI 단독" → "KOSPI+KOSDAQ 동시조건"으로 바꿨을 때 효과를,
특히 2024년 관점에서 재검토. 실제 KOSPI(KS11)/KOSDAQ(KQ11) 지수 데이터로 확인.

진단(매매 전):
  1) 2024년 중 'KOSPI강세(>200MA) AND KOSDAQ약세(<200MA)'인 날 수/비율
  2) 그 날들에 실제 매수된 종목 중 KOSDAQ 건수와, KOSDAQ vs KOSPI 최종손익 비교(base 기준)
  3) 2020~2026 전체·연도별로 같은 패턴(코스피강세+코스닥약세) 발생 비율 — 2024가 유독 많았나

비교(동일 DataStore·동일 날 → 드리프트 0, 차이는 오직 '시장필터'):
  · base       : KOSPI 단독 (원본 strategy_core)
  · dualindex  : KOSPI AND KOSDAQ 동시

정직성: 단일 경로(약 6.5년) 백테스트. 과최적화 위험 상존.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

import data_layer
import strategy_core as sc_base
from strategy_core import StockSnapshot, Position, MARKET_FILTER_MA_DAYS
from backtest_roe_eps_event import load_index, load_corp_cls

INITIAL_CASH = 100_000_000
BUY_COST = 0.00115
SELL_COST = 0.00295
TRADE_START = pd.Timestamp("2020-01-01")

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

WINDOWS = [
    ("2020", "2020-01-01", "2020-12-31"),
    ("2021", "2021-01-01", "2021-12-31"),
    ("2022(하락장)", "2022-01-01", "2022-12-31"),
    ("2023", "2023-01-01", "2023-12-31"),
    ("2024", "2024-01-01", "2024-12-31"),
    ("2025", "2025-01-01", "2025-12-31"),
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


def build_kosdaq(store):
    kq = load_index("kosdaq_index").reindex(store.trading_days).ffill().bfill()
    close = kq.to_numpy(dtype=float)
    ma = kq.rolling(MARKET_FILTER_MA_DAYS, min_periods=1).mean().to_numpy(dtype=float)
    return close, ma


def build_market_map(store):
    """{code: 'KOSPI'|'KOSDAQ'|'?'}  (Y=KOSPI, K=KOSDAQ)"""
    corps = list({store.code_to_corp[c] for c in store.valid_codes})
    cls = load_corp_cls(corps)   # corp_code -> 'Y'/'K'
    out = {}
    for c in store.valid_codes:
        v = cls.get(store.code_to_corp.get(c))
        out[c] = "KOSPI" if v == "Y" else ("KOSDAQ" if v == "K" else "?")
    return out


# ---------------------------------------------------------------------------
# 진단
# ---------------------------------------------------------------------------
def diagnostics(store, kq_close, kq_ma):
    td = store.trading_days
    kospi_bull = store.kospi_close_v > store.kospi_ma200_v
    kosdaq_bull = kq_close > kq_ma
    kospi_only = kospi_bull & (~kosdaq_bull)   # 코스피강세 + 코스닥약세

    print("\n" + "=" * 92)
    print(" (진단 1·3) 연도별 'KOSPI강세 AND KOSDAQ약세'(=코스피필터는 열렸는데 코스닥은 약세) 비율")
    print("=" * 92)
    rows = []
    for tag, ws, we in [("전체", "2020-01-01", "2026-07-14")] + WINDOWS:
        wsd = pd.Timestamp(ws); wed = pd.Timestamp(we)
        mask = (td >= wsd) & (td <= wed)
        ndays = int(mask.sum())
        nb = int(kospi_bull[mask].sum())                 # 코스피 강세일
        nko = int(kospi_only[mask].sum())                # 코스피강세+코스닥약세
        rows.append({
            "구간": tag, "거래일": ndays,
            "코스피강세일": nb,
            "코스피강세+코스닥약세": nko,
            "강세일中비율%": round(nko / nb * 100, 1) if nb else 0.0,
            "전체中비율%": round(nko / ndays * 100, 1) if ndays else 0.0,
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(os.path.join(RESULT_DIR, "dualindex_diag_days.csv"), index=False, encoding="utf-8-sig")
    return kospi_only


def log_buys_window(store, is_bull_arr, kospi_only_arr, market_of, ws, we):
    """base 전략을 [ws,we]에서 1억 리셋 실행하며 각 신규매수 기록(시장/코스닥약세일여부/최종손익)."""
    td = store.trading_days
    close_v = store.close_v; close_ff = store.close_ff
    c2c = store.code_to_col
    wsd = pd.Timestamp(ws); wed = pd.Timestamp(we)
    di_list = [di for di in range(len(td)) if wsd <= td[di] <= wed]
    end_di = di_list[-1]

    cash = float(INITIAL_CASH)
    positions = {}; rec = {}
    closed = []

    for di in di_list:
        today = td[di].date()
        is_bull = bool(is_bull_arr[di])
        sold_today = set()
        # 전량매도
        for t in list(positions.keys()):
            col = c2c[t]; px = close_v[di, col]
            if not np.isfinite(px):
                continue
            pos = positions[t]
            s, reason = sc_base.check_sell_condition(pos, _build_snap(store, col, di))
            if s:
                proceeds = px * pos.shares * (1.0 - SELL_COST); cash += proceeds
                r = rec[t]; r["proceeds"] += proceeds; r["pnl_pct"] = r["proceeds"]/r["invested"]-1.0
                r["reason"] = reason
                closed.append(r); del positions[t]; del rec[t]; sold_today.add(t)
        # 부분익절
        for t in list(positions.keys()):
            col = c2c[t]; px = close_v[di, col]
            if not np.isfinite(px) or px <= 0:
                continue
            pos = positions[t]
            if sc_base.check_partial_exit(pos, float(px)):
                sh = sc_base.apply_partial_exit(pos, float(px))
                proceeds = px*sh*(1.0-SELL_COST); cash += proceeds; rec[t]["proceeds"] += proceeds
        # 불타기
        if is_bull and positions:
            def cur_ret(t):
                col = c2c[t]; px = close_v[di, col]
                return px/positions[t].avg_price if np.isfinite(px) else -1.0
            for t in sorted(positions.keys(), key=cur_ret, reverse=True):
                col = c2c[t]; px = close_v[di, col]
                if not np.isfinite(px) or px <= 0:
                    continue
                pos = positions[t]
                if sc_base.check_pyramid(pos, float(px), today, is_bull, cash):
                    spent = sc_base.PYRAMID_ADD_AMOUNT_WON*(1.0+BUY_COST)
                    if cash < spent:
                        continue
                    sc_base.apply_pyramid(pos, float(px), today); cash -= spent; rec[t]["invested"] += spent
        # 신규매수
        free = sc_base.NUM_SLOTS - len(positions)
        if is_bull and free > 0:
            cands = []
            for t in store.get_universe(td[di]):
                if t in positions or t in sold_today:
                    continue
                col = c2c[t]; px = close_v[di, col]
                if not np.isfinite(px) or px <= 0:
                    continue
                s = _build_snap(store, col, di)
                ok, _ = sc_base.check_buy_conditions(s, True)
                if ok:
                    cands.append(s)
            for s in sc_base.rank_by_momentum(cands):
                if free <= 0:
                    break
                px = s.price; sh = int(sc_base.SLOT_AMOUNT_WON // px)
                if sh <= 0:
                    continue
                spent = sh*px*(1.0+BUY_COST)
                if spent > cash:
                    continue
                positions[s.ticker] = Position(ticker=s.ticker, entry_price=px, avg_price=px,
                                               shares=float(sh), entry_date=today)
                cash -= spent; free -= 1
                rec[s.ticker] = {"code": s.ticker, "market": market_of.get(s.ticker, "?"),
                                 "buy_date": str(today), "kosdaq_weak_day": bool(kospi_only_arr[di]),
                                 "invested": spent, "proceeds": 0.0,
                                 "pnl_pct": float("nan"), "reason": ""}
    # 기간말 보유중
    for t, pos in positions.items():
        col = c2c[t]; p = close_v[end_di, col]
        if not np.isfinite(p):
            p = close_ff[end_di, col]
            if not np.isfinite(p):
                p = pos.avg_price
        r = rec[t]; cur = p*pos.shares*(1.0-SELL_COST)
        r["pnl_pct"] = (r["proceeds"]+cur)/r["invested"]-1.0; r["reason"] = "보유중"
        closed.append(r)
    return closed


# ---------------------------------------------------------------------------
# 엔진 (시장필터 is_bull_arr 교체)
# ---------------------------------------------------------------------------
def run_engine(store, is_bull_arr, di_range):
    td = store.trading_days
    close_v = store.close_v; close_ff = store.close_ff
    c2c = store.code_to_col

    cash = float(INITIAL_CASH)
    positions = {}; cost = {}
    equity = []; days = []
    n_buy = n_add = n_partial = n_final = 0
    wins = 0; gross_w = gross_l = 0.0

    for di in di_range:
        today = td[di].date()
        is_bull = bool(is_bull_arr[di])
        sold_today = set()

        for t in list(positions.keys()):
            col = c2c[t]; px = close_v[di, col]
            if not np.isfinite(px):
                continue
            pos = positions[t]
            s, _ = sc_base.check_sell_condition(pos, _build_snap(store, col, di))
            if s:
                proceeds = px*pos.shares*(1.0-SELL_COST); cash += proceeds
                pnl = proceeds - cost[t]
                if pnl > 0: wins += 1; gross_w += pnl
                else: gross_l += pnl
                n_final += 1
                del positions[t]; del cost[t]; sold_today.add(t)

        for t in list(positions.keys()):
            col = c2c[t]; px = close_v[di, col]
            if not np.isfinite(px) or px <= 0:
                continue
            pos = positions[t]
            if sc_base.check_partial_exit(pos, float(px)):
                before = pos.shares
                sh = sc_base.apply_partial_exit(pos, float(px))
                ratio = sh/before
                proceeds = px*sh*(1.0-SELL_COST); cost_sold = cost[t]*ratio
                pnl = proceeds - cost_sold
                cash += proceeds; cost[t] -= cost_sold; n_partial += 1
                if pnl > 0: wins += 1; gross_w += pnl
                else: gross_l += pnl

        if is_bull and positions:
            def cur_ret(t):
                col = c2c[t]; px = close_v[di, col]
                return px/positions[t].avg_price if np.isfinite(px) else -1.0
            for t in sorted(positions.keys(), key=cur_ret, reverse=True):
                col = c2c[t]; px = close_v[di, col]
                if not np.isfinite(px) or px <= 0:
                    continue
                pos = positions[t]
                if sc_base.check_pyramid(pos, float(px), today, is_bull, cash):
                    spent = sc_base.PYRAMID_ADD_AMOUNT_WON*(1.0+BUY_COST)
                    if cash < spent:
                        continue
                    sc_base.apply_pyramid(pos, float(px), today)
                    cash -= spent; cost[t] += spent; n_add += 1

        free = sc_base.NUM_SLOTS - len(positions)
        if is_bull and free > 0:
            cands = []
            for t in store.get_universe(td[di]):
                if t in positions or t in sold_today:
                    continue
                col = c2c[t]; px = close_v[di, col]
                if not np.isfinite(px) or px <= 0:
                    continue
                s = _build_snap(store, col, di)
                ok, _ = sc_base.check_buy_conditions(s, True)
                if ok:
                    cands.append(s)
            for s in sc_base.rank_by_momentum(cands):
                if free <= 0:
                    break
                px = s.price; sh = int(sc_base.SLOT_AMOUNT_WON // px)
                if sh <= 0:
                    continue
                spent = sh*px*(1.0+BUY_COST)
                if spent > cash:
                    continue
                positions[s.ticker] = Position(ticker=s.ticker, entry_price=px, avg_price=px,
                                               shares=float(sh), entry_date=today)
                cost[s.ticker] = spent; cash -= spent; n_buy += 1; free -= 1

        hv = 0.0
        for t, pos in positions.items():
            col = c2c[t]; p = close_v[di, col]
            if not np.isfinite(p):
                p = close_ff[di, col]
                if not np.isfinite(p):
                    p = pos.avg_price
            hv += p*pos.shares
        equity.append(cash+hv); days.append(td[di])

    s = pd.Series(equity, index=pd.DatetimeIndex(days))
    final = float(s.iloc[-1]); years = (s.index[-1]-s.index[0]).days/365.25
    period_ret = final/INITIAL_CASH - 1.0
    cagr = (final/INITIAL_CASH)**(1.0/max(years,1e-9)) - 1.0
    peak = s.cummax(); mdd = float(((s-peak)/peak).min())
    rets = s.pct_change().fillna(0)
    sharpe = float((rets.mean()/(rets.std()+1e-12))*math.sqrt(252))
    calmar = cagr/abs(mdd) if mdd != 0 else float("inf")
    n_sell = n_partial + n_final
    return {"period_ret": period_ret, "CAGR": cagr, "MDD": mdd, "Sharpe": sharpe, "Calmar": calmar,
            "PL": abs(gross_w/gross_l) if gross_l != 0 else float("inf"),
            "win_rate": wins/n_sell if n_sell else 0.0,
            "n_trades": n_buy+n_add+n_partial+n_final, "n_buy": n_buy, "n_add": n_add,
            "start": str(s.index[0].date()), "end": str(s.index[-1].date())}


def main():
    store = data_layer.get_store()
    kq_close, kq_ma = build_kosdaq(store)
    market_of = build_market_map(store)

    kospi_only = diagnostics(store, kq_close, kq_ma)

    kospi_bull_arr = store.kospi_close_v > store.kospi_ma200_v
    dual_bull_arr = kospi_bull_arr & (kq_close > kq_ma)

    # (진단 2) 2024년 base 매수 종목 — 시장별/코스닥약세일별 손익
    logs = log_buys_window(store, kospi_bull_arr, kospi_only, market_of, "2024-01-01", "2024-12-31")
    dfl = pd.DataFrame(logs)
    dfl["pnl_pct"] = dfl["pnl_pct"] * 100.0
    dfl.to_csv(os.path.join(RESULT_DIR, "dualindex_2024_buys.csv"), index=False, encoding="utf-8-sig")
    print("\n" + "=" * 92)
    print(" (진단 2) 2024년 base 신규매수 — 시장 구분별 손익")
    print("=" * 92)
    g = dfl.groupby("market").agg(건수=("code", "size"), 평균손익=("pnl_pct", "mean"),
                                  손실건수=("pnl_pct", lambda x: int((x <= 0).sum())))
    print(g.to_string())
    print(f"\n  ▷ '코스닥약세일(KOSPI강세+KOSDAQ약세)'에 매수된 건: {int(dfl['kosdaq_weak_day'].sum())} / {len(dfl)}")
    gw = dfl.groupby(["kosdaq_weak_day"]).agg(건수=("code", "size"), 평균손익=("pnl_pct", "mean"))
    print(gw.to_string())
    # 코스닥약세일에 매수된 코스닥 종목만
    sub = dfl[(dfl["kosdaq_weak_day"]) & (dfl["market"] == "KOSDAQ")]
    if len(sub):
        print(f"\n  ▷ 그 중 KOSDAQ 종목: {len(sub)}건 · 평균손익 {sub['pnl_pct'].mean():.2f}% "
              f"(전체 KOSDAQ 평균 {dfl[dfl['market']=='KOSDAQ']['pnl_pct'].mean():.2f}%)")

    # (1) 전체기간 비교
    full = range(store.start_di, len(store.trading_days))
    mb = run_engine(store, kospi_bull_arr, full)
    md = run_engine(store, dual_bull_arr, full)
    rows = []
    for tag, m in [("base(KOSPI단독)", mb), ("dualindex(KOSPI+KOSDAQ)", md)]:
        rows.append({"버전": tag, "CAGR%": round(m["CAGR"]*100, 2), "MDD%": round(m["MDD"]*100, 2),
                     "Sharpe": round(m["Sharpe"], 2), "Calmar": round(m["Calmar"], 2),
                     "손익비": round(m["PL"], 2), "승률%": round(m["win_rate"]*100, 1),
                     "총거래": m["n_trades"], "신규매수": m["n_buy"], "불타기": m["n_add"]})
    dff = pd.DataFrame(rows)
    print("\n" + "=" * 92)
    print(f" (1) 전체기간 비교  ·  {mb['start']} ~ {mb['end']}  ·  1억 · 거래비용 반영")
    print("=" * 92)
    print(dff.to_string(index=False))
    print(f"  Δ CAGR(dual-base) = {(md['CAGR']-mb['CAGR'])*100:+.2f}%p")
    dff.to_csv(os.path.join(RESULT_DIR, "dualindex_full_compare.csv"), index=False, encoding="utf-8-sig")

    # (2) 연도별 독립
    yr = []
    td = store.trading_days
    for tag, ws, we in WINDOWS:
        wsd = pd.Timestamp(ws); wed = pd.Timestamp(we)
        di_list = [di for di in range(len(td)) if wsd <= td[di] <= wed]
        b = run_engine(store, kospi_bull_arr, di_list)
        d = run_engine(store, dual_bull_arr, di_list)
        yr.append({"구간": tag,
                   "base수익%": round(b["period_ret"]*100, 2), "dual수익%": round(d["period_ret"]*100, 2),
                   "Δ%p": round((d["period_ret"]-b["period_ret"])*100, 2),
                   "base_MDD%": round(b["MDD"]*100, 2), "dual_MDD%": round(d["MDD"]*100, 2),
                   "base매수": b["n_buy"], "dual매수": d["n_buy"]})
    dfy = pd.DataFrame(yr)
    print("\n" + "=" * 92)
    print(" (2) 연도별 독립 백테스트 (매년 1억 리셋)")
    print("=" * 92)
    print(dfy.to_string(index=False))
    dfy.to_csv(os.path.join(RESULT_DIR, "dualindex_yearly.csv"), index=False, encoding="utf-8-sig")

    print("\n" + "=" * 92)
    print(" ⚠ 정직성: 단일 경로(약 6.5년) 백테스트. 과최적화 위험 존재.")
    print("=" * 92)


if __name__ == "__main__":
    main()
