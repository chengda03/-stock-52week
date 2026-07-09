# -*- coding: utf-8 -*-
"""
backtest_v3.py
==============
strategy_core.py(전략 판정) + data_layer.py(데이터)를 조합한 백테스트.

목표: STRATEGY_FINAL.md §2 수치 재현
  - 기간   : 2020-01-02 ~ 2026-06-05 (데이터 있는 만큼)
  - 초기자금: 1억원 / 거래비용 매수 0.115% · 매도 0.295%
  - 목표    : 누적 +3,634.3%, CAGR +75.7%, MDD -30.9%,
             Sharpe 1.66, Calmar 2.45, 손익비 3.65, 승률 12.8%

★ 판정 로직은 전부 strategy_core.py 함수만 사용합니다.
  (백테스트와 실운영 앱이 '똑같은 규칙'을 쓰도록 — 규칙은 한 파일에만 존재)

★ 엔진(backtest_roe_eps_variants.py)과의 미세 차이 (완전 동일하진 않음, 방향성 검증용):
  - 불타기 체결수량: apply_pyramid는 소수주(=금액/가격), 원 엔진은 정수주(int)
  - 평단가 계산: strategy_core는 수수료 미포함 가격가중, 원 엔진은 수수료 포함
  → 데이터·핵심 규칙은 동일하므로 큰 자릿수/방향성은 재현되어야 정상.
"""
from __future__ import annotations

import json
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
    PARTIAL_EXIT_TRIGGER_PCT,
)
import data_layer
from backtest_roe_eps_event import load_index, fpct, log

# --- 운용 설정 -------------------------------------------------------------
INITIAL_CASH = 100_000_000
BUY_COST = 0.00115     # 매수 수수료 0.115%
SELL_COST = 0.00295    # 매도 수수료+세금 0.295%
TRADE_START = pd.Timestamp("2020-01-01")

RESULT_DIR = os.path.join("data", "cache_v3", "results")
os.makedirs(RESULT_DIR, exist_ok=True)

# 확정판(+8%@50% 부분익절) 목표 수치 (backtest_s1_partial 계산본과 대조용)
#   → strategy_core 관통 계산이 별도 스크립트 수치와 정확히 일치해야 정상.
TARGET = {
    "누적수익률": 39.662, "CAGR": 0.7805, "MDD": -0.2762,
    "Sharpe": 1.688, "Calmar": 2.826, "손익비": 3.879, "승률": 0.3353,
}


def _build_snap(store, col, di) -> StockSnapshot:
    """store 배열에서 di일·col종목의 StockSnapshot을 빠르게 조립."""
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


def run_backtest():
    log("===== backtest_v3 (strategy_core 기반) 시작 =====")
    store = data_layer.get_store()
    td = store.trading_days
    close_v = store.close_v
    close_ff = store.close_ff
    c2c = store.code_to_col
    start_di = store.start_di

    cash = float(INITIAL_CASH)
    positions: dict[str, Position] = {}   # ticker -> Position
    cost: dict[str, float] = {}           # ticker -> 누적 투입원가(수수료 포함)
    daily_total = np.empty(len(td))
    trades: list[dict] = []

    n_buy = n_add = n_partial = n_final = 0
    wins = 0
    gross_w = gross_l = 0.0

    for di in range(len(td)):
        today = td[di].date()

        if di < start_di:
            daily_total[di] = INITIAL_CASH
            continue

        kc = store.kospi_close_v[di]
        km = store.kospi_ma200_v[di]
        is_bull = is_bull_market(kc, km)
        sold_today: set[str] = set()

        # ---------------- 1) 전량매도 (90일선/하드손절, 시장상태 무관) ----------------
        for ticker in list(positions.keys()):
            col = c2c[ticker]
            px = close_v[di, col]
            if not np.isfinite(px):
                continue  # 거래정지 → 평가만
            pos = positions[ticker]
            snap = _build_snap(store, col, di)
            should_sell, reason = check_sell_condition(pos, snap)
            if should_sell:
                proceeds = px * pos.shares * (1.0 - SELL_COST)
                cash += proceeds
                pnl = proceeds - cost[ticker]
                if pnl > 0:
                    wins += 1; gross_w += pnl
                else:
                    gross_l += pnl
                n_final += 1
                trades.append({"date": today, "ticker": ticker,
                               "name": store.name_map.get(ticker, ticker),
                               "action": "매도", "reason": reason,
                               "price": float(px), "shares": pos.shares,
                               "pnl": float(pnl), "ret": float(px / pos.avg_price - 1.0)})
                del positions[ticker]; del cost[ticker]
                sold_today.add(ticker)

        # ---------------- 1-2) 부분익절 (+8% 도달시 50%, 전량매도 안 된 종목만) ----------------
        # 호출 순서 규칙: check_sell_condition(전량) → check_partial_exit(부분).
        for ticker in list(positions.keys()):
            col = c2c[ticker]
            px = close_v[di, col]
            if not np.isfinite(px) or px <= 0:
                continue
            pos = positions[ticker]
            if check_partial_exit(pos, float(px)):
                shares_before = pos.shares
                sell_sh = apply_partial_exit(pos, float(px))   # 50% 매도수량, pos.shares 차감·플래그 set
                ratio = sell_sh / shares_before
                proceeds = px * sell_sh * (1.0 - SELL_COST)
                cost_sold = cost[ticker] * ratio
                pnl = proceeds - cost_sold
                cash += proceeds
                cost[ticker] -= cost_sold      # 남은 50% 원가만 남김(전량 청산시 손익 정확)
                n_partial += 1
                if pnl > 0:
                    wins += 1; gross_w += pnl
                else:
                    gross_l += pnl
                trades.append({"date": today, "ticker": ticker,
                               "name": store.name_map.get(ticker, ticker),
                               "action": "부분익절", "reason": f"+{PARTIAL_EXIT_TRIGGER_PCT*100:.0f}% 트리거(50% 매도)",
                               "price": float(px), "shares": float(sell_sh),
                               "pnl": float(pnl), "ret": float(px / pos.entry_price - 1.0)})

        # ---------------- 2) 불타기 (강세장, 수익률 높은 순 우선) ----------------
        if is_bull and positions:
            def cur_ret(t):
                col = c2c[t]; px = close_v[di, col]
                return px / positions[t].avg_price if np.isfinite(px) else -1.0
            for ticker in sorted(positions.keys(), key=cur_ret, reverse=True):
                col = c2c[ticker]
                px = close_v[di, col]
                if not np.isfinite(px) or px <= 0:
                    continue
                pos = positions[ticker]
                if check_pyramid(pos, float(px), today, is_bull, cash):
                    spent = PYRAMID_ADD_AMOUNT_WON * (1.0 + BUY_COST)
                    if cash < spent:
                        continue
                    apply_pyramid(pos, float(px), today)   # 소수주 추가 + 평단가/카운트 갱신
                    cash -= spent
                    cost[ticker] += spent
                    n_add += 1
                    trades.append({"date": today, "ticker": ticker,
                                   "name": store.name_map.get(ticker, ticker),
                                   "action": "불타기", "reason": f"{pos.pyramid_count}단계 트리거",
                                   "price": float(px), "shares": pos.shares,
                                   "pnl": 0.0, "ret": float(px / pos.avg_price - 1.0)})

        # ---------------- 3) 신규매수 (강세장, 빈 슬롯) ----------------
        free = NUM_SLOTS - len(positions)
        if is_bull and free > 0:
            uni = store.get_universe(td[di])
            cands: list[StockSnapshot] = []
            for t in uni:
                if t in positions or t in sold_today:
                    continue
                col = c2c[t]
                px = close_v[di, col]
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
                cash -= spent
                n_buy += 1
                free -= 1
                trades.append({"date": today, "ticker": snap.ticker,
                               "name": store.name_map.get(snap.ticker, snap.ticker),
                               "action": "신규매수", "reason": "6조건 통과",
                               "price": float(px), "shares": float(sh),
                               "pnl": 0.0, "ret": 0.0})

        # ---------------- 4) 일별 평가 ----------------
        hv = 0.0
        for t, pos in positions.items():
            col = c2c[t]
            p = close_v[di, col]
            if not np.isfinite(p):
                p = close_ff[di, col]
                if not np.isfinite(p):
                    p = pos.avg_price
            hv += p * pos.shares
        daily_total[di] = cash + hv

    # ---------------- 성과지표 ----------------
    dt = pd.Series(daily_total, index=td)
    dt = dt[dt.index >= TRADE_START]
    final = float(dt.iloc[-1])
    years = (dt.index[-1] - dt.index[0]).days / 365.25
    total_ret = final / INITIAL_CASH - 1.0
    cagr = (final / INITIAL_CASH) ** (1.0 / max(years, 1e-9)) - 1.0
    peak = dt.cummax()
    mdd = float(((dt - peak) / peak).min())
    rets = dt.pct_change().fillna(0)
    sharpe = float((rets.mean() / (rets.std() + 1e-12)) * math.sqrt(252))
    calmar = cagr / abs(mdd) if mdd != 0 else float("inf")
    n_sell = n_partial + n_final   # 승률 분모 = 부분익절 + 전량매도
    win_rate = wins / n_sell if n_sell else 0.0
    pl = abs(gross_w / gross_l) if gross_l != 0 else float("inf")

    res = {"final": final, "total_ret": total_ret, "CAGR": cagr, "MDD": mdd,
           "Sharpe": sharpe, "Calmar": calmar, "PL": pl, "win_rate": win_rate,
           "n_buy": n_buy, "n_add": n_add, "n_partial": n_partial,
           "n_final": n_final, "n_sell": n_sell,
           "start": str(dt.index[0].date()), "end": str(dt.index[-1].date())}

    # 벤치마크
    bench = {}
    for nm, idxname in [("KOSPI", "kospi_index"), ("KOSDAQ", "kosdaq_index")]:
        try:
            bench[nm] = load_index(idxname).reindex(dt.index).ffill().bfill()
        except Exception:
            pass

    # 연도별 수익률
    yr_strat = dt.groupby(dt.index.year).apply(lambda g: g.iloc[-1] / g.iloc[0] - 1)
    ytab = {"전략": yr_strat}
    for nm, ser in bench.items():
        ytab[nm] = ser.groupby(ser.index.year).apply(lambda g: g.iloc[-1] / g.iloc[0] - 1)
    ydf = pd.DataFrame(ytab)

    _print_report(res, ydf, bench, dt)
    _save(res, ydf, dt, trades, bench)
    return res


def _print_report(res, ydf, bench, dt):
    print()
    print("=" * 88)
    print(f" backtest_v3 결과  ·  {res['start']} ~ {res['end']}  ·  1억 · 거래비용 반영")
    print("=" * 88)
    print(f"  최종자산 : {res['final']:,.0f} 원")
    print(f"  누적수익률: {fpct(res['total_ret'])}")
    print(f"  CAGR     : {fpct(res['CAGR'])}   MDD: {fpct(res['MDD'])}")
    print(f"  Sharpe   : {res['Sharpe']:.2f}   Calmar: {res['Calmar']:.2f}   손익비: {res['PL']:.2f}")
    print(f"  거래수   : 신규 {res['n_buy']} · 불타기 {res['n_add']} · 부분익절 {res['n_partial']}"
          f" · 최종매도 {res['n_final']} (승률 {res['win_rate']*100:.1f}%, 분모=부분익절+최종매도={res['n_sell']})")

    print("\n  [ STRATEGY_FINAL.md 목표 대비 재현도 ]")
    print(f"  {'지표':>10s} {'목표':>12s} {'재현':>12s} {'차이':>12s}")
    rows = [
        ("누적수익률", TARGET["누적수익률"], res["total_ret"], "배"),
        ("CAGR", TARGET["CAGR"], res["CAGR"], "%"),
        ("MDD", TARGET["MDD"], res["MDD"], "%"),
        ("Sharpe", TARGET["Sharpe"], res["Sharpe"], ""),
        ("Calmar", TARGET["Calmar"], res["Calmar"], ""),
        ("손익비", TARGET["손익비"], res["PL"], ""),
        ("승률", TARGET["승률"], res["win_rate"], "%"),
    ]
    for name, tgt, got, unit in rows:
        if unit == "%":
            print(f"  {name:>10s} {tgt*100:>10.1f}% {got*100:>10.1f}% {(got-tgt)*100:>+10.1f}%p")
        elif unit == "배":
            print(f"  {name:>10s} {tgt:>10.2f}배 {got:>10.2f}배 {got-tgt:>+10.2f}배")
        else:
            print(f"  {name:>10s} {tgt:>11.2f} {got:>11.2f} {got-tgt:>+11.2f}")

    print("\n  [ 벤치마크 (단순보유) ]")
    for nm, ser in bench.items():
        b_ret = ser.iloc[-1] / ser.iloc[0] - 1.0
        print(f"    {nm:7s} 누적 {fpct(b_ret):>10s}")

    print("\n  [ 연도별 수익률 ]")
    print(ydf.map(lambda v: fpct(v) if pd.notna(v) else "-").to_string())


def _save(res, ydf, dt, trades, bench):
    dt.to_frame("total").to_csv(os.path.join(RESULT_DIR, "equity.csv"), encoding="utf-8-sig")
    ydf.to_csv(os.path.join(RESULT_DIR, "yearly.csv"), encoding="utf-8-sig")
    if trades:
        pd.DataFrame(trades).to_csv(os.path.join(RESULT_DIR, "trades.csv"),
                                    index=False, encoding="utf-8-sig")
    with open(os.path.join(RESULT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    # 자산곡선 차트 (전략 vs KOSPI/KOSDAQ)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(dt.index, dt.values, label="S1 전략", lw=1.7, color="#3182F6")
        for nm, ser in bench.items():
            norm = ser / ser.iloc[0] * INITIAL_CASH
            ax.plot(norm.index, norm.values, label=nm, lw=1.0, ls="--")
        ax.set_yscale("log")
        ax.set_title("backtest_v3 : S1 (strategy_core) vs benchmarks")
        ax.legend(loc="upper left")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e8:.1f}억"))
        ax.grid(alpha=0.3, which="both")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, "equity_curve.png"), dpi=140)
        plt.close(fig)
    except Exception as e:
        print(f"  (그래프 생략: {e})")
    print(f"\n  결과 저장 : {RESULT_DIR}")


if __name__ == "__main__":
    run_backtest()
