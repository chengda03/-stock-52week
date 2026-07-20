# -*- coding: utf-8 -*-
"""
backtest_s1_mktfilter.py — KOSPI 시장필터 기준선(이동평균 일수) 스윕
============================================================================
strategy_core.py / backtest_s1_partial.py 는 전혀 수정하지 않습니다.
backtest_s1_partial.py 의 엔진(run_variant)과 동일한 매매 규칙을 그대로 복제하되,
"강세장 판정에 쓰는 KOSPI 이동평균 일수"만 파라미터로 바꿔가며 비교합니다.

[고정 전략]  = backtest_s1_partial 의 "+8%50%익절·전량손절" 변형
  · 매수 6조건(시장필터 제외 5개) + 시장필터(KOSPI 종가 > K일선)
  · 불타기(강세장·하루1회·복리+3%)
  · +8% 도달 시 보유수량 50% 부분익절
  · 매도조건(90일선 이탈 / 평단 -12% 하드손절) → 전량매도
  → 매수/불타기/익절/손절 판정 로직은 strategy_core 그대로. 바뀌는 건 시장필터 K뿐.

[스윕 대상 K]  100, 120, 150, 170, 200, 220, 240 (일)

[출력] 기준선별 CAGR / MDD / Sharpe / Calmar / 손익비 / 승률 / 신규매수 건수
        + 기준선별 연도별 수익률표
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
    NUM_SLOTS, SLOT_AMOUNT_WON, PYRAMID_ADD_AMOUNT_WON,
)
import data_layer
from backtest_roe_eps_event import load_index, fpct, log
# 엔진 상수/스냅샷 빌더는 backtest_s1_partial 에서 그대로 재사용(원본 미수정)
from backtest_s1_partial import (
    _build_snap, INITIAL_CASH, BUY_COST, SELL_COST, TRADE_START, DUST_WON,
)

MA_WINDOWS = [100, 120, 150, 170, 200, 220, 240]
TP_STAGES = [(0.08, 0.50)]   # +8% 도달 시 50% 부분익절 (고정)
STOP_FRAC = 1.0              # 매도조건 발동 시 전량매도 (고정)

RESULT_DIR = os.path.join("data", "cache_s1_mktfilter", "results")
os.makedirs(RESULT_DIR, exist_ok=True)


def build_kospi_ma(store, window: int) -> np.ndarray:
    """거래일에 정렬된 KOSPI 종가로 window일 이동평균 계산 (data_layer 방식과 동일).
    min_periods=1 → 초기 구간도 부분평균으로 채움(원본 kospi_ma200_v와 동일 규칙)."""
    ks = load_index("kospi_index").reindex(store.trading_days).ffill().bfill()
    return ks.rolling(window, min_periods=1).mean().to_numpy(dtype=float)


def run_filter(store, kospi_ma_v: np.ndarray, name: str):
    """backtest_s1_partial.run_variant 의 로직을 그대로 복제.
    유일한 차이: is_bull 판정에 store.kospi_ma200_v 대신 인자 kospi_ma_v 사용."""
    td = store.trading_days
    close_v = store.close_v
    close_ff = store.close_ff
    c2c = store.code_to_col
    start_di = store.start_di
    stages = TP_STAGES
    stop_frac = STOP_FRAC

    cash = float(INITIAL_CASH)
    positions: dict[str, Position] = {}
    cost: dict[str, float] = {}
    stage_idx: dict[str, int] = {}
    daily_total = np.empty(len(td))

    n_buy = n_add = n_partial = n_final = 0
    wins = 0
    gross_w = gross_l = 0.0

    for di in range(len(td)):
        today = td[di].date()
        if di < start_di:
            daily_total[di] = INITIAL_CASH
            continue

        is_bull = is_bull_market(store.kospi_close_v[di], kospi_ma_v[di])
        sold_today: set[str] = set()

        # 1) 최종매도 (90일선/하드손절, 전량)
        for ticker in list(positions.keys()):
            col = c2c[ticker]
            px = close_v[di, col]
            if not np.isfinite(px):
                continue
            pos = positions[ticker]
            snap = _build_snap(store, col, di)
            should_sell, reason = check_sell_condition(pos, snap)
            if should_sell:
                if stop_frac >= 1.0:
                    sell_sh = pos.shares
                else:
                    sell_sh = pos.shares * stop_frac
                    if (pos.shares - sell_sh) * px < DUST_WON:
                        sell_sh = pos.shares
                frac = sell_sh / pos.shares
                proceeds = px * sell_sh * (1.0 - SELL_COST)
                cost_sold = cost[ticker] * frac
                pnl = proceeds - cost_sold
                cash += proceeds
                if pnl > 0:
                    wins += 1; gross_w += pnl
                else:
                    gross_l += pnl
                n_final += 1
                if sell_sh >= pos.shares - 1e-9:
                    del positions[ticker]; del cost[ticker]
                    stage_idx.pop(ticker, None)
                    sold_today.add(ticker)
                else:
                    pos.shares -= sell_sh
                    cost[ticker] -= cost_sold

        # 2) 부분익절 (+8% 도달시 50%)
        if stages:
            for ticker in list(positions.keys()):
                col = c2c[ticker]
                px = close_v[di, col]
                if not np.isfinite(px) or px <= 0:
                    continue
                pos = positions[ticker]
                ret = px / pos.entry_price - 1.0
                while stage_idx.get(ticker, 0) < len(stages):
                    idx = stage_idx.get(ticker, 0)
                    trig, frac = stages[idx]
                    if ret < trig or pos.shares <= 0:
                        break
                    sell_sh = pos.shares * frac
                    proceeds = px * sell_sh * (1.0 - SELL_COST)
                    cost_sold = cost[ticker] * frac
                    pnl = proceeds - cost_sold
                    cash += proceeds
                    pos.shares -= sell_sh
                    cost[ticker] -= cost_sold
                    stage_idx[ticker] = idx + 1
                    n_partial += 1
                    if pnl > 0:
                        wins += 1; gross_w += pnl
                    else:
                        gross_l += pnl

        # 3) 불타기 (강세장, 수익률 높은 순)
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
                    apply_pyramid(pos, float(px), today)
                    cash -= spent
                    cost[ticker] += spent
                    n_add += 1

        # 4) 신규매수 (강세장, 빈 슬롯)
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
                stage_idx[snap.ticker] = 0
                cash -= spent
                n_buy += 1
                free -= 1

        # 5) 일별 평가
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

    # 성과지표
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
    n_sell = n_partial + n_final
    win_rate = wins / n_sell if n_sell else 0.0
    pl = abs(gross_w / gross_l) if gross_l != 0 else float("inf")

    # 연도별 수익률 (직전 연말 대비, 첫 해는 초기자본 대비)
    ye = dt.resample("YE").last()
    yearly: dict[int, float] = {}
    prev = float(INITIAL_CASH)
    for ts, val in ye.items():
        yearly[ts.year] = val / prev - 1.0
        prev = float(val)

    return {"name": name, "final": final, "total_ret": total_ret,
            "CAGR": cagr, "MDD": mdd, "Sharpe": sharpe, "Calmar": calmar,
            "PL": pl, "win_rate": win_rate, "n_buy": n_buy, "n_add": n_add,
            "n_partial": n_partial, "n_final": n_final, "n_sell": n_sell,
            "yearly": yearly, "equity": dt}


def main():
    import argparse
    ap = argparse.ArgumentParser(description="KOSPI 시장필터 기준선 스윕 백테스트")
    ap.add_argument("--end", type=str, default=None,
                    help="백테스트 종료일(YYYY-MM-DD). 지정 시 그 날짜까지만 사용.")
    args = ap.parse_args()
    end = pd.Timestamp(args.end) if args.end else None
    log("===== KOSPI 시장필터 기준선 스윕 ====="
        + (f" (종료일 {end.date()})" if end is not None else ""))
    store = data_layer.get_store(end_date=end)

    results = []
    for w in MA_WINDOWS:
        ma_v = build_kospi_ma(store, w)
        name = f"{w}일선"
        log(f"[mktfilter] {name} 백테스트 중...")
        results.append(run_filter(store, ma_v, name))

    # ---- 요약표 ----
    print()
    print("=" * 100)
    print(" KOSPI 시장필터 기준선(이동평균 일수)별 성과  [전략: +8%50%익절·전량손절 고정]")
    print("=" * 100)
    print(f"  {'기준선':>7s} {'CAGR':>9s} {'MDD':>9s} {'Sharpe':>7s} {'Calmar':>7s} "
          f"{'손익비':>6s} {'승률':>7s} {'신규매수':>7s} {'불타기':>6s}")
    for r in results:
        print(f"  {r['name']:>7s} {fpct(r['CAGR']):>9s} {fpct(r['MDD']):>9s} "
              f"{r['Sharpe']:>7.2f} {r['Calmar']:>7.2f} {r['PL']:>6.2f} "
              f"{r['win_rate']*100:>6.1f}% {r['n_buy']:>7d} {r['n_add']:>6d}")

    # ---- 연도별 수익률표 ----
    years = sorted({y for r in results for y in r["yearly"].keys()})
    print()
    print("=" * 100)
    print(" 기준선별 연도별 수익률")
    print("=" * 100)
    header = "  " + f"{'기준선':>7s}" + "".join(f"{y:>9d}" for y in years)
    print(header)
    for r in results:
        row = "  " + f"{r['name']:>7s}"
        for y in years:
            v = r["yearly"].get(y)
            row += f"{(fpct(v) if v is not None else '-'):>9s}"
        print(row)

    # ---- 최적 기준선 판정 ----
    print()
    best_cagr = max(results, key=lambda r: r["CAGR"])
    best_calmar = max(results, key=lambda r: r["Calmar"])
    best_sharpe = max(results, key=lambda r: r["Sharpe"])
    base200 = next(r for r in results if r["name"] == "200일선")
    print(f"  · CAGR 최고  : {best_cagr['name']} ({fpct(best_cagr['CAGR'])})")
    print(f"  · Calmar 최고: {best_calmar['name']} ({best_calmar['Calmar']:.2f})")
    print(f"  · Sharpe 최고: {best_sharpe['name']} ({best_sharpe['Sharpe']:.2f})")
    print(f"  · 기준(200일선): CAGR {fpct(base200['CAGR'])}, MDD {fpct(base200['MDD'])}, "
          f"Calmar {base200['Calmar']:.2f}, Sharpe {base200['Sharpe']:.2f}")

    _save(results, years)


def _save(results, years):
    rows = []
    for r in results:
        row = {k: r[k] for k in ("name", "total_ret", "CAGR", "MDD", "Sharpe",
                                 "Calmar", "PL", "win_rate", "n_buy", "n_add",
                                 "n_partial", "n_final", "n_sell")}
        for y in years:
            row[f"ret_{y}"] = r["yearly"].get(y)
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(RESULT_DIR, "mktfilter_results.csv"),
                              index=False, encoding="utf-8-sig")
    print(f"  결과 저장 : {RESULT_DIR}")


if __name__ == "__main__":
    main()
