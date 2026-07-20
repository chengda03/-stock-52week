# -*- coding: utf-8 -*-
"""
backtest_grid_ladder.py — "그리드 물타기형" 전략 백테스트
============================================================================
strategy_core.py / backtest_s1_partial.py 는 전혀 수정하지 않고 import만 합니다.
S1(추세추종: 오르는 종목을 불타기로 쫓아감)과 정반대 성격의
"하락 그리드 물타기 + 분할익절" 전략을 새로 구현·비교합니다.

[유니버스/후보]  S1과 동일 (시총상위500, 관리/스팩/우선주/리츠 제외, 매수 6조건,
                20일모멘텀 순 정렬) — data_layer / strategy_core 그대로 재사용.

[최초 매수]  빈 슬롯에 후보를 100만원어치 매수. 이 최초물량은 90일선 안전장치가
             걸리기 전까지 절대 팔지 않고 계속 보유.

[추가 매수(하락 그리드)]  진입가 × 0.97^n (n=1,2,3...) 지점을 하향 돌파할 때마다
             100만원씩 추가매수. 횟수 무제한, 현금≥100만, 강세장일 때만.
             각 회차는 독립 "물량(lot)"으로 관리.

[매도(개별 익절)]  각 추가매수 lot 은 자기 매입가 대비 +3% 오르면 그 lot만 매도.
             (최초물량은 제외 → 계속 보유)

[전량매도(안전장치)]  종가가 90일선을 3일 연속 이탈하면 그 종목 전 물량 청산.

[운영]  20종목 · 1억원 · 종목당 상한 없음(그리드 자연발생) ·
        거래비용 매수 0.115% / 매도 0.295%.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

from strategy_core import (
    StockSnapshot,
    is_bull_market, check_buy_conditions, rank_by_momentum,
    NUM_SLOTS,
)
import data_layer
from backtest_roe_eps_event import fpct, log
from backtest_s1_partial import (
    _build_snap, INITIAL_CASH, BUY_COST, SELL_COST, TRADE_START, run_variant,
)

INITIAL_BUY_WON = 1_000_000    # 최초 매수 금액
GRID_ADD_WON = 1_000_000       # 그리드 추가매수 1회 금액
GRID_STEP = 0.97               # -3% 복리 하락마다 추가매수
GRID_TP = 1.03                 # 추가매수 lot은 자기 매입가 +3%에 개별익절
MA_BREAK_DAYS = 3              # 90일선 연속 이탈 일수 → 전량매도

RESULT_DIR = os.path.join("data", "cache_grid_ladder", "results")
os.makedirs(RESULT_DIR, exist_ok=True)


class GridPos:
    """한 종목의 그리드 포지션. lot 리스트로 물량을 개별 관리."""
    __slots__ = ("entry_price", "next_n", "break_days", "lots", "peak_invested")

    def __init__(self, entry_price: float):
        self.entry_price = entry_price     # 최초 매수가 (그리드 기준점)
        self.next_n = 1                    # 다음에 노릴 그리드 단계 n
        self.break_days = 0                # 90일선 연속 이탈 일수
        # lot: dict(buy_price, shares, cost, initial:bool)
        self.lots: list[dict] = []
        self.peak_invested = 0.0           # 이 종목에 동시 투입된 원가의 최대치

    def invested(self) -> float:
        return sum(l["cost"] for l in self.lots)


def compute_yearly(equity: pd.Series) -> dict[int, float]:
    """연도별 수익률 (직전 연말 대비, 첫 해는 초기자본 대비)."""
    ye = equity.resample("YE").last()
    out: dict[int, float] = {}
    prev = float(INITIAL_CASH)
    for ts, val in ye.items():
        out[ts.year] = val / prev - 1.0
        prev = float(val)
    return out


def run_grid(store):
    td = store.trading_days
    close_v = store.close_v
    close_ff = store.close_ff
    ma_sell_v = store.ma_sell_v       # 90일 이동평균
    c2c = store.code_to_col
    start_di = store.start_di

    cash = float(INITIAL_CASH)
    positions: dict[str, GridPos] = {}
    daily_total = np.empty(len(td))

    n_init = n_grid = n_tp = n_stopall = 0   # 거래수 카운터
    wins = 0
    gross_w = gross_l = 0.0
    n_units = 0                               # 실현된 매매단위(lot 청산) 수
    peak_invested_by_code: dict[str, float] = {}   # 종목별 최대 동시투입액

    def realize(pos_ticker, lot, px):
        """lot을 px에 청산 → 현금/승패/손익비 집계."""
        nonlocal cash, wins, gross_w, gross_l, n_units
        proceeds = px * lot["shares"] * (1.0 - SELL_COST)
        pnl = proceeds - lot["cost"]
        cash += proceeds
        n_units += 1
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

        # ---------- 1) 90일선 3일연속 이탈 → 전량매도(안전장치) ----------
        for ticker in list(positions.keys()):
            col = c2c[ticker]
            px = close_v[di, col]
            if not np.isfinite(px) or px <= 0:
                continue   # 데이터 없는 날은 카운터 변화 없음
            pos = positions[ticker]
            ma = ma_sell_v[di, col]
            if np.isfinite(ma) and px < ma:
                pos.break_days += 1
            else:
                pos.break_days = 0
            if pos.break_days >= MA_BREAK_DAYS:
                for lot in pos.lots:
                    realize(ticker, lot, px)
                n_stopall += 1
                del positions[ticker]
                sold_today.add(ticker)

        # ---------- 2) 개별 익절: 추가매수 lot이 자기 매입가 +3% ----------
        for ticker in list(positions.keys()):
            col = c2c[ticker]
            px = close_v[di, col]
            if not np.isfinite(px) or px <= 0:
                continue
            pos = positions[ticker]
            keep: list[dict] = []
            for lot in pos.lots:
                if (not lot["initial"]) and px >= lot["buy_price"] * GRID_TP:
                    realize(ticker, lot, px)
                    n_tp += 1
                else:
                    keep.append(lot)
            pos.lots = keep

        # ---------- 3) 하락 그리드 추가매수 (강세장, 현금≥100만) ----------
        if is_bull:
            for ticker in list(positions.keys()):
                col = c2c[ticker]
                px = close_v[di, col]
                if not np.isfinite(px) or px <= 0:
                    continue
                pos = positions[ticker]
                # 도달한 하락단계들을 순서대로 집행 (하루에 여러 단계 갭하락 가능)
                while cash >= GRID_ADD_WON * (1.0 + BUY_COST):
                    trigger = pos.entry_price * (GRID_STEP ** pos.next_n)
                    if px > trigger:
                        break
                    sh = int(GRID_ADD_WON // px)
                    if sh <= 0:
                        pos.next_n += 1   # 너무 비싼 종목: 단계만 넘기고 skip
                        continue
                    spent = sh * px * (1.0 + BUY_COST)
                    if spent > cash:
                        break
                    cash -= spent
                    pos.lots.append({"buy_price": px, "shares": float(sh),
                                     "cost": spent, "initial": False})
                    pos.next_n += 1
                    n_grid += 1

        # ---------- 4) 신규 최초매수 (강세장, 빈 슬롯) ----------
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
                sh = int(INITIAL_BUY_WON // px)
                if sh <= 0:
                    continue
                spent = sh * px * (1.0 + BUY_COST)
                if spent > cash:
                    continue
                pos = GridPos(entry_price=px)
                pos.lots.append({"buy_price": px, "shares": float(sh),
                                 "cost": spent, "initial": True})
                positions[snap.ticker] = pos
                cash -= spent
                n_init += 1
                free -= 1

        # ---------- 5) 최대 투입액 추적 + 일별 평가 ----------
        hv = 0.0
        for t, pos in positions.items():
            inv = pos.invested()
            if inv > pos.peak_invested:
                pos.peak_invested = inv
            if inv > peak_invested_by_code.get(t, 0.0):
                peak_invested_by_code[t] = inv
            col = c2c[t]
            p = close_v[di, col]
            if not np.isfinite(p):
                p = close_ff[di, col]
            for lot in pos.lots:
                pp = p if np.isfinite(p) else lot["buy_price"]
                hv += pp * lot["shares"]
        daily_total[di] = cash + hv

    # ---------- 성과지표 ----------
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
    win_rate = wins / n_units if n_units else 0.0
    pl = abs(gross_w / gross_l) if gross_l != 0 else float("inf")

    # 종목당 최대 투입액 상위 5
    top_inv = sorted(peak_invested_by_code.items(), key=lambda kv: kv[1], reverse=True)[:5]

    return {"name": "그리드 물타기형", "final": final, "total_ret": total_ret,
            "CAGR": cagr, "MDD": mdd, "Sharpe": sharpe, "Calmar": calmar,
            "PL": pl, "win_rate": win_rate,
            "n_init": n_init, "n_grid": n_grid, "n_tp": n_tp, "n_stopall": n_stopall,
            "n_units": n_units, "top_inv": top_inv,
            "yearly": compute_yearly(dt), "equity": dt}


def main():
    import argparse
    ap = argparse.ArgumentParser(description="그리드 물타기형 백테스트")
    ap.add_argument("--end", type=str, default=None,
                    help="백테스트 종료일(YYYY-MM-DD).")
    args = ap.parse_args()
    end = pd.Timestamp(args.end) if args.end else None
    log("===== 그리드 물타기형 백테스트 ====="
        + (f" (종료일 {end.date()})" if end is not None else ""))
    store = data_layer.get_store(end_date=end)

    grid = run_grid(store)

    # 비교대상: S1 v2 확정판 (+8%50%익절·전량손절) — 검증된 엔진 그대로
    log("[compare] S1 v2 확정판(+8%50%익절·전량손절) 실행 중...")
    s1 = run_variant(store, [(0.08, 0.50)], "S1 v2(+8%50%·전량손절)", 1.0)
    s1["yearly"] = compute_yearly(s1["equity"])

    name_map = store.name_map

    # ---------- 요약 비교표 ----------
    print()
    print("=" * 96)
    print(" 전략 비교  (동일 기간·동일 데이터, 연도별 주식수 반영)")
    print("=" * 96)
    print(f"  {'지표':>10s} {'그리드 물타기형':>16s} {'S1 v2 확정판':>16s}")
    print("  " + "-" * 46)
    def row(lbl, gv, sv):
        print(f"  {lbl:>10s} {gv:>16s} {sv:>16s}")
    row("CAGR", fpct(grid["CAGR"]), fpct(s1["CAGR"]))
    row("MDD", fpct(grid["MDD"]), fpct(s1["MDD"]))
    row("Sharpe", f"{grid['Sharpe']:.2f}", f"{s1['Sharpe']:.2f}")
    row("Calmar", f"{grid['Calmar']:.2f}", f"{s1['Calmar']:.2f}")
    row("손익비", f"{grid['PL']:.2f}", f"{s1['PL']:.2f}")
    row("승률", f"{grid['win_rate']*100:.1f}%", f"{s1['win_rate']*100:.1f}%")
    row("누적수익", fpct(grid["total_ret"]), fpct(s1["total_ret"]))

    # ---------- 거래수 ----------
    print()
    print("=" * 96)
    print(" 그리드 물타기형 거래수")
    print("=" * 96)
    print(f"  최초매수 {grid['n_init']}건 · 추가매수(그리드) {grid['n_grid']}건 · "
          f"개별익절매도 {grid['n_tp']}건 · 90일선3일연속 전량매도 {grid['n_stopall']}건")
    print(f"  (실현 매매단위 {grid['n_units']}건 기준 승률 {grid['win_rate']*100:.1f}%)")

    # ---------- 연도별 수익률표 ----------
    yrs = sorted(set(grid["yearly"]) | set(s1["yearly"]))
    print()
    print("=" * 96)
    print(" 연도별 수익률 (나란히 비교)")
    print("=" * 96)
    print("  " + f"{'전략':>16s}" + "".join(f"{y:>9d}" for y in yrs))
    print("  " + f"{'그리드 물타기형':>16s}"
          + "".join(f"{fpct(grid['yearly'].get(y)) if y in grid['yearly'] else '-':>9s}" for y in yrs))
    print("  " + f"{'S1 v2 확정판':>16s}"
          + "".join(f"{fpct(s1['yearly'].get(y)) if y in s1['yearly'] else '-':>9s}" for y in yrs))

    # ---------- 종목당 최대 투입액 상위 5 ----------
    print()
    print("=" * 96)
    print(" 종목당 최대 동시투입액 상위 5 (그리드가 얼마나 깊게 들어갔나)")
    print("=" * 96)
    for t, amt in grid["top_inv"]:
        nm = name_map.get(t, "")
        print(f"  {nm}({t}) : {amt/1e4:,.0f}만원  (최초100만 기준 {amt/INITIAL_BUY_WON:.1f}배)")

    _save(grid, s1, yrs)


def _save(grid, s1, yrs):
    rows = []
    for r in (grid, s1):
        row = {"name": r["name"], "CAGR": r["CAGR"], "MDD": r["MDD"],
               "Sharpe": r["Sharpe"], "Calmar": r["Calmar"], "PL": r["PL"],
               "win_rate": r["win_rate"], "total_ret": r["total_ret"]}
        for y in yrs:
            row[f"ret_{y}"] = r["yearly"].get(y)
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(RESULT_DIR, "grid_vs_s1.csv"),
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
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(grid["equity"].index, grid["equity"].values, lw=1.6,
                label=f"그리드 물타기형 (CAGR {fpct(grid['CAGR'])})", color="#D9534F")
        ax.plot(s1["equity"].index, s1["equity"].values, lw=1.6,
                label=f"S1 v2 확정판 (CAGR {fpct(s1['CAGR'])})", color="#337AB7")
        ax.set_yscale("log")
        ax.set_title("자산곡선: 그리드 물타기형 vs S1 v2 확정판")
        ax.legend(fontsize=10); ax.grid(alpha=0.3, which="both")
        plt.tight_layout()
        png = os.path.join(RESULT_DIR, "grid_vs_s1.png")
        plt.savefig(png, dpi=140)
        plt.close(fig)
        print(f"\n  그래프 저장 : {png}")
    except Exception as e:
        print(f"\n  (그래프 생략: {e})")
    print(f"  결과 저장 : {RESULT_DIR}")


if __name__ == "__main__":
    main()
