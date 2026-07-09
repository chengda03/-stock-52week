# -*- coding: utf-8 -*-
"""
backtest_s1_final_report.py — S1 확정판(+8% 도달시 50% 부분익절) 성과 리포트
============================================================================
매도규칙 확정: 기존 매수·불타기·시장필터·(90일선/평단가-12% 하드손절) 전부 유지
             + "진입가 +8% 도달시 보유수량의 50% 부분익절" 추가.
계산 엔진은 backtest_s1_partial.run_variant 를 그대로 사용(=이전 비교와 동일 수치).

출력:
  1) 연도별 수익률 표 (전략 · KOSPI · KOSDAQ)
  2) 1억원 투자 시 연도별 연말잔고·누적수익금·누적수익률·연간수익금·연간수익률
  3) 전체기간 지표 (CAGR/MDD/Sharpe/Calmar/손익비/승률/거래수)
  4) 자산곡선 차트(로그스케일, KOSPI·KOSDAQ 동반)
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import data_layer
from backtest_s1_partial import run_variant, INITIAL_CASH
from backtest_roe_eps_event import load_index, fpct, log

RESULT_DIR = os.path.join("data", "cache_s1_final", "results")
os.makedirs(RESULT_DIR, exist_ok=True)


def won(x: float) -> str:
    return f"{x:,.0f}원"


def main():
    log("===== S1 확정판(+8%@50% 부분익절) 성과 리포트 =====")
    store = data_layer.get_store()
    r = run_variant(store, [(0.08, 0.50)], "+8%@50% 부분익절", stop_frac=1.0)
    dt = r["equity"]

    # 벤치마크
    bench = {}
    for nm, idxname in [("KOSPI", "kospi_index"), ("KOSDAQ", "kosdaq_index")]:
        try:
            bench[nm] = load_index(idxname).reindex(dt.index).ffill().bfill()
        except Exception:
            pass

    # -------- 1) 연도별 수익률 (연중 첫 거래일 → 마지막 거래일) --------
    yr_strat = dt.groupby(dt.index.year).apply(lambda g: g.iloc[-1] / g.iloc[0] - 1)
    ytab = {"전략": yr_strat}
    for nm, ser in bench.items():
        ytab[nm] = ser.groupby(ser.index.year).apply(lambda g: g.iloc[-1] / g.iloc[0] - 1)
    ydf = pd.DataFrame(ytab)

    # -------- 2) 1억 투자 연도별 결과 (연말잔고 기준) --------
    year_end = dt.groupby(dt.index.year).last()
    rows = []
    prev = float(INITIAL_CASH)
    rows.append({"연도": "시작 (2020-01)", "연말잔고": INITIAL_CASH,
                 "누적수익금": None, "누적수익률": None,
                 "연간수익금": None, "연간수익률": None})
    for y in year_end.index:
        bal = float(year_end.loc[y])
        cum_profit = bal - INITIAL_CASH
        cum_ret = bal / INITIAL_CASH - 1.0
        yr_profit = bal - prev
        yr_ret = bal / prev - 1.0
        star = "*" if y == year_end.index[-1] else ""
        rows.append({"연도": f"{y}{star}", "연말잔고": bal,
                     "누적수익금": cum_profit, "누적수익률": cum_ret,
                     "연간수익금": yr_profit, "연간수익률": yr_ret})
        prev = bal
    itab = pd.DataFrame(rows)

    _print_report(r, ydf, itab, bench)
    _save(r, ydf, itab, dt, bench)


def _print_report(r, ydf, itab, bench):
    print()
    print("=" * 92)
    print(f" S1 확정판(+8%@50% 부분익절)  ·  {r['equity'].index[0].date()} ~ {r['equity'].index[-1].date()}"
          f"  ·  1억 · 거래비용 반영")
    print("=" * 92)
    print(f"  최종자산 : {won(r['final'])}")
    print(f"  누적수익률: {fpct(r['total_ret'])}    CAGR: {fpct(r['CAGR'])}    MDD: {fpct(r['MDD'])}")
    print(f"  Sharpe   : {r['Sharpe']:.2f}    Calmar: {r['Calmar']:.2f}    손익비: {r['PL']:.2f}    승률: {fpct(r['win_rate'])}")
    print(f"  거래수   : 신규 {r['n_buy']} · 불타기 {r['n_add']} · 부분익절 {r['n_partial']} · 최종매도 {r['n_final']}"
          f"  (승률 분모 = 부분익절+최종매도 = {r['n_sell']})")

    print("\n  [ 1. 연도별 수익률 ]")
    print(ydf.map(lambda v: fpct(v) if pd.notna(v) else "-").to_string())

    print("\n  [ 2. 1억원 투자 시 연도별 결과 ]")
    print(f"  {'연도':>14s} {'연말잔고':>18s} {'누적수익금':>18s} {'누적수익률':>10s} {'연간수익금':>18s} {'연간수익률':>10s}")
    for _, row in itab.iterrows():
        bal = won(row["연말잔고"])
        if row["누적수익금"] is None:
            print(f"  {row['연도']:>14s} {bal:>18s} {'—':>18s} {'—':>10s} {'—':>18s} {'—':>10s}")
        else:
            print(f"  {row['연도']:>14s} {bal:>18s} {('+'+won(row['누적수익금'])):>18s} "
                  f"{fpct(row['누적수익률']):>10s} {('+'+won(row['연간수익금'])):>18s} {fpct(row['연간수익률']):>10s}")

    print("\n  [ 벤치마크 단순보유 누적 ]")
    for nm, ser in bench.items():
        print(f"    {nm:7s} {fpct(ser.iloc[-1]/ser.iloc[0]-1):>10s}")


def _save(r, ydf, itab, dt, bench):
    ydf.to_csv(os.path.join(RESULT_DIR, "yearly_returns.csv"), encoding="utf-8-sig")
    itab.to_csv(os.path.join(RESULT_DIR, "investment_100m.csv"), index=False, encoding="utf-8-sig")
    dt.to_frame("total").to_csv(os.path.join(RESULT_DIR, "equity.csv"), encoding="utf-8-sig")
    summary = {k: r[k] for k in ("final", "total_ret", "CAGR", "MDD", "Sharpe",
                                 "Calmar", "PL", "win_rate", "n_buy", "n_add",
                                 "n_partial", "n_final", "n_sell")}
    pd.Series(summary).to_csv(os.path.join(RESULT_DIR, "summary.csv"), encoding="utf-8-sig")
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
        ax.plot(dt.index, dt.values, label="S1 확정판(+8%@50% 부분익절)", lw=1.8, color="#3182F6")
        for nm, ser in bench.items():
            norm = ser / ser.iloc[0] * INITIAL_CASH
            ax.plot(norm.index, norm.values, label=nm, lw=1.0, ls="--")
        ax.set_yscale("log")
        ax.set_title("S1 확정판 자산곡선 (1억 시작, 로그스케일)")
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
    main()
