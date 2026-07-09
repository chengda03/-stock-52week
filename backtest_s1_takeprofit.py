# -*- coding: utf-8 -*-
"""
S1 + 익절(Take-Profit) 규칙 6종 비교
============================================================================
매수조건은 S1 그대로. 매도조건에 익절 규칙만 OR 로 추가.
기존 매도: 90일선 이탈 + 평단가 -12% 하드손절 (유지)

변형:
  S1     : 익절 없음 (기준)
  A +8%(진입가)  B +12%(진입가)  C +15%(진입가)  D +20%(진입가)
  E +10%(평단가)  F +15%(평단가)

출력: CAGR / MDD / 매매단위 승률 / 손익비 / 연도별 수익률 / '연 20%+ 달성 비율'
목표: 승률 ≥ 70% & CAGR ≥ 20% 조합 탐색.
"""
import os
import sys

import numpy as np
import pandas as pd

import backtest_roe_eps_variants as bt
from backtest_roe_eps_variants import Cfg, build_pre, run_config
from backtest_roe_eps_event import load_index, fpct, log

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ks_full = load_index("kospi_index")
bt.END_DATE = pd.Timestamp(ks_full.index.max()).normalize()

RESULT_DIR = os.path.join("data", "cache_s1_takeprofit", "results")
os.makedirs(RESULT_DIR, exist_ok=True)

TV30 = 3_000_000_000
RC_B, RC_S = 0.00115, 0.00295


def s1(name, **kw):
    base = dict(mom=20, ma_buy=60, ma_sell=90, trail_stop=0.99,
                roe_min_pct=15.0, select_by_mom=True, min_tv=TV30,
                buy_cost=RC_B, sell_cost=RC_S, mkt_ks200=True, mkt_block_pyr=True,
                universe_top=500, n_slots=20, max_adds_day=1, hard_stop_cost=0.12)
    base.update(kw)
    return Cfg(name, **base)


def yearly_returns(equity):
    return equity.groupby(equity.index.year).apply(lambda g: g.iloc[-1] / g.iloc[0] - 1.0)


def main():
    log("===== S1 + 익절 규칙 6종 비교 =====")
    targets = [
        ("S1 (익절없음)",       s1("S1")),
        ("A 진입가+8%",         s1("A", take_profit_entry=0.08)),
        ("B 진입가+12%",        s1("B", take_profit_entry=0.12)),
        ("C 진입가+15%",        s1("C", take_profit_entry=0.15)),
        ("D 진입가+20%",        s1("D", take_profit_entry=0.20)),
        ("E 평단가+10%",        s1("E", take_profit_avg=0.10)),
        ("F 평단가+15%",        s1("F", take_profit_avg=0.15)),
    ]
    bt.CONFIGS = [c for _, c in targets]
    pre = build_pre()

    results = []
    for label, c in targets:
        r = run_config(c, pre)
        yr = yearly_returns(r["equity"])
        n_years = len(yr)
        pct20 = float((yr >= 0.20).sum()) / n_years if n_years else 0.0
        results.append((label, r, yr, pct20))
        log(f"  {label:16s} CAGR {fpct(r['CAGR']):>8s}  MDD {fpct(r['MDD']):>8s}  "
            f"승률 {r['win_rate']*100:5.1f}%  손익비 {r['PL']:5.2f}  "
            f"20%+해비율 {pct20*100:4.0f}%  매도 {r['n_sell']}")

    idx = results[0][1]["equity"].index
    all_years = sorted(set().union(*[set(y.index) for _, _, y, _ in results]))

    # ---- 비교표 ----
    print()
    print("=" * 104)
    print(f" S1 + 익절 규칙 비교  ·  {idx[0].date()} ~ {idx[-1].date()}  ·  1억 · 거래비용 반영")
    print("=" * 104)
    print(f"  {'변형':>16s} {'CAGR':>8s} {'MDD':>8s} {'승률':>7s} {'손익비':>6s} "
          f"{'20%+해비율':>9s} {'매도수':>6s}")
    rows = []
    for label, r, yr, pct20 in results:
        print(f"  {label:>16s} {fpct(r['CAGR']):>8s} {fpct(r['MDD']):>8s} "
              f"{r['win_rate']*100:>6.1f}% {r['PL']:>6.2f} {pct20*100:>8.0f}% {r['n_sell']:>6d}")
        rows.append({"변형": label, "CAGR": fpct(r["CAGR"]), "MDD": fpct(r["MDD"]),
                     "승률": f"{r['win_rate']*100:.1f}%", "손익비": f"{r['PL']:.2f}",
                     "20%+해비율": f"{pct20*100:.0f}%", "매도수": r["n_sell"],
                     "누적수익률": fpct(r["total_ret"]), "Sharpe": f"{r['Sharpe']:.2f}"})

    # ---- 연도별 수익률 ----
    print("\n  [ 연도별 수익률 ]")
    header = "   연도 " + "".join(f"{lbl.split()[0]:>10s}" for lbl, _, _, _ in results)
    print(header)
    for y in all_years:
        line = f"  {y} "
        for _, _, yr, _ in results:
            v = yr.get(y, np.nan)
            line += f"{fpct(v):>10s}"
        print(line)

    # ---- 목표 판정 ----
    print("\n  [ 목표 판정: 승률 ≥ 70% & CAGR ≥ 20% ]")
    hit = [(lbl, r) for lbl, r, _, _ in results if r["win_rate"] >= 0.70 and r["CAGR"] >= 0.20]
    if hit:
        for lbl, r in hit:
            print(f"    ✅ {lbl}: 승률 {r['win_rate']*100:.1f}% · CAGR {fpct(r['CAGR'])}")
    else:
        print("    ❌ 승률 70% & CAGR 20% 동시 충족 조합 없음.")
        # 승률 70% 근처에서 CAGR 최고
        near_win = max(results, key=lambda t: (t[1]["win_rate"], t[1]["CAGR"]))
        best_win_cagr = max([t for t in results if t[1]["win_rate"] >= 0.60] or results,
                            key=lambda t: t[1]["CAGR"])
        # CAGR 20% 근처(이상)에서 승률 최고
        cagr_ok = [t for t in results if t[1]["CAGR"] >= 0.20]
        best_cagr_win = max(cagr_ok or results, key=lambda t: t[1]["win_rate"])
        print(f"    · 최고 승률 변형        : {near_win[0]} "
              f"(승률 {near_win[1]['win_rate']*100:.1f}%, CAGR {fpct(near_win[1]['CAGR'])})")
        print(f"    · 승률↑ 그룹 중 CAGR 최고: {best_win_cagr[0]} "
              f"(승률 {best_win_cagr[1]['win_rate']*100:.1f}%, CAGR {fpct(best_win_cagr[1]['CAGR'])})")
        print(f"    · CAGR≥20% 중 승률 최고  : {best_cagr_win[0]} "
              f"(승률 {best_cagr_win[1]['win_rate']*100:.1f}%, CAGR {fpct(best_cagr_win[1]['CAGR'])})")

    pd.DataFrame(rows).to_csv(os.path.join(RESULT_DIR, "summary.csv"),
                              index=False, encoding="utf-8-sig")
    print(f"\n  결과 저장 : {RESULT_DIR}")


if __name__ == "__main__":
    main()
