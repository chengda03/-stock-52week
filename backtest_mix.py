# -*- coding: utf-8 -*-
"""
backtest_mix.py — S1(챔피언) + MR3(안정형) 혼합 포트폴리오
============================================================================
두 전략은 성격이 정반대다.
  · S1  : ROE×EPS 불타기 → 고CAGR·저승률·고MDD (추세추종형, 큰 승부)
  · MR3 : 실적필터 평균회귀 → 중CAGR·저MDD·고Sharpe (안정형, 낮은 낙폭)
성격이 다르면 낙폭 시점이 어긋나 '분산효과'가 난다. 자금을 나눠 담았을 때
전체 MDD가 얼마나 줄고 위험조정 성과(Sharpe/Calmar)가 얼마나 좋아지는지 검증.

방식(리밸런싱 없는 2계좌 모델):
  전체 1억을 S1에 w, MR3에 (1-w) 비중으로 나눠 각자 굴린다.
  혼합자산(t) = w·S1정규화(t) + (1-w)·MR3정규화(t)   (각 곡선은 1억 기준)
  → 초기 배분만 하고 이후 각 슬리브는 독립적으로 성장(현실적 가정).

S1 곡선  : data/cache_v3/results/equity.csv (STRATEGY_FINAL 재현본)
MR3 곡선 : backtest_mr3.run_mr3_daily 로 최선안 재계산
           (ROE+모멘텀 · 트레일-10% · 시간15 = 최고 Sharpe 1.01, MDD -15.9%)
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd

import backtest_mr3 as mr3
from backtest_roe_eps_event import fpct, log

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

RESULT_DIR = os.path.join("data", "cache_mix", "results")
os.makedirs(RESULT_DIR, exist_ok=True)
S1_EQUITY = os.path.join("data", "cache_v3", "results", "equity.csv")


def metrics(norm: pd.Series) -> dict:
    """1.0에서 시작하는 정규화 자산곡선에 대한 성과지표."""
    final = float(norm.iloc[-1])
    years = (norm.index[-1] - norm.index[0]).days / 365.25
    cagr = final ** (1.0 / max(years, 1e-9)) - 1.0
    peak = norm.cummax()
    mdd = float(((norm - peak) / peak).min())
    rets = norm.pct_change().fillna(0)
    sharpe = float((rets.mean() / (rets.std() + 1e-12)) * math.sqrt(252))
    calmar = cagr / abs(mdd) if mdd != 0 else float("inf")
    return {"total_ret": final - 1.0, "CAGR": cagr, "MDD": mdd,
            "Sharpe": sharpe, "Calmar": calmar}


def main():
    log("===== S1 + MR3 혼합 포트폴리오 =====")

    # S1 자산곡선
    s1 = pd.read_csv(S1_EQUITY, index_col=0, parse_dates=True)["total"]
    s1.index = s1.index.normalize()

    # MR3 최선안 자산곡선 재계산
    D = mr3.MR3Data()
    em = D.elig_for("ROE+MOM")
    mr3_eq = mr3.run_mr3_daily(D, em, trail=0.10, time_stop=15)
    mr3_eq = mr3_eq[mr3_eq.index >= mr3.TRADE_START]
    mr3_eq.index = mr3_eq.index.normalize()

    # 공통 날짜 정렬 + 정규화(시작=1.0)
    idx = s1.index.intersection(mr3_eq.index)
    s1n = (s1.reindex(idx) / s1.reindex(idx).iloc[0])
    m3n = (mr3_eq.reindex(idx) / mr3_eq.reindex(idx).iloc[0])
    log(f"공통 구간 {idx[0].date()} ~ {idx[-1].date()} ({len(idx)}일)")

    # 일간수익률 상관 (분산효과 근거)
    corr = float(s1n.pct_change().fillna(0).corr(m3n.pct_change().fillna(0)))
    log(f"S1 · MR3 일간수익률 상관계수: {corr:.3f}  (낮을수록 분산효과 큼)")

    # 개별 + 혼합
    print()
    print("=" * 96)
    print(" 자금배분별 성과 (전체 1억, 리밸런싱 없음)")
    print("=" * 96)
    print(f"  {'S1:MR3 비중':>14s} {'총수익률':>10s} {'CAGR':>8s} {'MDD':>8s} "
          f"{'Sharpe':>7s} {'Calmar':>7s} {'최종자산(1억→)':>14s}")

    rows = []
    for w in [1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.0]:
        blended = w * s1n + (1.0 - w) * m3n
        mt = metrics(blended)
        mt["w_s1"] = w
        mt["final_won"] = blended.iloc[-1] * 1e8
        mt["curve"] = blended
        rows.append(mt)
        tag = f"{int(w*100)}:{int((1-w)*100)}"
        if w == 1.0:
            tag += " (S1)"
        elif w == 0.0:
            tag += " (MR3)"
        print(f"  {tag:>14s} {fpct(mt['total_ret']):>10s} {fpct(mt['CAGR']):>8s} "
              f"{fpct(mt['MDD']):>8s} {mt['Sharpe']:>7.2f} {mt['Calmar']:>7.2f} "
              f"{mt['final_won']/1e8:>12.2f}억")

    _verdict(rows)
    _save(rows, idx)


def _verdict(rows):
    print()
    print("=" * 96)
    print(" 종합 판정")
    print("=" * 96)
    s1_only = next(r for r in rows if r["w_s1"] == 1.0)
    mr3_only = next(r for r in rows if r["w_s1"] == 0.0)
    best_sharpe = max(rows, key=lambda x: x["Sharpe"])
    best_calmar = max(rows, key=lambda x: x["Calmar"])
    print(f"  · S1 단독 : CAGR {fpct(s1_only['CAGR'])}, MDD {fpct(s1_only['MDD'])}, Sharpe {s1_only['Sharpe']:.2f}, Calmar {s1_only['Calmar']:.2f}")
    print(f"  · MR3 단독: CAGR {fpct(mr3_only['CAGR'])}, MDD {fpct(mr3_only['MDD'])}, Sharpe {mr3_only['Sharpe']:.2f}, Calmar {mr3_only['Calmar']:.2f}")
    print(f"  · 최고 Sharpe 혼합: S1 {int(best_sharpe['w_s1']*100)}% → Sharpe {best_sharpe['Sharpe']:.2f} "
          f"(CAGR {fpct(best_sharpe['CAGR'])}, MDD {fpct(best_sharpe['MDD'])})")
    print(f"  · 최고 Calmar 혼합: S1 {int(best_calmar['w_s1']*100)}% → Calmar {best_calmar['Calmar']:.2f} "
          f"(CAGR {fpct(best_calmar['CAGR'])}, MDD {fpct(best_calmar['MDD'])})")
    # 예: 70:30이 S1단독 대비 MDD를 얼마나 줄이는가
    w70 = next(r for r in rows if abs(r["w_s1"] - 0.7) < 1e-9)
    print(f"\n  [예시] S1 70 : MR3 30")
    print(f"    CAGR {fpct(s1_only['CAGR'])} → {fpct(w70['CAGR'])}  (수익은 {fpct(w70['CAGR']-s1_only['CAGR'])}p 변화)")
    print(f"    MDD  {fpct(s1_only['MDD'])} → {fpct(w70['MDD'])}  (낙폭 {(1-w70['MDD']/s1_only['MDD'])*100:.0f}% 완화)")
    print(f"    Sharpe {s1_only['Sharpe']:.2f} → {w70['Sharpe']:.2f}")


def _save(rows, idx):
    tbl = pd.DataFrame([{k: r[k] for k in ("w_s1", "total_ret", "CAGR", "MDD",
                                           "Sharpe", "Calmar", "final_won")} for r in rows])
    tbl.to_csv(os.path.join(RESULT_DIR, "mix_results.csv"), index=False, encoding="utf-8-sig")
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
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 11), height_ratios=[2, 1])
        for r in rows:
            w = r["w_s1"]
            if w in (1.0, 0.7, 0.5, 0.0):
                lab = {1.0: "S1 100%", 0.7: "S1 70:MR3 30",
                       0.5: "S1 50:MR3 50", 0.0: "MR3 100%"}[w]
                ax1.plot(idx, r["curve"].values * 1e8 / 1e8, label=lab, lw=1.5)
        ax1.set_yscale("log")
        ax1.set_ylabel("자산 (1억 기준, 로그축)")
        ax1.set_title("S1+MR3 혼합 자산곡선")
        ax1.legend(); ax1.grid(alpha=0.3)

        ws = [r["w_s1"] * 100 for r in rows]
        ax2.plot(ws, [r["CAGR"] * 100 for r in rows], "o-", color="#E94545", label="CAGR(%)")
        ax2b = ax2.twinx()
        ax2b.plot(ws, [-r["MDD"] * 100 for r in rows], "s--", color="#3182F6", label="MDD 크기(%)")
        ax2.set_xlabel("S1 비중 (%)  ← 오른쪽=공격적 / 왼쪽=안정적")
        ax2.set_ylabel("CAGR (%)", color="#E94545")
        ax2b.set_ylabel("MDD 크기 (%)", color="#3182F6")
        ax2.set_title("S1 비중에 따른 수익(CAGR) vs 낙폭(MDD)")
        ax2.grid(alpha=0.3)
        plt.tight_layout()
        png = os.path.join(RESULT_DIR, "mix_curves.png")
        plt.savefig(png, dpi=140)
        plt.close(fig)
        print(f"\n  그래프 저장 : {png}")
    except Exception as e:
        print(f"\n  (그래프 생략: {e})")
    print(f"  결과 저장 : {RESULT_DIR}")


if __name__ == "__main__":
    main()
