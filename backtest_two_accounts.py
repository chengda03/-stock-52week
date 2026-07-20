# -*- coding: utf-8 -*-
"""
backtest_two_accounts.py — S1 v2 를 두 개의 독립 계좌로 동시 운용
============================================================================
strategy_core.py / backtest_s1_partial.py / backtest_s1_capwithdraw.py 는
수정하지 않고 import만 합니다. (엔진은 backtest_s1_capwithdraw.run_cap 재사용)

[계좌 A - 순수 성장전용]  1억. 매수6조건+불타기 무제한+90일선/-12% 전량손절.
  부분익절 없음, 비중상한 없음, 인출 없음. → run_cap(cap=1.0, 인출0, 고정슬롯)
  (cap=1.0이면 단일종목이 총자산을 넘을 수 없어 트리밍이 절대 발생하지 않음
   = 부분익절을 뺀 순수 S1 성장형과 동일)

[계좌 B - 인출전용]  1억. A와 동일 로직 + 비중상한 50%(구 총자산기준)
  초과분 매도 → 그 대금의 30%만 인출, 70% 재투입. → run_cap(cap=0.5,
  denom_incl_withdrawn=True, withdraw_frac=0.3, proportional=True)

두 계좌는 각자 1억·독립 포지션. 자금 이동 없음. 합산은 2억 기준으로 평가.

[참고]  기존 S1 v2(+8%@50%익절 있는 버전)도 같이 표기 → 계좌A가 더 나은지 확인.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

import data_layer
from backtest_roe_eps_event import fpct, log
from backtest_s1_capwithdraw import run_cap, compute_yearly, INITIAL_CASH, TRADE_START
from backtest_s1_partial import run_variant

RESULT_DIR = os.path.join("data", "cache_two_accounts", "results")
os.makedirs(RESULT_DIR, exist_ok=True)


def combined_metrics(store, a, b):
    """두 계좌 합산(2억) 성과 + 합산 최대 단일종목 비중."""
    td = a["td"]
    mask = td >= TRADE_START
    totA = a["daily_total_full"]
    totB = b["daily_total_full"]
    comb = (totA + totB)[mask]
    idx = td[mask]
    s = pd.Series(comb, index=idx)
    init = 2 * INITIAL_CASH
    final = float(s.iloc[-1])
    years = (idx[-1] - idx[0]).days / 365.25
    cagr = (final / init) ** (1.0 / max(years, 1e-9)) - 1.0
    peak = s.cummax()
    mdd = float(((s - peak) / peak).min())
    rets = s.pct_change().fillna(0)
    sharpe = float((rets.mean() / (rets.std() + 1e-12)) * math.sqrt(252))
    calmar = cagr / abs(mdd) if mdd != 0 else float("inf")

    # 합산 최대 단일종목 비중: (A평가 + B평가) / (A총자산 + B총자산)
    pm = a["posmat"] + b["posmat"]           # (T, N) 종목별 합산 평가액
    denom = (totA + totB)                     # (T,) 합산 총자산
    max_w = 0.0
    rows = np.where(mask)[0]
    for di in rows:
        d = denom[di]
        if d > 0:
            w = pm[di].max() / d
            if w > max_w:
                max_w = w
    return {"equity": s, "final": final, "CAGR": cagr, "MDD": mdd,
            "Sharpe": sharpe, "Calmar": calmar, "max_weight": max_w, "init": init}


def main():
    import argparse
    ap = argparse.ArgumentParser(description="S1 두 계좌 동시 운용")
    ap.add_argument("--end", type=str, default=None, help="종료일(YYYY-MM-DD).")
    args = ap.parse_args()
    end = pd.Timestamp(args.end) if args.end else None
    log("===== S1 두 계좌 동시 백테스트 ====="
        + (f" (종료일 {end.date()})" if end is not None else ""))
    store = data_layer.get_store(end_date=end)

    # 계좌 A: 순수 성장 (부분익절 X, 상한 X, 인출 X, 고정 500만 슬롯)
    log("[A] 순수 성장전용 실행...")
    A = run_cap(store, cap=1.0, denom_incl_withdrawn=False, withdraw_frac=0.0,
                proportional=False, track_posmat=True)
    A["yearly"] = compute_yearly(A["equity"])

    # 계좌 B: 인출전용 (상한50% 총자산기준, 인출30%, 자본비례 슬롯)
    log("[B] 인출전용 실행...")
    B = run_cap(store, cap=0.50, denom_incl_withdrawn=True, withdraw_frac=0.30,
                proportional=True, track_posmat=True)
    B["yearly"] = compute_yearly(B["equity"])

    # 참고: 기존 S1 v2 (+8%@50%익절)
    log("[ref] 기존 S1 v2(+8%50%익절) 실행...")
    ref = run_variant(store, [(0.08, 0.50)], "기존S1v2(+8%50%익절)", 1.0)
    ref["yearly"] = compute_yearly(ref["equity"])

    comb = combined_metrics(store, A, B)

    W = 96
    # ---------- 계좌 A ----------
    print()
    print("=" * W)
    print(" [계좌 A] 순수 성장전용 (1억, 부분익절 제거·상한없음·인출없음)")
    print("=" * W)
    print(f"  CAGR {fpct(A['CAGR'])} · MDD {fpct(A['MDD'])} · Sharpe {A['Sharpe']:.2f} · "
          f"Calmar {A['Calmar']:.2f} · 손익비 {A['PL']:.2f} · 승률 {A['win_rate']*100:.1f}%")
    print(f"  최대 단일종목 비중 {A['max_weight']*100:.1f}% · 최종자산 {A['final']/1e8:.2f}억")

    # ---------- 계좌 B ----------
    print()
    print("=" * W)
    print(" [계좌 B] 인출전용 (1억, 상한50%·총자산기준·인출30%)")
    print("=" * W)
    print(f"  CAGR {fpct(B['CAGR'])} · MDD {fpct(B['MDD'])} · Sharpe {B['Sharpe']:.2f} · "
          f"Calmar {B['Calmar']:.2f} · 손익비 {B['PL']:.2f} · 승률 {B['win_rate']*100:.1f}%")
    print(f"  최종 운용자산 {B['op_final']/1e8:.2f}억 · 누적 인출가능금 {B['withdrawn']/1e8:.2f}억 · "
          f"최대 단일종목 비중 {B['max_weight']*100:.1f}%")

    # ---------- 합산 & 참고 비교표 ----------
    print()
    print("=" * W)
    print(" 종합 비교표")
    print("=" * W)
    print(f"  {'항목':>20s} {'CAGR':>8s} {'MDD':>8s} {'Sharpe':>7s} {'Calmar':>7s} "
          f"{'손익비':>6s} {'승률':>6s} {'최대비중':>7s}")
    def prow(name, r):
        pl = r.get("PL", float('nan')); wr = r.get("win_rate", float('nan'))
        mw = r.get("max_weight")
        mw_s = f"{mw*100:>6.1f}%" if mw is not None else f"{'-':>7s}"
        print(f"  {name:>20s} {fpct(r['CAGR']):>8s} {fpct(r['MDD']):>8s} "
              f"{r['Sharpe']:>7.2f} {r['Calmar']:>7.2f} {pl:>6.2f} {wr*100:>5.1f}% {mw_s}")
    prow("계좌A(성장)", A)
    prow("계좌B(인출)", B)
    print(f"  {'A+B 합산(2억)':>20s} {fpct(comb['CAGR']):>8s} {fpct(comb['MDD']):>8s} "
          f"{comb['Sharpe']:>7.2f} {comb['Calmar']:>7.2f} {'-':>6s} {'-':>5s} "
          f"{comb['max_weight']*100:>6.1f}%")
    prow("[참고]기존S1v2(익절)", ref)

    # ---------- 연도별 신규 인출액 (계좌 B) ----------
    yrs = sorted(B["wd_year"].keys())
    print()
    print("=" * W)
    print(" [계좌 B] 연도별 신규 인출액 / 인출누적 (단위:억)")
    print("=" * W)
    print("  " + f"{'':>8s}" + "".join(f"{y:>13d}" for y in yrs))
    row = "  " + f"{'신규/누적':>8s}"
    for y in yrs:
        info = B["wd_year"][y]
        row += f"{info['added']/1e8:>6.2f}/{info['balance']/1e8:<6.2f}"
    print(row)

    # ---------- 연도별 수익률 ----------
    yrs2 = sorted(set(A["yearly"]) | set(B["yearly"]) | set(ref["yearly"]) | set(comb_yearly(comb)))
    print()
    print("=" * W)
    print(" 연도별 수익률")
    print("=" * W)
    print("  " + f"{'전략':>20s}" + "".join(f"{y:>9d}" for y in yrs2))
    def yrow(name, yd):
        return "  " + f"{name:>20s}" + "".join(
            f"{fpct(yd.get(y)) if y in yd else '-':>9s}" for y in yrs2)
    print(yrow("계좌A(성장)", A["yearly"]))
    print(yrow("계좌B(인출)", B["yearly"]))
    print(yrow("A+B 합산(2억)", comb_yearly(comb)))
    print(yrow("[참고]기존S1v2(익절)", ref["yearly"]))

    _save(A, B, comb, ref, yrs2)


def comb_yearly(comb):
    """합산 계좌 연도별 수익률 (첫 해 기준을 2억으로)."""
    ye = comb["equity"].resample("YE").last()
    out = {}
    prev = float(comb["init"])   # 2억
    for ts, val in ye.items():
        out[ts.year] = val / prev - 1.0
        prev = float(val)
    return out


def _save(A, B, comb, ref, yrs):
    rows = []
    for name, r, extra in [("A_growth", A, {}), ("B_withdraw", B, {}),
                           ("AB_combined", comb, {}), ("ref_S1v2", ref, {})]:
        yd = r["yearly"] if "yearly" in r else comb_yearly(r)
        row = {"name": name, "CAGR": r["CAGR"], "MDD": r["MDD"],
               "Sharpe": r["Sharpe"], "Calmar": r["Calmar"],
               "PL": r.get("PL"), "win_rate": r.get("win_rate"),
               "max_weight": r.get("max_weight"),
               "op_final": r.get("op_final"), "withdrawn": r.get("withdrawn"),
               "final": r["final"]}
        for y in yrs:
            row[f"ret_{y}"] = yd.get(y)
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(RESULT_DIR, "two_accounts.csv"),
                              index=False, encoding="utf-8-sig")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for fnt in ["Malgun Gothic", "NanumGothic", "AppleGothic", "Gulim"]:
            try:
                matplotlib.rcParams["font.family"] = fnt
                matplotlib.rcParams["axes.unicode_minus"] = False
                break
            except Exception:
                continue
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        ax1.plot(A["equity"].index, A["equity"].values / 1e8, lw=1.5,
                 label=f"계좌A 성장 ({fpct(A['CAGR'])})", color="#D9534F")
        ax1.plot(B["equity"].index, B["equity"].values / 1e8, lw=1.5,
                 label=f"계좌B 인출 ({fpct(B['CAGR'])})", color="#5CB85C")
        ax1.plot(comb["equity"].index, comb["equity"].values / 1e8, lw=1.8,
                 label=f"A+B 합산 ({fpct(comb['CAGR'])})", color="#333333")
        ax1.plot(ref["equity"].index, ref["equity"].values / 1e8, lw=1.1, ls="--",
                 label=f"[참고]기존S1v2 ({fpct(ref['CAGR'])})", color="#337AB7")
        ax1.set_yscale("log"); ax1.set_title("자산곡선 (억원)")
        ax1.legend(fontsize=9); ax1.grid(alpha=0.3, which="both")
        wd = pd.Series({pd.Timestamp(f"{y}-12-31"): v["balance"]
                        for y, v in B["wd_year"].items()})
        ax2.bar([str(y) for y in wd.index.year], wd.values / 1e8, color="#5CB85C")
        ax2.set_title("계좌B 누적 인출가능금 (억원)")
        ax2.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        png = os.path.join(RESULT_DIR, "two_accounts.png")
        plt.savefig(png, dpi=140); plt.close(fig)
        print(f"\n  그래프 저장 : {png}")
    except Exception as e:
        print(f"\n  (그래프 생략: {e})")
    print(f"  결과 저장 : {RESULT_DIR}")


if __name__ == "__main__":
    main()
