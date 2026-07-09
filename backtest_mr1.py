# -*- coding: utf-8 -*-
"""
backtest_mr1.py — "MR1: 단기 스윙 평균회귀" 전략 (신규, S1과 완전 별개)
============================================================================
S1(ROE×EPS 불타기)은 추세추종+불타기라 '작게 자주 잃고 크게 번다'(승률 낮음).
MR1은 정반대 컨셉: 상승추세 종목의 '단기 눌림(과매도)'을 사서 작은 목표수익에
빠르게 파는 평균회귀 스윙 → '높은 승률'을 노린다. 불타기 없음.

목표: 매매단위 승률 ≥ 70% & CAGR ≥ 20% 동시 만족 조합 탐색.
      없으면 승률-CAGR 트레이드오프 곡선(산점도)으로 확인.

데이터: data_layer.DataStore 재사용 (시총상위500·관리종목/스팩/우선주/리츠 제외,
        가격/거래대금/관리종목 필터 그대로). 지표(120MA/RSI14/5MA)는 여기서 계산.

매수조건 (모두 충족)
  1) 현재가 > 120일선            (장기 상승추세)
  2) 과매도(아래 중 택1, 각각 테스트)
       RSI  : RSI(14) < 30
       DEV3/5/7 : 현재가 < 5일선 × (1 - X),  X ∈ {3%,5%,7%}
  3) 20일평균거래대금 ≥ 10억     (S1의 30억보다 완화)

매도조건 (셋 중 먼저 닿는 것)
  1) 익절   : 진입가 × (1+Y),  Y ∈ {5%,8%,12%}
  2) 시간손절: 매수 후 N거래일 경과,  N ∈ {5,10,15}
  3) 손절   : 진입가 × (1-Z),  Z ∈ {5%,7%}

운영: 1억원, 슬롯당 = 1억/슬롯수, 정수주 매수, 불타기 없음.
      후보가 슬롯보다 많으면 '가장 과매도된' 순서로 채움.
거래비용: 매수 0.115% / 매도 0.295% (S1과 동일, 비교 가능하게)
기간: 2020-01-01 ~ 최신
"""
from __future__ import annotations

import itertools
import math
import os
import sys

import numpy as np
import pandas as pd

import data_layer
from backtest_roe_eps_event import fpct, log

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

RESULT_DIR = os.path.join("data", "cache_mr1", "results")
os.makedirs(RESULT_DIR, exist_ok=True)

INITIAL_CASH = 100_000_000
BUY_COST = 0.00115
SELL_COST = 0.00295
LIQ_MIN = 1_000_000_000          # 10억
TRADE_START = pd.Timestamp("2020-01-01")


# ---------------------------------------------------------------------------
# 지표 사전계산 (한 번만)
# ---------------------------------------------------------------------------
class MR1Data:
    def __init__(self):
        store = data_layer.get_store()
        self.store = store
        td = store.trading_days
        self.td = td
        self.close_v = store.close_v
        self.close_ff = store.close_ff
        self.name_map = store.name_map
        self.valid_codes = store.valid_codes
        T, N = self.close_v.shape
        self.T, self.N = T, N
        self.start_di = store.start_di

        close = pd.DataFrame(self.close_v, index=td, columns=store.valid_codes)

        # 이동평균
        ma120 = close.rolling(120, min_periods=120).mean().to_numpy(dtype=float)
        ma5 = close.rolling(5, min_periods=5).mean().to_numpy(dtype=float)

        # RSI(14) — Wilder 방식
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        rsi = (100.0 - 100.0 / (1.0 + rs)).to_numpy(dtype=float)

        tv = store.tv_v  # 20일 평균거래대금 (data_layer가 이미 계산)

        # 5일선 대비 하락률 (양수 = 5일선 아래)
        dev = (ma5 - self.close_v) / ma5

        # 공통 조건
        trend_ok = self.close_v > ma120
        liq_ok = np.isfinite(tv) & (tv >= LIQ_MIN)
        common = trend_ok & liq_ok

        # 관리종목 제외 포함한 시총 상위 500 마스크 (T,N)
        in_top = self._build_top500_mask()

        base = common & in_top

        # 과매도 방식별 최종 매수후보 마스크 + 랭킹점수(오름차순=더 과매도)
        self.methods = {}
        # RSI: 낮을수록 과매도 → score=rsi
        self.methods["RSI"] = (base & np.isfinite(rsi) & (rsi < 30.0), rsi)
        # DEV: 5일선 아래로 X% 이상 → score=-dev (클수록 과매도 → 오름차순 위해 음수)
        for x, tag in [(0.03, "DEV3"), (0.05, "DEV5"), (0.07, "DEV7")]:
            mask = base & np.isfinite(dev) & (dev >= x)
            self.methods[tag] = (mask, -dev)

    def _build_top500_mask(self):
        store = self.store
        mc = store.marcap_v.copy()
        # 관리종목: 지정일 이후는 제외
        desig = store.admin_desig_v  # (N,) datetime64, NaT=비관리
        isnat = np.isnat(desig)
        td64 = self.td.values.astype("datetime64[ns]")
        # (T,N) 관리 활성 마스크
        admin_active = (~isnat)[None, :] & (desig[None, :] <= td64[:, None])
        mc[~np.isfinite(mc)] = -1.0
        mc[mc <= 0] = -1.0
        mc[admin_active] = -1.0
        in_top = np.zeros((self.T, self.N), dtype=bool)
        for di in range(self.T):
            row = mc[di]
            vidx = np.where(row > 0)[0]
            if len(vidx) > 500:
                sel = vidx[np.argpartition(-row[vidx], 500)[:500]]
            else:
                sel = vidx
            in_top[di, sel] = True
        return in_top


# ---------------------------------------------------------------------------
# 백테스트 1회
# ---------------------------------------------------------------------------
def run_mr1(D: MR1Data, method: str, tp: float, time_stop: int, stop: float,
            slots: int) -> dict:
    elig_mask, score = D.methods[method]
    close_v = D.close_v
    close_ff = D.close_ff
    slot_amt = INITIAL_CASH / slots

    cash = float(INITIAL_CASH)
    port: dict[int, dict] = {}   # col -> {shares, entry, cost, buy_di}
    daily = np.empty(D.T)
    n_buy = n_sell = wins = 0
    gross_w = gross_l = 0.0
    hold_sum = 0

    for di in range(D.T):
        if di < D.start_di:
            daily[di] = INITIAL_CASH
            continue
        row = close_v[di]

        # 1) 매도 (익절 → 손절 → 시간손절)
        for col in list(port.keys()):
            px = row[col]
            if not np.isfinite(px):
                continue
            pos = port[col]
            reason = None
            if px >= pos["entry"] * (1.0 + tp):
                reason = "TP"
            elif px <= pos["entry"] * (1.0 - stop):
                reason = "SL"
            elif (di - pos["buy_di"]) >= time_stop:
                reason = "TIME"
            if reason is not None:
                proceeds = px * pos["shares"] * (1.0 - SELL_COST)
                pnl = proceeds - pos["cost"]
                cash += proceeds
                n_sell += 1
                hold_sum += di - pos["buy_di"]
                if pnl > 0:
                    wins += 1; gross_w += pnl
                else:
                    gross_l += pnl
                del port[col]

        # 2) 신규매수 (빈 슬롯, 가장 과매도된 순)
        free = slots - len(port)
        if free > 0:
            elig = elig_mask[di].copy()
            for col in port:
                elig[col] = False
            cand = np.where(elig)[0]
            if len(cand) > 0:
                order = cand[np.argsort(score[di][cand])]  # 오름차순=더 과매도
                for col in order:
                    if free <= 0:
                        break
                    p = row[col]
                    if not np.isfinite(p) or p <= 0:
                        continue
                    sh = int(slot_amt // p)
                    if sh <= 0:
                        continue
                    spent = sh * p * (1.0 + BUY_COST)
                    if spent > cash:
                        continue
                    port[col] = {"shares": sh, "entry": float(p),
                                 "cost": spent, "buy_di": di}
                    cash -= spent
                    n_buy += 1
                    free -= 1

        # 3) 평가
        hv = 0.0
        for col, pos in port.items():
            p = row[col]
            if not np.isfinite(p):
                p = close_ff[di, col]
                if not np.isfinite(p):
                    p = pos["entry"]
            hv += p * pos["shares"]
        daily[di] = cash + hv

    # 성과지표
    dt = pd.Series(daily, index=D.td)
    dt = dt[dt.index >= TRADE_START]
    final = float(dt.iloc[-1])
    years = (dt.index[-1] - dt.index[0]).days / 365.25
    cagr = (final / INITIAL_CASH) ** (1.0 / max(years, 1e-9)) - 1.0
    peak = dt.cummax()
    mdd = float(((dt - peak) / peak).min())
    rets = dt.pct_change().fillna(0)
    sharpe = float((rets.mean() / (rets.std() + 1e-12)) * math.sqrt(252))
    win_rate = wins / n_sell if n_sell else 0.0
    pl = abs(gross_w / gross_l) if gross_l != 0 else float("inf")
    avg_hold = hold_sum / n_sell if n_sell else 0.0

    return {"method": method, "tp": tp, "time": time_stop, "stop": stop,
            "slots": slots, "CAGR": cagr, "MDD": mdd, "win_rate": win_rate,
            "PL": pl, "Sharpe": sharpe, "total_ret": final / INITIAL_CASH - 1.0,
            "n_buy": n_buy, "n_sell": n_sell, "avg_hold": avg_hold, "final": final}


def label(r) -> str:
    return f"{r['method']}·익절{r['tp']*100:.0f}%·시간{r['time']}·손절{r['stop']*100:.0f}%·슬롯{r['slots']}"


def main():
    log("===== MR1 단기 스윙 평균회귀 백테스트 =====")
    D = MR1Data()
    log(f"거래일 {D.td[0].date()}~{D.td[-1].date()} · 종목 {D.N} · 매매시작 {D.td[D.start_di].date()}")

    all_results = []

    # ---------------- 1단계: 슬롯20·손절-7% 고정 ----------------
    log("[1단계] 슬롯20·손절-7% 고정, 과매도4 × 익절3 × 시간손절3 = 36조합")
    methods = ["RSI", "DEV3", "DEV5", "DEV7"]
    tps = [0.05, 0.08, 0.12]
    times = [5, 10, 15]
    stage1 = []
    for m, tp, tstop in itertools.product(methods, tps, times):
        r = run_mr1(D, m, tp, tstop, stop=0.07, slots=20)
        r["stage"] = 1
        stage1.append(r); all_results.append(r)
    _print_table("1단계 결과 (36조합)", stage1)

    # ---------------- 2단계: 1단계 Top3 확장 ----------------
    # 선정 기준: 목표(승률70%&CAGR20%) 우선, 없으면 '승률↑ + CAGR↑' 균형점수.
    def balance_score(r):
        # 승률·CAGR 둘 다 높을수록 좋게: 각 목표 대비 달성도(상한 1.5)를 곱
        w = min(r["win_rate"] / 0.70, 1.5)
        c = min(max(r["CAGR"], 0) / 0.20, 1.5)
        return w * c
    top3 = sorted(stage1, key=balance_score, reverse=True)[:3]
    log("[2단계] 1단계 상위 3조합을 슬롯(20/30/40)×손절(-5/-7%)로 확장")
    for t in top3:
        log(f"   선정: {label(t)}  (승률 {t['win_rate']*100:.1f}% · CAGR {fpct(t['CAGR'])})")
    stage2 = []
    seen = {(r["method"], r["tp"], r["time"], r["stop"], r["slots"]) for r in stage1}
    for t in top3:
        for slots in [20, 30, 40]:
            for stop in [0.05, 0.07]:
                key = (t["method"], t["tp"], t["time"], stop, slots)
                r = run_mr1(D, t["method"], t["tp"], t["time"], stop, slots)
                r["stage"] = 2
                stage2.append(r)
                if key not in seen:
                    all_results.append(r); seen.add(key)
    _print_table("2단계 결과 (Top3 확장)", stage2)

    # ---------------- 종합 판정 ----------------
    _verdict(all_results)
    _save_and_plot(all_results)


def _print_table(title, results):
    print()
    print("=" * 108)
    print(f" {title}")
    print("=" * 108)
    print(f"  {'조합':>44s} {'CAGR':>8s} {'MDD':>8s} {'승률':>7s} {'손익비':>6s} "
          f"{'Sharpe':>7s} {'매수':>5s} {'매도':>5s} {'보유일':>6s}")
    for r in sorted(results, key=lambda x: -x["win_rate"]):
        print(f"  {label(r):>44s} {fpct(r['CAGR']):>8s} {fpct(r['MDD']):>8s} "
              f"{r['win_rate']*100:>6.1f}% {r['PL']:>6.2f} {r['Sharpe']:>7.2f} "
              f"{r['n_buy']:>5d} {r['n_sell']:>5d} {r['avg_hold']:>6.1f}")


def _verdict(results):
    print()
    print("=" * 108)
    print(" 종합 판정")
    print("=" * 108)
    hit = [r for r in results if r["win_rate"] >= 0.70 and r["CAGR"] >= 0.20]
    if hit:
        print("  ✅ 승률≥70% & CAGR≥20% 동시 충족 조합:")
        for r in sorted(hit, key=lambda x: -x["CAGR"]):
            print(f"     {label(r)}  승률 {r['win_rate']*100:.1f}% · CAGR {fpct(r['CAGR'])}")
    else:
        print("  ❌ 승률≥70% & CAGR≥20% 동시 충족 조합 없음.\n")

    print("\n  [ 승률 ≥ 70% 이면서 CAGR 높은 Top5 ]")
    hi_win = sorted([r for r in results if r["win_rate"] >= 0.70],
                    key=lambda x: -x["CAGR"])[:5]
    if hi_win:
        for r in hi_win:
            print(f"    {label(r):>44s}  승률 {r['win_rate']*100:5.1f}%  CAGR {fpct(r['CAGR']):>8s}  MDD {fpct(r['MDD']):>8s}")
    else:
        print("    (승률 70% 이상 조합 없음)")

    print("\n  [ CAGR ≥ 20% 이면서 승률 높은 Top5 ]")
    hi_cagr = sorted([r for r in results if r["CAGR"] >= 0.20],
                     key=lambda x: -x["win_rate"])[:5]
    if hi_cagr:
        for r in hi_cagr:
            print(f"    {label(r):>44s}  CAGR {fpct(r['CAGR']):>8s}  승률 {r['win_rate']*100:5.1f}%  MDD {fpct(r['MDD']):>8s}")
    else:
        print("    (CAGR 20% 이상 조합 없음)")

    # 최고 승률 / 최고 CAGR 극단점
    best_win = max(results, key=lambda x: x["win_rate"])
    best_cagr = max(results, key=lambda x: x["CAGR"])
    print(f"\n  · 전체 최고 승률 : {label(best_win)} → 승률 {best_win['win_rate']*100:.1f}%, CAGR {fpct(best_win['CAGR'])}")
    print(f"  · 전체 최고 CAGR : {label(best_cagr)} → CAGR {fpct(best_cagr['CAGR'])}, 승률 {best_cagr['win_rate']*100:.1f}%")


def _save_and_plot(results):
    df = pd.DataFrame(results)
    df["조합"] = df.apply(label, axis=1)
    cols = ["조합", "method", "tp", "time", "stop", "slots", "CAGR", "MDD",
            "win_rate", "PL", "Sharpe", "total_ret", "n_buy", "n_sell", "avg_hold", "stage"]
    df[cols].to_csv(os.path.join(RESULT_DIR, "grid_results.csv"),
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
        fig, ax = plt.subplots(figsize=(11, 8))
        x = df["win_rate"] * 100
        y = df["CAGR"] * 100
        colors = ["#E94545" if m == "RSI" else "#3182F6" if m == "DEV3"
                  else "#22A06B" if m == "DEV5" else "#F5A623" for m in df["method"]]
        ax.scatter(x, y, c=colors, s=40, alpha=0.75, edgecolors="k", linewidths=0.3)
        ax.axvline(70, color="gray", ls="--", lw=1, label="승률 70% 목표선")
        ax.axhline(20, color="purple", ls="--", lw=1, label="CAGR 20% 목표선")
        # 목표 영역(우상단) 음영
        ax.axvspan(70, 100, ymin=0, ymax=1, alpha=0.04, color="green")
        ax.set_xlabel("매매단위 승률 (%)")
        ax.set_ylabel("CAGR (%)")
        ax.set_title("MR1 승률 vs CAGR 트레이드오프 (색=과매도방식)")
        # 범례(색상)
        from matplotlib.lines import Line2D
        leg = [Line2D([0], [0], marker='o', color='w', label=k,
                      markerfacecolor=c, markersize=8)
               for k, c in [("RSI", "#E94545"), ("DEV3", "#3182F6"),
                            ("DEV5", "#22A06B"), ("DEV7", "#F5A623")]]
        leg += [Line2D([0], [0], color="gray", ls="--", label="승률70%"),
                Line2D([0], [0], color="purple", ls="--", label="CAGR20%")]
        ax.legend(handles=leg, loc="best", fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        png = os.path.join(RESULT_DIR, "scatter_winrate_cagr.png")
        plt.savefig(png, dpi=140)
        plt.close(fig)
        print(f"\n  산점도 저장 : {png}")
    except Exception as e:
        print(f"\n  (산점도 생략: {e})")
    print(f"  결과 저장 : {RESULT_DIR}")


if __name__ == "__main__":
    main()
