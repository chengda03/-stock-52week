# -*- coding: utf-8 -*-
"""
backtest_mr2.py — "MR2: 평균회귀 개선판" (MR1의 두 약점을 정면으로 고침)
============================================================================
MR1의 문제
  (a) 떨어지는 칼날: '가장 많이 빠진 종목' 우선매수 → 하락 가속 종목만 잡아 폭망
  (b) 자금 미투입 : RSI<30 조건은 너무 드물어 슬롯 대부분 현금 → CAGR ~0%

MR2의 개선
  1) 시장레짐 필터: KOSPI가 120일선 위일 때만 신규매수 (하락장 회피)
  2) '얕게 눌린' 종목만: 5일선 대비 3~7% 사이로만 눌린 종목 (과대낙폭 제외)
     → 그 중 이격도가 '가장 작은(얕은)' 순서로 우선매수 (칼날 회피)
  3) 목표익절 제거 → 트레일링 스톱으로 수익을 늘려서 붙잡음(자금이 더 오래 굴러감)

매수조건 (모두 충족)
  1) 현재가 > 120일선
  2) KOSPI 종가 > KOSPI 120일선            ← 신규
  3) 3% ≤ (5일선 대비 하락률) ≤ 7%          ← 밴드 제한 + 얕은 순 랭킹
  4) 20일평균거래대금 ≥ 10억

매도조건 (셋 중 먼저 닿는 것)
  1) 손절   : 진입가 × 0.93 (하드 바닥, -7%)
  2) 트레일링: 진입 후 고점 대비 -X%,  X ∈ {5%,7%,10%}
  3) 시간손절: 매수 후 N거래일 경과 & 아직 손실,  N ∈ {10,15}

불타기 없음 · 1억원 · 슬롯 20 · 비용 매수0.115%/매도0.295%
기간 2020-01-01 ~ 최신
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

RESULT_DIR = os.path.join("data", "cache_mr2", "results")
os.makedirs(RESULT_DIR, exist_ok=True)

INITIAL_CASH = 100_000_000
BUY_COST = 0.00115
SELL_COST = 0.00295
LIQ_MIN = 1_000_000_000
SLOTS = 20
HARD_STOP = 0.07
DEV_LO, DEV_HI = 0.03, 0.07        # 얕게 눌림 밴드
KOSPI_MA = 120                     # 시장레짐 이동평균
TRADE_START = pd.Timestamp("2020-01-01")


class MR2Data:
    def __init__(self):
        store = data_layer.get_store()
        self.store = store
        td = store.trading_days
        self.td = td
        self.close_v = store.close_v
        self.close_ff = store.close_ff
        self.name_map = store.name_map
        T, N = self.close_v.shape
        self.T, self.N = T, N
        self.start_di = store.start_di

        close = pd.DataFrame(self.close_v, index=td, columns=store.valid_codes)
        ma120 = close.rolling(120, min_periods=120).mean().to_numpy(dtype=float)
        ma5 = close.rolling(5, min_periods=5).mean().to_numpy(dtype=float)
        tv = store.tv_v
        dev = (ma5 - self.close_v) / ma5           # 양수 = 5일선 아래

        # 시장레짐: KOSPI > KOSPI 120일선
        ks = pd.Series(store.kospi_close_v, index=td)
        ks_ma = ks.rolling(KOSPI_MA, min_periods=KOSPI_MA).mean()
        self.market_ok = (ks > ks_ma).to_numpy(dtype=bool)

        in_top = self._build_top500_mask()
        trend_ok = self.close_v > ma120
        liq_ok = np.isfinite(tv) & (tv >= LIQ_MIN)
        band_ok = np.isfinite(dev) & (dev >= DEV_LO) & (dev <= DEV_HI)

        self.elig = in_top & trend_ok & liq_ok & band_ok
        self.score = dev                           # 오름차순 = 얕게 눌린 순 우선

    def _build_top500_mask(self):
        store = self.store
        mc = store.marcap_v.copy()
        desig = store.admin_desig_v
        isnat = np.isnat(desig)
        td64 = self.td.values.astype("datetime64[ns]")
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


def run_mr2(D: MR2Data, trail: float, time_stop: int) -> dict:
    close_v = D.close_v
    close_ff = D.close_ff
    slot_amt = INITIAL_CASH / SLOTS
    cash = float(INITIAL_CASH)
    port: dict[int, dict] = {}   # col -> {shares, entry, cost, buy_di, peak}
    daily = np.empty(D.T)
    n_buy = n_sell = wins = 0
    gross_w = gross_l = 0.0
    hold_sum = 0
    reasons = {"SL": 0, "TRAIL": 0, "TIME": 0}

    for di in range(D.T):
        if di < D.start_di:
            daily[di] = INITIAL_CASH
            continue
        row = close_v[di]

        # 1) 매도
        for col in list(port.keys()):
            px = row[col]
            if not np.isfinite(px):
                continue
            pos = port[col]
            if px > pos["peak"]:
                pos["peak"] = px
            reason = None
            if px <= pos["entry"] * (1.0 - HARD_STOP):
                reason = "SL"
            elif px <= pos["peak"] * (1.0 - trail):
                reason = "TRAIL"
            elif (di - pos["buy_di"]) >= time_stop and px < pos["entry"]:
                reason = "TIME"
            if reason is not None:
                proceeds = px * pos["shares"] * (1.0 - SELL_COST)
                pnl = proceeds - pos["cost"]
                cash += proceeds
                n_sell += 1
                reasons[reason] += 1
                hold_sum += di - pos["buy_di"]
                if pnl > 0:
                    wins += 1; gross_w += pnl
                else:
                    gross_l += pnl
                del port[col]

        # 2) 신규매수 (시장레짐 ON일 때만, 얕게 눌린 순)
        free = SLOTS - len(port)
        if free > 0 and D.market_ok[di]:
            elig = D.elig[di].copy()
            for col in port:
                elig[col] = False
            cand = np.where(elig)[0]
            if len(cand) > 0:
                order = cand[np.argsort(D.score[di][cand])]  # 오름차순=얕은 순
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
                                 "cost": spent, "buy_di": di, "peak": float(p)}
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

    return {"trail": trail, "time": time_stop, "CAGR": cagr, "MDD": mdd,
            "win_rate": win_rate, "PL": pl, "Sharpe": sharpe,
            "total_ret": final / INITIAL_CASH - 1.0, "n_buy": n_buy,
            "n_sell": n_sell, "avg_hold": avg_hold, "final": final,
            "reasons": reasons}


def label(r) -> str:
    return f"트레일-{r['trail']*100:.0f}%·시간{r['time']}·손절-7%·슬롯20"


def main():
    log("===== MR2 평균회귀 개선판 백테스트 =====")
    D = MR2Data()
    log(f"거래일 {D.td[0].date()}~{D.td[-1].date()} · 매매시작 {D.td[D.start_di].date()}")
    on = D.market_ok[D.start_di:].mean()
    log(f"시장레짐(KOSPI>120일선) ON 비율(매매기간): {on*100:.1f}%")

    results = []
    for trail, tstop in itertools.product([0.05, 0.07, 0.10], [10, 15]):
        r = run_mr2(D, trail, tstop)
        results.append(r)

    print()
    print("=" * 104)
    print(" MR2 결과 (트레일링3 × 시간손절2 = 6조합, 슬롯20·손절-7% 고정)")
    print("=" * 104)
    print(f"  {'조합':>34s} {'CAGR':>8s} {'MDD':>8s} {'승률':>7s} {'손익비':>6s} "
          f"{'Sharpe':>7s} {'매수':>5s} {'매도':>5s} {'보유일':>6s}  매도사유(손절/트레일/시간)")
    for r in sorted(results, key=lambda x: -x["CAGR"]):
        rs = r["reasons"]
        print(f"  {label(r):>34s} {fpct(r['CAGR']):>8s} {fpct(r['MDD']):>8s} "
              f"{r['win_rate']*100:>6.1f}% {r['PL']:>6.2f} {r['Sharpe']:>7.2f} "
              f"{r['n_buy']:>5d} {r['n_sell']:>5d} {r['avg_hold']:>6.1f}  "
              f"{rs['SL']}/{rs['TRAIL']}/{rs['TIME']}")

    _verdict(results)
    _save_and_plot(results)


def _verdict(results):
    print()
    print("=" * 104)
    print(" 종합 판정 (MR1 대비 개선 여부)")
    print("=" * 104)
    best_cagr = max(results, key=lambda x: x["CAGR"])
    best_mdd = max(results, key=lambda x: x["MDD"])   # MDD는 음수라 최댓값=가장 얕음
    best_win = max(results, key=lambda x: x["win_rate"])
    print("  [MR1 기준선]")
    print("    · RSI 최고: CAGR +1.4% / MDD -5.4% / 승률 65.5%  (자금 미투입)")
    print("    · DEV 방식: CAGR -9% ~ -42% / MDD -63% ~ -97%   (떨어지는 칼날)")
    print("  [MR2 결과]")
    print(f"    · 최고 CAGR : {label(best_cagr)} → CAGR {fpct(best_cagr['CAGR'])}, MDD {fpct(best_cagr['MDD'])}, 승률 {best_cagr['win_rate']*100:.1f}%")
    print(f"    · 최저 MDD  : {label(best_mdd)} → MDD {fpct(best_mdd['MDD'])}, CAGR {fpct(best_mdd['CAGR'])}")
    print(f"    · 최고 승률 : {label(best_win)} → 승률 {best_win['win_rate']*100:.1f}%, CAGR {fpct(best_win['CAGR'])}")
    all_pos = all(r["CAGR"] > 0 for r in results)
    any_pos = any(r["CAGR"] > 0 for r in results)
    print("\n  [판정]")
    print(f"    · '떨어지는 칼날' 해소: DEV 폭망(MDD -97%) → MR2 최저 MDD {fpct(best_mdd['MDD'])} (개선)")
    print(f"    · CAGR 양수 구간: {'모든 조합 양수' if all_pos else ('일부 양수' if any_pos else '여전히 음수')}")
    hit = [r for r in results if r["win_rate"] >= 0.70 and r["CAGR"] >= 0.20]
    print(f"    · 목표(승률70%+CAGR20%): {'달성' if hit else '미달성'}")


def _save_and_plot(results):
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "reasons"} for r in results])
    df["조합"] = df.apply(label, axis=1)
    df.to_csv(os.path.join(RESULT_DIR, "grid_results.csv"),
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
        cmap = {0.05: "#3182F6", 0.07: "#22A06B", 0.10: "#E94545"}
        colors = [cmap[t] for t in df["trail"]]
        ax.scatter(x, y, c=colors, s=90, alpha=0.85, edgecolors="k", linewidths=0.4)
        for _, rr in df.iterrows():
            ax.annotate(f"트레일{rr['trail']*100:.0f}·시간{rr['time']}",
                        (rr["win_rate"] * 100, rr["CAGR"] * 100),
                        fontsize=8, xytext=(5, 4), textcoords="offset points")
        ax.axvline(70, color="gray", ls="--", lw=1)
        ax.axhline(20, color="purple", ls="--", lw=1)
        ax.axhline(0, color="black", lw=0.8)
        ax.axvspan(70, 100, alpha=0.04, color="green")
        ax.set_xlabel("매매단위 승률 (%)")
        ax.set_ylabel("CAGR (%)")
        ax.set_title("MR2 승률 vs CAGR (색=트레일링폭)")
        from matplotlib.lines import Line2D
        leg = [Line2D([0], [0], marker='o', color='w', label=f"트레일 {int(k*100)}%",
                      markerfacecolor=v, markersize=9) for k, v in cmap.items()]
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
