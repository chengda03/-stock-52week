# -*- coding: utf-8 -*-
"""
backtest_mr3.py — "MR3: 종목선정 강화판" (MR2 + 실적·수급 필터)
============================================================================
MR2까지의 결론
  · 시장레짐(KOSPI>120일선) + 얕은 눌림(5일선 -3~7%) + 트레일링 스톱으로
    '떨어지는 칼날'·'자금 미투입'은 고쳤고 CAGR이 전 구간 양수(+2.5~+11.6%)로 전환.
  · 그러나 승률은 27~35%로 낮음. 남은 과제 = "종목 품질"을 올려 승패 자체를 개선.

MR3 아이디어
  MR2 기계장치(매수·매도 규칙)는 그대로 두고, '무엇을 살지'만 강화한다.
    · 실적 필터 : ROE(%) ≥ 10  (적자/부실주 배제, 룩어헤드 방지 재무 사용)
    · 수급 필터 : 60일 모멘텀 > 0  (중기 상승 흐름이 살아있는 종목만)
  두 필터를 토글(끔/ROE/모멘텀/둘다)로 넣어 MR2 대비 승률·CAGR이
  '동시에' 올라가는지 확인한다.

매수조건 (모두 충족)
  1) 현재가 > 120일선
  2) KOSPI 종가 > KOSPI 120일선
  3) 3% ≤ (5일선 대비 하락률) ≤ 7%   (얕게 눌린 순 우선매수)
  4) 20일평균거래대금 ≥ 10억
  5) [토글] ROE(%) ≥ 10
  6) [토글] 60일 모멘텀 > 0

매도조건 (셋 중 먼저)
  1) 손절 -7% (하드)  2) 트레일링 -X% (고점대비)  3) 시간손절 N일&손실
불타기 없음 · 1억 · 슬롯20 · 비용 매수0.115%/매도0.295% · 2020-01-01~최신
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

RESULT_DIR = os.path.join("data", "cache_mr3", "results")
os.makedirs(RESULT_DIR, exist_ok=True)

INITIAL_CASH = 100_000_000
BUY_COST = 0.00115
SELL_COST = 0.00295
LIQ_MIN = 1_000_000_000
SLOTS = 20
HARD_STOP = 0.07
DEV_LO, DEV_HI = 0.03, 0.07
KOSPI_MA = 120
ROE_MIN = 10.0
MOM_WIN = 60
TRADE_START = pd.Timestamp("2020-01-01")


class MR3Data:
    def __init__(self):
        store = data_layer.get_store()
        self.store = store
        td = store.trading_days
        self.td = td
        self.close_v = store.close_v
        self.close_ff = store.close_ff
        T, N = self.close_v.shape
        self.T, self.N = T, N
        self.start_di = store.start_di

        close = pd.DataFrame(self.close_v, index=td, columns=store.valid_codes)
        ma120 = close.rolling(120, min_periods=120).mean().to_numpy(dtype=float)
        ma5 = close.rolling(5, min_periods=5).mean().to_numpy(dtype=float)
        mom60 = (close / close.shift(MOM_WIN) - 1.0).to_numpy(dtype=float)
        tv = store.tv_v
        dev = (ma5 - self.close_v) / ma5

        ks = pd.Series(store.kospi_close_v, index=td)
        ks_ma = ks.rolling(KOSPI_MA, min_periods=KOSPI_MA).mean()
        self.market_ok = (ks > ks_ma).to_numpy(dtype=bool)

        # 룩어헤드 방지 ROE(%) 일별 패널: 그 날 사용가능한 사업보고서 연도의 ROE
        roe_day = np.full((T, N), np.nan)
        for yr in np.unique(store.day_fin_year):
            a = store.roe_by_year.get(int(yr))
            if a is None:
                continue
            rows = np.where(store.day_fin_year == yr)[0]
            roe_day[rows, :] = a[None, :]

        in_top = self._build_top500_mask()
        trend_ok = self.close_v > ma120
        liq_ok = np.isfinite(tv) & (tv >= LIQ_MIN)
        band_ok = np.isfinite(dev) & (dev >= DEV_LO) & (dev <= DEV_HI)
        self.base = in_top & trend_ok & liq_ok & band_ok
        self.score = dev

        # 토글 필터 마스크
        self.roe_ok = np.isfinite(roe_day) & (roe_day >= ROE_MIN)
        self.mom_ok = np.isfinite(mom60) & (mom60 > 0.0)

    def elig_for(self, quality: str) -> np.ndarray:
        m = self.base
        if quality in ("ROE", "ROE+MOM"):
            m = m & self.roe_ok
        if quality in ("MOM", "ROE+MOM"):
            m = m & self.mom_ok
        return m

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


def run_mr3(D: MR3Data, elig_mask, quality: str, trail: float, time_stop: int) -> dict:
    close_v = D.close_v
    close_ff = D.close_ff
    slot_amt = INITIAL_CASH / SLOTS
    cash = float(INITIAL_CASH)
    port: dict[int, dict] = {}
    daily = np.empty(D.T)
    n_buy = n_sell = wins = 0
    gross_w = gross_l = 0.0
    hold_sum = 0

    for di in range(D.T):
        if di < D.start_di:
            daily[di] = INITIAL_CASH
            continue
        row = close_v[di]

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
                hold_sum += di - pos["buy_di"]
                if pnl > 0:
                    wins += 1; gross_w += pnl
                else:
                    gross_l += pnl
                del port[col]

        free = SLOTS - len(port)
        if free > 0 and D.market_ok[di]:
            elig = elig_mask[di].copy()
            for col in port:
                elig[col] = False
            cand = np.where(elig)[0]
            if len(cand) > 0:
                order = cand[np.argsort(D.score[di][cand])]
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

    return {"quality": quality, "trail": trail, "time": time_stop, "CAGR": cagr,
            "MDD": mdd, "win_rate": win_rate, "PL": pl, "Sharpe": sharpe,
            "total_ret": final / INITIAL_CASH - 1.0, "n_buy": n_buy,
            "n_sell": n_sell, "avg_hold": avg_hold, "final": final}


def run_mr3_daily(D: "MR3Data", elig_mask, trail: float, time_stop: int) -> pd.Series:
    """혼합 포트폴리오용: 한 조합의 일별 자산곡선(pd.Series, 1억 기준)을 반환."""
    close_v = D.close_v
    close_ff = D.close_ff
    slot_amt = INITIAL_CASH / SLOTS
    cash = float(INITIAL_CASH)
    port: dict[int, dict] = {}
    daily = np.empty(D.T)
    for di in range(D.T):
        if di < D.start_di:
            daily[di] = INITIAL_CASH
            continue
        row = close_v[di]
        for col in list(port.keys()):
            px = row[col]
            if not np.isfinite(px):
                continue
            pos = port[col]
            if px > pos["peak"]:
                pos["peak"] = px
            if (px <= pos["entry"] * (1.0 - HARD_STOP) or
                    px <= pos["peak"] * (1.0 - trail) or
                    ((di - pos["buy_di"]) >= time_stop and px < pos["entry"])):
                cash += px * pos["shares"] * (1.0 - SELL_COST)
                del port[col]
        free = SLOTS - len(port)
        if free > 0 and D.market_ok[di]:
            elig = elig_mask[di].copy()
            for col in port:
                elig[col] = False
            cand = np.where(elig)[0]
            if len(cand) > 0:
                for col in cand[np.argsort(D.score[di][cand])]:
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
                    free -= 1
        hv = 0.0
        for col, pos in port.items():
            p = row[col]
            if not np.isfinite(p):
                p = close_ff[di, col]
                if not np.isfinite(p):
                    p = pos["entry"]
            hv += p * pos["shares"]
        daily[di] = cash + hv
    return pd.Series(daily, index=D.td)


QLABEL = {"none": "필터없음", "ROE": "ROE≥10", "MOM": "모멘텀>0", "ROE+MOM": "ROE+모멘텀"}


def label(r) -> str:
    return f"{QLABEL[r['quality']]}·트레일-{r['trail']*100:.0f}%·시간{r['time']}"


def main():
    log("===== MR3 종목선정 강화판 (MR2 + 실적·수급 필터) =====")
    D = MR3Data()
    log(f"거래일 {D.td[0].date()}~{D.td[-1].date()} · 매매시작 {D.td[D.start_di].date()}")

    qualities = ["none", "ROE", "MOM", "ROE+MOM"]
    trails = [0.07, 0.10]
    times = [10, 15]
    results = []
    for q in qualities:
        em = D.elig_for(q)
        for trail, tstop in itertools.product(trails, times):
            results.append(run_mr3(D, em, q, trail, tstop))

    print()
    print("=" * 108)
    print(" MR3 결과 (품질필터4 × 트레일2 × 시간2 = 16조합, 슬롯20·손절-7% 고정)")
    print("=" * 108)
    print(f"  {'조합':>30s} {'CAGR':>8s} {'MDD':>8s} {'승률':>7s} {'손익비':>6s} "
          f"{'Sharpe':>7s} {'매수':>5s} {'매도':>5s} {'보유일':>6s}")
    for r in sorted(results, key=lambda x: -x["CAGR"]):
        print(f"  {label(r):>30s} {fpct(r['CAGR']):>8s} {fpct(r['MDD']):>8s} "
              f"{r['win_rate']*100:>6.1f}% {r['PL']:>6.2f} {r['Sharpe']:>7.2f} "
              f"{r['n_buy']:>5d} {r['n_sell']:>5d} {r['avg_hold']:>6.1f}")

    _verdict(results)
    _save_and_plot(results)


def _verdict(results):
    print()
    print("=" * 108)
    print(" 종합 판정 (MR2 대비 필터 효과)")
    print("=" * 108)
    # 동일 매도조건(trail·time)에서 필터 유무 비교
    print("  [같은 매도조건에서 '필터없음' → 'ROE+모멘텀' 변화]")
    for trail, tstop in [(0.10, 15), (0.10, 10), (0.07, 15), (0.07, 10)]:
        base = next(r for r in results if r["quality"] == "none" and r["trail"] == trail and r["time"] == tstop)
        both = next(r for r in results if r["quality"] == "ROE+MOM" and r["trail"] == trail and r["time"] == tstop)
        print(f"    트레일-{trail*100:.0f}%·시간{tstop}: "
              f"승률 {base['win_rate']*100:.1f}%→{both['win_rate']*100:.1f}%  "
              f"CAGR {fpct(base['CAGR'])}→{fpct(both['CAGR'])}  "
              f"MDD {fpct(base['MDD'])}→{fpct(both['MDD'])}  "
              f"거래 {base['n_buy']}→{both['n_buy']}")

    best_cagr = max(results, key=lambda x: x["CAGR"])
    best_win = max(results, key=lambda x: x["win_rate"])
    best_sharpe = max(results, key=lambda x: x["Sharpe"])
    print(f"\n  · 최고 CAGR  : {label(best_cagr)} → CAGR {fpct(best_cagr['CAGR'])}, 승률 {best_cagr['win_rate']*100:.1f}%, MDD {fpct(best_cagr['MDD'])}")
    print(f"  · 최고 승률  : {label(best_win)} → 승률 {best_win['win_rate']*100:.1f}%, CAGR {fpct(best_win['CAGR'])}")
    print(f"  · 최고 Sharpe: {label(best_sharpe)} → Sharpe {best_sharpe['Sharpe']:.2f}, CAGR {fpct(best_sharpe['CAGR'])}, 승률 {best_sharpe['win_rate']*100:.1f}%")
    hit = [r for r in results if r["win_rate"] >= 0.70 and r["CAGR"] >= 0.20]
    print(f"\n  · 목표(승률70%+CAGR20%): {'달성' if hit else '미달성'}")


def _save_and_plot(results):
    df = pd.DataFrame(results)
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
        cmap = {"none": "#8E8E8E", "ROE": "#3182F6", "MOM": "#22A06B", "ROE+MOM": "#E94545"}
        colors = [cmap[q] for q in df["quality"]]
        ax.scatter(df["win_rate"] * 100, df["CAGR"] * 100, c=colors, s=90,
                   alpha=0.85, edgecolors="k", linewidths=0.4)
        ax.axvline(70, color="gray", ls="--", lw=1)
        ax.axhline(20, color="purple", ls="--", lw=1)
        ax.axhline(0, color="black", lw=0.8)
        ax.axvspan(70, 100, alpha=0.04, color="green")
        ax.set_xlabel("매매단위 승률 (%)")
        ax.set_ylabel("CAGR (%)")
        ax.set_title("MR3 승률 vs CAGR (색=품질필터)")
        from matplotlib.lines import Line2D
        leg = [Line2D([0], [0], marker='o', color='w', label=QLABEL[k],
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
