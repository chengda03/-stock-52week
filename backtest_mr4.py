# -*- coding: utf-8 -*-
"""
backtest_mr4.py — "MR4: 고승률 재시도" (매도를 작은 익절+얕은 손절로 되돌림)
============================================================================
MR3 결론: 실적(ROE) 필터로 MDD 반토막·Sharpe 1.0까지 왔지만, 트레일링 스톱
때문에 승률은 30% 부근에 고정. 승률은 '매도방식'이 결정한다는 것을 확인.

MR4 질문: 매도를 '작은 목표익절 + 얕은 손절'로 되돌려 승률을 끌어올리되,
          ROE 필터로 CAGR 손실을 얼마나 보전할 수 있는가?
   (MR1의 익절판은 필터가 없어 CAGR이 죽었음 → 여기선 ROE 필터를 얹어 재확인)

베이스(MR3와 동일): KOSPI>120일선 + 현재가>120일선 + 5일선 -3~7% 얕은눌림
                    + 20일평균거래대금 10억 + [ROE≥10 토글] + 얕은 순 우선매수
매도(셋 중 먼저): 익절 +Y%(진입가) / 손절 -Z%(진입가) / 시간손절 N일 경과
   Y ∈ {3%, 5%},  Z ∈ {3%, 5%},  N ∈ {5, 10},  ROE토글 {끔, ROE≥10}
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

RESULT_DIR = os.path.join("data", "cache_mr4", "results")
os.makedirs(RESULT_DIR, exist_ok=True)

INITIAL_CASH = 100_000_000
BUY_COST = 0.00115
SELL_COST = 0.00295
LIQ_MIN = 1_000_000_000
SLOTS = 20
DEV_LO, DEV_HI = 0.03, 0.07
KOSPI_MA = 120
ROE_MIN = 10.0
TRADE_START = pd.Timestamp("2020-01-01")


class MR4Data:
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
        tv = store.tv_v
        dev = (ma5 - self.close_v) / ma5

        ks = pd.Series(store.kospi_close_v, index=td)
        ks_ma = ks.rolling(KOSPI_MA, min_periods=KOSPI_MA).mean()
        self.market_ok = (ks > ks_ma).to_numpy(dtype=bool)

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
        self.roe_ok = np.isfinite(roe_day) & (roe_day >= ROE_MIN)

    def elig_for(self, use_roe: bool) -> np.ndarray:
        return self.base & self.roe_ok if use_roe else self.base

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


def run_mr4(D, elig_mask, use_roe, tp, stop, time_stop) -> dict:
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
                                 "cost": spent, "buy_di": di}
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

    return {"roe": use_roe, "tp": tp, "stop": stop, "time": time_stop,
            "CAGR": cagr, "MDD": mdd, "win_rate": win_rate, "PL": pl,
            "Sharpe": sharpe, "n_buy": n_buy, "n_sell": n_sell,
            "avg_hold": avg_hold}


def label(r) -> str:
    q = "ROE≥10" if r["roe"] else "필터없음"
    return f"{q}·익절+{r['tp']*100:.0f}%·손절-{r['stop']*100:.0f}%·시간{r['time']}"


def main():
    log("===== MR4 고승률 재시도 (작은 익절+얕은 손절 +ROE) =====")
    D = MR4Data()
    log(f"거래일 {D.td[0].date()}~{D.td[-1].date()} · 매매시작 {D.td[D.start_di].date()}")

    results = []
    for use_roe in [False, True]:
        em = D.elig_for(use_roe)
        for tp, stop, tstop in itertools.product([0.03, 0.05], [0.03, 0.05], [5, 10]):
            results.append(run_mr4(D, em, use_roe, tp, stop, tstop))

    print()
    print("=" * 110)
    print(" MR4 결과 (ROE토글2 × 익절2 × 손절2 × 시간2 = 16조합)")
    print("=" * 110)
    print(f"  {'조합':>38s} {'CAGR':>8s} {'MDD':>8s} {'승률':>7s} {'손익비':>6s} "
          f"{'Sharpe':>7s} {'매수':>5s} {'보유일':>6s}")
    for r in sorted(results, key=lambda x: -x["win_rate"]):
        print(f"  {label(r):>38s} {fpct(r['CAGR']):>8s} {fpct(r['MDD']):>8s} "
              f"{r['win_rate']*100:>6.1f}% {r['PL']:>6.2f} {r['Sharpe']:>7.2f} "
              f"{r['n_buy']:>5d} {r['avg_hold']:>6.1f}")

    print()
    print("=" * 110)
    print(" 종합 판정")
    print("=" * 110)
    best_win = max(results, key=lambda x: x["win_rate"])
    print(f"  · 최고 승률: {label(best_win)} → 승률 {best_win['win_rate']*100:.1f}%, CAGR {fpct(best_win['CAGR'])}")
    # 같은 조건에서 ROE 유무가 CAGR/승률에 준 영향
    print("\n  [같은 매도조건, 필터없음 → ROE≥10]")
    for tp, stop, tstop in itertools.product([0.03, 0.05], [0.03, 0.05], [5, 10]):
        b = next(r for r in results if not r["roe"] and r["tp"] == tp and r["stop"] == stop and r["time"] == tstop)
        g = next(r for r in results if r["roe"] and r["tp"] == tp and r["stop"] == stop and r["time"] == tstop)
        print(f"    익절+{tp*100:.0f}·손절-{stop*100:.0f}·시간{tstop}: "
              f"승률 {b['win_rate']*100:.1f}%→{g['win_rate']*100:.1f}%  "
              f"CAGR {fpct(b['CAGR'])}→{fpct(g['CAGR'])}")
    hit = [r for r in results if r["win_rate"] >= 0.70 and r["CAGR"] >= 0.20]
    hi_win = [r for r in results if r["win_rate"] >= 0.70]
    print(f"\n  · 승률≥70% 조합: {len(hi_win)}개", end="")
    if hi_win:
        bw = max(hi_win, key=lambda x: x["CAGR"])
        print(f" (그 중 최고 CAGR {fpct(bw['CAGR'])} @ {label(bw)})")
    else:
        print()
    print(f"  · 목표(승률70%+CAGR20%): {'달성' if hit else '미달성'}")

    df = pd.DataFrame(results)
    df["조합"] = df.apply(label, axis=1)
    df.to_csv(os.path.join(RESULT_DIR, "grid_results.csv"), index=False, encoding="utf-8-sig")
    _plot(df)


def _plot(df):
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
        colors = ["#E94545" if r else "#8E8E8E" for r in df["roe"]]
        ax.scatter(df["win_rate"] * 100, df["CAGR"] * 100, c=colors, s=90,
                   alpha=0.85, edgecolors="k", linewidths=0.4)
        ax.axvline(70, color="gray", ls="--", lw=1)
        ax.axhline(20, color="purple", ls="--", lw=1)
        ax.axhline(0, color="black", lw=0.8)
        ax.axvspan(70, 100, alpha=0.04, color="green")
        ax.set_xlabel("매매단위 승률 (%)")
        ax.set_ylabel("CAGR (%)")
        ax.set_title("MR4 승률 vs CAGR (빨강=ROE≥10, 회색=필터없음)")
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
