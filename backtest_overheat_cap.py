# -*- coding: utf-8 -*-
"""
backtest_overheat_cap.py
========================
[실험] 매수 시점 '과열 방지 상한'(모멘텀 상한 / 이격도 상한 / 결합)이 타당한지 비교.

배경: 2024년 최대손실 5건 분석에서 실패 종목 일부가 매수시점에 이미 과열(모멘텀 56~86%,
이격도 45~85%)이었음 → "너무 오른 곳에서 사서 터진다"는 가설. 이번엔 매수조건에 상한을 걸어
과열 종목을 '신규매수'에서만 제외한다(불타기는 원본 그대로).

비교(전부 같은 DataStore·같은 날 → 데이터 드리프트 0, 차이는 오직 check_buy_conditions):
  · base        : 현재 확정 strategy_core (상한 없음)
  · momcap30/50 : 0 < 20일모멘텀 ≤ 30% / 50%
  · dispcap30/50: 2.5% ≤ 이격도 ≤ 30% / 50%
  · combined    : 모멘텀≤30% + 이격도≤30% 동시

산출:
  (0) 사전확인: base 통과 후보 중 각 상한에 새로 걸리는 비율 + 2024 최대손실 5건 개별 판정
  (1) 전체기간(2020-01~) CAGR/MDD/Sharpe/Calmar/손익비/승률/총거래
  (2) 연도별 독립(매년 1억 리셋) 기간수익률 비교표

정직성: 단일 경로(약 6.5년) 백테스트. 과최적화 위험 상존.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

import data_layer
import strategy_core as sc_base
from strategy_core import StockSnapshot, Position

import strategy_core_momcap30 as v_m30
import strategy_core_momcap50 as v_m50
import strategy_core_dispcap30 as v_d30
import strategy_core_dispcap50 as v_d50
import strategy_core_combined_cap as v_cmb

INITIAL_CASH = 100_000_000
BUY_COST = 0.00115
SELL_COST = 0.00295
TRADE_START = pd.Timestamp("2020-01-01")
RESTOP_WINDOW = 5

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

# 2024 최대손실 5건 (종목명, 코드, 매수일)  ← 앞선 상세분석 results/trades_2024_worst5.csv
WORST5 = [
    ("DB하이텍", "000990", "2024-07-17"),
    ("AP시스템", "265520", "2024-04-22"),
    ("화신",     "010690", "2024-01-02"),
    ("사조대림", "003960", "2024-07-11"),
    ("리튬포어스", "073570", "2024-04-03"),
]

WINDOWS = [
    ("전체(참고)", "2020-01-01", "2026-07-14"),
    ("2020", "2020-01-01", "2020-12-31"),
    ("2021", "2021-01-01", "2021-12-31"),
    ("2022(하락장)", "2022-01-01", "2022-12-31"),
    ("2023", "2023-01-01", "2023-12-31"),
    ("2024", "2024-01-01", "2024-12-31"),
    ("2025", "2025-01-01", "2025-12-31"),
    ("2026(~7/14)", "2026-01-01", "2026-07-14"),
]


def _build_snap(store, col, di) -> StockSnapshot:
    yr = int(store.day_fin_year[di])
    roe_a = store.roe_by_year.get(yr)
    eps_a = store.eps_by_year.get(yr)
    roe = float(roe_a[col]) if roe_a is not None else float("nan")
    eps = float(eps_a[col]) if eps_a is not None else float("nan")
    return StockSnapshot(
        ticker=store.valid_codes[col],
        date=store.trading_days[di].date(),
        price=float(store.close_v[di, col]),
        ma_buy=float(store.ma_buy_v[di, col]),
        ma_sell=float(store.ma_sell_v[di, col]),
        roe_pct=roe, eps=eps,
        momentum_20d=float(store.mom_v[di, col]),
        trading_value_20d_avg=float(store.tv_v[di, col]),
    )


def _disp(snap) -> float:
    if snap.ma_sell and snap.ma_sell > 0:
        return (snap.price - snap.ma_sell) / snap.ma_sell * 100.0
    return float("-inf")


# ---------------------------------------------------------------------------
# (0) 사전확인
# ---------------------------------------------------------------------------
def precheck(store):
    td = store.trading_days
    c2c = store.code_to_col
    close_v = store.close_v
    start_di = store.start_di

    base_pass = 0
    cnt_m30 = cnt_m50 = cnt_d30 = cnt_d50 = cnt_cmb = 0
    for di in range(start_di, len(td)):
        if not sc_base.is_bull_market(store.kospi_close_v[di], store.kospi_ma200_v[di]):
            continue
        for t in store.get_universe(td[di]):
            col = c2c[t]
            px = close_v[di, col]
            if not np.isfinite(px) or px <= 0:
                continue
            snap = _build_snap(store, col, di)
            ok, _ = sc_base.check_buy_conditions(snap, True)
            if not ok:
                continue
            base_pass += 1
            mom = snap.momentum_20d * 100.0
            dp = _disp(snap)
            over_m30 = mom > 30.0
            over_m50 = mom > 50.0
            over_d30 = dp > 30.0
            over_d50 = dp > 50.0
            cnt_m30 += over_m30
            cnt_m50 += over_m50
            cnt_d30 += over_d30
            cnt_d50 += over_d50
            cnt_cmb += (over_m30 or over_d30)

    print("\n" + "=" * 90)
    print(" (0) 사전확인 — base 통과 '후보(종목×강세일)' 중 각 상한에 새로 걸려 탈락하는 비율")
    print("=" * 90)
    print(f"  base 통과 후보-건수(강세일×종목): {base_pass:,}")
    def pct(n): return f"{n:,} ({n/base_pass*100:.1f}%)" if base_pass else "0"
    print(f"  모멘텀 > 30% 로 신규탈락 : {pct(cnt_m30)}")
    print(f"  모멘텀 > 50% 로 신규탈락 : {pct(cnt_m50)}")
    print(f"  이격도 > 30% 로 신규탈락 : {pct(cnt_d30)}")
    print(f"  이격도 > 50% 로 신규탈락 : {pct(cnt_d50)}")
    print(f"  결합(모멘텀>30 또는 이격>30): {pct(cnt_cmb)}")

    # 2024 최대손실 5건 개별 판정
    print("\n" + "-" * 90)
    print(" [핵심검증] 2024 최대손실 5건이 각 상한에 걸려 '신규매수 제외'되는가?")
    print("-" * 90)
    print(f"  {'종목':<10s} {'매수일':<11s} {'모멘텀%':>8s} {'이격도%':>8s}   "
          f"{'m30':>4s} {'m50':>4s} {'d30':>4s} {'d50':>4s} {'결합':>4s}")
    rows = []
    for name, code, dstr in WORST5:
        col = c2c.get(code)
        if col is None:
            print(f"  {name:<10s} {dstr}  (코드 {code} 유니버스 없음)")
            continue
        ts = pd.Timestamp(dstr)
        di = int(np.searchsorted(td.values, ts.to_datetime64()))
        if di >= len(td) or td[di].normalize() != ts:
            # 정확 일자 없으면 근접 거래일
            di = min(di, len(td) - 1)
        snap = _build_snap(store, col, di)
        mom = snap.momentum_20d * 100.0
        dp = _disp(snap)
        ex_m30 = "제외" if mom > 30 else "통과"
        ex_m50 = "제외" if mom > 50 else "통과"
        ex_d30 = "제외" if dp > 30 else "통과"
        ex_d50 = "제외" if dp > 50 else "통과"
        ex_cmb = "제외" if (mom > 30 or dp > 30) else "통과"
        print(f"  {name:<10s} {str(td[di].date()):<11s} {mom:>8.1f} {dp:>8.1f}   "
              f"{ex_m30:>4s} {ex_m50:>4s} {ex_d30:>4s} {ex_d50:>4s} {ex_cmb:>4s}")
        rows.append({"종목": name, "코드": code, "매수일": str(td[di].date()),
                     "모멘텀%": round(mom, 1), "이격도%": round(dp, 1),
                     "momcap30": ex_m30, "momcap50": ex_m50,
                     "dispcap30": ex_d30, "dispcap50": ex_d50, "combined": ex_cmb})
    pd.DataFrame(rows).to_csv(os.path.join(RESULT_DIR, "overheat_worst5_check.csv"),
                              index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# 공통 엔진 (buy_fn만 교체) — 전체기간 & 윈도우 겸용
# ---------------------------------------------------------------------------
def run_engine(store, check_buy_fn, di_range, reset=True):
    td = store.trading_days
    close_v = store.close_v
    close_ff = store.close_ff
    c2c = store.code_to_col

    cash = float(INITIAL_CASH)
    positions: dict[str, Position] = {}
    cost: dict[str, float] = {}
    entry_di: dict[str, int] = {}
    equity: list[float] = []
    days: list[pd.Timestamp] = []

    n_buy = n_add = n_partial = n_final = 0
    wins = 0
    gross_w = gross_l = 0.0
    restop_5d = 0
    bought_set: set[str] = set()

    for di in di_range:
        today = td[di].date()
        is_bull = sc_base.is_bull_market(store.kospi_close_v[di], store.kospi_ma200_v[di])
        sold_today: set[str] = set()

        # 1) 전량매도
        for t in list(positions.keys()):
            col = c2c[t]; px = close_v[di, col]
            if not np.isfinite(px):
                continue
            pos = positions[t]; snap = _build_snap(store, col, di)
            s, _ = sc_base.check_sell_condition(pos, snap)
            if s:
                proceeds = px * pos.shares * (1.0 - SELL_COST)
                cash += proceeds
                pnl = proceeds - cost[t]
                if pnl > 0: wins += 1; gross_w += pnl
                else: gross_l += pnl
                n_final += 1
                if di - entry_di.get(t, di) <= RESTOP_WINDOW:
                    restop_5d += 1
                del positions[t]; del cost[t]; entry_di.pop(t, None)
                sold_today.add(t)

        # 1-2) 부분익절
        for t in list(positions.keys()):
            col = c2c[t]; px = close_v[di, col]
            if not np.isfinite(px) or px <= 0:
                continue
            pos = positions[t]
            if sc_base.check_partial_exit(pos, float(px)):
                before = pos.shares
                sell_sh = sc_base.apply_partial_exit(pos, float(px))
                ratio = sell_sh / before
                proceeds = px * sell_sh * (1.0 - SELL_COST)
                cost_sold = cost[t] * ratio
                pnl = proceeds - cost_sold
                cash += proceeds; cost[t] -= cost_sold
                n_partial += 1
                if pnl > 0: wins += 1; gross_w += pnl
                else: gross_l += pnl

        # 2) 불타기 (원본 규칙 — 상한 미적용)
        if is_bull and positions:
            def cur_ret(t):
                col = c2c[t]; px = close_v[di, col]
                return px / positions[t].avg_price if np.isfinite(px) else -1.0
            for t in sorted(positions.keys(), key=cur_ret, reverse=True):
                col = c2c[t]; px = close_v[di, col]
                if not np.isfinite(px) or px <= 0:
                    continue
                pos = positions[t]
                if sc_base.check_pyramid(pos, float(px), today, is_bull, cash):
                    spent = sc_base.PYRAMID_ADD_AMOUNT_WON * (1.0 + BUY_COST)
                    if cash < spent:
                        continue
                    sc_base.apply_pyramid(pos, float(px), today)
                    cash -= spent; cost[t] += spent; n_add += 1

        # 3) 신규매수 (buy_fn 교체 지점)
        free = sc_base.NUM_SLOTS - len(positions)
        if is_bull and free > 0:
            cands: list[StockSnapshot] = []
            for t in store.get_universe(td[di]):
                if t in positions or t in sold_today:
                    continue
                col = c2c[t]; px = close_v[di, col]
                if not np.isfinite(px) or px <= 0:
                    continue
                snap = _build_snap(store, col, di)
                ok, _ = check_buy_fn(snap, is_bull)
                if ok:
                    cands.append(snap)
            for snap in sc_base.rank_by_momentum(cands):
                if free <= 0:
                    break
                px = snap.price
                sh = int(sc_base.SLOT_AMOUNT_WON // px)
                if sh <= 0:
                    continue
                spent = sh * px * (1.0 + BUY_COST)
                if spent > cash:
                    continue
                positions[snap.ticker] = Position(ticker=snap.ticker, entry_price=px,
                                                  avg_price=px, shares=float(sh), entry_date=today)
                cost[snap.ticker] = spent; entry_di[snap.ticker] = di
                cash -= spent; n_buy += 1; free -= 1
                bought_set.add(snap.ticker)

        # 4) 평가
        hv = 0.0
        for t, pos in positions.items():
            col = c2c[t]; p = close_v[di, col]
            if not np.isfinite(p):
                p = close_ff[di, col]
                if not np.isfinite(p):
                    p = pos.avg_price
            hv += p * pos.shares
        equity.append(cash + hv)
        days.append(td[di])

    s = pd.Series(equity, index=pd.DatetimeIndex(days))
    final = float(s.iloc[-1])
    years = (s.index[-1] - s.index[0]).days / 365.25
    period_ret = final / INITIAL_CASH - 1.0
    cagr = (final / INITIAL_CASH) ** (1.0 / max(years, 1e-9)) - 1.0
    peak = s.cummax()
    mdd = float(((s - peak) / peak).min())
    rets = s.pct_change().fillna(0)
    sharpe = float((rets.mean() / (rets.std() + 1e-12)) * math.sqrt(252))
    calmar = cagr / abs(mdd) if mdd != 0 else float("inf")
    n_sell = n_partial + n_final
    return {
        "period_ret": period_ret, "CAGR": cagr, "MDD": mdd, "Sharpe": sharpe, "Calmar": calmar,
        "PL": abs(gross_w / gross_l) if gross_l != 0 else float("inf"),
        "win_rate": wins / n_sell if n_sell else 0.0,
        "n_trades": n_buy + n_add + n_partial + n_final, "n_buy": n_buy, "n_add": n_add,
        "restop_5d_ratio": restop_5d / n_buy if n_buy else 0.0,
        "bought_set": bought_set,
        "start": str(s.index[0].date()), "end": str(s.index[-1].date()),
    }


VARIANTS = [
    ("base", sc_base.check_buy_conditions),
    ("momcap30", v_m30.check_buy_conditions),
    ("momcap50", v_m50.check_buy_conditions),
    ("dispcap30", v_d30.check_buy_conditions),
    ("dispcap50", v_d50.check_buy_conditions),
    ("combined(m30+d30)", v_cmb.check_buy_conditions),
]


def _jaccard(a, b):
    return len(a & b) / len(a | b) if (a | b) else 1.0


def main():
    store = data_layer.get_store()
    td = store.trading_days

    precheck(store)

    # (1) 전체기간
    full_range = range(store.start_di, len(td))
    full = {}
    for tag, fn in VARIANTS:
        full[tag] = run_engine(store, fn, full_range)
    base_set = full["base"]["bought_set"]

    rows = []
    for tag, _ in VARIANTS:
        m = full[tag]
        rows.append({
            "버전": tag,
            "CAGR%": round(m["CAGR"] * 100, 2),
            "MDD%": round(m["MDD"] * 100, 2),
            "Sharpe": round(m["Sharpe"], 2),
            "Calmar": round(m["Calmar"], 2),
            "손익비": round(m["PL"], 2),
            "승률%": round(m["win_rate"] * 100, 1),
            "총거래": m["n_trades"],
            "신규매수": m["n_buy"],
            "불타기": m["n_add"],
            "5일내재손절%": round(m["restop_5d_ratio"] * 100, 1),
            "base겹침%": round(_jaccard(m["bought_set"], base_set) * 100, 1),
        })
    df_full = pd.DataFrame(rows)
    print("\n" + "=" * 110)
    print(f" (1) 전체기간 비교  ·  {full['base']['start']} ~ {full['base']['end']}  ·  1억 · 거래비용 반영")
    print("=" * 110)
    print(df_full.to_string(index=False))
    df_full.to_csv(os.path.join(RESULT_DIR, "overheat_full_compare.csv"),
                   index=False, encoding="utf-8-sig")

    # (2) 연도별 독립 (기간수익률%)
    yr_rows = []
    for wtag, ws, we in WINDOWS:
        wsd = pd.Timestamp(ws); wed = pd.Timestamp(we)
        di_list = [di for di in range(len(td)) if wsd <= td[di] <= wed]
        row = {"구간": wtag}
        row_mdd = {"구간": wtag}
        for tag, fn in VARIANTS:
            m = run_engine(store, fn, di_list)
            row[tag] = round(m["period_ret"] * 100, 2)
            row_mdd[tag] = round(m["MDD"] * 100, 2)
        yr_rows.append((row, row_mdd))

    df_yr = pd.DataFrame([r for r, _ in yr_rows])
    df_yr_mdd = pd.DataFrame([r for _, r in yr_rows])
    print("\n" + "=" * 110)
    print(" (2) 연도별 독립 백테스트 — 기간수익률%  (매년 1억 리셋)")
    print("=" * 110)
    print(df_yr.to_string(index=False))
    print("\n [참고] 연도별 MDD%")
    print(df_yr_mdd.to_string(index=False))
    df_yr.to_csv(os.path.join(RESULT_DIR, "overheat_yearly_ret.csv"),
                 index=False, encoding="utf-8-sig")
    df_yr_mdd.to_csv(os.path.join(RESULT_DIR, "overheat_yearly_mdd.csv"),
                     index=False, encoding="utf-8-sig")

    print("\n" + "=" * 110)
    print(" ⚠ 정직성: 단일 경로(약 6.5년) 백테스트. 상한값(30/50%)은 후행적으로 정한 값이라 과최적화 위험 존재.")
    print("=" * 110)


if __name__ == "__main__":
    main()
