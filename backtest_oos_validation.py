# -*- coding: utf-8 -*-
"""
backtest_oos_validation.py
==========================
[아웃오브샘플 검증] 지금까지 '전체기간(2020-01~2026-07) 그리드서치'로 확정한 조건 조합이
훈련/검증 분할에서도 유지되는지 정직하게 검증한다. strategy_core.py는 건드리지 않는다.

구간:
  · 훈련(In-sample)     : 2020-01-01 ~ 2023-06-30 (약 3.5년)
  · 검증(Out-of-sample) : 2023-07-01 ~ 2026-07-14 (약 3년)

파라미터 4종(나머지는 새 기준선 고정: 60일선삭제/거래대금30억/저평가ROE×EPS/20슬롯×500만/
무제한불타기/부분익절+8%@50%/시장필터 KOSPI 200일선):
  · 90일선 이격도 : [0, 1, 2.5, 5] %
  · ROE 문턱      : [10, 15, 20] %
  · 모멘텀 lookback: [10, 20, 30] 일   (매수조건 'N일>0' + 매수우선순위 'N일 높은순')
  · 하드손절       : [-8, -10, -12, -15, -20] %

단계1: 훈련구간만 보고 '좌표별(한 파라미터씩, 나머지는 새 기준선 고정)' CAGR 최적값 도출.
       → 검증구간 데이터는 이 단계에서 절대 참조하지 않음.
단계2: 훈련 최적조합을 '고정'해 검증구간에 적용. 전체기간확정조합과 비교.
       + (참고·실전불가) 검증구간 자체 재최적화 → 과최적화 갭 측정.

정직성: 훈련/검증 1분할은 한 가지 방식일 뿐. 결과는 있는 그대로 보고.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

import data_layer
import strategy_core as sc_base
from strategy_core import StockSnapshot, Position

INITIAL_CASH = 100_000_000
BUY_COST = 0.00115
SELL_COST = 0.00295

TRAIN_START = pd.Timestamp("2020-01-01")
TRAIN_END = pd.Timestamp("2023-06-30")
TEST_START = pd.Timestamp("2023-07-01")
TEST_END = pd.Timestamp("2026-07-14")

# 새 기준선(전체기간 확정) 파라미터
CONFIRMED = {"disp": 2.5, "roe": 15.0, "mom": 20, "sl": 12.0}

GRID = {
    "disp": [0.0, 1.0, 2.5, 5.0],
    "roe": [10.0, 15.0, 20.0],
    "mom": [10, 20, 30],
    "sl": [8.0, 10.0, 12.0, 15.0, 20.0],
}

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

_MOM_CACHE: dict[int, np.ndarray] = {}


def mom_panel(store, lookback: int) -> np.ndarray:
    if lookback not in _MOM_CACHE:
        close_df = pd.DataFrame(store.close_v)
        _MOM_CACHE[lookback] = (close_df / close_df.shift(lookback) - 1.0).to_numpy(dtype=float)
    return _MOM_CACHE[lookback]


def _metrics(equity: list[float], days: list[pd.Timestamp]) -> dict:
    s = pd.Series(equity, index=pd.DatetimeIndex(days))
    final = float(s.iloc[-1])
    years = (s.index[-1] - s.index[0]).days / 365.25
    cagr = (final / INITIAL_CASH) ** (1.0 / max(years, 1e-9)) - 1.0
    peak = s.cummax()
    mdd = float(((s - peak) / peak).min())
    rets = s.pct_change().fillna(0)
    sharpe = float((rets.mean() / (rets.std() + 1e-12)) * math.sqrt(252))
    calmar = cagr / abs(mdd) if mdd != 0 else float("inf")
    return {"CAGR": cagr, "MDD": mdd, "Sharpe": sharpe, "Calmar": calmar}


def run(store, params: dict, win_start, win_end) -> dict:
    """params={disp,roe,mom,sl}. [win_start,win_end] 구간만 1억으로 새로 시작해 백테스트."""
    td = store.trading_days
    close_v = store.close_v
    close_ff = store.close_ff
    c2c = store.code_to_col
    yr_arr = store.day_fin_year
    roe_by_year = store.roe_by_year
    eps_by_year = store.eps_by_year
    tv_v = store.tv_v
    ma_sell_v = store.ma_sell_v
    kc_v = store.kospi_close_v
    km_v = store.kospi_ma200_v

    disp_min = params["disp"]
    roe_min = params["roe"]
    sl_frac = params["sl"] / 100.0
    momP = mom_panel(store, params["mom"])

    # 윈도우 인덱스 범위
    di_list = [di for di in range(len(td)) if win_start <= td[di] <= win_end]
    if not di_list:
        raise ValueError("empty window")
    first_di = di_list[0]

    cash = float(INITIAL_CASH)
    positions: dict[str, Position] = {}
    cost: dict[str, float] = {}

    n_buy = n_add = n_partial = n_final = 0
    wins = 0
    gross_w = gross_l = 0.0
    equity: list[float] = []
    days: list[pd.Timestamp] = []

    for di in di_list:
        today = td[di].date()
        is_bull = sc_base.is_bull_market(kc_v[di], km_v[di])
        sold_today: set[str] = set()

        # 1) 전량매도: 90선 이탈 또는 하드손절(먼저 닿는 쪽)
        for t in list(positions.keys()):
            col = c2c[t]; px = close_v[di, col]
            if not np.isfinite(px):
                continue
            pos = positions[t]
            ms = ma_sell_v[di, col]
            sell = False
            if np.isfinite(ms) and px < ms:
                sell = True
            elif px <= pos.avg_price * (1.0 - sl_frac):
                sell = True
            if sell:
                proceeds = px * pos.shares * (1.0 - SELL_COST)
                cash += proceeds
                pnl = proceeds - cost[t]
                if pnl > 0: wins += 1; gross_w += pnl
                else: gross_l += pnl
                n_final += 1
                del positions[t]; del cost[t]; sold_today.add(t)

        # 1-2) 부분익절 (+8%@50%, 원본)
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

        # 2) 불타기 (무제한, 원본)
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

        # 3) 신규매수 (파라미터화된 매수조건 + 모멘텀 lookback 정렬)
        free = sc_base.NUM_SLOTS - len(positions)
        if is_bull and free > 0:
            yr = int(yr_arr[di])
            roe_a = roe_by_year.get(yr); eps_a = eps_by_year.get(yr)
            cands = []  # (mom_val, ticker, price)
            for t in store.get_universe(td[di]):
                if t in positions or t in sold_today:
                    continue
                col = c2c[t]; px = close_v[di, col]
                if not np.isfinite(px) or px <= 0:
                    continue
                roe = float(roe_a[col]) if roe_a is not None else float("nan")
                eps = float(eps_a[col]) if eps_a is not None else float("nan")
                # 저평가
                if not (px < roe * eps):
                    continue
                # ROE
                if not (roe >= roe_min):
                    continue
                # 이격도
                ms = ma_sell_v[di, col]
                if not (np.isfinite(ms) and ms > 0):
                    continue
                if not (((px - ms) / ms * 100.0) >= disp_min):
                    continue
                # 모멘텀 > 0
                mv = momP[di, col]
                if not (mv > 0):
                    continue
                # 거래대금
                if not (tv_v[di, col] >= sc_base.TRADING_VALUE_MIN_WON):
                    continue
                cands.append((float(mv), t, float(px)))
            cands.sort(key=lambda x: x[0], reverse=True)
            for mv, t, px in cands:
                if free <= 0:
                    break
                sh = int(sc_base.SLOT_AMOUNT_WON // px)
                if sh <= 0:
                    continue
                spent = sh * px * (1.0 + BUY_COST)
                if spent > cash:
                    continue
                positions[t] = Position(ticker=t, entry_price=px, avg_price=px,
                                        shares=float(sh), entry_date=today)
                cost[t] = spent
                cash -= spent; n_buy += 1; free -= 1

        # 4) 일별 평가
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

    m = _metrics(equity, days)
    n_sell = n_partial + n_final
    m.update({
        "PL": abs(gross_w / gross_l) if gross_l != 0 else float("inf"),
        "win_rate": wins / n_sell if n_sell else 0.0,
        "n_trades": n_buy + n_add + n_partial + n_final,
        "params": dict(params),
    })
    return m


def coordinate_search(store, win_start, win_end, base: dict) -> dict:
    """한 파라미터씩 sweep(나머지는 base 고정), CAGR 최적값 채택. 결과와 최적조합 반환."""
    sweeps = {}
    best = dict(base)
    for name in ["disp", "roe", "mom", "sl"]:
        rows = []
        for v in GRID[name]:
            p = dict(base); p[name] = v
            m = run(store, p, win_start, win_end)
            rows.append({"value": v, "CAGR": m["CAGR"], "MDD": m["MDD"],
                         "Sharpe": m["Sharpe"], "Calmar": m["Calmar"]})
        rows.sort(key=lambda r: r["CAGR"], reverse=True)
        sweeps[name] = rows
        best[name] = rows[0]["value"]
    return {"sweeps": sweeps, "best": best}


def _fmt_params(p: dict) -> str:
    return f"이격도{p['disp']}%/ROE{p['roe']:.0f}%/모멘텀{p['mom']}일/손절-{p['sl']:.0f}%"


def _mrow(tag, m) -> dict:
    return {
        "구분": tag,
        "CAGR%": round(m["CAGR"] * 100, 2),
        "MDD%": round(m["MDD"] * 100, 2),
        "Sharpe": round(m["Sharpe"], 2),
        "Calmar": round(m["Calmar"], 2),
        "손익비": round(m["PL"], 2),
        "승률%": round(m["win_rate"] * 100, 1),
        "총거래": m["n_trades"],
    }


def main():
    store = data_layer.get_store()

    # ===== 단계 1: 훈련구간 좌표별 그리드서치 =====
    print("\n" + "=" * 100)
    print(" 단계1  훈련구간(2020-01-01~2023-06-30)에서만 좌표별 그리드서치 (검증구간 미참조)")
    print("        각 파라미터 sweep 시 나머지는 새 기준선 고정(이격도2.5/ROE15/모멘텀20/손절-12)")
    print("=" * 100)
    tr = coordinate_search(store, TRAIN_START, TRAIN_END, CONFIRMED)
    label = {"disp": "90일선이격도(%)", "roe": "ROE문턱(%)", "mom": "모멘텀lookback(일)", "sl": "하드손절(-%)"}
    for name in ["disp", "roe", "mom", "sl"]:
        print(f"\n  [{label[name]}]  (CAGR 내림차순, ★=훈련 최적)")
        for i, r in enumerate(tr["sweeps"][name]):
            star = " ★" if i == 0 else "  "
            print(f"   {star} {name}={r['value']:>5}  CAGR {r['CAGR']*100:6.2f}%  "
                  f"MDD {r['MDD']*100:7.2f}%  Sharpe {r['Sharpe']:.2f}  Calmar {r['Calmar']:.2f}")

    train_best = tr["best"]
    print("\n  ------------------------------------------------------------------")
    print(f"  훈련구간 도출 최적조합 : {_fmt_params(train_best)}")
    print(f"  전체기간 확정조합      : {_fmt_params(CONFIRMED)}")
    same = train_best == CONFIRMED
    print(f"  → 일치 여부: {'완전 일치' if same else '불일치(아래 차이)'}")
    if not same:
        for k in ["disp", "roe", "mom", "sl"]:
            if train_best[k] != CONFIRMED[k]:
                print(f"     · {label[k]}: 훈련최적={train_best[k]}  vs  전체확정={CONFIRMED[k]}")

    # ===== 단계 2: 검증구간 성과 =====
    print("\n" + "=" * 100)
    print(" 단계2  검증구간(2023-07-01~2026-07-14) 성과 — 파라미터 재조정 없이 고정 적용")
    print("=" * 100)

    rows = []
    # 참고: 훈련 인샘플
    rows.append(_mrow(f"[참고]훈련 인샘플·확정조합", run(store, CONFIRMED, TRAIN_START, TRAIN_END)))
    if not same:
        rows.append(_mrow(f"[참고]훈련 인샘플·훈련최적", run(store, train_best, TRAIN_START, TRAIN_END)))
    # 검증: 전체기간 확정조합
    rows.append(_mrow(f"검증 OOS·전체확정조합", run(store, CONFIRMED, TEST_START, TEST_END)))
    # 검증: 훈련 도출조합
    if not same:
        rows.append(_mrow(f"검증 OOS·훈련도출조합", run(store, train_best, TEST_START, TEST_END)))
    else:
        rows.append(_mrow(f"검증 OOS·훈련도출조합(=확정과 동일)", run(store, train_best, TEST_START, TEST_END)))

    # 참고·실전불가: 검증구간 자체 재최적화
    te = coordinate_search(store, TEST_START, TEST_END, CONFIRMED)
    test_best = te["best"]
    rows.append(_mrow(f"[참고·실전불가]검증 자체재최적화", run(store, test_best, TEST_START, TEST_END)))

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))
    df.to_csv(os.path.join(RESULT_DIR, "oos_validation_summary.csv"), index=False, encoding="utf-8-sig")

    print("\n  ------------------------------------------------------------------")
    print(f"  검증구간 자체 재최적화 조합(참고) : {_fmt_params(test_best)}")
    print("   [검증구간 좌표별 sweep 최적값]")
    for name in ["disp", "roe", "mom", "sl"]:
        r0 = te["sweeps"][name][0]
        print(f"     · {label[name]}: 최적={r0['value']} (CAGR {r0['CAGR']*100:.2f}%)")

    print("\n" + "=" * 100)
    print(" ⚠ 정직성: 훈련/검증 1분할은 한 가지 방식일 뿐(구간을 다르게 나누면 결과가 달라질 수 있음).")
    print("=" * 100)
    return tr, te, rows


if __name__ == "__main__":
    main()
