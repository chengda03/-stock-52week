# -*- coding: utf-8 -*-
"""
backtest_mom_compare.py
=======================
[실험] '새 기준선'(60일선삭제+90일선이격도2.5%+ROE≥15%+거래대금30억+저평가ROE×EPS+시장필터,
20슬롯×500만·무제한불타기) 위에서 모멘텀 lookback 기간만 바꾼다.

  · new_base : 20일 모멘텀 (현재 확정 = 원본 strategy_core, store.mom_v)
  · mom10    : 10일 모멘텀
  · mom30    : 30일 모멘텀
  · mom50    : 50일 모멘텀
  · mom60    : 60일 모멘텀

바뀌는 것은 딱 두 곳: 매수조건 "N일 수익률>0"과 매수우선순위 "N일 수익률 높은 순".
구현: lookback별 모멘텀 패널 mom_N = close/close.shift(N)-1 을 만들어 snapshot.momentum_20d에 주입 →
원본 check_buy_conditions(>0 판정)과 rank_by_momentum(정렬)을 그대로 재사용(공정 비교). N=20이면 store.mom_v와 동일.

계측: CAGR/MDD/Sharpe/Calmar/손익비/승률/총거래/신규매수/통과평균종목수/
     20일기준 매수종목군과의 Jaccard/매수후5일내재손절비율.

정직성: 단일 경로(약 6.5년) 백테스트. '더 늦은 진입=악화'(정배열/90일선) vs '적절한 필터강화=개선'(이격도2.5%)
        중 어느 패턴에 가까운지 판단. 과최적화 위험 유의.
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
TRADE_START = pd.Timestamp("2020-01-01")
RESTOP_WINDOW = 5

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)
CSV_PATH = os.path.join(RESULT_DIR, "momentum_lookback_compare.csv")


def build_mom_panel(store, lookback: int) -> np.ndarray:
    """N일 수익률 패널: close/close.shift(N) - 1.  (N=20이면 store.mom_v와 동일)"""
    close_df = pd.DataFrame(store.close_v)
    return (close_df / close_df.shift(lookback) - 1.0).to_numpy(dtype=float)


def _build_snap(store, col, di, mom_panel) -> StockSnapshot:
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
        momentum_20d=float(mom_panel[di, col]),   # ★lookback별 모멘텀 주입
        trading_value_20d_avg=float(store.tv_v[di, col]),
    )


def _metrics(daily_total, td) -> dict:
    s = pd.Series(daily_total, index=td)
    s = s[s.index >= TRADE_START]
    final = float(s.iloc[-1])
    years = (s.index[-1] - s.index[0]).days / 365.25
    cagr = (final / INITIAL_CASH) ** (1.0 / max(years, 1e-9)) - 1.0
    peak = s.cummax()
    mdd = float(((s - peak) / peak).min())
    rets = s.pct_change().fillna(0)
    sharpe = float((rets.mean() / (rets.std() + 1e-12)) * math.sqrt(252))
    calmar = cagr / abs(mdd) if mdd != 0 else float("inf")
    return {"final": final, "total_ret": final / INITIAL_CASH - 1.0,
            "CAGR": cagr, "MDD": mdd, "Sharpe": sharpe, "Calmar": calmar,
            "start": str(s.index[0].date()), "end": str(s.index[-1].date())}


def run_variant(lookback: int, tag: str, store) -> dict:
    td = store.trading_days
    close_v = store.close_v
    close_ff = store.close_ff
    c2c = store.code_to_col
    start_di = store.start_di
    mom_panel = build_mom_panel(store, lookback)

    cash = float(INITIAL_CASH)
    positions: dict[str, Position] = {}
    cost: dict[str, float] = {}
    entry_di: dict[str, int] = {}
    daily_total = np.empty(len(td))

    n_buy = n_add = n_partial = n_final = 0
    wins = 0
    gross_w = gross_l = 0.0
    restop_5d = 0
    pass_cnt_sum = 0
    pass_days = 0
    bought_set: set[str] = set()

    for di in range(len(td)):
        today = td[di].date()
        if di < start_di:
            daily_total[di] = INITIAL_CASH
            continue

        is_bull = sc_base.is_bull_market(store.kospi_close_v[di], store.kospi_ma200_v[di])
        sold_today: set[str] = set()

        # 1) 전량매도
        for t in list(positions.keys()):
            col = c2c[t]; px = close_v[di, col]
            if not np.isfinite(px):
                continue
            pos = positions[t]; snap = _build_snap(store, col, di, mom_panel)
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

        # 2) 불타기 (무제한, 원본 로직)
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

        # 3) 신규매수 (모멘텀 N일: 조건 N일>0 + 정렬 N일 높은 순)
        free = sc_base.NUM_SLOTS - len(positions)
        if is_bull:
            uni = store.get_universe(td[di])
            day_uni = 0; day_pass = 0
            cands: list[StockSnapshot] = []
            for t in uni:
                col = c2c[t]; px = close_v[di, col]
                if not np.isfinite(px) or px <= 0:
                    continue
                day_uni += 1
                snap = _build_snap(store, col, di, mom_panel)
                ok, _ = sc_base.check_buy_conditions(snap, is_bull)
                if ok:
                    day_pass += 1
                    if t not in positions and t not in sold_today:
                        cands.append(snap)
            if day_uni > 0:
                pass_cnt_sum += day_pass
                pass_days += 1
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
                cost[snap.ticker] = spent
                entry_di[snap.ticker] = di
                cash -= spent; n_buy += 1; free -= 1
                bought_set.add(snap.ticker)

        # 4) 일별 평가
        hv = 0.0
        for t, pos in positions.items():
            col = c2c[t]; p = close_v[di, col]
            if not np.isfinite(p):
                p = close_ff[di, col]
                if not np.isfinite(p):
                    p = pos.avg_price
            hv += p * pos.shares
        daily_total[di] = cash + hv

    m = _metrics(daily_total, td)
    n_sell = n_partial + n_final
    m.update({
        "tag": tag,
        "PL": abs(gross_w / gross_l) if gross_l != 0 else float("inf"),
        "win_rate": wins / n_sell if n_sell else 0.0,
        "n_buy": n_buy, "n_add": n_add, "n_partial": n_partial, "n_final": n_final,
        "n_trades": n_buy + n_add + n_partial + n_final,
        "restop_5d_ratio": restop_5d / n_buy if n_buy else 0.0,
        "avg_pass_cnt": pass_cnt_sum / pass_days if pass_days else 0.0,
        "bought_set": bought_set,
    })
    return m


def _row(m: dict, base_set: set) -> dict:
    a = m["bought_set"]
    jac = len(a & base_set) / len(a | base_set) * 100 if (a | base_set) else 0.0
    return {
        "버전": m["tag"],
        "CAGR%": round(m["CAGR"] * 100, 2),
        "MDD%": round(m["MDD"] * 100, 2),
        "Sharpe": round(m["Sharpe"], 2),
        "Calmar": round(m["Calmar"], 2),
        "손익비": round(m["PL"], 2),
        "승률%": round(m["win_rate"] * 100, 1),
        "총거래": m["n_trades"],
        "신규매수": m["n_buy"],
        "통과평균개수": round(m["avg_pass_cnt"], 1),
        "20일기준Jaccard%": round(jac, 1),
        "5일내재손절%": round(m["restop_5d_ratio"] * 100, 1),
    }


def main():
    store = data_layer.get_store()

    specs = [(20, "new_base(20일)"), (10, "mom10(10일)"), (30, "mom30(30일)"),
             (50, "mom50(50일)"), (60, "mom60(60일)")]
    variants = [run_variant(lb, tag, store) for lb, tag in specs]
    base_set = variants[0]["bought_set"]

    df = pd.DataFrame([_row(m, base_set) for m in variants])

    print("\n" + "=" * 122)
    print(f" 모멘텀 lookback 실험 (10/20/30/50/60일)  ·  {variants[0]['start']} ~ {variants[0]['end']}"
          "  ·  1억 · 거래비용 반영 · 같은 데이터/같은 날")
    print("=" * 122)
    print(df.to_string(index=False))
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"\n  CSV 저장: {CSV_PATH}")

    print("\n" + "=" * 122)
    print(" ⚠ 정직성: 단일 경로 백테스트. 20일이 최적점 근처인지/양극단 경향성 판단 필요. 과최적화 위험 존재.")
    print("=" * 122)
    return variants


if __name__ == "__main__":
    main()
