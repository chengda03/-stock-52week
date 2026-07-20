# -*- coding: utf-8 -*-
"""
backtest_margin_compare.py
==========================
[실험] '새 기준선'(2026-07-14 반영: 60일선 삭제 + 90일선 이격도 2.5%) 위에 저평가
'안전마진'(margin 10/20/30%)을 얹으면 성과가 어떻게 변하나.

    마진(%) = (적정가 - 현재가) / 현재가 × 100,   적정가 = ROE(%) × EPS
    · new_base : 마진 0% (적정가가 현재가보다 조금이라도 높으면 통과) = 원본 strategy_core(현재 확정)
    · margin10 : 마진 ≥ 10%
    · margin20 : 마진 ≥ 20%
    · margin30 : 마진 ≥ 30%

전부 같은 프로세스·같은 DataStore·같은 날짜로 실행 → 데이터 드리프트 0.
new_base는 방금 정식 반영된 strategy_core.check_buy_conditions를 그대로 쓴다.
변형 간 차이는 오로지 '저평가 마진 문턱' 하나뿐(나머지 조건·매도·불타기·부분익절·비용 동일).

계측(매매 영향 없음): backtest_disp_compare.py 와 동일 정의(재손절5일·통과개수·Jaccard·현금유휴).
정직성: 단일 경로(약 6.5년) 백테스트. 통과 종목이 급감하면 표본 불안정 + 실전 슬롯 미충원 문제.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

import data_layer
import strategy_core as sc_base            # 새 기준선(=new_base) + 매도/불타기/부분익절 규칙
from strategy_core import StockSnapshot, Position

import strategy_core_margin10 as vm10
import strategy_core_margin20 as vm20
import strategy_core_margin30 as vm30

INITIAL_CASH = 100_000_000
BUY_COST = 0.00115
SELL_COST = 0.00295
TRADE_START = pd.Timestamp("2020-01-01")
RESTOP_WINDOW = 5

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)
CSV_PATH = os.path.join(RESULT_DIR, "margin_filter_compare.csv")


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


def _metrics(daily_total: np.ndarray, td: pd.DatetimeIndex) -> dict:
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


def run_variant(check_buy_fn, tag: str, store) -> dict:
    td = store.trading_days
    close_v = store.close_v
    close_ff = store.close_ff
    c2c = store.code_to_col
    start_di = store.start_di

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
    cash_ratio_sum = 0.0
    eval_days = 0
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

        # 2) 불타기
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

        # 3) 신규매수 + 통과개수 계측
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
                snap = _build_snap(store, col, di)
                ok, _ = check_buy_fn(snap, is_bull)
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

        # 4) 일별 평가 + 현금유휴비율
        hv = 0.0
        for t, pos in positions.items():
            col = c2c[t]; p = close_v[di, col]
            if not np.isfinite(p):
                p = close_ff[di, col]
                if not np.isfinite(p):
                    p = pos.avg_price
            hv += p * pos.shares
        total = cash + hv
        daily_total[di] = total
        if td[di] >= TRADE_START and total > 0:
            cash_ratio_sum += cash / total
            eval_days += 1

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
        "cash_idle_ratio": cash_ratio_sum / eval_days if eval_days else 0.0,
        "bought_set": bought_set,
    })
    return m


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if (a | b) else 0.0


def _row(m: dict, base_set: set) -> dict:
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
        "newbase겹침(Jaccard)%": round(_jaccard(m["bought_set"], base_set) * 100, 1),
        "현금유휴%": round(m["cash_idle_ratio"] * 100, 1),
        "5일내재손절%": round(m["restop_5d_ratio"] * 100, 1),
    }


def main():
    store = data_layer.get_store()

    variants = [
        run_variant(sc_base.check_buy_conditions, "new_base(마진0%)", store),
        run_variant(vm10.check_buy_conditions, "margin10(마진≥10%)", store),
        run_variant(vm20.check_buy_conditions, "margin20(마진≥20%)", store),
        run_variant(vm30.check_buy_conditions, "margin30(마진≥30%)", store),
    ]

    base_set = variants[0]["bought_set"]
    df = pd.DataFrame([_row(m, base_set) for m in variants])

    print("\n" + "=" * 124)
    print(f" 저평가 마진 실험 (새 기준선 위)  ·  {variants[0]['start']} ~ {variants[0]['end']}"
          "  ·  1억 · 거래비용 반영 · 같은 데이터/같은 날(드리프트 0)")
    print("=" * 124)
    print(df.to_string(index=False))
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"\n  CSV 저장: {CSV_PATH}")

    print("\n" + "-" * 124)
    print(" [종목군 겹침 상세] new_base가 신규매수한 종목집합 대비")
    print("-" * 124)
    for m in variants:
        a = m["bought_set"]
        print(f"  {m['tag']:<20s} 매수종목수 {len(a):>4d} · 공통 {len(a & base_set):>4d} · "
              f"이 버전에만 {len(a - base_set):>4d} · Jaccard {_jaccard(a, base_set)*100:>5.1f}%")

    print("\n" + "=" * 124)
    print(" ⚠ 정직성: 단일 경로 백테스트. 통과 종목이 급감하면 표본 불안정 + 실전 슬롯 미충원 문제. 과최적화 위험.")
    print("=" * 124)
    return variants


if __name__ == "__main__":
    main()
