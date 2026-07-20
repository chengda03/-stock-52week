# -*- coding: utf-8 -*-
"""
backtest_pyramid_compare.py
===========================
[실험] '새 기준선' 위에서 무제한 불타기에 '단계 수 상한'을 도입하면 어떻게 되나.
쏠림을 '비중%'가 아니라 '불타기 단계 수'로 제한한다(인계문서의 비중%상한 기각 실험과는 별개 방식).

  · new_base   : 불타기 무제한       (현재 확정 = 원본 strategy_core)
  · pyramid10  : 최대 10단계(≈+34%)
  · pyramid15  : 최대 15단계(≈+56%)
  · pyramid20  : 최대 20단계(≈+81%)

차이는 오직 check_pyramid(불타기 판정) 하나뿐. 매수/매도/부분익절/불타기 실행(apply_pyramid,
1스텝 500만) 및 슬롯(20×500만)은 전부 원본 strategy_core와 동일.

계측: CAGR/MDD/Sharpe/Calmar/손익비/승률/총거래/불타기건수/최대단일종목비중/
     단일종목비중 30%초과 일수비율(인계문서 60.7%와 대조).
추가: new_base에서 '단계별로 몇 건이나 불타기했는지' 분포(원래 몇 단계까지 갔었나) 집계.

정직성: 단일 경로(약 6.5년) 백테스트. 인계문서의 비중%상한(40%상한 -8~30%p 손실)과
        트레이드오프를 비교. 과최적화 위험 유의.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

import data_layer
import strategy_core as sc_base
from strategy_core import StockSnapshot, Position
import strategy_core_pyramid10 as p10
import strategy_core_pyramid15 as p15
import strategy_core_pyramid20 as p20

INITIAL_CASH = 100_000_000
BUY_COST = 0.00115
SELL_COST = 0.00295
TRADE_START = pd.Timestamp("2020-01-01")
WEIGHT_THRESH = 0.30   # 단일종목비중 30% 초과 일수 집계 기준

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)
CSV_PATH = os.path.join(RESULT_DIR, "pyramid_cap_compare.csv")


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


def run_variant(check_pyramid_fn, tag: str, store) -> dict:
    td = store.trading_days
    close_v = store.close_v
    close_ff = store.close_ff
    c2c = store.code_to_col
    start_di = store.start_di

    cash = float(INITIAL_CASH)
    positions: dict[str, Position] = {}
    cost: dict[str, float] = {}
    daily_total = np.empty(len(td))

    n_buy = n_add = n_partial = n_final = 0
    wins = 0
    gross_w = gross_l = 0.0
    max_weight = 0.0
    days_over_thresh = 0
    eval_days = 0
    peak_steps: list[int] = []   # 청산된(또는 잔존) 포지션들이 도달한 최종 불타기 단계 수

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
                peak_steps.append(pos.pyramid_count)
                del positions[t]; del cost[t]; sold_today.add(t)

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

        # 2) 불타기 (변형 check_pyramid_fn 사용, 실행은 원본 apply_pyramid)
        if is_bull and positions:
            def cur_ret(t):
                col = c2c[t]; px = close_v[di, col]
                return px / positions[t].avg_price if np.isfinite(px) else -1.0
            for t in sorted(positions.keys(), key=cur_ret, reverse=True):
                col = c2c[t]; px = close_v[di, col]
                if not np.isfinite(px) or px <= 0:
                    continue
                pos = positions[t]
                if check_pyramid_fn(pos, float(px), today, is_bull, cash):
                    spent = sc_base.PYRAMID_ADD_AMOUNT_WON * (1.0 + BUY_COST)
                    if cash < spent:
                        continue
                    sc_base.apply_pyramid(pos, float(px), today)
                    cash -= spent; cost[t] += spent; n_add += 1

        # 3) 신규매수
        free = sc_base.NUM_SLOTS - len(positions)
        if is_bull:
            uni = store.get_universe(td[di])
            cands: list[StockSnapshot] = []
            for t in uni:
                if t in positions or t in sold_today:
                    continue
                col = c2c[t]; px = close_v[di, col]
                if not np.isfinite(px) or px <= 0:
                    continue
                snap = _build_snap(store, col, di)
                ok, _ = sc_base.check_buy_conditions(snap, is_bull)
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
                cost[snap.ticker] = spent
                cash -= spent; n_buy += 1; free -= 1

        # 4) 일별 평가 + 최대 단일비중 + 30%초과 일수
        hv = 0.0; vmax = 0.0
        for t, pos in positions.items():
            col = c2c[t]; p = close_v[di, col]
            if not np.isfinite(p):
                p = close_ff[di, col]
                if not np.isfinite(p):
                    p = pos.avg_price
            v = p * pos.shares; hv += v
            if v > vmax:
                vmax = v
        total = cash + hv
        daily_total[di] = total
        if td[di] >= TRADE_START and total > 0:
            w = vmax / total
            if w > max_weight:
                max_weight = w
            if w > WEIGHT_THRESH:
                days_over_thresh += 1
            eval_days += 1

    # 잔존 포지션의 단계 수도 분포에 포함
    for pos in positions.values():
        peak_steps.append(pos.pyramid_count)

    m = _metrics(daily_total, td)
    n_sell = n_partial + n_final
    m.update({
        "tag": tag,
        "PL": abs(gross_w / gross_l) if gross_l != 0 else float("inf"),
        "win_rate": wins / n_sell if n_sell else 0.0,
        "n_buy": n_buy, "n_add": n_add, "n_partial": n_partial, "n_final": n_final,
        "n_trades": n_buy + n_add + n_partial + n_final,
        "max_weight": max_weight,
        "over30_ratio": days_over_thresh / eval_days if eval_days else 0.0,
        "peak_steps": peak_steps,
    })
    return m


def _row(m: dict) -> dict:
    return {
        "버전": m["tag"],
        "CAGR%": round(m["CAGR"] * 100, 2),
        "MDD%": round(m["MDD"] * 100, 2),
        "Sharpe": round(m["Sharpe"], 2),
        "Calmar": round(m["Calmar"], 2),
        "손익비": round(m["PL"], 2),
        "승률%": round(m["win_rate"] * 100, 1),
        "총거래": m["n_trades"],
        "불타기": m["n_add"],
        "최대단일비중%": round(m["max_weight"] * 100, 1),
        "비중30%초과일수%": round(m["over30_ratio"] * 100, 1),
    }


def main():
    store = data_layer.get_store()

    variants = [
        run_variant(sc_base.check_pyramid, "new_base(무제한)", store),
        run_variant(p10.check_pyramid, "pyramid10(≤10단계)", store),
        run_variant(p15.check_pyramid, "pyramid15(≤15단계)", store),
        run_variant(p20.check_pyramid, "pyramid20(≤20단계)", store),
    ]
    df = pd.DataFrame([_row(m) for m in variants])

    print("\n" + "=" * 118)
    print(f" 불타기 단계 상한 실험  ·  {variants[0]['start']} ~ {variants[0]['end']}"
          "  ·  1억 · 거래비용 반영 · 같은 데이터/같은 날")
    print("=" * 118)
    print(df.to_string(index=False))
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"\n  CSV 저장: {CSV_PATH}")

    # new_base의 '원래 몇 단계까지 불타기했나' 분포
    steps = np.array(variants[0]["peak_steps"])
    if len(steps):
        print("\n" + "-" * 118)
        print(" [new_base 불타기 단계 분포] 청산·잔존 포지션이 도달한 최종 불타기 단계 수 (총 포지션 %d개)" % len(steps))
        print("-" * 118)
        print(f"   0단계(불타기 안 함) {int((steps==0).sum())}건 · 1~9단계 {int(((steps>=1)&(steps<=9)).sum())}건 · "
              f"10~14단계 {int(((steps>=10)&(steps<=14)).sum())}건 · 15~19단계 {int(((steps>=15)&(steps<=19)).sum())}건 · "
              f"20단계 이상 {int((steps>=20).sum())}건")
        print(f"   최대 단계 = {int(steps.max())}단계 · 평균 = {steps.mean():.2f}단계 · "
              f"10단계 초과 포지션 {int((steps>10).sum())}건 ({(steps>10).mean()*100:.1f}%)")

    print("\n" + "=" * 118)
    print(" ⚠ 정직성: 단일 경로 백테스트. 인계문서 비중%상한(40%상한 -8~30%p)과 트레이드오프 비교 필요. 과최적화 주의.")
    print("=" * 118)
    return variants


if __name__ == "__main__":
    main()
