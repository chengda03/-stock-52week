# -*- coding: utf-8 -*-
"""
backtest_sl_compare.py
======================
[실험] '새 기준선' 위에서 하드손절 임계값만 바꾼다(-8/-10/-12/-15/-20%).

  · new_base : -12%  (현재 확정 = 원본 strategy_core.check_sell_condition)
  · sl8/10/15/20 : -8/-10/-15/-20%

매도판정은 원본과 동일하게 "먼저 닿는 쪽": ① 90일선 이탈 우선 → ② 하드손절.
→ 매도사유가 '하드손절'인 건수 = 90선으로는 안 걸렸는데 하드손절로만 '결정적'으로 잘린 케이스.

계측: CAGR/MDD/Sharpe/Calmar/손익비/승률/총거래/하드손절건수/90선이탈건수/평균 손실거래당 손실률.

정직성: 단일 경로(약 6.5년) 백테스트. '더 빡빡/느슨한 손절이 항상 좋은 게 아니다'라는
        트레이드오프가 이번에도 나타나는지 판단. 과최적화 위험 유의.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

import data_layer
import strategy_core as sc_base
from strategy_core import StockSnapshot, Position
import strategy_core_sl8 as sl8
import strategy_core_sl10 as sl10
import strategy_core_sl15 as sl15
import strategy_core_sl20 as sl20

INITIAL_CASH = 100_000_000
BUY_COST = 0.00115
SELL_COST = 0.00295
TRADE_START = pd.Timestamp("2020-01-01")

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)
CSV_PATH = os.path.join(RESULT_DIR, "hard_stop_compare.csv")


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


def run_variant(check_sell_fn, tag: str, store) -> dict:
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
    n_hardstop = n_ma_break = 0
    wins = 0
    gross_w = gross_l = 0.0
    loss_rets: list[float] = []    # 손실로 청산된 '전량매도'의 손실률(음수)

    for di in range(len(td)):
        today = td[di].date()
        if di < start_di:
            daily_total[di] = INITIAL_CASH
            continue

        is_bull = sc_base.is_bull_market(store.kospi_close_v[di], store.kospi_ma200_v[di])
        sold_today: set[str] = set()

        # 1) 전량매도 (변형 check_sell_fn 사용)
        for t in list(positions.keys()):
            col = c2c[t]; px = close_v[di, col]
            if not np.isfinite(px):
                continue
            pos = positions[t]; snap = _build_snap(store, col, di)
            s, why = check_sell_fn(pos, snap)
            if s:
                proceeds = px * pos.shares * (1.0 - SELL_COST)
                cash += proceeds
                pnl = proceeds - cost[t]
                ret = proceeds / cost[t] - 1.0
                if pnl > 0: wins += 1; gross_w += pnl
                else: gross_l += pnl; loss_rets.append(ret)
                n_final += 1
                if why and why.startswith("하드손절"):
                    n_hardstop += 1
                else:
                    n_ma_break += 1
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
        "n_hardstop": n_hardstop, "n_ma_break": n_ma_break,
        "avg_loss_ret": float(np.mean(loss_rets)) if loss_rets else 0.0,
        "n_loss": len(loss_rets),
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
        "하드손절": m["n_hardstop"],
        "90선이탈": m["n_ma_break"],
        "평균손실률%": round(m["avg_loss_ret"] * 100, 2),
    }


def main():
    store = data_layer.get_store()

    variants = [
        run_variant(sc_base.check_sell_condition, "new_base(-12%)", store),
        run_variant(sl8.check_sell_condition, "sl8(-8%)", store),
        run_variant(sl10.check_sell_condition, "sl10(-10%)", store),
        run_variant(sl15.check_sell_condition, "sl15(-15%)", store),
        run_variant(sl20.check_sell_condition, "sl20(-20%)", store),
    ]
    df = pd.DataFrame([_row(m) for m in variants])

    print("\n" + "=" * 116)
    print(f" 하드손절 임계값 실험 (-8/-10/-12/-15/-20%, 새 기준선 위)  ·  {variants[0]['start']} ~ {variants[0]['end']}"
          "  ·  1억 · 거래비용 반영 · 같은 데이터/같은 날")
    print("=" * 116)
    print(df.to_string(index=False))
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"\n  CSV 저장: {CSV_PATH}")

    print("\n" + "-" * 116)
    print(" [진단] 매도사유 구성 (하드손절 = 90선으로는 안 걸렸는데 하드손절로만 '결정적'으로 잘린 케이스)")
    print("-" * 116)
    for m in variants:
        tot = m["n_final"]
        print(f"   {m['tag']:<16s} 전량매도 {tot}건 = 하드손절 {m['n_hardstop']}건 "
              f"({m['n_hardstop']/max(tot,1)*100:.1f}%) + 90선이탈 {m['n_ma_break']}건 "
              f"({m['n_ma_break']/max(tot,1)*100:.1f}%) · 손실청산 {m['n_loss']}건 평균 {m['avg_loss_ret']*100:.2f}%")

    print("\n" + "=" * 116)
    print(" ⚠ 정직성: 단일 경로 백테스트. 손절 빡빡/느슨의 트레이드오프 판단 필요. 과최적화 위험 존재.")
    print("=" * 116)
    return variants


if __name__ == "__main__":
    main()
