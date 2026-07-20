# -*- coding: utf-8 -*-
"""
backtest_slot_compare.py
========================
[실험] '새 기준선' 위에서 매수조건은 그대로 두고 '슬롯 구조'만 바꾸면 어떻게 되나.
같은 자본(1억)을 더 적은 종목에 나눠 담는 방식으로 슬롯 미충원 문제에 접근한다.

  · new_base : 20슬롯 × 500만원   (현재 확정 = 원본 strategy_core)
  · slot15   : 15슬롯 × 667만원   (strategy_core_slot15)
  · slot10   : 10슬롯 × 1000만원  (strategy_core_slot10)

세 버전 모두 총 1억원. 매수/매도/불타기/부분익절 '판정'은 전부 원본 strategy_core와 동일.
차이는 오직 (슬롯 개수, 슬롯당 금액, 불타기 1스텝 금액) 세 값뿐.
  ★ 불타기 1스텝 금액 = 슬롯 금액(원본 설계). 슬롯을 키우면 불타기도 커진다 → 쏠림 관찰 포인트.

계측: CAGR/MDD/Sharpe/Calmar/손익비/승률/총거래/신규매수/불타기건수/슬롯미충원율/최대단일종목비중.
정직성: 단일 경로(약 6.5년) 백테스트. 트레이드오프(집중도↑ 위험) 여부를 명확히 볼 것. 과최적화 주의.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

import data_layer
import strategy_core as sc_base
from strategy_core import StockSnapshot, Position, PYRAMID_TRIGGER_PCT
import strategy_core_slot15 as s15
import strategy_core_slot10 as s10

INITIAL_CASH = 100_000_000
BUY_COST = 0.00115
SELL_COST = 0.00295
TRADE_START = pd.Timestamp("2020-01-01")

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)
CSV_PATH = os.path.join(RESULT_DIR, "slot_structure_compare.csv")


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


def _check_pyramid(pos, price, cur_date, is_bull, cash, pyramid_add) -> bool:
    """strategy_core.check_pyramid과 동일 로직 + 현금게이트/스텝금액만 변형값 사용."""
    if not is_bull:
        return False
    if pos.last_pyramid_date == cur_date:
        return False
    if cash < pyramid_add:
        return False
    next_step = pos.pyramid_count + 1
    trigger = pos.entry_price * ((1 + PYRAMID_TRIGGER_PCT) ** next_step)
    return price >= trigger


def _apply_pyramid(pos, price, cur_date, pyramid_add) -> None:
    """strategy_core.apply_pyramid과 동일하되 추가금액만 변형값 사용."""
    add_shares = pyramid_add / price
    old_cost = pos.avg_price * pos.shares
    new_cost = old_cost + pyramid_add
    new_shares = pos.shares + add_shares
    pos.avg_price = new_cost / new_shares
    pos.shares = new_shares
    pos.pyramid_count += 1
    pos.last_pyramid_date = cur_date


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


def run_variant(num_slots: int, slot_amount: float, pyramid_add: float, tag: str, store) -> dict:
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
    bull_days = 0
    slots_unfilled_days = 0

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
                del positions[t]; del cost[t]; sold_today.add(t)

        # 1-2) 부분익절 (+8%@50%)
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

        # 2) 불타기 (슬롯금액=불타기금액 변형값 사용)
        if is_bull and positions:
            def cur_ret(t):
                col = c2c[t]; px = close_v[di, col]
                return px / positions[t].avg_price if np.isfinite(px) else -1.0
            for t in sorted(positions.keys(), key=cur_ret, reverse=True):
                col = c2c[t]; px = close_v[di, col]
                if not np.isfinite(px) or px <= 0:
                    continue
                pos = positions[t]
                if _check_pyramid(pos, float(px), today, is_bull, cash, pyramid_add):
                    spent = pyramid_add * (1.0 + BUY_COST)
                    if cash < spent:
                        continue
                    _apply_pyramid(pos, float(px), today, pyramid_add)
                    cash -= spent; cost[t] += spent; n_add += 1

        # 3) 신규매수 (빈 슬롯) — 슬롯 개수/금액 변형값 사용
        free = num_slots - len(positions)
        if is_bull:
            bull_days += 1
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
                sh = int(slot_amount // px)
                if sh <= 0:
                    continue
                spent = sh * px * (1.0 + BUY_COST)
                if spent > cash:
                    continue
                positions[snap.ticker] = Position(ticker=snap.ticker, entry_price=px,
                                                  avg_price=px, shares=float(sh), entry_date=today)
                cost[snap.ticker] = spent
                cash -= spent; n_buy += 1; free -= 1
            if len(positions) < num_slots:
                slots_unfilled_days += 1

        # 4) 일별 평가 + 최대 단일종목 비중
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

    m = _metrics(daily_total, td)
    n_sell = n_partial + n_final
    m.update({
        "tag": tag,
        "PL": abs(gross_w / gross_l) if gross_l != 0 else float("inf"),
        "win_rate": wins / n_sell if n_sell else 0.0,
        "n_buy": n_buy, "n_add": n_add, "n_partial": n_partial, "n_final": n_final,
        "n_trades": n_buy + n_add + n_partial + n_final,
        "max_weight": max_weight,
        "bull_days": bull_days,
        "slots_unfilled_days": slots_unfilled_days,
        "unfilled_rate": slots_unfilled_days / bull_days if bull_days else 0.0,
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
        "신규매수": m["n_buy"],
        "불타기": m["n_add"],
        "슬롯미충원율%": round(m["unfilled_rate"] * 100, 1),
        "최대단일비중%": round(m["max_weight"] * 100, 1),
    }


def main():
    store = data_layer.get_store()

    variants = [
        run_variant(sc_base.NUM_SLOTS, sc_base.SLOT_AMOUNT_WON, sc_base.PYRAMID_ADD_AMOUNT_WON,
                    "new_base(20슬롯×500만)", store),
        run_variant(s15.NUM_SLOTS, s15.SLOT_AMOUNT_WON, s15.PYRAMID_ADD_AMOUNT_WON,
                    "slot15(15슬롯×667만)", store),
        run_variant(s10.NUM_SLOTS, s10.SLOT_AMOUNT_WON, s10.PYRAMID_ADD_AMOUNT_WON,
                    "slot10(10슬롯×1000만)", store),
    ]
    df = pd.DataFrame([_row(m) for m in variants])

    print("\n" + "=" * 118)
    print(f" 슬롯 구조 실험 (20 vs 15 vs 10, 새 기준선·매수조건 동일)  ·  {variants[0]['start']} ~ {variants[0]['end']}"
          "  ·  1억 · 거래비용 반영 · 같은 데이터/같은 날")
    print("=" * 118)
    print(df.to_string(index=False))
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"\n  CSV 저장: {CSV_PATH}")

    print("\n" + "-" * 118)
    print(" [진단] 강세장일 대비 '슬롯 미충원(보유<슬롯수)'일수")
    print("-" * 118)
    for m in variants:
        print(f"   {m['tag']:<24s} 강세장 {m['bull_days']}일 중 미충원 {m['slots_unfilled_days']}일 "
              f"({m['unfilled_rate']*100:.1f}%)  ·  최대단일종목비중 {m['max_weight']*100:.1f}%")

    print("\n" + "=" * 118)
    print(" ⚠ 정직성: 단일 경로 백테스트. 슬롯↓ → 미충원↓ 이득 vs 집중도↑ 위험의 트레이드오프. 과최적화 주의.")
    print("=" * 118)
    return variants


if __name__ == "__main__":
    main()
