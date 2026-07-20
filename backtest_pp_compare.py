# -*- coding: utf-8 -*-
"""
backtest_pp_compare.py
======================
[실험] 부분익절 트리거를 "+8%@50%"(현재 확정) → "+6%@50%"로 낮추면 성과가 어떻게 변하나.

비교 대상 (같은 프로세스·같은 DataStore·같은 날짜 → 데이터 드리프트 0):
  · base : 부분익절 +8% 도달시 50% 매도 (현재 확정 = 원본 strategy_core)
  · pp6  : 부분익절 +6% 도달시 50% 매도 (strategy_core_pp6)

두 버전의 유일한 차이는 '부분익절 트리거 비율'(0.08 vs 0.06) 하나뿐이다.
매수조건(base 7조건)·매도조건·불타기·사이징·비용은 전부 원본 strategy_core.py 재사용.

★ 엔진 규칙은 backtest_v3.py / bt_valuation_engine.py 와 100% 동일:
  - 기간 2020-01-01 ~ 데이터최신, 초기 1억, 매수 0.115% / 매도 0.295%
  - 강세장(KOSPI 200일선 위)에서만 신규매수/불타기
  - 전량매도(90일선/-12%) → 부분익절 → 불타기(+3%복리) → 신규매수(빈 슬롯)
  - 슬롯 20개, 슬롯당 500만원(정수주)

계측: CAGR/MDD/Sharpe/Calmar/손익비/승률/총거래/부분익절건수/전량매도건수/5일내 재손절비율.

정직성: 단일 경로(약 6.5년) 백테스트. 표본이 하나뿐이라 작은 차이는 노이즈일 수 있다.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

import data_layer
import strategy_core as sc_base
from strategy_core import StockSnapshot, Position
import strategy_core_pp6 as vpp6

INITIAL_CASH = 100_000_000
BUY_COST = 0.00115
SELL_COST = 0.00295
TRADE_START = pd.Timestamp("2020-01-01")
RESTOP_WINDOW = 5

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)
CSV_PATH = os.path.join(RESULT_DIR, "partial_exit_compare.csv")


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


def run_variant(check_partial_fn, apply_partial_fn, tag: str, store) -> dict:
    """부분익절 판정/실행 함수만 갈아끼워 백테스트. (그 외 규칙은 전부 원본 strategy_core)"""
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

    for di in range(len(td)):
        today = td[di].date()
        if di < start_di:
            daily_total[di] = INITIAL_CASH
            continue

        is_bull = sc_base.is_bull_market(store.kospi_close_v[di], store.kospi_ma200_v[di])
        sold_today: set[str] = set()

        # 1) 전량매도 (90일선/-12%)
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

        # 1-2) 부분익절 (트리거만 변형, 50% 매도)
        for t in list(positions.keys()):
            col = c2c[t]; px = close_v[di, col]
            if not np.isfinite(px) or px <= 0:
                continue
            pos = positions[t]
            if check_partial_fn(pos, float(px)):
                before = pos.shares
                sell_sh = apply_partial_fn(pos, float(px))
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

        # 3) 신규매수 (base 7조건)
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
                entry_di[snap.ticker] = di
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
        "n_sell": n_sell,
        "n_trades": n_buy + n_add + n_partial + n_final,
        "restop_5d": restop_5d,
        "restop_5d_ratio": restop_5d / n_buy if n_buy else 0.0,
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
        "부분익절건수": m["n_partial"],
        "전량매도건수": m["n_final"],
        "5일내재손절%": round(m["restop_5d_ratio"] * 100, 1),
    }


def main():
    store = data_layer.get_store()

    variants = [
        run_variant(sc_base.check_partial_exit, sc_base.apply_partial_exit,
                    "base(+8%@50%)", store),
        run_variant(vpp6.check_partial_exit, vpp6.apply_partial_exit,
                    "pp6(+6%@50%)", store),
    ]

    df = pd.DataFrame([_row(m) for m in variants])

    print("\n" + "=" * 104)
    print(f" 부분익절 트리거 실험 (+8% vs +6%)  ·  {variants[0]['start']} ~ {variants[0]['end']}"
          "  ·  1억 · 거래비용 반영 · 같은 데이터/같은 날(드리프트 0)")
    print("=" * 104)
    print(df.to_string(index=False))
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"\n  CSV 저장: {CSV_PATH}")

    # 승률 분해(부분익절/전량매도별 승리)를 참고로 출력
    print("\n" + "-" * 104)
    print(" [참고] 승률 분모 = 부분익절 + 전량매도. 부분익절은 진입가 대비 트리거라 사실상 항상 '이익 실현'.")
    for m in variants:
        print(f"   {m['tag']:<14s} 승률분모(매도이벤트) {m['n_sell']:>4d}  "
              f"= 부분익절 {m['n_partial']:>4d} + 전량매도 {m['n_final']:>4d}  → 승률 {m['win_rate']*100:.1f}%")

    print("\n" + "=" * 104)
    print(" ⚠ 정직성: 단일 경로 백테스트. 작은 차이는 노이즈일 수 있으니 '왜'가 설명되는지 함께 볼 것.")
    print("=" * 104)
    return variants


if __name__ == "__main__":
    main()
