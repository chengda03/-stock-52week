# -*- coding: utf-8 -*-
"""
backtest_v3_combined.py
=======================
[실험] 4가지 변경을 '동시에' 적용한 통합본(strategy_core_v3_combined) 백테스트 및 분해.

비교 대상 (같은 프로세스·같은 DataStore·같은 날짜 → 데이터 드리프트 0):
  · base        : 원본 확정 (60선+90선, +8%익절, KOSPI 단독필터)      = 원본 strategy_core
  · v3_mid      : (1)60선삭제 + (2)이격도2.5% + (4)+6%익절  … 시장필터는 기존 KOSPI 단독
  · v3_combined : 위 3가지 + (3)코스피+코스닥 동시 시장필터

  → v3_mid vs v3_combined 의 차이 = 오직 '동시 시장필터'의 순효과(요소 분해).

★ 시장필터 적용 규칙(스펙 그대로):
  - 신규매수 게이트 : v3_combined는 'KOSPI>200 AND KOSDAQ>200'. 그 외는 KOSPI 단독.
  - 불타기 게이트   : 세 버전 모두 KOSPI 단독(원본 로직) — 동시필터와 무관.
  - 매도            : 항상 작동(시장필터 무관).

★ 엔진 규칙은 backtest_v3.py 와 100% 동일: 1억 · 매수0.115%/매도0.295% ·
  전량매도(90선/-12%) → 부분익절 → 불타기(+3%복리) → 신규매수(빈슬롯20·슬롯500만·정수주).

계측: CAGR/MDD/Sharpe/Calmar/손익비/승률/총거래/신규매수/부분익절/5일내재손절/통과평균개수/현금유휴.
정직성: 단일 경로(약 6.5년) 백테스트 + 여러 조건 동시변경. 요소 상호작용/과최적화 주의.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

import data_layer
import strategy_core as sc_base
from strategy_core import StockSnapshot, Position
import strategy_core_v3_combined as sv
from backtest_roe_eps_event import load_index

INITIAL_CASH = 100_000_000
BUY_COST = 0.00115
SELL_COST = 0.00295
TRADE_START = pd.Timestamp("2020-01-01")
RESTOP_WINDOW = 5
MARKET_FILTER_MA_DAYS = 200

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)
CSV_PATH = os.path.join(RESULT_DIR, "v3_combined_compare.csv")


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


def _kosdaq_arrays(store):
    """KOSDAQ 종가/200일선을 store의 KOSPI와 '동일 방식'(reindex·ffill·bfill·rolling200·min_periods=1)으로 계산."""
    kq = load_index("kosdaq_index").reindex(store.trading_days).ffill().bfill()
    close = kq.to_numpy(dtype=float)
    ma200 = kq.rolling(MARKET_FILTER_MA_DAYS, min_periods=1).mean().to_numpy(dtype=float)
    return close, ma200


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


def run_variant(check_buy_fn, check_partial_fn, apply_partial_fn, dual_filter: bool,
                tag: str, store, kospi_bull_v, dual_bull_v) -> dict:
    """
    dual_filter=False → 신규매수 게이트 = KOSPI 단독(kospi_bull_v)  (base·v3_mid)
    dual_filter=True  → 신규매수 게이트 = 코스피+코스닥 동시(dual_bull_v)  (v3_combined)
    불타기 게이트는 항상 KOSPI 단독(kospi_bull_v).
    """
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

    for di in range(len(td)):
        today = td[di].date()
        if di < start_di:
            daily_total[di] = INITIAL_CASH
            continue

        is_bull_pyr = bool(kospi_bull_v[di])                                   # 불타기 게이트(KOSPI 단독)
        is_bull_buy = bool(dual_bull_v[di]) if dual_filter else bool(kospi_bull_v[di])  # 신규매수 게이트
        sold_today: set[str] = set()

        # 1) 전량매도 (시장필터 무관)
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

        # 1-2) 부분익절 (트리거만 변형)
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

        # 2) 불타기 (KOSPI 단독 게이트)
        if is_bull_pyr and positions:
            def cur_ret(t):
                col = c2c[t]; px = close_v[di, col]
                return px / positions[t].avg_price if np.isfinite(px) else -1.0
            for t in sorted(positions.keys(), key=cur_ret, reverse=True):
                col = c2c[t]; px = close_v[di, col]
                if not np.isfinite(px) or px <= 0:
                    continue
                pos = positions[t]
                if sc_base.check_pyramid(pos, float(px), today, is_bull_pyr, cash):
                    spent = sc_base.PYRAMID_ADD_AMOUNT_WON * (1.0 + BUY_COST)
                    if cash < spent:
                        continue
                    sc_base.apply_pyramid(pos, float(px), today)
                    cash -= spent; cost[t] += spent; n_add += 1

        # 3) 신규매수 (신규매수 게이트) + 통과개수 계측
        free = sc_base.NUM_SLOTS - len(positions)
        if is_bull_buy:
            uni = store.get_universe(td[di])
            day_uni = 0; day_pass = 0
            cands: list[StockSnapshot] = []
            for t in uni:
                col = c2c[t]; px = close_v[di, col]
                if not np.isfinite(px) or px <= 0:
                    continue
                day_uni += 1
                snap = _build_snap(store, col, di)
                ok, _ = check_buy_fn(snap, True)   # 게이트는 이미 통과 → 매수조건만 평가
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
        "부분익절건수": m["n_partial"],
        "5일내재손절%": round(m["restop_5d_ratio"] * 100, 1),
        "통과평균개수": round(m["avg_pass_cnt"], 1),
        "현금유휴%": round(m["cash_idle_ratio"] * 100, 1),
    }


def main():
    store = data_layer.get_store()
    kospi_close = store.kospi_close_v
    kospi_ma200 = store.kospi_ma200_v
    kosdaq_close, kosdaq_ma200 = _kosdaq_arrays(store)

    kospi_bull_v = kospi_close > kospi_ma200
    kosdaq_bull_v = kosdaq_close > kosdaq_ma200
    dual_bull_v = kospi_bull_v & kosdaq_bull_v

    variants = [
        run_variant(sc_base.check_buy_conditions, sc_base.check_partial_exit,
                    sc_base.apply_partial_exit, False, "base(원본확정)",
                    store, kospi_bull_v, dual_bull_v),
        run_variant(sv.check_buy_conditions, sv.check_partial_exit,
                    sv.apply_partial_exit, False, "v3_mid(3변경·KOSPI단독필터)",
                    store, kospi_bull_v, dual_bull_v),
        run_variant(sv.check_buy_conditions, sv.check_partial_exit,
                    sv.apply_partial_exit, True, "v3_combined(4변경·동시필터)",
                    store, kospi_bull_v, dual_bull_v),
    ]

    df = pd.DataFrame([_row(m) for m in variants])

    print("\n" + "=" * 122)
    print(f" 통합 변형(4가지) 실험  ·  {variants[0]['start']} ~ {variants[0]['end']}"
          "  ·  1억 · 거래비용 반영 · 같은 데이터/같은 날(드리프트 0)")
    print("=" * 122)
    print(df.to_string(index=False))
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"\n  CSV 저장: {CSV_PATH}")

    # 시장필터 진단: 코스닥이 신규매수를 '추가로' 얼마나 막는가
    mask = store.trading_days >= TRADE_START
    n_days = int(mask.sum())
    kospi_bull_days = int((kospi_bull_v & mask).sum())
    dual_bull_days = int((dual_bull_v & mask).sum())
    blocked_by_kosdaq = int((kospi_bull_v & ~kosdaq_bull_v & mask).sum())
    print("\n" + "-" * 122)
    print(" [시장필터 진단] 매매기간(2020~) 거래일 기준")
    print("-" * 122)
    print(f"   전체 거래일 {n_days}일")
    print(f"   KOSPI 단독 강세일(신규매수 허용, base·v3_mid) : {kospi_bull_days}일 ({kospi_bull_days/n_days*100:.1f}%)")
    print(f"   코스피+코스닥 동시 강세일(v3_combined 허용)    : {dual_bull_days}일 ({dual_bull_days/n_days*100:.1f}%)")
    print(f"   → 코스닥 때문에 '추가로' 신규매수 막힌 날        : {blocked_by_kosdaq}일 "
          f"(KOSPI 강세일의 {blocked_by_kosdaq/max(kospi_bull_days,1)*100:.1f}%)")

    print("\n" + "=" * 122)
    print(" ⚠ 정직성: 4개 조건 동시변경 + 단일 경로 백테스트. 요소 상호작용/과최적화 위험 존재.")
    print("=" * 122)
    return variants


if __name__ == "__main__":
    main()
