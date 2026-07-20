# -*- coding: utf-8 -*-
"""
backtest_roe_compare.py
=======================
[실험] '새 기준선'(60일선 삭제 + 90일선 이격도 2.5%, 거래대금 30억 유지) 위에서
ROE 퀄리티 문턱을 15% → 10%로 낮추면 어떻게 되나.

  · new_base : ROE ≥ 15%  (현재 확정 = 원본 strategy_core)
  · roe10    : ROE ≥ 10%  (strategy_core_roe10)

두 버전의 유일한 차이는 ROE 문턱(15% vs 10%) 하나뿐.

구성:
  A) [백테스트 전] ROE 15% vs 10% 통과 종목군 분석
     - Jaccard, 추가 편입 종목 수(= ROE 10~15% 구간)
     - '추가 편입 종목'의 평균 시총·주가·거래대금·20일모멘텀 vs '기존 ROE≥15% 종목'
       (모멘텀 비교: 매수우선순위가 모멘텀이라, 이 구간이 실제 슬롯을 얼마나 차지할지 가늠)
  B) [백테스트] 같은 프로세스·같은 DataStore·같은 날짜로 두 버전 실행(드리프트 0)

정직성: 단일 경로(약 6.5년) 백테스트 + 조건 누적 상태에서의 추가 변경. 상호작용/과최적화 주의.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

import data_layer
import strategy_core as sc_base
from strategy_core import StockSnapshot, Position
import strategy_core_roe10 as vroe

INITIAL_CASH = 100_000_000
BUY_COST = 0.00115
SELL_COST = 0.00295
TRADE_START = pd.Timestamp("2020-01-01")
RESTOP_WINDOW = 5
ROE15 = 15.0
ROE10 = 10.0

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)
CSV_PATH = os.path.join(RESULT_DIR, "roe_filter_compare.csv")


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


def _passes_non_roe(snap) -> bool:
    """ROE 문턱을 '제외한' 나머지 매수조건(저평가·이격도·모멘텀·거래대금)을 모두 만족하는가."""
    if not (snap.roe_pct == snap.roe_pct):   # nan 체크
        return False
    if not (snap.price < snap.roe_pct * snap.eps):
        return False
    if snap.ma_sell and snap.ma_sell > 0:
        disp = (snap.price - snap.ma_sell) / snap.ma_sell * 100.0
    else:
        disp = float("-inf")
    if not (disp >= sc_base.BUY_DISPARITY_MIN_PCT):
        return False
    if not (snap.momentum_20d > 0):
        return False
    if not (snap.trading_value_20d_avg >= sc_base.TRADING_VALUE_MIN_WON):
        return False
    return True


def analyze_filters(store) -> None:
    td = store.trading_days
    c2c = store.code_to_col
    start_di = store.start_di

    set15: set[str] = set()
    set10: set[str] = set()
    inc_mc = []; inc_px = []; inc_tv = []; inc_mom = []   # ROE≥15 (기존)
    ext_mc = []; ext_px = []; ext_tv = []; ext_mom = []   # ROE 10~15 (추가)

    for di in range(start_di, len(td)):
        if not sc_base.is_bull_market(store.kospi_close_v[di], store.kospi_ma200_v[di]):
            continue
        for t in store.get_universe(td[di]):
            col = c2c[t]
            px = store.close_v[di, col]
            if not np.isfinite(px) or px <= 0:
                continue
            snap = _build_snap(store, col, di)
            if not _passes_non_roe(snap):
                continue
            roe = snap.roe_pct
            mc = store.marcap_v[di, col]
            if roe >= ROE15:
                set15.add(t); set10.add(t)
                inc_mc.append(mc); inc_px.append(px); inc_tv.append(snap.trading_value_20d_avg); inc_mom.append(snap.momentum_20d)
            elif roe >= ROE10:
                set10.add(t)
                ext_mc.append(mc); ext_px.append(px); ext_tv.append(snap.trading_value_20d_avg); ext_mom.append(snap.momentum_20d)

    jac = len(set15 & set10) / len(set15 | set10) * 100 if (set15 or set10) else 0.0

    def _avg(a):
        return float(np.nanmean(a)) if a else float("nan")

    print("\n" + "=" * 112)
    print(" [백테스트 전] ROE 15% vs 10% 필터 통과 종목군 분석  (강세장일·유니버스, ROE 외 매수조건 통과분)")
    print("=" * 112)
    print(f"  ROE≥15% 통과 고유종목수 : {len(set15)}")
    print(f"  ROE≥10% 통과 고유종목수 : {len(set10)}  (추가 편입 {len(set10 - set15)}종목 = ROE 10~15% 구간)")
    print(f"  Jaccard(15%∩10% / 합집합) : {jac:.1f}%   (10%는 15%의 상위집합이라 15%⊂10%)")
    print(f"  pass-event 수 : 기존(ROE≥15%) {len(inc_mc):,}건 · 추가(ROE 10~15%) {len(ext_mc):,}건")
    print("\n  [ 평균 프로파일 (pass-event 기준) ]")
    print(f"  {'구분':<20s}{'평균 시총':>14s}{'평균 주가':>13s}{'평균 거래대금':>15s}{'평균 20일모멘텀':>16s}")
    print(f"  {'기존(ROE≥15%)':<20s}{_avg(inc_mc)/1e8:>11,.0f}억{_avg(inc_px):>11,.0f}원{_avg(inc_tv)/1e8:>12,.0f}억{_avg(inc_mom)*100:>14.1f}%")
    print(f"  {'추가(ROE 10~15%)':<20s}{_avg(ext_mc)/1e8:>11,.0f}억{_avg(ext_px):>11,.0f}원{_avg(ext_tv)/1e8:>12,.0f}억{_avg(ext_mom)*100:>14.1f}%")


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
    slots_unfilled_days = 0
    bought_set: set[str] = set()

    for di in range(len(td)):
        today = td[di].date()
        if di < start_di:
            daily_total[di] = INITIAL_CASH
            continue

        is_bull = sc_base.is_bull_market(store.kospi_close_v[di], store.kospi_ma200_v[di])
        sold_today: set[str] = set()

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
            if free > 0 and len(cands) < free:
                slots_unfilled_days += 1
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
        "slots_unfilled_days": slots_unfilled_days,
        "bull_days": pass_days,
        "bought_set": bought_set,
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
        "통과평균개수": round(m["avg_pass_cnt"], 1),
        "현금유휴%": round(m["cash_idle_ratio"] * 100, 1),
        "5일내재손절%": round(m["restop_5d_ratio"] * 100, 1),
    }


def main():
    store = data_layer.get_store()

    analyze_filters(store)

    variants = [
        run_variant(sc_base.check_buy_conditions, "new_base(ROE≥15%)", store),
        run_variant(vroe.check_buy_conditions, "roe10(ROE≥10%)", store),
    ]
    df = pd.DataFrame([_row(m) for m in variants])

    print("\n" + "=" * 114)
    print(f" ROE 필터 실험 (15% vs 10%, 새 기준선 위)  ·  {variants[0]['start']} ~ {variants[0]['end']}"
          "  ·  1억 · 거래비용 반영 · 같은 데이터/같은 날")
    print("=" * 114)
    print(df.to_string(index=False))
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"\n  CSV 저장: {CSV_PATH}")

    a, b = variants[0]["bought_set"], variants[1]["bought_set"]
    jac = len(a & b) / len(a | b) * 100 if (a | b) else 0.0
    print("\n" + "-" * 114)
    print(" [진단]")
    for m in variants:
        print(f"   {m['tag']:<20s} 강세장일 {m['bull_days']}일 중 '슬롯 미충원(후보<빈슬롯)'일수 {m['slots_unfilled_days']}일 "
              f"({m['slots_unfilled_days']/max(m['bull_days'],1)*100:.1f}%)")
    print(f"   실제 신규매수 종목군 겹침 Jaccard(new_base vs roe10) = {jac:.1f}%  "
          f"(new_base {len(a)}종목 · roe10 {len(b)}종목 · 공통 {len(a & b)})")

    print("\n" + "=" * 114)
    print(" ⚠ 정직성: 단일 경로 백테스트 + 조건 누적 상태에서의 추가 변경. 상호작용/과최적화 위험 존재.")
    print("=" * 114)
    return variants


if __name__ == "__main__":
    main()
