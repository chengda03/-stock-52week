# -*- coding: utf-8 -*-
"""
backtest_disp_compare.py
========================
[실험] "매수조건에서 60일선을 빼고 90일선 이격도로 교체"가 타당한지 비교.

비교 대상 (전부 같은 프로세스·같은 DataStore·같은 날짜 → 데이터 드리프트 0):
  · base  : 현재 확정 전략 (현재가>60일선 AND 현재가>90일선 둘 다) = 원본 strategy_core
  · disp0 : 60일선 빼고 90일선 이격도 ≥ 0%
  · disp2 : 60일선 빼고 90일선 이격도 ≥ 2%
  · disp3 : 60일선 빼고 90일선 이격도 ≥ 3%
  · disp5 : 60일선 빼고 90일선 이격도 ≥ 5%

    이격도(%) = (현재가 - 90일선) / 90일선 × 100

★ 엔진 규칙은 backtest_v3.py / bt_valuation_engine.py 와 100% 동일:
  - 기간 2020-01-01 ~ 데이터최신, 초기 1억, 매수 0.115% / 매도 0.295%
  - 강세장(KOSPI 200일선 위)에서만 신규매수/불타기
  - 전량매도(90일선/-12%) → 부분익절(+8%@50%) → 불타기(+3%복리) → 신규매수(빈 슬롯)
  - 슬롯 20개, 슬롯당 500만원(정수주)
  변형 간 차이는 오로지 '매수조건 판정 함수(check_buy_conditions)' 하나뿐이다.
  (매도/불타기/부분익절/사이징/비용은 전부 원본 strategy_core.py를 그대로 재사용)

추가 계측(비교표용, 매매에는 영향 없음):
  - immediate_sell   : 매수 당일 종가 기준 매도조건이 이미 켜진 건수(whipsaw 직접 지표)
  - restop_5d_ratio  : 신규매수 중 '5거래일 이내 전량매도(재손절)'로 청산된 비율
  - avg_pass_cnt     : 강세장일마다 '유니버스 중 매수조건 통과 종목 개수'의 평균
  - jaccard_vs_base  : base가 신규매수한 종목집합과의 자카드 유사도(교집합/합집합)
  - cash_idle_ratio  : 매매기간 일별 (현금/총자산)의 평균 = 현금 유휴 비율

정직성: 이것은 단일 경로(약 6.5년) 위의 백테스트다. 특정 임계값이 좋아 보여도
        '왜'가 논리적으로 설명되는지 함께 봐야 하며, 표본이 하나뿐이라 과최적화 위험이 있다.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

import data_layer
import strategy_core as sc_base            # 매도/불타기/부분익절/사이징 = 원본 규칙
from strategy_core import StockSnapshot, Position

import strategy_core_disp0 as v0
import strategy_core_disp2 as v2
import strategy_core_disp3 as v3
import strategy_core_disp5 as v5
from backtest_roe_eps_event import fpct

INITIAL_CASH = 100_000_000
BUY_COST = 0.00115
SELL_COST = 0.00295
TRADE_START = pd.Timestamp("2020-01-01")
RESTOP_WINDOW = 5   # '매수 후 N거래일 이내' 재손절 판정 창

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)
CSV_PATH = os.path.join(RESULT_DIR, "disparity_filter_compare.csv")


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
    """매수조건 함수 하나만 갈아끼워 백테스트. (그 외 규칙은 전부 원본 strategy_core)"""
    td = store.trading_days
    close_v = store.close_v
    close_ff = store.close_ff
    c2c = store.code_to_col
    start_di = store.start_di

    cash = float(INITIAL_CASH)
    positions: dict[str, Position] = {}
    cost: dict[str, float] = {}
    entry_di: dict[str, int] = {}          # 신규매수 시점의 거래일 인덱스(5일내 재손절 판정용)
    daily_total = np.empty(len(td))

    n_buy = n_add = n_partial = n_final = 0
    wins = 0
    gross_w = gross_l = 0.0
    immediate_sell = 0
    restop_5d = 0                           # 매수 후 RESTOP_WINDOW 거래일 이내 전량매도된 건수
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

        # 2) 불타기 (강세장, 수익률 높은 순)
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

        # 3) 신규매수 (강세장, 빈 슬롯) + 통과개수 계측
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
                new_pos = Position(ticker=snap.ticker, entry_price=px, avg_price=px,
                                   shares=float(sh), entry_date=today)
                imm, _ = sc_base.check_sell_condition(new_pos, snap)
                if imm:
                    immediate_sell += 1
                positions[snap.ticker] = new_pos
                cost[snap.ticker] = spent
                entry_di[snap.ticker] = di
                cash -= spent; n_buy += 1; free -= 1
                bought_set.add(snap.ticker)

        # 4) 일별 평가 + 현금유휴비율 누적
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
        "immediate_sell": immediate_sell,
        "restop_5d": restop_5d,
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
        "5일내재손절%": round(m["restop_5d_ratio"] * 100, 1),
        "통과평균개수": round(m["avg_pass_cnt"], 1),
        "base겹침(Jaccard)%": round(_jaccard(m["bought_set"], base_set) * 100, 1),
        "현금유휴%": round(m["cash_idle_ratio"] * 100, 1),
    }


def main():
    store = data_layer.get_store()   # 1회 로드 → 모든 변형 공유(공정 비교)

    variants = [
        run_variant(sc_base.check_buy_conditions, "base(60선+90선)", store),
        run_variant(v0.check_buy_conditions, "disp0(이격도≥0%)", store),
        run_variant(v2.check_buy_conditions, "disp2(이격도≥2%)", store),
        run_variant(v3.check_buy_conditions, "disp3(이격도≥3%)", store),
        run_variant(v5.check_buy_conditions, "disp5(이격도≥5%)", store),
    ]

    base_set = variants[0]["bought_set"]
    df = pd.DataFrame([_row(m, base_set) for m in variants])

    print("\n" + "=" * 118)
    print(f" 60일선 제거 + 90일선 이격도 교체 실험  ·  {variants[0]['start']} ~ {variants[0]['end']}"
          "  ·  1억 · 거래비용 반영 · 같은 데이터/같은 날(드리프트 0)")
    print("=" * 118)
    print(df.to_string(index=False))
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"\n  CSV 저장: {CSV_PATH}")

    # base 대비 종목군 겹침 상세
    print("\n" + "-" * 118)
    print(" [종목군 겹침 상세] base가 신규매수한 종목집합 대비")
    print("-" * 118)
    for m in variants:
        a = m["bought_set"]
        inter = a & base_set
        print(f"  {m['tag']:<16s} 매수종목수 {len(a):>4d} · 공통 {len(inter):>4d} · "
              f"이 버전에만 {len(a - base_set):>4d} · base에만 {len(base_set - a):>4d} · "
              f"Jaccard {_jaccard(a, base_set)*100:>5.1f}%")

    print("\n" + "=" * 118)
    print(" ⚠ 정직성: 단일 경로 백테스트. 임계값이 좋아 보여도 표본이 하나뿐이라 과최적화 위험 존재.")
    print("=" * 118)
    return variants


if __name__ == "__main__":
    main()
