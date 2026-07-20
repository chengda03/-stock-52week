# -*- coding: utf-8 -*-
"""
backtest_partial_grid.py — 부분익절 (임계값 × 매도비율) 그리드 백테스트
============================================================================
현재 S1 v2 에 고정돼 있는 "진입가 +8% 도달 시 50% 매도"가 정말 최적인지
검증하기 위해, 임계값과 매도비율을 격자(grid)로 바꿔가며 전부 돌려봅니다.

★ 설계 원칙 (지시사항 준수)
  - 기존 파일(strategy_core.py / backtest_v3.py / backtest_s1_partial.py)은
    "단일 진실소스"이므로 절대 수정하지 않습니다.
  - 매수/불타기/매도 '판정 로직'은 strategy_core.py 의 함수를 그대로 import 재사용.
  - strategy_core 의 부분익절 조건은 상수(0.08 / 0.5)로 하드코딩돼 있으므로,
    여기서는 그 상수를 건드리지 않고 백테스트 루프 안에서 (임계값, 매도비율)을
    '함수 인자'로 직접 주입하는 방식(wrapping)으로 처리합니다.
    → 이 루프는 backtest_s1_partial.run_variant 와 동일한 규칙을 그대로 복제하되,
      단일 (임계값, 매도비율)만 파라미터로 받고 '최대 단일종목 비중'을 추가 집계합니다.
  - 데이터는 data_layer 의 캐시(가격 100% 커버리지, 연도별 발행주식수)를 재사용하며
    새로 내려받지 않습니다. store 를 한 번만 만들어 18개 조합에 공유합니다(룩어헤드/재현성).
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

# --- strategy_core: '판정 로직'만 그대로 재사용 (수정 없음) -------------------
from strategy_core import (
    StockSnapshot, Position,
    is_bull_market, check_buy_conditions, rank_by_momentum,
    check_sell_condition, check_pyramid, apply_pyramid,
    NUM_SLOTS, SLOT_AMOUNT_WON, PYRAMID_ADD_AMOUNT_WON,
)
# --- 데이터 계층 + 검증된 엔진의 보조 함수/상수 재사용 (수정 없음) ------------
import data_layer
from backtest_roe_eps_event import fpct, log
from backtest_s1_partial import _build_snap, BUY_COST, SELL_COST


# ===== 그리드 설정 (여기만 바꾸면 됨) =====
THRESHOLD_GRID = [0.04, 0.06, 0.08, 0.10, 0.12, 0.15]  # 부분익절 임계값 (진입가 대비 +N%)
SELL_RATIO_GRID = [0.3, 0.5, 0.7]                       # 부분익절 시 매도비율
START_DATE = "2020-01-01"
END_DATE = None  # None이면 데이터 최신 시점까지
INITIAL_CAPITAL = 100_000_000  # 1억원
# ==========================================

# 현재 S1 v2 가 채택한 값 (표에서 ★현재채택★ 으로 표시)
CURRENT_THRESHOLD = 0.08
CURRENT_SELL_RATIO = 0.50

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

# 부분매도 후 잔량 평가액이 이 값 미만이면 전량 청산(먼지 방지) — run_variant 와 동일
DUST_WON = 500_000


def run_one(store, threshold: float, sell_ratio: float) -> dict:
    """
    (임계값 threshold, 매도비율 sell_ratio) 한 조합을 백테스트합니다.

    규칙 (backtest_s1_partial.run_variant 의 '+8%50%익절·전량손절' 변형과 동일):
      · 매수 6조건 + 20일모멘텀 정렬 (strategy_core 그대로)
      · 불타기: 강세장·하루1회·+3%복리 트리거 (strategy_core 그대로)
      · 부분익절: 진입가 대비 +threshold 도달 시, 그 시점 보유수량의 sell_ratio 매도 (1회)
      · 전량매도: 90일선 이탈 or 평단 -12% 하드손절 (strategy_core.check_sell_condition)
      · 처리순서: 전량매도 → 부분익절 → 불타기 → 신규매수 → 일별평가

    반환: 성과지표 dict (CAGR/MDD/Sharpe/Calmar/손익비/승률/최대단일종목비중 등)
    """
    td = store.trading_days
    close_v = store.close_v          # (T,N) 종가
    close_ff = store.close_ff        # 결측 보정 종가(평가용)
    c2c = store.code_to_col
    start_di = store.start_di

    cash = float(INITIAL_CAPITAL)
    positions: dict[str, Position] = {}
    cost: dict[str, float] = {}          # 종목별 실제 투입원가(수수료 포함)
    partial_done: dict[str, bool] = {}   # 종목별 부분익절 이미 했는지
    daily_total = np.empty(len(td))

    n_buy = n_add = n_partial = n_final = 0
    wins = 0
    gross_w = gross_l = 0.0
    max_weight = 0.0

    for di in range(len(td)):
        today = td[di].date()
        # 준비기간(2019-06~2019-12)에는 매매하지 않고 초기자본 그대로 둠
        if di < start_di:
            daily_total[di] = INITIAL_CAPITAL
            continue

        is_bull = is_bull_market(store.kospi_close_v[di], store.kospi_ma200_v[di])
        sold_today: set[str] = set()

        # ---------- 1) 전량매도 (90일선 이탈 / -12% 하드손절) ----------
        for ticker in list(positions.keys()):
            col = c2c[ticker]
            px = close_v[di, col]
            if not np.isfinite(px):
                continue
            pos = positions[ticker]
            snap = _build_snap(store, col, di)
            should_sell, _ = check_sell_condition(pos, snap)
            if should_sell:
                proceeds = px * pos.shares * (1.0 - SELL_COST)
                pnl = proceeds - cost[ticker]
                cash += proceeds
                n_final += 1
                if pnl > 0:
                    wins += 1; gross_w += pnl
                else:
                    gross_l += pnl
                del positions[ticker]; del cost[ticker]
                partial_done.pop(ticker, None)
                sold_today.add(ticker)

        # ---------- 2) 부분익절 (진입가 +threshold 도달 시 sell_ratio 매도, 1회) ----------
        for ticker in list(positions.keys()):
            col = c2c[ticker]
            px = close_v[di, col]
            if not np.isfinite(px) or px <= 0:
                continue
            pos = positions[ticker]
            if partial_done.get(ticker):
                continue
            # 기준가격은 평단가가 아니라 '최초 진입가(entry_price)' (strategy_core 규칙과 동일)
            if px < pos.entry_price * (1.0 + threshold):
                continue
            sell_sh = pos.shares * sell_ratio
            # 잔량이 먼지 수준이면 전량 매도
            if (pos.shares - sell_sh) * px < DUST_WON:
                sell_sh = pos.shares
            frac = sell_sh / pos.shares
            proceeds = px * sell_sh * (1.0 - SELL_COST)
            cost_sold = cost[ticker] * frac
            pnl = proceeds - cost_sold
            cash += proceeds
            n_partial += 1
            if pnl > 0:
                wins += 1; gross_w += pnl
            else:
                gross_l += pnl
            if sell_sh >= pos.shares - 1e-9:
                del positions[ticker]; del cost[ticker]
                partial_done.pop(ticker, None)
            else:
                pos.shares -= sell_sh
                cost[ticker] -= cost_sold
                partial_done[ticker] = True   # 부분익절은 1회만

        # ---------- 3) 불타기 (강세장, 현재 수익률 높은 종목부터) ----------
        if is_bull and positions:
            def cur_ret(t):
                col = c2c[t]; px = close_v[di, col]
                return px / positions[t].avg_price if np.isfinite(px) else -1.0
            for ticker in sorted(positions.keys(), key=cur_ret, reverse=True):
                col = c2c[ticker]
                px = close_v[di, col]
                if not np.isfinite(px) or px <= 0:
                    continue
                pos = positions[ticker]
                if check_pyramid(pos, float(px), today, is_bull, cash):
                    spent = PYRAMID_ADD_AMOUNT_WON * (1.0 + BUY_COST)
                    if cash < spent:
                        continue
                    apply_pyramid(pos, float(px), today)
                    cash -= spent
                    cost[ticker] += spent
                    n_add += 1

        # ---------- 4) 신규매수 (강세장, 빈 슬롯) ----------
        free = NUM_SLOTS - len(positions)
        if is_bull and free > 0:
            uni = store.get_universe(td[di])          # 시총 상위 500 (그날 기준)
            cands: list[StockSnapshot] = []
            for t in uni:
                if t in positions or t in sold_today:
                    continue
                col = c2c[t]
                px = close_v[di, col]
                if not np.isfinite(px) or px <= 0:
                    continue
                snap = _build_snap(store, col, di)
                ok, _ = check_buy_conditions(snap, is_bull)
                if ok:
                    cands.append(snap)
            for snap in rank_by_momentum(cands):      # 20일 모멘텀 높은 순
                if free <= 0:
                    break
                px = snap.price
                sh = int(SLOT_AMOUNT_WON // px)
                if sh <= 0:
                    continue
                spent = sh * px * (1.0 + BUY_COST)
                if spent > cash:
                    continue
                positions[snap.ticker] = Position(
                    ticker=snap.ticker, entry_price=px, avg_price=px,
                    shares=float(sh), entry_date=today)
                cost[snap.ticker] = spent
                partial_done[snap.ticker] = False
                cash -= spent
                n_buy += 1
                free -= 1

        # ---------- 5) 일별 평가 + 최대 단일종목 비중 집계 ----------
        hv = 0.0
        max_e = 0.0
        for t, pos in positions.items():
            col = c2c[t]
            p = close_v[di, col]
            if not np.isfinite(p):
                p = close_ff[di, col]
                if not np.isfinite(p):
                    p = pos.avg_price
            e = p * pos.shares
            hv += e
            if e > max_e:
                max_e = e
        total = cash + hv
        daily_total[di] = total
        if total > 0:
            w = max_e / total
            if w > max_weight:
                max_weight = w

    # ---------- 성과지표 계산 (매매 시작일 이후 구간만) ----------
    dt = pd.Series(daily_total, index=td)
    dt = dt[dt.index >= pd.Timestamp(START_DATE)]
    final = float(dt.iloc[-1])
    years = (dt.index[-1] - dt.index[0]).days / 365.25
    cagr = (final / INITIAL_CAPITAL) ** (1.0 / max(years, 1e-9)) - 1.0
    peak = dt.cummax()
    mdd = float(((dt - peak) / peak).min())
    rets = dt.pct_change().fillna(0)
    sharpe = float((rets.mean() / (rets.std() + 1e-12)) * math.sqrt(252))
    calmar = cagr / abs(mdd) if mdd != 0 else float("inf")
    n_sell = n_partial + n_final
    win_rate = wins / n_sell if n_sell else 0.0
    pl = abs(gross_w / gross_l) if gross_l != 0 else float("inf")

    return {
        "임계값": threshold, "매도비율": sell_ratio,
        "CAGR": cagr, "MDD": mdd, "Sharpe": sharpe, "Calmar": calmar,
        "손익비": pl, "승률": win_rate, "최대단일종목비중": max_weight,
        "final": final, "n_buy": n_buy, "n_partial": n_partial, "n_final": n_final,
    }


def main():
    """18개(6×3) 조합을 전부 돌리고, DataFrame 으로 정리 → CSV 저장 + 콘솔 출력."""
    end = pd.Timestamp(END_DATE) if END_DATE else None
    log("===== 부분익절 그리드 백테스트 ====="
        + (f" (종료일 {end.date()})" if end is not None else " (데이터 최신까지)"))

    # ★ 데이터는 캐시에서 한 번만 로드하여 18개 조합에 공유 (새로 내려받지 않음)
    store = data_layer.get_store(end_date=end)

    combos = [(t, r) for t in THRESHOLD_GRID for r in SELL_RATIO_GRID]
    total = len(combos)
    rows = []
    for i, (t, r) in enumerate(combos, start=1):
        res = run_one(store, t, r)
        rows.append(res)
        # 진행 상황 출력
        tag = "  ★현재채택★" if (abs(t - CURRENT_THRESHOLD) < 1e-9
                                and abs(r - CURRENT_SELL_RATIO) < 1e-9) else ""
        log(f"[{i}/{total}] 임계값 {int(t*100)}%, 매도비율 {int(r*100)}% 완료 "
            f"- CAGR {fpct(res['CAGR'])}{tag}")

    # DataFrame 정리 (CAGR 내림차순)
    df = pd.DataFrame(rows)
    df = df.sort_values("CAGR", ascending=False).reset_index(drop=True)

    # 현재채택 표시 컬럼
    df["비고"] = np.where(
        (np.abs(df["임계값"] - CURRENT_THRESHOLD) < 1e-9)
        & (np.abs(df["매도비율"] - CURRENT_SELL_RATIO) < 1e-9),
        "★현재채택★", "")

    # CSV 저장
    out_cols = ["임계값", "매도비율", "CAGR", "MDD", "Sharpe", "Calmar",
                "손익비", "승률", "최대단일종목비중", "비고"]
    csv_path = os.path.join(RESULT_DIR, "partial_grid_result.csv")
    df[out_cols].to_csv(csv_path, index=False, encoding="utf-8-sig")

    # ---------- 콘솔 표 출력 ----------
    print()
    print("=" * 104)
    print(" 부분익절 그리드 결과 (CAGR 내림차순) — 상위 5개는 [TOP] 표시")
    print("=" * 104)
    print(f"  {'순위':>3s} {'임계값':>5s} {'매도비율':>6s} {'CAGR':>8s} {'MDD':>8s} "
          f"{'Sharpe':>7s} {'Calmar':>7s} {'손익비':>6s} {'승률':>6s} {'최대비중':>7s}  비고")
    for rank, row in df.iterrows():
        top = "[TOP]" if rank < 5 else "     "
        note = row["비고"]
        print(f"  {top}{rank+1:>2d} {int(row['임계값']*100):>4d}% {int(row['매도비율']*100):>5d}% "
              f"{fpct(row['CAGR']):>8s} {fpct(row['MDD']):>8s} {row['Sharpe']:>7.2f} "
              f"{row['Calmar']:>7.2f} {row['손익비']:>6.2f} {row['승률']*100:>5.1f}% "
              f"{row['최대단일종목비중']*100:>6.1f}%  {note}")

    print(f"\n  CSV 저장: {csv_path}")

    # 현재채택 vs 최고 조합 요약
    cur = df[df["비고"] == "★현재채택★"].iloc[0]
    cur_rank = int(df.index[df["비고"] == "★현재채택★"][0]) + 1
    best = df.iloc[0]
    print(f"\n  · 현재채택 8%@50% : {cur_rank}위 / {total}개, CAGR {fpct(cur['CAGR'])}, "
          f"Calmar {cur['Calmar']:.2f}")
    print(f"  · 최고 CAGR 조합   : 임계값 {int(best['임계값']*100)}% @ 매도 {int(best['매도비율']*100)}%, "
          f"CAGR {fpct(best['CAGR'])}, Calmar {best['Calmar']:.2f}")


if __name__ == "__main__":
    main()
