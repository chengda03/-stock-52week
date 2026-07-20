# -*- coding: utf-8 -*-
"""
backtest_yearend_withdraw.py — "연말 정률인출" 방식 백테스트
============================================================================
목적
  기존 "비중상한 + 초과분 인출" 방식은 비중이 상한을 안 넘는 해에는 인출금이
  0원이 되는 문제가 있었다. 이번엔 비중과 무관하게
      "매년 12월 마지막 거래일에 무조건 총자산의 W%를 인출금으로 떼어낸다"
  는 규칙을 테스트한다. 매년 반드시 인출이 발생하므로 "생활비처럼 꾸준히"라는
  목적에 더 맞는지 확인한다.

★ 설계 원칙 (지시 준수)
  · strategy_core.py / backtest_s1_partial.py / backtest_v3.py 는 절대 수정하지 않음.
  · S1 v2 판정 로직(매수 6조건·매도 2조건·불타기·+8%@50% 부분익절)은 그대로 재사용.
    바꾸는 것은 오직 '인출 규칙'뿐 → 엔진은 backtest_partial_grid 와 동일한
    (임계값 8%, 매도비율 50%, 전량손절) S1 v2 를 여기서 복제하고, 연말 인출만 추가.
  · 데이터는 data_layer 의 검증된 캐시를 그대로 재사용(새 수집 없음).

★ 인출 규칙(핵심)
  · 매년 12월 마지막 거래일에:
      총자산 = 보유주식평가 + 운용현금 + 누적인출금   ("구 총자산기준")
      인출액 = 총자산 × W%
      → 운용계좌(현금 우선, 부족분은 '비중 큰 종목부터' 매도)에서 빼서
        인출계좌로 이동. 인출계좌 돈은 이후 절대 재투자하지 않음.
  · 평상시(연말 아님)에는 인출 없이 S1 v2 와 완전히 동일하게 매매.

★ 정직성(§7)
  · 같은 데이터에 대한 그리드서치 → 과최적화 위험 있음(결과 하단에 명시).
  · 이번 실험의 성패 기준 = "인출 0원 연도 수"가 기존(비중상한형)보다 확실히 적은가.
    이 숫자를 눈에 띄게 출력한다.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

from strategy_core import (
    StockSnapshot, Position,
    is_bull_market, check_buy_conditions, rank_by_momentum,
    check_sell_condition, check_pyramid, apply_pyramid,
    NUM_SLOTS, SLOT_AMOUNT_WON, PYRAMID_ADD_AMOUNT_WON,
)
import data_layer
from backtest_roe_eps_event import fpct, log
from backtest_s1_partial import _build_snap, INITIAL_CASH, BUY_COST, SELL_COST, TRADE_START
# 기존(비중상한50%+인출30%) 참고행을 '같은 코드'로 재현하기 위해 import (수정 아님)
from backtest_s1_capwithdraw import run_cap

# ===== 그리드 설정 (여기만 바꾸면 됨) =====
WITHDRAW_RATE_GRID = [0.10, 0.15, 0.20]  # 연말 인출비율 10%/15%/20%
START_DATE = "2020-01-01"
END_DATE = None
INITIAL_CAPITAL = 100_000_000
# ==========================================

# 부분익절 잔량이 이 값 미만이면 전량청산(먼지 방지) — S1 파생 엔진 관례와 동일
DUST_WON = 500_000

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)


def _is_year_last_trading_day(td, di) -> bool:
    """td[di] 가 '그 해의 마지막 거래일'이고 실제 12월인지 (연말 인출 시점)."""
    if td[di].month != 12:
        return False
    return (di == len(td) - 1) or (td[di + 1].year != td[di].year)


def run_yearend(store, W: float) -> dict:
    """
    S1 v2(+8%@50% 부분익절·전량손절) 그대로 + 매년 연말 총자산 W% 정률인출.

    반환: CAGR(운용자산기준)·MDD(총자산기준)·최종운용자산·누적인출금·연도별 인출액 등.
    """
    td = store.trading_days
    close_v = store.close_v
    close_ff = store.close_ff
    c2c = store.code_to_col
    start_di = store.start_di

    cash = float(INITIAL_CAPITAL)   # 운용현금(재투입 가능)
    withdrawn = 0.0                 # 인출계좌(재투입 불가, 누적)
    positions: dict[str, Position] = {}
    cost: dict[str, float] = {}
    partial_done: dict[str, bool] = {}

    daily_total = np.empty(len(td))   # 총자산 = 현금+평가액+인출금 (MDD용)
    daily_op = np.empty(len(td))      # 운용자산 = 현금+평가액 (CAGR·최종운용자산용)
    yearly_wd: dict[int, float] = {}  # 그 해 12월에 새로 인출한 금액

    n_buy = n_add = n_partial = n_final = 0
    hold_counts = []

    threshold, sell_ratio = 0.08, 0.50   # S1 v2 부분익절 (고정)

    def eval_price(col, di):
        p = close_v[di, col]
        if not np.isfinite(p):
            p = close_ff[di, col]
        return p

    for di in range(len(td)):
        today = td[di].date()
        if di < start_di:
            daily_total[di] = INITIAL_CAPITAL
            daily_op[di] = INITIAL_CAPITAL
            continue

        is_bull = is_bull_market(store.kospi_close_v[di], store.kospi_ma200_v[di])
        sold_today: set[str] = set()

        # ---------- 1) 전량매도 (90일선 이탈 / -12% 하드손절) ----------
        for ticker in list(positions.keys()):
            col = c2c[ticker]
            px = close_v[di, col]
            if not np.isfinite(px):
                continue
            snap = _build_snap(store, col, di)
            should_sell, _ = check_sell_condition(positions[ticker], snap)
            if should_sell:
                cash += px * positions[ticker].shares * (1.0 - SELL_COST)
                n_final += 1
                del positions[ticker]; del cost[ticker]
                partial_done.pop(ticker, None)
                sold_today.add(ticker)

        # ---------- 2) 부분익절 (진입가 +8% 도달 시 50% 매도, 1회) ----------
        for ticker in list(positions.keys()):
            col = c2c[ticker]
            px = close_v[di, col]
            if not np.isfinite(px) or px <= 0 or partial_done.get(ticker):
                continue
            pos = positions[ticker]
            if px < pos.entry_price * (1.0 + threshold):
                continue
            sell_sh = pos.shares * sell_ratio
            if (pos.shares - sell_sh) * px < DUST_WON:
                sell_sh = pos.shares
            frac = sell_sh / pos.shares
            cash += px * sell_sh * (1.0 - SELL_COST)
            cost[ticker] -= cost[ticker] * frac
            n_partial += 1
            if sell_sh >= pos.shares - 1e-9:
                del positions[ticker]; del cost[ticker]
                partial_done.pop(ticker, None)
            else:
                pos.shares -= sell_sh
                partial_done[ticker] = True

        # ---------- 3) 불타기 (강세장, 수익률 높은 순) ----------
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

        # ---------- 4) 신규매수 (강세장, 빈 슬롯, 500만원 고정) ----------
        free = NUM_SLOTS - len(positions)
        if is_bull and free > 0:
            cands: list[StockSnapshot] = []
            for t in store.get_universe(td[di]):
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
            for snap in rank_by_momentum(cands):
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

        # ---------- 5) 일별 평가 ----------
        hv = 0.0
        for t, pos in positions.items():
            hv += eval_price(c2c[t], di) * pos.shares

        # ---------- 6) 연말 정률인출 (12월 마지막 거래일에만) ----------
        if _is_year_last_trading_day(td, di):
            total = cash + hv + withdrawn          # 구 총자산기준
            target_cash = W * total                # 이번에 떼어낼 금액
            target_cash = min(target_cash, cash + hv * (1.0 - SELL_COST))  # 실현가능 한도
            raised = 0.0
            # (a) 현금 먼저 사용
            take = min(cash, target_cash)
            cash -= take
            raised += take
            # (b) 부족분은 '비중(평가액) 큰 종목부터' 필요한 만큼만 매도
            need = target_cash - raised
            if need > 1e-6 and positions:
                order = sorted(positions.keys(),
                               key=lambda t: eval_price(c2c[t], di) * positions[t].shares,
                               reverse=True)
                for ticker in order:
                    if need <= 1e-6:
                        break
                    col = c2c[ticker]
                    px = eval_price(col, di)
                    if not np.isfinite(px) or px <= 0:
                        continue
                    pos = positions[ticker]
                    net_per_share = px * (1.0 - SELL_COST)
                    sh_needed = int(math.ceil(need / net_per_share))
                    sell_sh = min(sh_needed, int(pos.shares))
                    if sell_sh <= 0:
                        continue
                    frac = sell_sh / pos.shares
                    proceeds = px * sell_sh * (1.0 - SELL_COST)
                    cost[ticker] -= cost[ticker] * frac
                    raised += proceeds
                    need -= proceeds
                    if sell_sh >= pos.shares - 1e-9:
                        del positions[ticker]; del cost[ticker]
                        partial_done.pop(ticker, None)
                    else:
                        pos.shares -= sell_sh
            withdrawn += raised
            yearly_wd[td[di].year] = yearly_wd.get(td[di].year, 0.0) + raised
            # 인출 후 평가액 재계산
            hv = sum(eval_price(c2c[t], di) * positions[t].shares for t in positions)
            hold_counts.append(len(positions))

        daily_op[di] = cash + hv
        daily_total[di] = cash + hv + withdrawn

    # ---------- 성과지표 ----------
    idx = td
    op = pd.Series(daily_op, index=idx)
    tot = pd.Series(daily_total, index=idx)
    m = op.index >= pd.Timestamp(START_DATE)
    op = op[m]; tot = tot[m]

    op_final = float(op.iloc[-1])
    years = (op.index[-1] - op.index[0]).days / 365.25
    # CAGR: 운용자산 기준(인출금 제외) — 인출로 빠져나간 돈은 성장에서 제외됨(의도)
    cagr_op = (op_final / INITIAL_CAPITAL) ** (1.0 / max(years, 1e-9)) - 1.0
    # MDD: 총자산 기준(인출금 포함) — 연말 인출 이동을 낙폭으로 오해하지 않도록
    peak = tot.cummax()
    mdd = float(((tot - peak) / peak).min())

    # 완료된 연말(2020~2025 등 실제 12월 인출이 있었던 해) 중 인출 0원 연도 수
    zero_years = sum(1 for v in yearly_wd.values() if v <= 1.0)

    return {
        "W": W, "CAGR": cagr_op, "MDD": mdd,
        "op_final": op_final, "withdrawn": withdrawn,
        "yearly_wd": yearly_wd, "zero_years": zero_years,
        "n_buy": n_buy, "n_partial": n_partial, "n_final": n_final,
        "avg_holdings": float(np.mean(hold_counts)) if hold_counts else 0.0,
        "op_curve": op, "total_curve": tot,
    }


def _reference_row(store) -> dict:
    """참고행: 기존 '비중상한50% + 인출30%'(구 총자산기준, 자본비례) — 같은 코드로 재현.
    지시문에 제시된 값(CAGR38.06%/MDD-28.76%/운용5.50억/인출2.44억)과 대조용."""
    r = run_cap(store, cap=0.50, denom_incl_withdrawn=True,
                withdraw_frac=0.30, proportional=True)
    op_final = r["op_final"]
    yrs = (r["equity"].index[-1] - r["equity"].index[0]).days / 365.25
    cagr_op = (op_final / INITIAL_CAPITAL) ** (1.0 / max(yrs, 1e-9)) - 1.0
    # 연도별 신규 인출액 → 0원 연도 수 (2020~ 마지막해 직전까지 완료연도)
    zero = sum(1 for y, info in r["wd_year"].items()
               if info["added"] <= 1.0 and y < r["equity"].index[-1].year)
    added = {y: info["added"] for y, info in r["wd_year"].items()}
    return {"name": "비중상한50%+인출30%(기존,참고)",
            "CAGR": cagr_op, "MDD": r["MDD"], "op_final": op_final,
            "withdrawn": r["withdrawn"], "zero_years": zero, "yearly_wd": added}


def main():
    import argparse
    ap = argparse.ArgumentParser(description="연말 정률인출 백테스트")
    ap.add_argument("--end", type=str, default=END_DATE, help="종료일 YYYY-MM-DD")
    args = ap.parse_args()
    end = pd.Timestamp(args.end) if args.end else None

    log("===== 연말 정률인출 백테스트 =====")
    store = data_layer.get_store(end_date=end)

    # 참고행(기존 방식) + 연말정률 3종
    log("[ref] 기존 비중상한50%+인출30% 재현 중...")
    ref = _reference_row(store)
    results = []
    for i, W in enumerate(WITHDRAW_RATE_GRID, 1):
        r = run_yearend(store, W)
        results.append(r)
        log(f"[{i}/{len(WITHDRAW_RATE_GRID)}] 연말정률 {int(W*100)}% 완료 "
            f"- CAGR(운용) {fpct(r['CAGR'])}, 누적인출 {r['withdrawn']/1e8:.2f}억, "
            f"인출0원연도 {r['zero_years']}개")

    last_year = store.trading_days[-1].year
    all_years = list(range(2020, last_year + 1))

    # ---------- 비교표 (§5) ----------
    W_ = 96
    print("\n" + "=" * W_)
    print(" 연말 정률인출 vs 기존 비중상한형 (2020-01-01 ~ %s, 초기자본 1억)"
          % store.trading_days[-1].date())
    print("=" * W_)
    print(f"  {'방식':<26s} {'CAGR':>8s} {'MDD':>8s} {'최종운용자산':>11s} "
          f"{'누적인출금':>10s} {'인출0원연도':>9s}")

    def prow(name, d):
        print(f"  {name:<26s} {fpct(d['CAGR']):>8s} {fpct(d['MDD']):>8s} "
              f"{d['op_final']/1e8:>9,.2f}억 {d['withdrawn']/1e8:>8,.2f}억 "
              f"{d['zero_years']:>7d}개")

    prow(ref["name"], ref)
    print("  " + "-" * (W_ - 2))
    for r in results:
        prow(f"연말정률 {int(r['W']*100)}%", r)

    # ---------- 연도별 인출금 표 (§4 핵심) ----------
    print("\n" + "=" * W_)
    print(" ★ 연도별 인출금 (단위:억) — 0원 나오는 해가 있는지 반드시 확인")
    print("=" * W_)
    header = "  " + f"{'방식':<26s}" + "".join(f"{y:>8d}" for y in all_years)
    print(header)

    def wd_row(name, ywd):
        row = "  " + f"{name:<26s}"
        for y in all_years:
            v = ywd.get(y)
            row += (f"{v/1e8:>8.2f}" if v is not None else f"{'-':>8s}")
        return row

    print(wd_row(ref["name"], ref["yearly_wd"]))
    print("  " + "-" * (W_ - 2))
    for r in results:
        print(wd_row(f"연말정률 {int(r['W']*100)}%", r["yearly_wd"]))
    print(f"\n  (참고) {last_year}년은 12월 이전 데이터 종료로 연말인출 미발생 → '-' 표기.")

    # ---------- CSV 저장 ----------
    rows = []
    for name, d in [(ref["name"], ref)] + [(f"연말정률{int(r['W']*100)}%", r) for r in results]:
        row = {"방식": name, "CAGR": d["CAGR"], "MDD": d["MDD"],
               "최종운용자산": d["op_final"], "누적인출금": d["withdrawn"],
               "인출0원연도수": d["zero_years"]}
        for y in all_years:
            row[f"인출_{y}"] = d["yearly_wd"].get(y, 0.0)
        rows.append(row)
    csv_path = os.path.join(RESULT_DIR, "yearend_withdraw_result.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n  CSV 저장: {csv_path}")

    # ---------- 정직성 경고 (§7) ----------
    print("\n" + "-" * W_)
    print("  ⚠ 과최적화 주의: 위 결과는 '같은 과거 데이터'에 대한 그리드서치입니다.")
    print("    인출비율(W)을 사후에 고른 것이므로 미래 성과를 보장하지 않습니다.")
    print("  · 이번 실험의 성패 기준 = '인출 0원 연도 수'가 기존 비중상한형보다 적은가.")
    best_zero = min(r["zero_years"] for r in results)
    print(f"    → 연말정률 방식 인출0원 연도 수(최소) {best_zero}개  vs  "
          f"기존 비중상한형 {ref['zero_years']}개")


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
