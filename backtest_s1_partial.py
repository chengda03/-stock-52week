# -*- coding: utf-8 -*-
"""
backtest_s1_partial.py — S1(ROE×EPS 불타기) + "부분익절" 변형 비교
============================================================================
backtest_v3.py를 복사·확장한 별도 파일입니다. (원본 backtest_v3.py는 보존)
strategy_core.py는 전혀 수정하지 않고 그대로 import합니다.

[유지] 매수 6조건·불타기(하루1회·복리+3%)·시장필터(KOSPI 200일선) 전부 동일
[유지] 기존 매도조건: ① 90일선 이탈  ② 평단가 -12% 하드손절
       → 판정은 strategy_core.check_sell_condition 그대로 사용

[추가] 부분익절 규칙 (백테스트 루프에서만 처리, 코어 미변경)
  · 포지션의 현재가가 "최초 진입가 대비 +Y% 이상"이고, 아직 부분익절 안 했으면
    → 보유수량의 50%를 그날 종가로 매도 (나머지 50%는 계속 보유)
    → 남은 50%는 기존 매도조건(90일선/하드손절)으로 최종 청산
  · Y ∈ {15%, 20%, 25%, 30%}  (+ 부분익절 없는 기존 S1 = 비교 기준)

[승률 집계 방식]  (사용자 지정)
  · 부분익절 = 매도 1건. 손익 = (50% 매도대금) - (원가의 50%). 양수면 승, 음수면 패.
  · 최종 잔량 청산 = 별도 매도 1건.
  · 승률 = 이긴 매도건수 / (부분익절 건수 + 최종매도 건수)

[처리순서]  매도(최종) → 부분익절 → 불타기 → 신규매수   (기존 S1 순서 유지)
  · 90일선/하드손절이 걸리면 부분익절 여부와 무관하게 전량 최종청산.
  · 부분익절한 잔량에도 불타기 규칙은 그대로 적용(기존 규칙 유지).
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
from backtest_roe_eps_event import load_index, fpct, log

INITIAL_CASH = 100_000_000
BUY_COST = 0.00115
SELL_COST = 0.00295
TRADE_START = pd.Timestamp("2020-01-01")

RESULT_DIR = os.path.join("data", "cache_s1_partial", "results")
os.makedirs(RESULT_DIR, exist_ok=True)

PARTIAL_FRACTION = 0.50   # 부분익절시 매도 비율


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


DUST_WON = 500_000   # 부분매도 후 잔량 평가액이 이 값 미만이면 전량 청산(먼지 방지)


def run_variant(store, stages, name, stop_frac=1.0):
    """
    stages: 부분익절 단계 리스트 [(트리거수익률, 그때보유수량중매도비율), ...]
      · 트리거수익률은 '최초 진입가 대비' 기준.
      · 매도비율은 '그 시점 보유수량' 기준(잔량 기준).
      · 예) [(0.10, 0.5), (0.20, 0.5)] = +10%에서 절반(원수량50%),
            +20%에서 잔량의 절반(원수량25%), 나머지 원수량25% 계속 보유.
      · 빈 리스트 [] = 기존 S1 (부분익절 없음).
    각 종목은 stages를 순서대로 하나씩 진행(하루에 여러 단계 도달시 순차 실행).

    stop_frac: 매도조건(90일선/하드손절) 발동 시 '잔량 중' 매도 비율.
      · 1.0 = 기존 전량손절.
      · 0.5 = 부분손절(잔량의 50%만 매도, 나머지 보유 → 다음날에도 조건 유지되면 또 50%).
      · 잔량 평가액이 DUST_WON 미만이 되면 전량 청산.
    """
    td = store.trading_days
    close_v = store.close_v
    close_ff = store.close_ff
    c2c = store.code_to_col
    start_di = store.start_di

    cash = float(INITIAL_CASH)
    positions: dict[str, Position] = {}
    cost: dict[str, float] = {}
    stage_idx: dict[str, int] = {}   # 종목별 '다음에 실행할 부분익절 단계' 인덱스
    daily_total = np.empty(len(td))

    n_buy = n_add = n_partial = n_final = 0
    wins = 0
    gross_w = gross_l = 0.0

    for di in range(len(td)):
        today = td[di].date()
        if di < start_di:
            daily_total[di] = INITIAL_CASH
            continue

        is_bull = is_bull_market(store.kospi_close_v[di], store.kospi_ma200_v[di])
        sold_today: set[str] = set()

        # ---------- 1) 최종매도 (90일선/하드손절, 전량) ----------
        for ticker in list(positions.keys()):
            col = c2c[ticker]
            px = close_v[di, col]
            if not np.isfinite(px):
                continue
            pos = positions[ticker]
            snap = _build_snap(store, col, di)
            should_sell, reason = check_sell_condition(pos, snap)
            if should_sell:
                # 부분손절: 잔량의 stop_frac만 매도(잔량 dust면 전량)
                if stop_frac >= 1.0:
                    sell_sh = pos.shares
                else:
                    sell_sh = pos.shares * stop_frac
                    if (pos.shares - sell_sh) * px < DUST_WON:
                        sell_sh = pos.shares
                frac = sell_sh / pos.shares
                proceeds = px * sell_sh * (1.0 - SELL_COST)
                cost_sold = cost[ticker] * frac
                pnl = proceeds - cost_sold
                cash += proceeds
                if pnl > 0:
                    wins += 1; gross_w += pnl
                else:
                    gross_l += pnl
                n_final += 1
                if sell_sh >= pos.shares - 1e-9:
                    del positions[ticker]; del cost[ticker]
                    stage_idx.pop(ticker, None)
                    sold_today.add(ticker)
                else:
                    pos.shares -= sell_sh
                    cost[ticker] -= cost_sold

        # ---------- 2) 부분익절 (진입가 +트리거% 도달시 단계별 매도) ----------
        if stages:
            for ticker in list(positions.keys()):
                col = c2c[ticker]
                px = close_v[di, col]
                if not np.isfinite(px) or px <= 0:
                    continue
                pos = positions[ticker]
                ret = px / pos.entry_price - 1.0
                # 도달한 다음 단계들을 순서대로 실행 (하루에 여러 단계 도달 가능)
                while stage_idx.get(ticker, 0) < len(stages):
                    idx = stage_idx.get(ticker, 0)
                    trig, frac = stages[idx]
                    if ret < trig or pos.shares <= 0:
                        break
                    sell_sh = pos.shares * frac
                    proceeds = px * sell_sh * (1.0 - SELL_COST)
                    cost_sold = cost[ticker] * frac
                    pnl = proceeds - cost_sold
                    cash += proceeds
                    pos.shares -= sell_sh
                    cost[ticker] -= cost_sold
                    stage_idx[ticker] = idx + 1
                    n_partial += 1
                    if pnl > 0:
                        wins += 1; gross_w += pnl
                    else:
                        gross_l += pnl

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

        # ---------- 4) 신규매수 (강세장, 빈 슬롯) ----------
        free = NUM_SLOTS - len(positions)
        if is_bull and free > 0:
            uni = store.get_universe(td[di])
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
                stage_idx[snap.ticker] = 0
                cash -= spent
                n_buy += 1
                free -= 1

        # ---------- 5) 일별 평가 ----------
        hv = 0.0
        for t, pos in positions.items():
            col = c2c[t]
            p = close_v[di, col]
            if not np.isfinite(p):
                p = close_ff[di, col]
                if not np.isfinite(p):
                    p = pos.avg_price
            hv += p * pos.shares
        daily_total[di] = cash + hv

    # ---------- 성과지표 ----------
    dt = pd.Series(daily_total, index=td)
    dt = dt[dt.index >= TRADE_START]
    final = float(dt.iloc[-1])
    years = (dt.index[-1] - dt.index[0]).days / 365.25
    total_ret = final / INITIAL_CASH - 1.0
    cagr = (final / INITIAL_CASH) ** (1.0 / max(years, 1e-9)) - 1.0
    peak = dt.cummax()
    mdd = float(((dt - peak) / peak).min())
    rets = dt.pct_change().fillna(0)
    sharpe = float((rets.mean() / (rets.std() + 1e-12)) * math.sqrt(252))
    calmar = cagr / abs(mdd) if mdd != 0 else float("inf")
    n_sell = n_partial + n_final
    win_rate = wins / n_sell if n_sell else 0.0
    pl = abs(gross_w / gross_l) if gross_l != 0 else float("inf")

    return {"name": name, "final": final, "total_ret": total_ret,
            "CAGR": cagr, "MDD": mdd, "Sharpe": sharpe, "Calmar": calmar,
            "PL": pl, "win_rate": win_rate, "n_buy": n_buy, "n_add": n_add,
            "n_partial": n_partial, "n_final": n_final, "n_sell": n_sell,
            "equity": dt}


def label(r) -> str:
    return r["name"]


def main():
    import argparse
    ap = argparse.ArgumentParser(description="S1 부분익절 변형 백테스트")
    ap.add_argument("--end", type=str, default=None,
                    help="백테스트 종료일(YYYY-MM-DD). 지정 시 그 날짜까지만 사용(기간 통제).")
    args = ap.parse_args()
    end = pd.Timestamp(args.end) if args.end else None
    log("===== S1 부분익절 변형 백테스트 ====="
        + (f" (종료일 {end.date()})" if end is not None else ""))
    store = data_layer.get_store(end_date=end)

    # (이름, 부분익절 단계[(진입가대비 트리거, 그시점 잔량중 매도비율), ...])
    # 익절은 '+8%에서 50% 부분익절'로 고정하고, 손절 방식만 바꿔 비교.
    # (이름, 익절단계, 손절매도비율)
    tp = [(0.08, 0.50)]   # 진입가 +8% 도달시 잔량의 50% 익절
    variants = [
        ("기존 S1(익절X·전량손절)", [], 1.0),
        ("+8%50%익절·전량손절", tp, 1.0),
        ("+8%50%익절·부분손절50%", tp, 0.50),
        ("+8%50%익절·부분손절30%", tp, 0.30),
    ]
    results = [run_variant(store, stages, name, sf) for name, stages, sf in variants]

    print()
    print("=" * 112)
    print(" S1 부분익절 변형 비교 (매수·불타기·시장필터 동일, 매도에 50% 부분익절만 추가)")
    print("=" * 112)
    print(f"  {'변형':>18s} {'CAGR':>8s} {'MDD':>8s} {'승률':>7s} {'손익비':>6s} "
          f"{'Sharpe':>7s} {'Calmar':>7s} | {'신규':>4s} {'불타기':>5s} {'익절':>6s} {'손절매도':>6s}")
    for r in results:
        print(f"  {label(r):>18s} {fpct(r['CAGR']):>8s} {fpct(r['MDD']):>8s} "
              f"{r['win_rate']*100:>6.1f}% {r['PL']:>6.2f} {r['Sharpe']:>7.2f} {r['Calmar']:>7.2f} | "
              f"{r['n_buy']:>4d} {r['n_add']:>5d} {r['n_partial']:>6d} {r['n_final']:>6d}")

    # 기존 대비 변화
    base = results[0]
    print()
    print("=" * 112)
    print(" 기존 S1 대비 변화 (Δ승률 / ΔCAGR)")
    print("=" * 112)
    for r in results[1:]:
        d_win = (r["win_rate"] - base["win_rate"]) * 100
        d_cagr = (r["CAGR"] - base["CAGR"]) * 100
        print(f"  {label(r):>18s} : 승률 {base['win_rate']*100:.1f}%→{r['win_rate']*100:.1f}% "
              f"({d_win:+.1f}%p)   CAGR {fpct(base['CAGR'])}→{fpct(r['CAGR'])} ({d_cagr:+.1f}%p)   "
              f"MDD {fpct(base['MDD'])}→{fpct(r['MDD'])}")

    # 목표: 승률↑ & CAGR 손실 최소 조합 판정
    print()
    improved = [r for r in results[1:] if r["win_rate"] > base["win_rate"]]
    if improved:
        best = max(improved, key=lambda r: r["CAGR"])   # 승률 오른 것 중 CAGR 손실 최소
        print(f"  · 승률이 오른 변형 중 CAGR 손실이 가장 작은 것: {label(best)} "
              f"(승률 +{(best['win_rate']-base['win_rate'])*100:.1f}%p, CAGR {(best['CAGR']-base['CAGR'])*100:+.1f}%p)")
    else:
        print("  · 어떤 Y값도 기존 S1보다 승률을 올리지 못함.")

    _save(results, base)


def _save(results, base):
    rows = [{k: r[k] for k in ("name", "total_ret", "CAGR", "MDD", "Sharpe",
                               "Calmar", "PL", "win_rate", "n_buy", "n_add",
                               "n_partial", "n_final", "n_sell")} for r in results]
    pd.DataFrame(rows).to_csv(os.path.join(RESULT_DIR, "partial_results.csv"),
                              index=False, encoding="utf-8-sig")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for f in ["Malgun Gothic", "NanumGothic", "AppleGothic", "Gulim"]:
            try:
                matplotlib.rcParams["font.family"] = f
                matplotlib.rcParams["axes.unicode_minus"] = False
                break
            except Exception:
                continue
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        # (좌) 자산곡선
        for r in results:
            ax1.plot(r["equity"].index, r["equity"].values, lw=1.4, label=label(r))
        ax1.set_yscale("log")
        ax1.set_title("자산곡선 (부분익절 Y값별)")
        ax1.legend(fontsize=9); ax1.grid(alpha=0.3, which="both")
        # (우) 승률 vs CAGR
        xs = [r["win_rate"] * 100 for r in results]
        ys = [r["CAGR"] * 100 for r in results]
        ax2.plot(xs, ys, "o-", color="#8E8E8E", alpha=0.6)
        for r in results:
            ax2.annotate(label(r), (r["win_rate"] * 100, r["CAGR"] * 100),
                         fontsize=9, xytext=(6, 4), textcoords="offset points")
        ax2.set_xlabel("매매단위 승률 (%)"); ax2.set_ylabel("CAGR (%)")
        ax2.set_title("승률 vs CAGR (부분익절 트레이드오프)")
        ax2.grid(alpha=0.3)
        plt.tight_layout()
        png = os.path.join(RESULT_DIR, "partial_compare.png")
        plt.savefig(png, dpi=140)
        plt.close(fig)
        print(f"\n  그래프 저장 : {png}")
    except Exception as e:
        print(f"\n  (그래프 생략: {e})")
    print(f"  결과 저장 : {RESULT_DIR}")


if __name__ == "__main__":
    main()
