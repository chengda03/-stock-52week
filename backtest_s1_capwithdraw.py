# -*- coding: utf-8 -*-
"""
backtest_s1_capwithdraw.py — S1 v2 매도규칙 재설계: "비중상한 초과분 인출"
============================================================================
strategy_core.py / backtest_s1_partial.py 는 전혀 수정하지 않고 import만 합니다.

[매도규칙 변경]
  · 기존 "+8% 도달시 50% 익절" 제거.
  · 대신 종목 비중(그 종목 평가금액 / 총자산)이 X% 초과 시 초과분만 매도,
    그 매도대금은 '인출가능금(withdrawable)'으로 분리 → 다시는 매매에 사용 안 함.
  · 90일선 이탈 / 평단 -12% 하드손절 시 전량매도는 기존 그대로 유지
    (strategy_core.check_sell_condition 그대로 = 90일선 '당일' 이탈 or -12%).
    이 현금은 정상적으로 운용현금에 복귀(재투입).

[총자산 정의 (사용자 지정)]  총자산 = 운용현금 + 보유주식 평가액 + 인출가능금.
  · 비중 계산·성과지표 모두 이 총자산 기준.

[비교]
  1. 기존 S1 v2 (+8%@50%익절, 인출 없음)   ← backtest_s1_partial.run_variant 그대로
  2~5. 비중상한 30 / 40 / 50 / 60% + 인출
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
    _pyramid_trigger_price,
    NUM_SLOTS, SLOT_AMOUNT_WON, PYRAMID_ADD_AMOUNT_WON,
)
import data_layer
from backtest_roe_eps_event import fpct, log
from backtest_s1_partial import (
    _build_snap, INITIAL_CASH, BUY_COST, SELL_COST, TRADE_START, run_variant,
)

CAP_LEVELS = [0.30, 0.40, 0.50, 0.60]

RESULT_DIR = os.path.join("data", "cache_s1_capwithdraw", "results")
os.makedirs(RESULT_DIR, exist_ok=True)


def compute_yearly(equity: pd.Series) -> dict[int, float]:
    ye = equity.resample("YE").last()
    out: dict[int, float] = {}
    prev = float(INITIAL_CASH)
    for ts, val in ye.items():
        out[ts.year] = val / prev - 1.0
        prev = float(val)
    return out


def run_cap(store, cap: float, denom_incl_withdrawn: bool = False,
            withdraw_frac: float = 1.0, proportional: bool = False,
            track_posmat: bool = False):
    """비중상한 cap 초과분을 매도해 일부를 인출가능금으로 빼내는 S1 변형 (부분익절 없음).

    denom_incl_withdrawn:
      · False(기본, 신 방식) → 비중 = 종목평가 / (운용현금+평가액)  [인출금 제외]
      · True (구 방식)       → 비중 = 종목평가 / (운용현금+평가액+인출가능금)
    withdraw_frac:
      · 초과분 매도대금 중 인출가능금으로 분리할 비율.
      · 1.0 = 전액인출(재투입 없음), 0.5 = 절반만 인출·나머지 운용현금 재투입.
    proportional:
      · False → 기존 S1처럼 신규매수/불타기 금액을 500만원 고정.
      · True  → 신규매수/불타기 금액을 '현재 운용자본/슬롯수'에 비례.
        (인출로 운용자본이 줄어도 슬롯보다 작아져 매매가 멈추는 '동결' 방지)
    """
    td = store.trading_days
    close_v = store.close_v
    close_ff = store.close_ff
    c2c = store.code_to_col
    start_di = store.start_di

    cash = float(INITIAL_CASH)      # 운용현금 (재투입 가능)
    withdrawn = 0.0                 # 인출가능금 (재투입 불가, 누적)
    positions: dict[str, Position] = {}
    cost: dict[str, float] = {}
    daily_total = np.empty(len(td))     # 총자산 = cash + 평가액 + withdrawn
    daily_withdrawn = np.empty(len(td))
    daily_cash = np.zeros(len(td))
    daily_posval = np.zeros(len(td))
    daily_npos = np.zeros(len(td))
    trim_by_year: dict[int, float] = {}   # 연도별 트리밍 총매도대금(원)
    buy_by_year: dict[int, int] = {}      # 연도별 신규매수 건수

    n_buy = n_add = n_trim = n_final = 0
    wins = 0
    gross_w = gross_l = 0.0
    n_units = 0
    max_weight = 0.0
    N = len(store.valid_codes)
    posmat = np.zeros((len(td), N)) if track_posmat else None

    def eval_of(ticker, di):
        col = c2c[ticker]
        p = close_v[di, col]
        if not np.isfinite(p):
            p = close_ff[di, col]
            if not np.isfinite(p):
                p = positions[ticker].avg_price
        return p, positions[ticker].shares * p

    for di in range(len(td)):
        today = td[di].date()
        if di < start_di:
            daily_total[di] = INITIAL_CASH
            daily_withdrawn[di] = 0.0
            continue

        is_bull = is_bull_market(store.kospi_close_v[di], store.kospi_ma200_v[di])
        sold_today: set[str] = set()

        # ---------- 1) 전량매도 (90일선/하드손절) → 운용현금 복귀 ----------
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
                n_final += 1; n_units += 1
                if pnl > 0:
                    wins += 1; gross_w += pnl
                else:
                    gross_l += pnl
                del positions[ticker]; del cost[ticker]
                sold_today.add(ticker)

        # ---- 이번 날의 1슬롯 투입금액(unit) 결정 ----
        #   proportional=True: 현재 운용자본(현금+평가액)/슬롯수 → 규모에 맞춰 자동 축소
        #   proportional=False: 기존 S1과 동일하게 500만원 고정
        if proportional:
            _pv = 0.0
            for t in positions:
                col = c2c[t]
                p = close_v[di, col]
                if not np.isfinite(p):
                    p = close_ff[di, col]
                    if not np.isfinite(p):
                        p = positions[t].avg_price
                _pv += p * positions[t].shares
            unit = max((cash + _pv) / NUM_SLOTS, 0.0)
        else:
            unit = float(SLOT_AMOUNT_WON)

        # ---------- 2) 불타기 (강세장, 수익률 높은 순) ----------
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
                if proportional:
                    # 기존 트리거(진입가×1.03^n, 하루1회)는 유지, 투입금액만 unit로
                    if pos.last_pyramid_date == today:
                        continue
                    next_step = pos.pyramid_count + 1
                    if float(px) < _pyramid_trigger_price(pos.entry_price, next_step):
                        continue
                    spent = unit * (1.0 + BUY_COST)
                    if unit <= 0 or cash < spent:
                        continue
                    old_cost = pos.avg_price * pos.shares
                    pos.shares += unit / float(px)
                    pos.avg_price = (old_cost + unit) / pos.shares
                    pos.pyramid_count = next_step
                    pos.last_pyramid_date = today
                    cash -= spent
                    cost[ticker] += spent
                    n_add += 1
                else:
                    if check_pyramid(pos, float(px), today, is_bull, cash):
                        spent = PYRAMID_ADD_AMOUNT_WON * (1.0 + BUY_COST)
                        if cash < spent:
                            continue
                        apply_pyramid(pos, float(px), today)
                        cash -= spent
                        cost[ticker] += spent
                        n_add += 1

        # ---------- 3) 신규매수 (강세장, 빈 슬롯) ----------
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
                sh = int(unit // px)
                if sh <= 0:
                    continue
                spent = sh * px * (1.0 + BUY_COST)
                if spent > cash:
                    continue
                positions[snap.ticker] = Position(
                    ticker=snap.ticker, entry_price=px, avg_price=px,
                    shares=float(sh), entry_date=today)
                cost[snap.ticker] = spent
                cash -= spent
                n_buy += 1
                buy_by_year[today.year] = buy_by_year.get(today.year, 0) + 1
                free -= 1

        # ---------- 4) 비중상한 초과분 인출 ----------
        #   신 방식: base = 운용현금 + 평가액        (인출금 제외 → 후반에도 계속 인출)
        #   구 방식: base = 운용현금 + 평가액 + 인출금 (인출금 포함 → 후반 인출 자기억제)
        evals = {t: eval_of(t, di) for t in positions}   # t -> (px, eval)
        pos_val = sum(v for _, v in evals.values())
        base = cash + pos_val + (withdrawn if denom_incl_withdrawn else 0.0)
        if base > 0:
            for ticker in list(positions.keys()):
                px, e = evals[ticker]
                if px <= 0 or e <= cap * base:
                    continue
                excess = e - cap * base
                sell_sh = int(np.ceil(excess / px))
                pos = positions[ticker]
                if sell_sh <= 0:
                    continue
                sell_sh = min(sell_sh, int(pos.shares))
                if sell_sh <= 0:
                    continue
                frac = sell_sh / pos.shares
                proceeds = px * sell_sh * (1.0 - SELL_COST)
                cost_sold = cost[ticker] * frac
                pnl = proceeds - cost_sold
                withdrawn += proceeds * withdraw_frac          # 일부만 인출
                cash += proceeds * (1.0 - withdraw_frac)       # 나머지 운용현금 재투입
                trim_by_year[today.year] = trim_by_year.get(today.year, 0.0) + proceeds
                n_trim += 1; n_units += 1
                if pnl > 0:
                    wins += 1; gross_w += pnl
                else:
                    gross_l += pnl
                if sell_sh >= pos.shares - 1e-9:
                    del positions[ticker]; del cost[ticker]
                else:
                    pos.shares -= sell_sh
                    cost[ticker] -= cost_sold

        # ---------- 5) 일별 평가 + 비중 상한 준수 확인 ----------
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
            if posmat is not None:
                posmat[di, col] = e
        total_now = cash + hv + withdrawn
        daily_total[di] = total_now
        daily_withdrawn[di] = withdrawn
        daily_cash[di] = cash
        daily_posval[di] = hv
        daily_npos[di] = len(positions)
        # 최대비중은 '상한 판정에 쓰는 분모'로 측정 (신:운용자산 / 구:총자산)
        denom_now = (cash + hv + withdrawn) if denom_incl_withdrawn else (cash + hv)
        if denom_now > 0:
            w = max_e / denom_now
            if w > max_weight:
                max_weight = w

    # ---------- 성과지표 (총자산 기준) ----------
    dt = pd.Series(daily_total, index=td)
    wd = pd.Series(daily_withdrawn, index=td)
    mask = dt.index >= TRADE_START
    dt = dt[mask]; wd = wd[mask]
    final = float(dt.iloc[-1])
    years = (dt.index[-1] - dt.index[0]).days / 365.25
    total_ret = final / INITIAL_CASH - 1.0
    cagr = (final / INITIAL_CASH) ** (1.0 / max(years, 1e-9)) - 1.0
    peak = dt.cummax()
    mdd = float(((dt - peak) / peak).min())
    rets = dt.pct_change().fillna(0)
    sharpe = float((rets.mean() / (rets.std() + 1e-12)) * math.sqrt(252))
    calmar = cagr / abs(mdd) if mdd != 0 else float("inf")
    win_rate = wins / n_units if n_units else 0.0
    pl = abs(gross_w / gross_l) if gross_l != 0 else float("inf")

    # 연도별 인출가능금 잔액 & 증가분
    wd_ye = wd.resample("YE").last()
    wd_year = {}
    prev = 0.0
    for ts, val in wd_ye.items():
        wd_year[ts.year] = {"balance": float(val), "added": float(val) - prev}
        prev = float(val)

    op_final = final - withdrawn   # 운용중 자산 = 총자산 - 인출가능금
    tag = "운용기준" if not denom_incl_withdrawn else "총자산기준"
    name = f"{int(cap*100)}%상한/인출{int(round(withdraw_frac*100))}%"

    return {"name": name, "cap": cap, "tag": tag,
            "withdraw_frac": withdraw_frac,
            "denom_incl_withdrawn": denom_incl_withdrawn,
            "final": final, "total_ret": total_ret,
            "CAGR": cagr, "MDD": mdd, "Sharpe": sharpe, "Calmar": calmar,
            "PL": pl, "win_rate": win_rate,
            "n_buy": n_buy, "n_add": n_add, "n_trim": n_trim, "n_final": n_final,
            "n_units": n_units, "max_weight": max_weight,
            "withdrawn": withdrawn, "op_final": op_final,
            "wd_year": wd_year, "yearly": compute_yearly(dt), "equity": dt,
            "trim_by_year": trim_by_year, "buy_by_year": buy_by_year,
            "daily_cash": pd.Series(daily_cash, index=td),
            "daily_posval": pd.Series(daily_posval, index=td),
            "daily_npos": pd.Series(daily_npos, index=td),
            "posmat": posmat, "daily_total_full": daily_total, "td": td}


def fracsweep(store, cap=0.60, fracs=(0.30, 0.50, 0.70, 1.00),
              denom_incl_withdrawn=False):
    """비중상한 cap 고정, 인출비율(withdraw_frac)만 바꿔 공정 비교 (proportional 사이징)."""
    W = 100
    denom_txt = "총자산 기준(인출금 포함)" if denom_incl_withdrawn else "운용자산 기준"
    log(f"[fracsweep] 상한 {int(cap*100)}% 고정, 인출비율 {[int(f*100) for f in fracs]}%, "
        f"분모={denom_txt}")
    base = run_variant(store, [(0.08, 0.50)], "기존S1v2(인출없음)", 1.0)
    base["yearly"] = compute_yearly(base["equity"])
    res = [run_cap(store, cap, denom_incl_withdrawn=denom_incl_withdrawn, withdraw_frac=f,
                   proportional=True) for f in fracs]

    print()
    print("=" * W)
    print(f" 인출비율 비교  [비중상한 {int(cap*100)}% 고정 · 슬롯금액 자본비례 · 분모={denom_txt}]")
    print("=" * W)
    print(f"  {'인출비율':>8s} {'CAGR':>8s} {'MDD':>8s} {'Sharpe':>7s} {'Calmar':>7s} "
          f"{'손익비':>6s} {'최종운용자산':>10s} {'누적인출금':>10s}")
    print(f"  {'(인출없음)':>8s} {fpct(base['CAGR']):>8s} {fpct(base['MDD']):>8s} "
          f"{base['Sharpe']:>7.2f} {base['Calmar']:>7.2f} {base['PL']:>6.2f} "
          f"{base['final']/1e8:>8,.2f}억 {0.0:>8,.2f}억")
    for f, r in zip(fracs, res):
        print(f"  {int(f*100):>7d}% {fpct(r['CAGR']):>8s} {fpct(r['MDD']):>8s} "
              f"{r['Sharpe']:>7.2f} {r['Calmar']:>7.2f} {r['PL']:>6.2f} "
              f"{r['op_final']/1e8:>8,.2f}억 {r['withdrawn']/1e8:>8,.2f}억")

    # 연도별 신규 인출액
    yrs = sorted(set().union(*[set(r["wd_year"]) for r in res]))
    print()
    print("=" * W)
    print(" ★ 연도별 신규 인출액 (단위:억) — 후반까지 꾸준히 나오는지")
    print("=" * W)
    print("  " + f"{'인출비율':>8s}" + "".join(f"{y:>9d}" for y in yrs))
    for f, r in zip(fracs, res):
        row = "  " + f"{int(f*100):>7d}%"
        for y in yrs:
            info = r["wd_year"].get(y)
            cell = f"{info['added']/1e8:.2f}" if info else "-"
            row += f"{cell:>9s}"
        print(row)

    # 연도별 수익률
    yrs2 = sorted(set(base["yearly"]) | set().union(*[set(r["yearly"]) for r in res]))
    print()
    print("=" * W)
    print(" 연도별 수익률 (총자산 기준)")
    print("=" * W)
    print("  " + f"{'인출비율':>8s}" + "".join(f"{y:>9d}" for y in yrs2))
    print("  " + f"{'(없음)':>8s}" + "".join(
        f"{fpct(base['yearly'].get(y)) if y in base['yearly'] else '-':>9s}" for y in yrs2))
    for f, r in zip(fracs, res):
        row = "  " + f"{int(f*100):>7d}%"
        for y in yrs2:
            row += f"{fpct(r['yearly'].get(y)) if y in r['yearly'] else '-':>9s}"
        print(row)

    # CSV + 차트
    rows = []
    for f, r in zip(fracs, res):
        row = {"withdraw_frac": f, "CAGR": r["CAGR"], "MDD": r["MDD"],
               "Sharpe": r["Sharpe"], "Calmar": r["Calmar"], "PL": r["PL"],
               "op_final": r["op_final"], "withdrawn": r["withdrawn"], "final": r["final"]}
        for y in yrs:
            wy = r["wd_year"].get(y)
            row[f"wd_added_{y}"] = (wy["added"] if wy else 0.0)
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(RESULT_DIR, "fracsweep_results.csv"),
                              index=False, encoding="utf-8-sig")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for fnt in ["Malgun Gothic", "NanumGothic", "AppleGothic", "Gulim"]:
            try:
                matplotlib.rcParams["font.family"] = fnt
                matplotlib.rcParams["axes.unicode_minus"] = False
                break
            except Exception:
                continue
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        ax1.plot(base["equity"].index, base["equity"].values, lw=1.6, color="#333333",
                 label=f"인출없음 ({fpct(base['CAGR'])})")
        for f, r in zip(fracs, res):
            ax1.plot(r["equity"].index, r["equity"].values, lw=1.3,
                     label=f"인출{int(f*100)}% ({fpct(r['CAGR'])})")
        ax1.set_yscale("log"); ax1.set_title(f"총자산 곡선 (상한 {int(cap*100)}%)")
        ax1.legend(fontsize=9); ax1.grid(alpha=0.3, which="both")
        for f, r in zip(fracs, res):
            wd = pd.Series({pd.Timestamp(f"{y}-12-31"): v["balance"]
                            for y, v in r["wd_year"].items()})
            ax2.plot(wd.index, wd.values / 1e8, "o-", lw=1.4, label=f"인출{int(f*100)}%")
        ax2.set_title("누적 인출가능금 (억원)")
        ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
        plt.tight_layout()
        png = os.path.join(RESULT_DIR, "fracsweep_compare.png")
        plt.savefig(png, dpi=140); plt.close(fig)
        print(f"\n  그래프 저장 : {png}")
    except Exception as e:
        print(f"\n  (그래프 생략: {e})")
    print(f"  결과 저장 : {RESULT_DIR}")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="S1 비중상한+인출 백테스트")
    ap.add_argument("--end", type=str, default=None, help="종료일(YYYY-MM-DD).")
    ap.add_argument("--fracsweep", action="store_true",
                    help="비중상한 고정, 인출비율 스윕 비교.")
    ap.add_argument("--cap", type=float, default=0.60, help="fracsweep에서 쓸 비중상한.")
    ap.add_argument("--denomtotal", action="store_true",
                    help="fracsweep 비중분모를 총자산기준(인출금 포함)으로.")
    ap.add_argument("--withzero", action="store_true",
                    help="fracsweep에 인출비율 0 퍼센트(인출없음) 포함.")
    args = ap.parse_args()
    end = pd.Timestamp(args.end) if args.end else None
    log("===== S1 비중상한+인출 백테스트 ====="
        + (f" (종료일 {end.date()})" if end is not None else ""))
    store = data_layer.get_store(end_date=end)

    if args.fracsweep:
        fr = (0.0, 0.30, 0.50, 0.70, 1.00) if args.withzero else (0.30, 0.50, 0.70, 1.00)
        fracsweep(store, cap=args.cap, fracs=fr, denom_incl_withdrawn=args.denomtotal)
        return

    caps = [0.40, 0.50, 0.60]

    # 비교기준: 기존 S1 v2 (+8%@50%익절, 인출 없음)
    log("[base] 기존 S1 v2(+8%50%익절·전량손절) 실행 중...")
    base = run_variant(store, [(0.08, 0.50)], "기존S1v2(인출없음)", 1.0)
    base["yearly"] = compute_yearly(base["equity"])

    # 이번(부분인출 50%) + 참고(전액인출 100%), 둘 다 운용자산 기준 분모.
    # ★ 동결버그 수정: 신규매수/불타기 금액을 운용자본에 비례(proportional=True).
    new_res = [run_cap(store, c, denom_incl_withdrawn=False, withdraw_frac=0.5,
                       proportional=True) for c in caps]
    ref_res = [run_cap(store, c, denom_incl_withdrawn=False, withdraw_frac=1.0,
                       proportional=True) for c in caps]
    log("[cap] 인출50% / 인출100% 각 3개 상한 완료")

    W = 108

    def perf_row(r, show_w=True):
        w = f"{r['max_weight']*100:>6.1f}%" if show_w else f"{'-':>7s}"
        print(f"  {r['name']:>18s} {fpct(r['CAGR']):>8s} {fpct(r['MDD']):>8s} "
              f"{r['Sharpe']:>7.2f} {r['Calmar']:>7.2f} {r['PL']:>6.2f} "
              f"{r['win_rate']*100:>6.1f}% {w}")

    # ---------- 성과 요약 ----------
    print()
    print("=" * W)
    print(" 비중상한+부분인출 비교 (운용자산 기준 분모 고정, 동일 기간·데이터)")
    print("=" * W)
    print(f"  {'전략':>18s} {'CAGR':>8s} {'MDD':>8s} {'Sharpe':>7s} {'Calmar':>7s} "
          f"{'손익비':>6s} {'승률':>7s} {'최대비중':>7s}")
    perf_row(base, show_w=False)
    print("  " + "- 이번: 초과분 매도대금 50%만 인출 " + "-" * 30)
    for r in new_res:
        perf_row(r)
    print("  " + "- 참고: 초과분 매도대금 100% 전액인출 " + "-" * 27)
    for r in ref_res:
        perf_row(r)

    # ---------- 최종 자산 분해 ----------
    print()
    print("=" * W)
    print(" 전체기간 종료 후 최종결과 (운용중 자산 / 누적 인출가능금)")
    print("=" * W)
    print(f"  {'전략':>18s} {'총자산':>12s} {'운용중자산':>12s} {'누적인출가능금':>14s} {'인출/총자산':>9s}")
    print(f"  {base['name']:>18s} {base['final']/1e8:>10,.2f}억 {base['final']/1e8:>10,.2f}억 "
          f"{0.0:>10,.2f}억 {'0.0%':>9s}")
    for r in new_res + ref_res:
        print(f"  {r['name']:>18s} {r['final']/1e8:>10,.2f}억 {r['op_final']/1e8:>10,.2f}억 "
              f"{r['withdrawn']/1e8:>12,.2f}억 {r['withdrawn']/r['final']*100:>8.1f}%")

    # ---------- 연도별 신규 인출액 (핵심) ----------
    yrs = sorted(set().union(*[set(r["wd_year"]) for r in new_res + ref_res]))
    print()
    print("=" * W)
    print(" ★ 연도별 '신규 인출액' (그 해에 새로 인출된 금액, 단위:억) — 매년 꾸준히 나오는지 확인")
    print("=" * W)
    print("  " + f"{'전략':>18s}" + "".join(f"{y:>9d}" for y in yrs))

    def add_row(r):
        row = "  " + f"{r['name']:>18s}"
        for y in yrs:
            info = r["wd_year"].get(y)
            cell = f"{info['added']/1e8:.2f}" if info else "-"
            row += f"{cell:>9s}"
        print(row)

    print("  " + "- 이번(인출50%) " + "-" * 28)
    for r in new_res:
        add_row(r)
    print("  " + "- 참고(인출100%) " + "-" * 27)
    for r in ref_res:
        add_row(r)

    # ---------- 연도별 수익률표 ----------
    yrs2 = sorted(set(base["yearly"]) | set().union(*[set(r["yearly"]) for r in new_res + ref_res]))
    print()
    print("=" * W)
    print(" 연도별 수익률 (총자산 기준)")
    print("=" * W)
    print("  " + f"{'전략':>18s}" + "".join(f"{y:>9d}" for y in yrs2))
    def yrow(r):
        return "  " + f"{r['name']:>18s}" + "".join(
            f"{fpct(r['yearly'].get(y)) if y in r['yearly'] else '-':>9s}" for y in yrs2)
    print(yrow(base))
    print("  " + "- 이번(인출50%) " + "-" * 28)
    for r in new_res:
        print(yrow(r))
    print("  " + "- 참고(인출100%) " + "-" * 27)
    for r in ref_res:
        print(yrow(r))

    _save(base, new_res, ref_res, yrs2)


def _save(base, new_res, old_res, yrs):
    rows = []
    for r in [base] + new_res + old_res:
        row = {"name": r["name"], "tag": r.get("tag", "base"),
               "CAGR": r["CAGR"], "MDD": r["MDD"],
               "Sharpe": r["Sharpe"], "Calmar": r["Calmar"], "PL": r["PL"],
               "win_rate": r["win_rate"],
               "withdrawn": r.get("withdrawn", 0.0),
               "op_final": r.get("op_final", r["final"]),
               "final": r["final"], "max_weight": r.get("max_weight", float("nan"))}
        for y in yrs:
            row[f"ret_{y}"] = r["yearly"].get(y)
            wy = r.get("wd_year", {}).get(y)
            row[f"wd_added_{y}"] = (wy["added"] if wy else 0.0)
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(RESULT_DIR, "capwithdraw_results.csv"),
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
        ax1.plot(base["equity"].index, base["equity"].values, lw=1.6,
                 label=f"기존 S1 v2 ({fpct(base['CAGR'])})", color="#333333")
        for r in new_res:
            ax1.plot(r["equity"].index, r["equity"].values, lw=1.3,
                     label=f"{r['name']} ({fpct(r['CAGR'])})")
        ax1.set_yscale("log"); ax1.set_title("총자산 곡선 (실선=인출50%)")
        ax1.legend(fontsize=8); ax1.grid(alpha=0.3, which="both")
        # 인출50%(실선) vs 인출100%(점선) 누적 인출가능금 비교
        for r in new_res:
            wd = pd.Series({pd.Timestamp(f"{y}-12-31"): v["balance"]
                            for y, v in r["wd_year"].items()})
            ax2.plot(wd.index, wd.values / 1e8, "o-", lw=1.5, label=r["name"])
        for r in old_res:
            wd = pd.Series({pd.Timestamp(f"{y}-12-31"): v["balance"]
                            for y, v in r["wd_year"].items()})
            ax2.plot(wd.index, wd.values / 1e8, "o--", lw=1.0, alpha=0.5, label=r["name"])
        ax2.set_title("누적 인출가능금 (억원): 인출50%(실선) vs 인출100%(점선)")
        ax2.legend(fontsize=7, ncol=2); ax2.grid(alpha=0.3)
        plt.tight_layout()
        png = os.path.join(RESULT_DIR, "capwithdraw_compare.png")
        plt.savefig(png, dpi=140); plt.close(fig)
        print(f"\n  그래프 저장 : {png}")
    except Exception as e:
        print(f"\n  (그래프 생략: {e})")
    print(f"  결과 저장 : {RESULT_DIR}")


if __name__ == "__main__":
    main()
