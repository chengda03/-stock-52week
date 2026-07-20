# -*- coding: utf-8 -*-
"""
backtest_qv.py — "가치우량주 정배열"(전략2) 백테스트 엔진
============================================================================
S1(strategy_core/backtest_v3)와 완전히 독립된 전략2 엔진입니다.
데이터는 data_layer_qv, 판정은 strategy_qv 를 사용합니다.

운용(§6):
  · 리밸런싱: 매월 첫 거래일
  · 매수조건 8개 통과 종목 중 PER 낮은 순 상위 20종목, 동일비중(각 5%)
  · 20종목 못 채우면 현금(억지로 안 채움)
매도(§5, 매월 첫 거래일 점검):
  · 1) 현재가<20일선  2) -10% 손절  (매월)
  · 3) 분기(1/4/7/10월) 첫달: §2 8개 재점검 → 탈락 시 매도
리밸런싱 상위20 유지/교체(§6):
  · 보유종목이 이번달 '통과+PER 상위20' 안이면 유지, 밖이면 매도(밀려남)
비용(§7): 매수 0.115%(수수료0.015+슬리피지0.1), 매도 0.295%(+거래세0.18)

★ 정직성(§9)
  · 각 리밸런싱에서 종목이 '실제로 어느 FY 재무를, 실접수일/추정공시일 중
    무엇으로' 썼는지 샘플 10건을 로그로 출력(룩어헤드 육안검증).
  · 유니버스는 현재 상장사 기준 → 상장폐지 종목 누락(생존편향) 있음.
  · 이 전략은 아직 검증되지 않은 신규 설계안임.
"""
from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd

import data_layer_qv as dlq
import strategy_qv as sq
from strategy_qv import (
    check_buy_conditions, check_sell_basic, sort_key_per,
    NUM_STOCKS, WEIGHT_PER_STOCK, INITIAL_CAPITAL,
)
from backtest_roe_eps_event import log, fpct

TRADE_START = pd.Timestamp("2020-01-01")
BUY_COST = 0.00115     # 수수료0.015% + 슬리피지0.1%
SELL_COST = 0.00295    # 수수료0.015% + 거래세0.18% + 슬리피지0.1%


def _metrics(equity: pd.Series, initial: float) -> dict:
    """자산곡선 → CAGR/MDD/Sharpe/Calmar/총수익률."""
    eq = equity[equity.index >= TRADE_START]
    final = float(eq.iloc[-1])
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (final / initial) ** (1.0 / max(years, 1e-9)) - 1.0
    peak = eq.cummax()
    mdd = float(((eq - peak) / peak).min())
    rets = eq.pct_change().fillna(0)
    sharpe = float((rets.mean() / (rets.std() + 1e-12)) * math.sqrt(252))
    calmar = cagr / abs(mdd) if mdd != 0 else float("inf")
    total = final / initial - 1.0
    return {"CAGR": cagr, "MDD": mdd, "Sharpe": sharpe, "Calmar": calmar,
            "total": total, "final": final}


def run_qv(store, end_date=None, pit_samples: int = 10) -> dict:
    """전략2 백테스트 본체. 반환: 성과지표 + 자산곡선 + PIT 샘플 로그."""
    td = store.trading_days
    close_v = store.close_v
    close_ff = store.close_ff
    c2c = store.code_to_col
    start_di = int(np.searchsorted(td.values, TRADE_START.to_datetime64()))

    cash = float(INITIAL_CAPITAL)
    pos: dict[str, dict] = {}      # ticker -> {shares, entry_price}
    cost: dict[str, float] = {}    # ticker -> 투입원가(수수료 포함)
    equity_arr = np.empty(len(td))

    n_buy = n_sell = wins = 0
    gross_w = gross_l = 0.0
    hold_counts = []
    pit_log = []                   # (날짜, 종목, FY, estimated, per, roe/roe_prev)

    def sell(t, price, reason):
        nonlocal cash, n_sell, wins, gross_w, gross_l
        proceeds = price * pos[t]["shares"] * (1.0 - SELL_COST)
        pnl = proceeds - cost[t]
        cash += proceeds
        n_sell += 1
        if pnl > 0:
            wins += 1; gross_w += pnl
        else:
            gross_l += pnl
        del pos[t]; del cost[t]

    for di in range(len(td)):
        d = td[di]
        if di < start_di:
            equity_arr[di] = INITIAL_CAPITAL
            continue

        # 이번 거래일이 '이 달의 첫 거래일'인가?
        is_month_first = (di == start_di) or (d.month != td[di - 1].month) \
            or (d.year != td[di - 1].year)

        if is_month_first:
            as_of = d
            is_q = d.month in (1, 4, 7, 10)   # 분기 첫달(§5-3 재점검)
            sold_today = set()

            # ---- A. 매도규칙(§5) : 20일선 이탈 / -10% 손절 / (분기)조건탈락 ----
            for t in list(pos.keys()):
                col = c2c[t]
                pf = store.get_price_features(t, as_of)
                if pf is None:
                    continue
                do_sell, reason = check_sell_basic(
                    pos[t]["entry_price"], pf["price"], pf["ma20"])
                if (not do_sell) and is_q:
                    fin = store.get_financials(t, as_of)
                    ok, _ = check_buy_conditions(pf, fin, store.is_financial.get(t, False))
                    if not ok:
                        do_sell, reason = True, "분기재점검탈락"
                if do_sell:
                    sell(t, pf["price"], reason)
                    sold_today.add(t)

            # ---- B. 매수후보 산출(§2 8개 통과) + PER 오름차순 ----
            passers = []
            for t in store.get_universe(as_of):
                if t in sold_today:
                    continue
                pf = store.get_price_features(t, as_of)
                if pf is None:
                    continue
                fin = store.get_financials(t, as_of)
                ok, info = check_buy_conditions(pf, fin, store.is_financial.get(t, False))
                if ok:
                    passers.append((t, pf, info))
            passers.sort(key=lambda x: sort_key_per(x[2]))
            target = [t for (t, _, _) in passers[:NUM_STOCKS]]
            target_set = set(target)

            # ---- C. 상위20 밖으로 밀려난 보유종목 매도(§6 교체) ----
            for t in list(pos.keys()):
                if t not in target_set:
                    col = c2c[t]
                    px = close_v[di, col]
                    if not np.isfinite(px):
                        px = close_ff[di, col]
                    sell(t, float(px), "상위20이탈")

            # ---- D. 빈 슬롯을 상위 후보로 채움(동일비중 5%) ----
            hv = sum(pos[t]["shares"] * close_ff[di, c2c[t]] for t in pos)
            total_eq = cash + hv
            for (t, pf, info) in passers:
                if len(pos) >= NUM_STOCKS:
                    break
                if t in pos:
                    continue
                price = pf["price"]
                target_val = WEIGHT_PER_STOCK * total_eq
                sh = int(target_val // price)
                if sh > 0:
                    spent = sh * price * (1.0 + BUY_COST)
                    if spent > cash:  # 현금 부족 시 가능한 만큼 축소
                        sh = int((cash / (1.0 + BUY_COST)) // price)
                        spent = sh * price * (1.0 + BUY_COST)
                    if sh > 0 and spent <= cash:
                        pos[t] = {"shares": float(sh), "entry_price": price}
                        cost[t] = spent
                        cash -= spent
                        n_buy += 1
                        if len(pit_log) < pit_samples:
                            cur = info  # info 에 per/year/estimated 포함
                            prev_roe = None
                            fin = store.get_financials(t, as_of)
                            if fin and fin["prev"]:
                                prev_roe = fin["prev"].get("roe")
                            cur_roe = fin["cur"].get("roe") if fin else None
                            pit_log.append({
                                "date": as_of.date(), "ticker": t,
                                "name": store.name_map.get(t, ""),
                                "FY": info["year"], "estimated": info["estimated"],
                                "per": info["per"], "pbr": info["pbr"],
                                "roe": cur_roe, "roe_prev": prev_roe,
                            })

            hold_counts.append(len(pos))

        # ---- 매일 평가 ----
        hv = 0.0
        for t in pos:
            col = c2c[t]
            p = close_v[di, col]
            if not np.isfinite(p):
                p = close_ff[di, col]
                if not np.isfinite(p):
                    p = pos[t]["entry_price"]
            hv += p * pos[t]["shares"]
        equity_arr[di] = cash + hv

    equity = pd.Series(equity_arr, index=td)
    m = _metrics(equity, INITIAL_CAPITAL)
    n_sold = n_sell
    m.update({
        "n_buy": n_buy, "n_sell": n_sold,
        "win_rate": wins / n_sold if n_sold else 0.0,
        "profit_factor": abs(gross_w / gross_l) if gross_l != 0 else float("inf"),
        "avg_holdings": float(np.mean(hold_counts)) if hold_counts else 0.0,
        "equity": equity, "pit_log": pit_log,
    })
    return m


def _kospi_bh(store) -> dict:
    """KOSPI 지수 Buy&Hold (같은 기간, 초기자본 1억)."""
    ks = pd.Series(store.kospi_close_v, index=store.trading_days)
    ks = ks[ks.index >= TRADE_START]
    curve = ks / ks.iloc[0] * INITIAL_CAPITAL
    return _metrics(curve, INITIAL_CAPITAL) | {"equity": curve}


def _run_s1(end_date):
    """비교용 S1 v2(+8%@50% 부분익절) — 같은 기간/초기자본으로 실행."""
    try:
        import data_layer as dl
        from backtest_s1_partial import run_variant
        s1_store = dl.get_store(end_date=end_date)
        return run_variant(s1_store, [(0.08, 0.50)], "S1 v2", 1.0)
    except Exception as e:
        log(f"[qv] ⚠ S1 비교 실행 실패({e}) — S1 열은 생략")
        return None


def main():
    import argparse
    ap = argparse.ArgumentParser(description="전략2(가치우량주 정배열) 백테스트")
    ap.add_argument("--end", type=str, default=None, help="종료일 YYYY-MM-DD")
    ap.add_argument("--smoke", action="store_true",
                    help="QV 부가필드(영업이익/부채) 미수집이어도 강제 실행(결과 무의미)")
    args = ap.parse_args()
    end = pd.Timestamp(args.end) if args.end else None

    log("===== 전략2: 가치우량주 정배열 백테스트 =====")
    store = dlq.get_store(end_date=end)

    if not store.has_qv_extras and not args.smoke:
        log("=" * 76)
        log("⛔ QV 부가필드(영업이익/부채총계/공시접수일)가 아직 수집되지 않았습니다.")
        log("   → 매수조건 §2-3(영업이익↑)·§2-6(부채비율)을 판정할 수 없어")
        log("     지금 실행하면 사실상 아무 종목도 매수하지 못합니다(무의미한 결과).")
        log("   먼저 아래를 실행해 데이터를 채운 뒤 다시 돌려주세요(DART 일일한도 주의):")
        log("     python fetch_qv_extras.py --workers 2 --sleep 0.4")
        log("   (강제로 확인만 하려면 --smoke 옵션)")
        log("=" * 76)
        return

    m = run_qv(store, end_date=end)
    bh = _kospi_bh(store)
    s1 = _run_s1(end)

    # ---- PIT 샘플(룩어헤드 육안검증) ----
    print("\n" + "=" * 92)
    print(" [PIT 검증] 각 매수 시점에 실제로 사용한 재무 (샘플 %d건)" % len(m["pit_log"]))
    print("=" * 92)
    print(f"  {'리밸일':>10s} {'종목':>7s} {'종목명':<10s} {'FY':>5s} {'공시':>6s} "
          f"{'PER':>6s} {'PBR':>5s} {'ROE':>6s} {'전년ROE':>7s}")
    for r in m["pit_log"]:
        disc = "추정" if r["estimated"] else "실제"
        per = f"{r['per']:.1f}" if r['per'] is not None else "-"
        pbr = f"{r['pbr']:.2f}" if r['pbr'] is not None else "-"
        roe = f"{r['roe']:.1f}" if r['roe'] is not None else "-"
        rpv = f"{r['roe_prev']:.1f}" if r['roe_prev'] is not None else "-"
        print(f"  {str(r['date']):>10s} {r['ticker']:>7s} {r['name'][:10]:<10s} "
              f"{r['FY']:>5d} {disc:>6s} {per:>6s} {pbr:>5s} {roe:>6s} {rpv:>7s}")

    # ---- 성과 비교표 ----
    def row(label, d):
        if d is None:
            return f"  {label:<16s} {'-':>9s}"
        return (f"  {label:<16s} {fpct(d['CAGR']):>9s} {fpct(d['MDD']):>9s} "
                f"{d.get('Sharpe',float('nan')):>7.2f} {d.get('Calmar',float('nan')):>7.2f} "
                f"{fpct(d['total']):>10s}")

    print("\n" + "=" * 92)
    print(" 전략2 vs KOSPI vs S1 v2  (2020-01-01 ~ %s, 초기자본 1억)"
          % store.trading_days[-1].date())
    print("=" * 92)
    print(f"  {'전략':<16s} {'CAGR':>9s} {'MDD':>9s} {'Sharpe':>7s} {'Calmar':>7s} {'총수익률':>10s}")
    print(row("전략2(QV)", m))
    print(row("KOSPI B&H", bh))
    if s1:
        print(row("S1 v2(+8%@50%)", s1))

    print("\n  [전략2 상세]")
    print(f"    · 매수 {m['n_buy']}회 / 매도 {m['n_sell']}회, 승률 {m['win_rate']*100:.1f}%, "
          f"손익비 {m['profit_factor']:.2f}")
    print(f"    · 평균 보유종목수 {m['avg_holdings']:.1f} / {NUM_STOCKS}")
    print(f"    · 최종자산 {m['final']/1e8:.2f}억  (KOSPI B&H {bh['final']/1e8:.2f}억)")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
