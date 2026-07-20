# -*- coding: utf-8 -*-
"""
backtest_v3b.py  (90일선 매수필터 변형 백테스트 + 기존 S1v2 비교)
================================================================
backtest_v3.py 를 복사·변형한 스크립트입니다.
목적: 매수조건에 "현재가 > 90일선"을 추가한 변형(strategy_core_v2b)이
      기존 S1 v2(strategy_core) 대비 성과가 어떻게 달라지는지 '같은 엔진'으로
      공정 비교하는 것.

★ 기존 파일은 절대 수정하지 않습니다.
  - 판정 로직 원본:  strategy_core.py       (기존, 필터 없음)
  - 판정 로직 변형:  strategy_core_v2b.py   (90일선 매수필터 추가)
  - 매도/불타기/부분익절 판정은 두 버전이 완전히 동일하므로,
    엔진은 그 함수들을 strategy_core(=sc)에서 공용으로 쓰고
    '매수조건 함수만' 갈아끼워 두 번 돌립니다.

핵심 확인 포인트
----------------
"매수 즉시 매도" = 신규매수한 그 종가 기준으로 이미 매도조건(90일선 이탈 등)이
켜져 있는 경우. = 사자마자 다음날 팔릴 whipsaw 후보. 이 건수를 필터 OFF/ON
양쪽에서 세어 비교합니다. (필터 ON이면 원칙적으로 0에 수렴해야 함)

⚠️ 정직성: 이것도 조건을 추가/제거하는 그리드서치성 실험이라 과최적화 위험이
   있습니다. CAGR이 좋아지든 나빠지든 있는 그대로 보고합니다.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

# 공용(두 버전 동일) 판정 함수 & 상수는 원본 strategy_core 에서 가져온다.
import strategy_core as sc
# 매수조건만 다른 두 버전
from strategy_core import check_buy_conditions as buy_baseline      # 필터 없음(기존)
from strategy_core_v2b import check_buy_conditions as buy_filter90  # 90일선 필터 추가

import data_layer
from backtest_roe_eps_event import load_index, fpct, log

# --- 운용 설정 (backtest_v3.py 와 동일) ------------------------------------
INITIAL_CASH = 100_000_000
BUY_COST = 0.00115     # 매수 수수료 0.115%
SELL_COST = 0.00295    # 매도 수수료+세금 0.295%
TRADE_START = pd.Timestamp("2020-01-01")

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

# 사용자가 제시한 '기존 확정 전략' 참고 수치 (표의 기준 컬럼 참고용)
REF_BASELINE = {
    "CAGR": 0.6904, "MDD": -0.3198, "Sharpe": 1.54,
    "Calmar": 2.16, "손익비": 3.65, "승률": 0.387,
}


def _build_snap(store, col, di) -> sc.StockSnapshot:
    """store 배열에서 di일·col종목의 StockSnapshot 조립 (backtest_v3.py 와 동일)."""
    yr = int(store.day_fin_year[di])
    roe_a = store.roe_by_year.get(yr)
    eps_a = store.eps_by_year.get(yr)
    roe = float(roe_a[col]) if roe_a is not None else float("nan")
    eps = float(eps_a[col]) if eps_a is not None else float("nan")
    return sc.StockSnapshot(
        ticker=store.valid_codes[col],
        date=store.trading_days[di].date(),
        price=float(store.close_v[di, col]),
        ma_buy=float(store.ma_buy_v[di, col]),
        ma_sell=float(store.ma_sell_v[di, col]),
        roe_pct=roe, eps=eps,
        momentum_20d=float(store.mom_v[di, col]),
        trading_value_20d_avg=float(store.tv_v[di, col]),
    )


def run_engine(store, check_buy_fn, tag: str) -> dict:
    """
    S1 v2 엔진 1회 실행. check_buy_fn 만 갈아끼워 필터 OFF/ON 을 구분한다.
    (매도·부분익절·불타기는 두 버전 동일 → sc.* 공용 사용)

    추가 계측:
      - immediate_sell : 신규매수한 '그 종가' 기준으로 이미 매도조건이 켜진 건수
                         (= 사자마자 매도 대상이 되는 whipsaw 후보)
    """
    td = store.trading_days
    close_v = store.close_v
    close_ff = store.close_ff
    c2c = store.code_to_col
    start_di = store.start_di

    cash = float(INITIAL_CASH)
    positions: dict[str, sc.Position] = {}
    cost: dict[str, float] = {}
    daily_total = np.empty(len(td))

    n_buy = n_add = n_partial = n_final = 0
    wins = 0
    gross_w = gross_l = 0.0
    immediate_sell = 0            # ★ 핵심 계측
    immediate_list: list[dict] = []

    for di in range(len(td)):
        today = td[di].date()
        if di < start_di:
            daily_total[di] = INITIAL_CASH
            continue

        kc = store.kospi_close_v[di]
        km = store.kospi_ma200_v[di]
        is_bull = sc.is_bull_market(kc, km)
        sold_today: set[str] = set()

        # 1) 전량매도 (90일선/하드손절)
        for ticker in list(positions.keys()):
            col = c2c[ticker]
            px = close_v[di, col]
            if not np.isfinite(px):
                continue
            pos = positions[ticker]
            snap = _build_snap(store, col, di)
            should_sell, reason = sc.check_sell_condition(pos, snap)
            if should_sell:
                proceeds = px * pos.shares * (1.0 - SELL_COST)
                cash += proceeds
                pnl = proceeds - cost[ticker]
                if pnl > 0:
                    wins += 1; gross_w += pnl
                else:
                    gross_l += pnl
                n_final += 1
                del positions[ticker]; del cost[ticker]
                sold_today.add(ticker)

        # 1-2) 부분익절 (+8% 도달시 50%)
        for ticker in list(positions.keys()):
            col = c2c[ticker]
            px = close_v[di, col]
            if not np.isfinite(px) or px <= 0:
                continue
            pos = positions[ticker]
            if sc.check_partial_exit(pos, float(px)):
                shares_before = pos.shares
                sell_sh = sc.apply_partial_exit(pos, float(px))
                ratio = sell_sh / shares_before
                proceeds = px * sell_sh * (1.0 - SELL_COST)
                cost_sold = cost[ticker] * ratio
                pnl = proceeds - cost_sold
                cash += proceeds
                cost[ticker] -= cost_sold
                n_partial += 1
                if pnl > 0:
                    wins += 1; gross_w += pnl
                else:
                    gross_l += pnl

        # 2) 불타기 (강세장, 수익률 높은 순)
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
                if sc.check_pyramid(pos, float(px), today, is_bull, cash):
                    spent = sc.PYRAMID_ADD_AMOUNT_WON * (1.0 + BUY_COST)
                    if cash < spent:
                        continue
                    sc.apply_pyramid(pos, float(px), today)
                    cash -= spent
                    cost[ticker] += spent
                    n_add += 1

        # 3) 신규매수 (강세장, 빈 슬롯)
        free = sc.NUM_SLOTS - len(positions)
        if is_bull and free > 0:
            uni = store.get_universe(td[di])
            cands: list[sc.StockSnapshot] = []
            for t in uni:
                if t in positions or t in sold_today:
                    continue
                col = c2c[t]
                px = close_v[di, col]
                if not np.isfinite(px) or px <= 0:
                    continue
                snap = _build_snap(store, col, di)
                ok, _ = check_buy_fn(snap, is_bull)   # ★ 필터 OFF/ON 갈아끼우는 지점
                if ok:
                    cands.append(snap)
            for snap in sc.rank_by_momentum(cands):
                if free <= 0:
                    break
                px = snap.price
                sh = int(sc.SLOT_AMOUNT_WON // px)
                if sh <= 0:
                    continue
                spent = sh * px * (1.0 + BUY_COST)
                if spent > cash:
                    continue
                new_pos = sc.Position(ticker=snap.ticker, entry_price=px, avg_price=px,
                                      shares=float(sh), entry_date=today)
                # ★ 매수 즉시 매도 계측: 방금 산 종가 기준으로 매도조건이 이미 켜졌나?
                imm, imm_reason = sc.check_sell_condition(new_pos, snap)
                if imm:
                    immediate_sell += 1
                    immediate_list.append({"date": str(today), "ticker": snap.ticker,
                                            "name": store.name_map.get(snap.ticker, snap.ticker),
                                            "price": float(px), "ma_sell": float(snap.ma_sell),
                                            "reason": imm_reason})
                positions[snap.ticker] = new_pos
                cost[snap.ticker] = spent
                cash -= spent
                n_buy += 1
                free -= 1

        # 4) 일별 평가
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

    # ---- 성과지표 (backtest_v3.py 와 동일 방식) ----
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
    total_trades = n_buy + n_add + n_partial + n_final

    log(f"[{tag}] 완료: CAGR {cagr*100:.2f}% · 신규 {n_buy} · 매수즉시매도 {immediate_sell}")
    return {
        "tag": tag, "final": final, "total_ret": total_ret, "CAGR": cagr, "MDD": mdd,
        "Sharpe": sharpe, "Calmar": calmar, "손익비": pl, "승률": win_rate,
        "n_buy": n_buy, "n_add": n_add, "n_partial": n_partial, "n_final": n_final,
        "total_trades": total_trades, "immediate_sell": immediate_sell,
        "immediate_list": immediate_list,
        "start": str(dt.index[0].date()), "end": str(dt.index[-1].date()),
    }


def _fmt_pct(x):
    return f"{x*100:.2f}%"


def main():
    log("===== backtest_v3b (90일선 매수필터 변형 vs 기존 S1v2) 시작 =====")
    store = data_layer.get_store()

    base = run_engine(store, buy_baseline, tag="필터없음(기존 S1v2)")
    filt = run_engine(store, buy_filter90, tag="90일선 필터 추가")

    # ---------------- 비교표 ----------------
    rows = [
        ("CAGR", _fmt_pct(base["CAGR"]), _fmt_pct(filt["CAGR"])),
        ("MDD", _fmt_pct(base["MDD"]), _fmt_pct(filt["MDD"])),
        ("Sharpe", f"{base['Sharpe']:.2f}", f"{filt['Sharpe']:.2f}"),
        ("Calmar", f"{base['Calmar']:.2f}", f"{filt['Calmar']:.2f}"),
        ("손익비", f"{base['손익비']:.2f}", f"{filt['손익비']:.2f}"),
        ("승률", _fmt_pct(base["승률"]), _fmt_pct(filt["승률"])),
        ("총 거래횟수", f"{base['total_trades']}", f"{filt['total_trades']}"),
        ("  ├ 신규매수", f"{base['n_buy']}", f"{filt['n_buy']}"),
        ("  ├ 불타기", f"{base['n_add']}", f"{filt['n_add']}"),
        ("  ├ 부분익절", f"{base['n_partial']}", f"{filt['n_partial']}"),
        ("  └ 전량매도", f"{base['n_final']}", f"{filt['n_final']}"),
        ("매수 즉시 매도 건수", f"{base['immediate_sell']}", f"{filt['immediate_sell']}"),
    ]

    print()
    print("=" * 78)
    print(f" 90일선 매수필터 실험  ·  {base['start']} ~ {base['end']}  ·  1억 · 거래비용 반영")
    print("=" * 78)
    print(f" {'항목':<22s}{'기존 S1v2(필터없음)':>20s}{'90일선 필터추가':>18s}")
    print("-" * 78)
    for name, b, f in rows:
        print(f" {name:<22s}{b:>20s}{f:>18s}")
    print("-" * 78)
    print(" [참고] 사용자 제시 '기존 확정 전략' 수치: "
          f"CAGR {REF_BASELINE['CAGR']*100:.2f}% · MDD {REF_BASELINE['MDD']*100:.2f}% · "
          f"Sharpe {REF_BASELINE['Sharpe']:.2f} · Calmar {REF_BASELINE['Calmar']:.2f} · "
          f"손익비 {REF_BASELINE['손익비']:.2f} · 승률 {REF_BASELINE['승률']*100:.1f}%")
    print("        (위 '기존 S1v2(필터없음)' 열은 동일 엔진 재실행값이라 미세차이 가능)")

    print("\n [핵심] '매수 즉시 매도(당일 매수 + 그 종가로 이미 매도조건 발동)' 건수")
    print(f"   · 필터 없음(기존) : {base['immediate_sell']} 건   ← 지엔씨에너지 같은 경계매수 사례")
    print(f"   · 90일선 필터추가 : {filt['immediate_sell']} 건   ← 매수단계에서 90일선 위만 사므로 차단 기대")
    if base["immediate_list"]:
        print("   · (필터없음) 대표 사례 최대 5건:")
        for d in base["immediate_list"][:5]:
            print(f"       {d['date']} {d['name']}({d['ticker']}) "
                  f"종가 {d['price']:,.0f} ≤ 90일선 {d['ma_sell']:,.0f} · {d['reason']}")

    print("\n ⚠️ 과최적화 주의: 이 실험은 매수조건을 하나 더 얹은 그리드서치성 변형입니다.")
    print("    표본기간(2020~) 한정 결과이며, 조건 추가가 특정 구간에 유리/불리했을 수 있습니다.")
    print("    아래 CSV와 함께 '있는 그대로' 해석하세요.")

    # ---------------- CSV 저장 ----------------
    out = pd.DataFrame(
        [(n, b, f) for (n, b, f) in rows],
        columns=["항목", "기존_S1v2_필터없음", "90일선_필터추가"],
    )
    csv_path = os.path.join(RESULT_DIR, "buy90ma_filter_compare.csv")
    out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    # 매수 즉시 매도 상세도 별도 저장(있으면)
    if base["immediate_list"]:
        pd.DataFrame(base["immediate_list"]).to_csv(
            os.path.join(RESULT_DIR, "buy90ma_immediate_sell_baseline.csv"),
            index=False, encoding="utf-8-sig")
    print(f"\n 결과 CSV 저장: {csv_path}")


if __name__ == "__main__":
    main()
