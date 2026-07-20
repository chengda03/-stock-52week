# -*- coding: utf-8 -*-
"""
backtest_pbrvalue.py
====================
[독립 전략 백테스트] 저PBR 가치투자전략(strategy_core_pbrvalue.py) 전용 엔진.
SF투자법과 '병합'이 아니라, 완전히 다른 두 전략의 성과를 나란히 비교하기 위한 것.

데이터/유니버스는 data_layer를 재사용(허용). 비교용으로 strategy_core를 import하지만(허용),
저PBR 전략의 매수판정에는 전혀 섞지 않는다(SF는 오직 'SF의 종목군/모멘텀'을 얻어 Jaccard·모멘텀
비교 진단에만 사용).

PIT PBR:
  bps_day[di,col] = BPS(그날 적용 회계연도) = EPS×100/ROE(%)   (data_layer 저장 배열에서 유도)
  pbr_v[di,col]   = 종가 / bps_day                             (그날까지 공시분만 사용 → 룩어헤드 없음)
  avg3y[di,col]   = pbr_v의 과거방향 756거래일 롤링평균(min 120일)
  매수 = pbr_v ≤ avg3y × 0.7,  우선순위 = pbr_v/avg3y 오름차순

산출: CAGR/MDD/Sharpe/Calmar/손익비/승률/총거래/통과평균종목수/현금유휴비율/최대단일종목비중
     + 연도별 독립(매년 1억 리셋) 수익률.
정직성: 약 6.5년 단일 경로. 데이터 시작(2019-06)한계로 초기엔 3년평균이 실제 3년 미만.
        BPS는 흑자(ROE>0·EPS>0) 기업만 유도 가능 → 적자기업은 유니버스에서 자동 제외(구현상 한계).
        우월/열위 단정 금지 — 성격이 다른 전략의 특징 비교. 과최적화 위험 유의.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

import data_layer
import strategy_core_pbrvalue as pv

INITIAL_CASH = 100_000_000
BUY_COST = 0.00115
SELL_COST = 0.00295

# SF투자법 base (사용자 제공, 전체기간 2020-01~2026-07)
SF_BASE = {"CAGR": 68.51, "MDD": -35.36, "Sharpe": 1.48, "Calmar": 1.94,
           "PL": 3.63, "win": 39.9}

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

WINDOWS = [
    ("전체", "2020-01-01", "2026-07-14"),
    ("2020", "2020-01-01", "2020-12-31"),
    ("2021", "2021-01-01", "2021-12-31"),
    ("2022", "2022-01-01", "2022-12-31"),
    ("2023", "2023-01-01", "2023-12-31"),
    ("2024", "2024-01-01", "2024-12-31"),
    ("2025", "2025-01-01", "2025-12-31"),
    ("2026", "2026-01-01", "2026-07-14"),
]


def build_pbr_panels(store):
    """BPS(그날 적용연도) → 일별 PBR → 3년 롤링평균 PBR 패널 생성 (전부 PIT)."""
    T = len(store.trading_days); N = len(store.valid_codes)
    # 연도별 BPS 배열
    bps_by_year = {}
    for yr, roe_a in store.roe_by_year.items():
        eps_a = store.eps_by_year.get(yr)
        if eps_a is None:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            bps = np.where((roe_a > 0) & (eps_a > 0), eps_a * 100.0 / roe_a, np.nan)
        bps_by_year[yr] = bps
    # 그날 적용연도 BPS 패널
    bps_day = np.full((T, N), np.nan)
    for yr, bps in bps_by_year.items():
        mask = (store.day_fin_year == yr)
        if mask.any():
            bps_day[mask, :] = bps[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        pbr_v = np.where(bps_day > 0, store.close_v / bps_day, np.nan)
    avg3y = (pd.DataFrame(pbr_v)
             .rolling(pv.PBR_AVG_TRADING_DAYS, min_periods=pv.PBR_AVG_MIN_DAYS)
             .mean().to_numpy(dtype=float))
    return pbr_v, avg3y


def _metrics(equity, days):
    s = pd.Series(equity, index=pd.DatetimeIndex(days))
    final = float(s.iloc[-1])
    years = (s.index[-1] - s.index[0]).days / 365.25
    period_ret = final / INITIAL_CASH - 1.0
    cagr = (final / INITIAL_CASH) ** (1.0 / max(years, 1e-9)) - 1.0
    peak = s.cummax()
    mdd = float(((s - peak) / peak).min())
    rets = s.pct_change().fillna(0)
    sharpe = float((rets.mean() / (rets.std() + 1e-12)) * math.sqrt(252))
    calmar = cagr / abs(mdd) if mdd != 0 else float("inf")
    return {"period_ret": period_ret, "CAGR": cagr, "MDD": mdd,
            "Sharpe": sharpe, "Calmar": calmar,
            "start": str(s.index[0].date()), "end": str(s.index[-1].date())}


def run_pbr(store, win_start, win_end, pbr_v, avg3y, collect=False) -> dict:
    td = store.trading_days
    close_v = store.close_v; close_ff = store.close_ff
    c2c = store.code_to_col
    ma_sell_v = store.ma_sell_v
    mom_v = store.mom_v

    ws = pd.Timestamp(win_start); we = pd.Timestamp(win_end)
    di_list = [di for di in range(len(td)) if ws <= td[di] <= we]

    cash = float(INITIAL_CASH)
    positions: dict[str, pv.PBRPosition] = {}
    cost: dict[str, float] = {}
    equity = []; days = []

    n_buy = n_add = n_partial = n_final = 0
    wins = 0; gross_w = gross_l = 0.0
    pass_sum = 0; pass_days = 0
    cash_ratio_sum = 0.0; eval_days = 0
    max_weight = 0.0
    bought_set: set[str] = set()
    buy_mom: list[float] = []

    for di in di_list:
        today = td[di].date()
        sold_today: set[str] = set()

        # 1) 전량매도
        for t in list(positions.keys()):
            col = c2c[t]; px = close_v[di, col]
            if not np.isfinite(px):
                continue
            pos = positions[t]
            ok, _ = pv.check_sell(float(px), pos.avg_price, float(ma_sell_v[di, col]))
            if ok:
                proceeds = px * pos.shares * (1.0 - SELL_COST)
                cash += proceeds
                pnl = proceeds - cost[t]
                if pnl > 0: wins += 1; gross_w += pnl
                else: gross_l += pnl
                n_final += 1
                del positions[t]; del cost[t]; sold_today.add(t)

        # 1-2) 부분익절
        for t in list(positions.keys()):
            col = c2c[t]; px = close_v[di, col]
            if not np.isfinite(px) or px <= 0:
                continue
            pos = positions[t]
            if pv.check_partial_exit(pos, float(px)):
                before = pos.shares
                sell_sh = pv.apply_partial_exit(pos, float(px))
                ratio = sell_sh / before
                proceeds = px * sell_sh * (1.0 - SELL_COST)
                cost_sold = cost[t] * ratio
                pnl = proceeds - cost_sold
                cash += proceeds; cost[t] -= cost_sold
                n_partial += 1
                if pnl > 0: wins += 1; gross_w += pnl
                else: gross_l += pnl

        # 2) 불타기 (시장필터 없음)
        if positions:
            def cur_ret(t):
                col = c2c[t]; px = close_v[di, col]
                return px / positions[t].avg_price if np.isfinite(px) else -1.0
            for t in sorted(positions.keys(), key=cur_ret, reverse=True):
                col = c2c[t]; px = close_v[di, col]
                if not np.isfinite(px) or px <= 0:
                    continue
                pos = positions[t]
                if pv.check_pyramid(pos, float(px), today, cash):
                    spent = pv.PYRAMID_ADD_AMOUNT_WON * (1.0 + BUY_COST)
                    if cash < spent:
                        continue
                    pv.apply_pyramid(pos, float(px), today)
                    cash -= spent; cost[t] += spent; n_add += 1

        # 3) 신규매수 (저PBR, 시장필터 없음)
        free = pv.NUM_SLOTS - len(positions)
        uni = store.get_universe(td[di])
        day_pass = 0
        cands = []  # (disc, ticker, price)
        for t in uni:
            col = c2c[t]; px = close_v[di, col]
            if not np.isfinite(px) or px <= 0:
                continue
            cur_pbr = pbr_v[di, col]; a3 = avg3y[di, col]
            if pv.check_buy_pbr(cur_pbr, a3):
                day_pass += 1
                if t not in positions and t not in sold_today:
                    cands.append((pv.discount_ratio(cur_pbr, a3), t, float(px), float(mom_v[di, col])))
        pass_sum += day_pass; pass_days += 1
        if free > 0:
            cands.sort(key=lambda x: x[0])   # 할인율 큰(=낮은 비율) 순
            for disc, t, px, mom in cands:
                if free <= 0:
                    break
                sh = int(pv.SLOT_AMOUNT_WON // px)
                if sh <= 0:
                    continue
                spent = sh * px * (1.0 + BUY_COST)
                if spent > cash:
                    continue
                positions[t] = pv.PBRPosition(ticker=t, entry_price=px, avg_price=px,
                                              shares=float(sh), entry_date=today)
                cost[t] = spent
                cash -= spent; n_buy += 1; free -= 1
                bought_set.add(t)
                if np.isfinite(mom):
                    buy_mom.append(mom)

        # 4) 일별 평가
        hv = 0.0; vmax = 0.0
        for t, pos in positions.items():
            col = c2c[t]; p = close_v[di, col]
            if not np.isfinite(p):
                p = close_ff[di, col]
                if not np.isfinite(p):
                    p = pos.avg_price
            v = p * pos.shares; hv += v
            if v > vmax:
                vmax = v
        total = cash + hv
        equity.append(total); days.append(td[di])
        if total > 0:
            cash_ratio_sum += cash / total
            if vmax / total > max_weight:
                max_weight = vmax / total
            eval_days += 1

    m = _metrics(equity, days)
    n_sell = n_partial + n_final
    m.update({
        "PL": abs(gross_w / gross_l) if gross_l != 0 else float("inf"),
        "win_rate": wins / n_sell if n_sell else 0.0,
        "n_buy": n_buy, "n_add": n_add, "n_partial": n_partial, "n_final": n_final,
        "n_trades": n_buy + n_add + n_partial + n_final,
        "avg_pass": pass_sum / pass_days if pass_days else 0.0,
        "cash_idle": cash_ratio_sum / eval_days if eval_days else 0.0,
        "max_weight": max_weight,
        "bought_set": bought_set, "buy_mom": buy_mom,
    })
    return m


def run_sf_base(store):
    """[비교 진단 전용] SF투자법 종목군/모멘텀 수집. strategy_core는 import만(데이터 비교용)."""
    import strategy_core as sc

    td = store.trading_days
    close_v = store.close_v; c2c = store.code_to_col
    start_di = store.start_di

    def snap(col, di):
        yr = int(store.day_fin_year[di])
        roe_a = store.roe_by_year.get(yr); eps_a = store.eps_by_year.get(yr)
        roe = float(roe_a[col]) if roe_a is not None else float("nan")
        eps = float(eps_a[col]) if eps_a is not None else float("nan")
        return sc.StockSnapshot(ticker=store.valid_codes[col], date=td[di].date(),
                                price=float(close_v[di, col]),
                                ma_buy=float(store.ma_buy_v[di, col]),
                                ma_sell=float(store.ma_sell_v[di, col]),
                                roe_pct=roe, eps=eps,
                                momentum_20d=float(store.mom_v[di, col]),
                                trading_value_20d_avg=float(store.tv_v[di, col]))

    cash = float(INITIAL_CASH); positions = {}; cost = {}
    bought_set = set(); buy_mom = []
    for di in range(start_di, len(td)):
        today = td[di].date()
        is_bull = sc.is_bull_market(store.kospi_close_v[di], store.kospi_ma200_v[di])
        sold = set()
        for t in list(positions.keys()):
            col = c2c[t]; px = close_v[di, col]
            if not np.isfinite(px):
                continue
            s, _ = sc.check_sell_condition(positions[t], snap(col, di))
            if s:
                cash += px * positions[t].shares * (1 - SELL_COST)
                del positions[t]; del cost[t]; sold.add(t)
        for t in list(positions.keys()):
            col = c2c[t]; px = close_v[di, col]
            if not np.isfinite(px) or px <= 0:
                continue
            if sc.check_partial_exit(positions[t], float(px)):
                sh = sc.apply_partial_exit(positions[t], float(px))
                cash += px * sh * (1 - SELL_COST)
        if is_bull and positions:
            for t in sorted(positions.keys(),
                            key=lambda t: (close_v[di, c2c[t]] / positions[t].avg_price)
                            if np.isfinite(close_v[di, c2c[t]]) else -1, reverse=True):
                col = c2c[t]; px = close_v[di, col]
                if not np.isfinite(px) or px <= 0:
                    continue
                if sc.check_pyramid(positions[t], float(px), today, is_bull, cash):
                    spent = sc.PYRAMID_ADD_AMOUNT_WON * (1 + BUY_COST)
                    if cash < spent:
                        continue
                    sc.apply_pyramid(positions[t], float(px), today)
                    cash -= spent; cost[t] += spent
        free = sc.NUM_SLOTS - len(positions)
        if is_bull and free > 0:
            cands = []
            for t in store.get_universe(td[di]):
                if t in positions or t in sold:
                    continue
                col = c2c[t]; px = close_v[di, col]
                if not np.isfinite(px) or px <= 0:
                    continue
                s = snap(col, di)
                ok, _ = sc.check_buy_conditions(s, is_bull)
                if ok:
                    cands.append(s)
            for s in sc.rank_by_momentum(cands):
                if free <= 0:
                    break
                px = s.price; sh = int(sc.SLOT_AMOUNT_WON // px)
                if sh <= 0:
                    continue
                spent = sh * px * (1 + BUY_COST)
                if spent > cash:
                    continue
                positions[s.ticker] = sc.Position(ticker=s.ticker, entry_price=px, avg_price=px,
                                                  shares=float(sh), entry_date=today)
                cost[s.ticker] = spent
                cash -= spent; free -= 1
                bought_set.add(s.ticker); buy_mom.append(s.momentum_20d)
    return bought_set, buy_mom


def main():
    store = data_layer.get_store()
    pbr_v, avg3y = build_pbr_panels(store)

    full = run_pbr(store, "2020-01-01", "2026-07-14", pbr_v, avg3y, collect=True)
    yearly = {tag: run_pbr(store, ws, we, pbr_v, avg3y)
              for tag, ws, we in WINDOWS if tag != "전체"}

    # ---- 저PBR 전체기간 지표 ----
    print("\n" + "=" * 100)
    print(f" [저PBR 가치투자전략] 전체기간 {full['start']}~{full['end']}  ·  1억 · 거래비용 반영")
    print("=" * 100)
    print(f"  CAGR {full['CAGR']*100:.2f}%  MDD {full['MDD']*100:.2f}%  Sharpe {full['Sharpe']:.2f}  "
          f"Calmar {full['Calmar']:.2f}  손익비 {full['PL']:.2f}  승률 {full['win_rate']*100:.1f}%")
    print(f"  총거래 {full['n_trades']} (신규 {full['n_buy']}·불타기 {full['n_add']}·부분익절 {full['n_partial']}·최종매도 {full['n_final']})")
    print(f"  통과 평균종목수 {full['avg_pass']:.1f}  ·  현금유휴비율 {full['cash_idle']*100:.1f}%  ·  최대단일종목비중 {full['max_weight']*100:.1f}%")

    # ---- SF투자법과 나란히 비교 (병합 아님, 서로 다른 두 전략) ----
    print("\n" + "=" * 100)
    print(" [비교] 완전히 다른 두 전략의 전체기간 성과 (병합 아님)")
    print("=" * 100)
    cmp = pd.DataFrame([
        {"전략": "SF투자법 base", "CAGR%": SF_BASE["CAGR"], "MDD%": SF_BASE["MDD"],
         "Sharpe": SF_BASE["Sharpe"], "Calmar": SF_BASE["Calmar"], "손익비": SF_BASE["PL"], "승률%": SF_BASE["win"]},
        {"전략": "저PBR 가치투자", "CAGR%": round(full["CAGR"]*100, 2), "MDD%": round(full["MDD"]*100, 2),
         "Sharpe": round(full["Sharpe"], 2), "Calmar": round(full["Calmar"], 2),
         "손익비": round(full["PL"], 2), "승률%": round(full["win_rate"]*100, 1)},
    ])
    print(cmp.to_string(index=False))

    # ---- 연도별 독립 ----
    print("\n" + "=" * 100)
    print(" [저PBR] 연도별 독립 백테스트 (매년 1억 리셋)")
    print("=" * 100)
    yrows = [{"연도": tag, "기간수익률%": round(yearly[tag]["period_ret"]*100, 2),
              "MDD%": round(yearly[tag]["MDD"]*100, 2), "신규매수": yearly[tag]["n_buy"],
              "총거래": yearly[tag]["n_trades"]}
             for tag in ["2020", "2021", "2022", "2023", "2024", "2025", "2026"]]
    print(pd.DataFrame(yrows).to_string(index=False))

    # ---- 진단: SF와 종목겹침(Jaccard) / 모멘텀 비교 ----
    print("\n" + "=" * 100)
    print(" [진단] SF투자법과의 성격 차이 (SF는 비교용 재현, 로직 혼합 아님)")
    print("=" * 100)
    sf_set, sf_mom = run_sf_base(store)
    pbr_set = full["bought_set"]
    jac = len(pbr_set & sf_set) / len(pbr_set | sf_set) * 100 if (pbr_set | sf_set) else 0.0
    print(f"  매수 종목군 Jaccard(저PBR ∩ SF / 합집합) = {jac:.1f}%")
    print(f"    · 저PBR 매수 고유종목 {len(pbr_set)}개 · SF 매수 고유종목 {len(sf_set)}개 · 공통 {len(pbr_set & sf_set)}개")
    pm = np.array(full["buy_mom"]); sm = np.array(sf_mom)
    print(f"  매수시점 20일모멘텀 평균: 저PBR {pm.mean()*100:+.2f}%  vs  SF {sm.mean()*100:+.2f}%  "
          f"(저PBR이 {'낮음(인기식은 종목)' if pm.mean() < sm.mean() else '높음'})")
    print(f"  승률/손익비 패턴: 저PBR 승률 {full['win_rate']*100:.1f}%·손익비 {full['PL']:.2f}  "
          f"vs  SF 승률 {SF_BASE['win']}%·손익비 {SF_BASE['PL']}")

    pd.DataFrame(yrows).to_csv(os.path.join(RESULT_DIR, "pbrvalue_yearly.csv"), index=False, encoding="utf-8-sig")
    cmp.to_csv(os.path.join(RESULT_DIR, "pbrvalue_vs_sf.csv"), index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print(" ⚠ 정직성: 약 6.5년 단일경로. 초기엔 3년평균PBR이 데이터한계로 3년 미만. 적자기업은 BPS유도불가로 제외.")
    print("            우월/열위 단정 아님 — 성격이 다른 전략. 과최적화 위험 존재.")
    print("=" * 100)
    return full, yearly


if __name__ == "__main__":
    main()
