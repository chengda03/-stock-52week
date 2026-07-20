# -*- coding: utf-8 -*-
"""
analyze_2024_trades.py
======================
[조회/분석 전용] SF투자법 2024년 신규매수 69건의 상세 로그 추출 + 분석.
strategy_core.py / backtest_v3.py는 '수정 없이' import만 하여 판정 로직을 그대로 재현.
2024년 독립창(1월1일 자산 1억 리셋 — 앞선 연도별 분석의 '2024 -42.45% · 신규매수 69건'과 동일 기준)을
재현하며, 각 신규매수의 생애주기(진입→불타기/부분익절→전량매도/기간종료)를 기록한다.

업종/섹터: FinanceDataReader StockListing('KRX-DESC')의 Sector(KRX 표준산업분류 기반) 사용, CSV 캐시.
결과 CSV: results/trades_2024_detail.csv / _sector.csv / _monthly.csv
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

import data_layer
import strategy_core as sc   # ★ import만(수정 없음). 판정 로직 그대로 사용.

INITIAL_CASH = 100_000_000
BUY_COST = 0.00115
SELL_COST = 0.00295
WIN_START = pd.Timestamp("2024-01-01")
WIN_END = pd.Timestamp("2024-12-31")

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)
# FDR StockListing('KRX-DESC')의 'Industry' = KRX 표준산업분류 업종명(통계청 표준산업분류 기반)
INDUSTRY_CACHE = os.path.join("data", "cache_52w_bt", "krx_industry.csv")

# 업종 텍스트 → 굵은 테마 버킷(쏠림 판정용). 키워드 우선순위 순서대로 매칭.
THEME_RULES = [
    ("반도체", ("반도체", "웨이퍼", "디스플레이", "전자부품", "전자집적")),
    ("2차전지/소재", ("전지", "축전지", "합성고무 및 플라스틱", "기초 화학", "1차 비철금속")),
    ("바이오/제약", ("의약", "제약", "바이오", "생물학", "자연과학 및 공학 연구개발")),
    ("화학", ("화학",)),
    ("기계/장비", ("기계", "장비 제조", "정밀기기", "측정")),
    ("전기장비", ("전기장비", "전기 장비")),
    ("자동차/부품", ("자동차", "차체", "운송장비")),
    ("조선/방산/항공", ("선박", "항공", "무기", "방위")),
    ("금융", ("금융", "은행", "보험", "증권", "지주")),
    ("건설/건자재", ("건설", "건물", "시멘트", "토목")),
    ("철강/금속", ("철강", "금속", "제철")),
    ("식품/음료", ("식료품", "음료", "담배", "가공")),
    ("유통/소비재", ("소매", "도매", "유통", "의복", "봉제")),
    ("미디어/엔터/IT서비스", ("방송", "영화", "소프트웨어", "게임", "정보서비스", "포털", "통신")),
    ("운송/물류", ("운송", "물류", "항만")),
]


def theme_of(industry: str, name: str) -> str:
    txt = f"{industry} {name}"
    for label, kws in THEME_RULES:
        if any(k in txt for k in kws):
            return label
    return "기타"


def load_sector_map(name_map: dict) -> dict:
    """{code: 업종명(Industry)}. FDR StockListing('KRX-DESC')의 Industry 컬럼. 실패시 캐시/미분류."""
    if os.path.exists(INDUSTRY_CACHE):
        df = pd.read_csv(INDUSTRY_CACHE, dtype={"code": str})
        return {str(c).zfill(6): (s if isinstance(s, str) and s and s != "nan" else "")
                for c, s in zip(df["code"], df["industry"].astype(str))}
    smap = {}
    try:
        import FinanceDataReader as fdr
        lst = fdr.StockListing("KRX-DESC")
        for _, r in lst.iterrows():
            code = str(r.get("Code", "")).strip().zfill(6)
            ind = r.get("Industry", "")
            smap[code] = str(ind).strip() if isinstance(ind, str) else ""
        recs = [{"code": c, "name": name_map.get(c, c), "industry": smap.get(c, "")}
                for c in name_map]
        pd.DataFrame(recs).to_csv(INDUSTRY_CACHE, index=False, encoding="utf-8-sig")
    except Exception as e:
        print(f"  ⚠ FDR 업종 조회 실패: {e} → 업종 '미분류'")
    return smap


def _snap(store, col, di):
    yr = int(store.day_fin_year[di])
    roe_a = store.roe_by_year.get(yr); eps_a = store.eps_by_year.get(yr)
    roe = float(roe_a[col]) if roe_a is not None else float("nan")
    eps = float(eps_a[col]) if eps_a is not None else float("nan")
    return sc.StockSnapshot(ticker=store.valid_codes[col], date=store.trading_days[di].date(),
                            price=float(store.close_v[di, col]),
                            ma_buy=float(store.ma_buy_v[di, col]),
                            ma_sell=float(store.ma_sell_v[di, col]),
                            roe_pct=roe, eps=eps,
                            momentum_20d=float(store.mom_v[di, col]),
                            trading_value_20d_avg=float(store.tv_v[di, col]))


def run_and_log(store, sector_map):
    td = store.trading_days
    close_v = store.close_v; close_ff = store.close_ff
    c2c = store.code_to_col

    di_list = [di for di in range(len(td)) if WIN_START <= td[di] <= WIN_END]
    end_di = di_list[-1]

    cash = float(INITIAL_CASH)
    positions = {}          # ticker -> sc.Position
    rec = {}                # ticker -> 현재 진입 기록(dict)
    closed = []             # 완료된 매수건 기록

    for di in di_list:
        today = td[di].date()
        is_bull = sc.is_bull_market(store.kospi_close_v[di], store.kospi_ma200_v[di])
        sold_today = set()

        # 1) 전량매도
        for t in list(positions.keys()):
            col = c2c[t]; px = close_v[di, col]
            if not np.isfinite(px):
                continue
            pos = positions[t]
            should, reason = sc.check_sell_condition(pos, _snap(store, col, di))
            if should:
                proceeds = px * pos.shares * (1.0 - SELL_COST)
                cash += proceeds
                r = rec[t]; r["proceeds"] += proceeds
                r["exit_date"] = str(today); r["exit_price"] = float(px)
                r["reason"] = reason; r["exit_di"] = di
                r["pnl_pct"] = r["proceeds"] / r["invested"] - 1.0
                r["hold_days"] = di - r["entry_di"]
                closed.append(r)
                del positions[t]; del rec[t]; sold_today.add(t)

        # 1-2) 부분익절
        for t in list(positions.keys()):
            col = c2c[t]; px = close_v[di, col]
            if not np.isfinite(px) or px <= 0:
                continue
            pos = positions[t]
            if sc.check_partial_exit(pos, float(px)):
                before = pos.shares
                sell_sh = sc.apply_partial_exit(pos, float(px))
                proceeds = px * sell_sh * (1.0 - SELL_COST)
                cash += proceeds
                rec[t]["proceeds"] += proceeds
                rec[t]["partial"] = "Y"

        # 2) 불타기
        if is_bull and positions:
            def cur_ret(t):
                col = c2c[t]; px = close_v[di, col]
                return px / positions[t].avg_price if np.isfinite(px) else -1.0
            for t in sorted(positions.keys(), key=cur_ret, reverse=True):
                col = c2c[t]; px = close_v[di, col]
                if not np.isfinite(px) or px <= 0:
                    continue
                pos = positions[t]
                if sc.check_pyramid(pos, float(px), today, is_bull, cash):
                    spent = sc.PYRAMID_ADD_AMOUNT_WON * (1.0 + BUY_COST)
                    if cash < spent:
                        continue
                    sc.apply_pyramid(pos, float(px), today)
                    cash -= spent; rec[t]["invested"] += spent; rec[t]["pyr"] += 1

        # 3) 신규매수
        free = sc.NUM_SLOTS - len(positions)
        if is_bull and free > 0:
            cands = []
            for t in store.get_universe(td[di]):
                if t in positions or t in sold_today:
                    continue
                col = c2c[t]; px = close_v[di, col]
                if not np.isfinite(px) or px <= 0:
                    continue
                s = _snap(store, col, di)
                ok, _ = sc.check_buy_conditions(s, is_bull)
                if ok:
                    cands.append(s)
            for s in sc.rank_by_momentum(cands):
                if free <= 0:
                    break
                px = s.price; sh = int(sc.SLOT_AMOUNT_WON // px)
                if sh <= 0:
                    continue
                spent = sh * px * (1.0 + BUY_COST)
                if spent > cash:
                    continue
                col = c2c[s.ticker]
                ms = store.ma_sell_v[di, col]
                disp = (px - ms) / ms * 100.0 if (np.isfinite(ms) and ms > 0) else float("nan")
                positions[s.ticker] = sc.Position(ticker=s.ticker, entry_price=px, avg_price=px,
                                                  shares=float(sh), entry_date=today)
                cash -= spent; free -= 1
                _nm = store.name_map.get(s.ticker, s.ticker)
                _ind = sector_map.get(s.ticker, "") or "미분류"
                rec[s.ticker] = {
                    "code": s.ticker, "name": _nm,
                    "industry": _ind, "theme": theme_of(_ind, _nm),
                    "buy_date": str(today), "buy_price": float(px),
                    "buy_mom_pct": float(s.momentum_20d) * 100.0,
                    "buy_disp_pct": float(disp),
                    "entry_di": di, "invested": spent, "proceeds": 0.0,
                    "pyr": 0, "partial": "N",
                    "exit_date": "", "exit_price": float("nan"), "reason": "", "exit_di": None,
                    "pnl_pct": float("nan"), "hold_days": None,
                }

        # 4) 평가(현금 흐름만 필요 — 생략 가능하나 원엔진과 동일하게 유지)
        # (자산평가는 이 분석에 불필요하므로 계산 생략)

    # 기간말 보유중 처리
    for t, pos in positions.items():
        col = c2c[t]; p = close_v[end_di, col]
        if not np.isfinite(p):
            p = close_ff[end_di, col]
            if not np.isfinite(p):
                p = pos.avg_price
        r = rec[t]
        cur_val = p * pos.shares * (1.0 - SELL_COST)
        r["exit_date"] = "보유중"; r["exit_price"] = float(p)
        r["reason"] = "기간종료(보유중)"; r["exit_di"] = end_di
        r["pnl_pct"] = (r["proceeds"] + cur_val) / r["invested"] - 1.0
        r["hold_days"] = end_di - r["entry_di"]
        closed.append(r)

    return closed


def main():
    store = data_layer.get_store()
    sector_map = load_sector_map(store.name_map)

    trades = run_and_log(store, sector_map)
    df = pd.DataFrame(trades)
    df = df.sort_values("buy_date").reset_index(drop=True)
    keep = ["name", "code", "theme", "industry", "buy_date", "buy_price", "exit_date", "exit_price",
            "reason", "partial", "pnl_pct", "buy_mom_pct", "buy_disp_pct", "hold_days", "pyr"]
    df = df[keep]
    df["pnl_pct"] = df["pnl_pct"] * 100.0
    df.to_csv(os.path.join(RESULT_DIR, "trades_2024_detail.csv"), index=False, encoding="utf-8-sig")

    print(f"\n총 신규매수 건수: {len(df)}건 (기대: 69건)")
    losers = df[df["pnl_pct"] <= 0]; winners = df[df["pnl_pct"] > 0]
    print(f"  손실 {len(losers)}건 · 이익 {len(winners)}건 · 평균손익률 {df['pnl_pct'].mean():.2f}%")

    # 1) 섹터 집중도 (테마 버킷 + 원시 업종 둘 다)
    sec = (df.groupby("theme")
           .agg(건수=("code", "size"), 평균손익=("pnl_pct", "mean"),
                손실건수=("pnl_pct", lambda x: int((x <= 0).sum())))
           .sort_values("건수", ascending=False))
    sec["비율%"] = (sec["건수"] / len(df) * 100).round(1)
    sec.to_csv(os.path.join(RESULT_DIR, "trades_2024_sector.csv"), encoding="utf-8-sig")
    ind = (df.groupby("industry")
           .agg(건수=("code", "size"), 평균손익=("pnl_pct", "mean"))
           .sort_values("건수", ascending=False))
    ind.to_csv(os.path.join(RESULT_DIR, "trades_2024_industry.csv"), encoding="utf-8-sig")

    # 2) 실패 패턴: 승/패 그룹 통계
    def stat(g):
        return dict(n=len(g), mom=g["buy_mom_pct"].mean(), disp=g["buy_disp_pct"].mean(),
                    hold=g["hold_days"].mean())
    print("\n[실패패턴] 승/패 그룹 매수시점 지표 평균")
    print(f"  손실({len(losers)}): 모멘텀 {losers['buy_mom_pct'].mean():.2f}% · 이격도 {losers['buy_disp_pct'].mean():.2f}% · 보유 {losers['hold_days'].mean():.1f}일")
    print(f"  이익({len(winners)}): 모멘텀 {winners['buy_mom_pct'].mean():.2f}% · 이격도 {winners['buy_disp_pct'].mean():.2f}% · 보유 {winners['hold_days'].mean():.1f}일")
    print(f"  손실건 보유기간 분포: 중앙값 {losers['hold_days'].median():.0f}일 · 5일이내 {int((losers['hold_days']<=5).sum())}건 · 10일이내 {int((losers['hold_days']<=10).sum())}건")
    print("  매도사유 분포:")
    print(df["reason"].value_counts().to_string())

    # 3) 월별 분포 + KOSPI 상태
    df["month"] = df["buy_date"].str[:7]
    td = store.trading_days; kc = store.kospi_close_v
    mrows = []
    for m in sorted(df["month"].unique()):
        cnt = int((df["month"] == m).sum())
        yr, mo = int(m[:4]), int(m[5:7])
        idx = [di for di in range(len(td)) if td[di].year == yr and td[di].month == mo]
        if idx:
            chg = (kc[idx[-1]] / kc[idx[0]] - 1) * 100
            state = "상승" if chg > 1.5 else ("하락" if chg < -1.5 else "횡보")
        else:
            chg, state = float("nan"), "-"
        mrows.append({"월": m, "신규매수": cnt, "KOSPI월변동%": round(chg, 2), "상태": state})
    mdf = pd.DataFrame(mrows)
    mdf.to_csv(os.path.join(RESULT_DIR, "trades_2024_monthly.csv"), index=False, encoding="utf-8-sig")
    print("\n[월별]");  print(mdf.to_string(index=False))

    # 4) 대표 사례
    worst = df.nsmallest(5, "pnl_pct")
    best = df.nlargest(5, "pnl_pct")
    worst.to_csv(os.path.join(RESULT_DIR, "trades_2024_worst5.csv"), index=False, encoding="utf-8-sig")
    best.to_csv(os.path.join(RESULT_DIR, "trades_2024_best5.csv"), index=False, encoding="utf-8-sig")

    print("\n  상세/섹터/월별/대표사례 CSV 저장 완료 (results/)")
    return df


if __name__ == "__main__":
    main()
