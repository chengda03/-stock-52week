# -*- coding: utf-8 -*-
"""
backtest_valuation_compare.py
=============================
저평가 필터 교체 실험 비교 드라이버.
  · 같은 프로세스 · 같은 DataStore · 같은 날짜로 모든 변형을 실행 → 데이터 드리프트 0.
  · 콘솔 비교표 + results/valuation_filter_compare.csv 저장.
  · base vs PER 선택 종목 '겹침(overlap)' 분석.

★ 부채비율 변형(3·4)은 '부채총계(liabilities)' 데이터가 있어야 실행 가능.
  현재 캐시(financials_full)엔 순이익·자본만 있고 부채총계가 없으므로,
  fetch_qv_extras.py로 수집되기 전까지는 이 드라이버가 자동으로 base/PER만 실행한다.
  (부채 데이터가 준비되면 debt/per_debt 모듈을 등록하는 자리를 아래에 표시해 둠.)

정직성: 이 실험은 6.4년 단일 경로 위의 그리드서치다. 특정 변형이 좋아도
        '왜' 좋아졌는지 논리가 서는지 함께 봐야 하며, 바로 확정하면 과최적화 위험.
"""
import os
import pandas as pd

import data_layer
import backtest_base
import backtest_per
from backtest_roe_eps_event import fpct

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)
CSV_PATH = os.path.join(RESULT_DIR, "valuation_filter_compare.csv")


def _row(m: dict) -> dict:
    return {
        "버전": m["tag"],
        "CAGR": round(m["CAGR"] * 100, 2),
        "MDD": round(m["MDD"] * 100, 2),
        "Sharpe": round(m["Sharpe"], 2),
        "Calmar": round(m["Calmar"], 2),
        "손익비": round(m["PL"], 2),
        "승률%": round(m["win_rate"] * 100, 1),
        "총거래": m["n_trades"],
        "신규매수": m["n_buy"],
        "통과평균개수": round(m["avg_pass_cnt"], 1),
        "통과율%": round(m["pass_rate"] * 100, 1),
        "최대단일비중%": round(m["max_weight"] * 100, 1),
    }


def main():
    store = data_layer.get_store()   # 1회 로드 → 모든 변형이 공유(공정 비교)

    # ---- 실행할 변형 등록 (부채 데이터 준비되면 debt/per_debt 추가) ----
    variants = [backtest_base.run(store), backtest_per.run(store)]
    # 예: import backtest_debt, backtest_per_debt 후
    #     variants += [backtest_debt.run(store), backtest_per_debt.run(store)]

    df = pd.DataFrame([_row(m) for m in variants])
    df = df.sort_values("CAGR", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 100)
    print(" 저평가 필터 교체 실험 비교  ·  " + f"{variants[0]['start']} ~ {variants[0]['end']}"
          + "  ·  같은 데이터·같은 날 실행(드리프트 0)")
    print("=" * 100)
    print(df.to_string(index=False))
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"\n  CSV 저장: {CSV_PATH}")

    # ---- base vs PER 선택 종목 겹침 분석 ----
    byname = {m["tag"]: m for m in variants}
    base = next(m for m in variants if m["tag"].startswith("1"))
    per = next(m for m in variants if m["tag"].startswith("2"))
    a, b = base["bought_set"], per["bought_set"]
    inter = a & b
    print("\n" + "-" * 100)
    print(" [base vs PER] 신규매수한 '종목군'이 얼마나 겹치나")
    print("-" * 100)
    print(f"   base 매수종목수 {len(a)} · PER 매수종목수 {len(b)} · 공통 {len(inter)}")
    if a or b:
        print(f"   Jaccard(교집합/합집합) = {len(inter)/len(a|b)*100:.1f}%")
        print(f"   base에만 있는 종목 {len(a-b)}개 · PER에만 있는 종목 {len(b-a)}개")

    print("\n" + "=" * 100)
    print(" ⚠ 정직성: 6.4년 단일 경로 그리드서치 결과. 한 버전이 좋아도 '왜'가 설명되는지 확인 필요(과최적화 위험).")
    print("=" * 100)


if __name__ == "__main__":
    main()
