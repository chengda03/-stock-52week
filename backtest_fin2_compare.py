# -*- coding: utf-8 -*-
"""
backtest_fin2_compare.py
========================
base(현재 확정 로직) vs 영업이익률≥5% vs 부채비율≤100%(금융업 면제) 비교 드라이버.
  · 같은 프로세스·같은 DataStore·같은 날 → 데이터 드리프트 0.
  · 여러 조건을 '한 번에 섞지 않고' 하나씩 분리해서 base와 비교(지난번 원칙).
  · 콘솔 비교표 + results/fin2_filter_compare.csv.

정직성: 6.4년 단일 경로 그리드서치. 특정 변형이 좋아도 '왜'가 설명되는지 확인 필요(과최적화 위험).
"""
import os
import pandas as pd

import data_layer
import backtest_base
import backtest_opmargin
import backtest_debt2

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)
CSV_PATH = os.path.join(RESULT_DIR, "fin2_filter_compare.csv")


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
    store = data_layer.get_store()   # 1회 로드 → 세 변형 공유

    base = backtest_base.run(store)
    opm = backtest_opmargin.run(store)
    debt = backtest_debt2.run(store)
    variants = [base, opm, debt]

    df = pd.DataFrame([_row(m) for m in variants])

    print("\n" + "=" * 104)
    print(" 재무필터 실험 비교(하나씩 분리)  ·  " + f"{base['start']} ~ {base['end']}"
          + "  ·  같은 데이터·같은 날(드리프트 0)")
    print("=" * 104)
    print(df.to_string(index=False))
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"\n  CSV 저장: {CSV_PATH}")

    # base 대비 종목군 겹침
    print("\n" + "-" * 104)
    print(" [base 대비] 각 필터가 신규매수한 '종목군' 비교")
    print("-" * 104)
    b = base["bought_set"]
    for m in (opm, debt):
        s = m["bought_set"]
        inter = b & s
        uni = b | s
        print(f"   {m['tag']:>12s}: 매수종목 {len(s):3d} · base와 공통 {len(inter):3d} · "
              f"Jaccard {len(inter)/len(uni)*100:4.1f}%  (이 필터에서 새로 빠진 종목 {len(b - s)})")

    print("\n" + "=" * 104)
    print(" ⚠ 정직성: 6.4년 단일 경로 그리드서치. 임계값(5%/100%)도 임의 1점. 좋아도 '왜'가 서는지 확인 필요.")
    print("=" * 104)


if __name__ == "__main__":
    main()
