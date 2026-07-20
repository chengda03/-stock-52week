# -*- coding: utf-8 -*-
"""
strategy_core_combined_ab.py
============================
[노출 방식 변형 C · 실험용] A+B 결합: 하루 신규매수 3종목 제한(dailycap3) + 분할진입(splitentry).
나머지는 새 기준선 그대로. strategy_core.py는 건드리지 않는다(드라이버에서 두 규칙 동시 적용).
"""
DAILY_NEW_CAP = 3
SPLIT_ENTRY = True
INITIAL_TRANCHE_WON = 2_500_000
SECOND_TRANCHE_WON = 2_500_000
SECOND_TRANCHE_DAYS = 5

if __name__ == "__main__":
    assert DAILY_NEW_CAP == 3 and INITIAL_TRANCHE_WON + SECOND_TRANCHE_WON == 5_000_000
    print(f"[PASS] strategy_core_combined_ab (하루 {DAILY_NEW_CAP}종목 + 분할진입 {INITIAL_TRANCHE_WON:,}→+{SECOND_TRANCHE_WON:,}) 선언 확인")
