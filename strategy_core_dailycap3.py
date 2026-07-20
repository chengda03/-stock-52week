# -*- coding: utf-8 -*-
"""
strategy_core_dailycap3.py
==========================
[노출 방식 변형 A · 실험용] 하루 신규매수 최대 3종목까지만 허용(나머지 이월, 우선순위=20일모멘텀).
나머지는 새 기준선 그대로. strategy_core.py는 건드리지 않는다(드라이버에서 적용).
"""
DAILY_NEW_CAP = 3

if __name__ == "__main__":
    assert DAILY_NEW_CAP == 3
    print(f"[PASS] strategy_core_dailycap3 (하루 신규매수 최대 {DAILY_NEW_CAP}종목) 선언 확인")
