# -*- coding: utf-8 -*-
"""
strategy_core_dailycap2.py
==========================
[노출 방식 변형 A · 실험용] 확정 조건 위에서 '하루 신규매수 종목 수'만 제한.
    매수조건 통과 종목이 여러 개여도, 하루 신규진입(불타기 아닌 첫 진입)은 최대 2종목까지만.
    나머지는 다음날로 이월(그날도 조건 통과하면 그때 매수). 우선순위는 기존과 동일(20일모멘텀 높은 순).
불타기/매도/부분익절/시장필터 등 나머지는 새 기준선 그대로. strategy_core.py는 건드리지 않는다
(하루 제한은 드라이버 신규매수 루프에서 카운트로 적용).
"""
DAILY_NEW_CAP = 2

if __name__ == "__main__":
    assert DAILY_NEW_CAP == 2
    print(f"[PASS] strategy_core_dailycap2 (하루 신규매수 최대 {DAILY_NEW_CAP}종목) 선언 확인")
