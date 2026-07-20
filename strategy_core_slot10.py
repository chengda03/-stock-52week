# -*- coding: utf-8 -*-
"""
strategy_core_slot10.py
=======================
[슬롯 구조 변형 · 실험용] '새 기준선' 위에서 매수/매도/불타기/부분익절 조건은 '전부 동일'하게
두고 슬롯 개수와 슬롯당 배분금액만 바꾼 버전.

    (새 기준선) 20슬롯 × 500만원  = 1억원
    (이 변형)   10슬롯 × 1000만원 = 1억원   ← 슬롯을 절반으로 줄이고 슬롯당 금액 2배

★ 불타기 1회 추가금액도 슬롯 금액과 동일(PYRAMID_ADD_AMOUNT_WON = SLOT_AMOUNT_WON).
★ strategy_core.py는 건드리지 않는다. 이 파일은 '슬롯 사이징 3개 상수'만 재정의한다.
"""
from strategy_core import check_buy_conditions  # noqa: F401

NUM_SLOTS = 10
SLOT_AMOUNT_WON = 10_000_000          # 1억 ÷ 10 = 1000만
PYRAMID_ADD_AMOUNT_WON = 10_000_000   # 불타기 1스텝 금액 = 슬롯 금액과 동일


if __name__ == "__main__":
    assert NUM_SLOTS == 10
    assert NUM_SLOTS * SLOT_AMOUNT_WON == 100_000_000
    assert PYRAMID_ADD_AMOUNT_WON == SLOT_AMOUNT_WON
    print(f"[PASS] strategy_core_slot10: {NUM_SLOTS}슬롯 × {SLOT_AMOUNT_WON:,}원 "
          f"= {NUM_SLOTS*SLOT_AMOUNT_WON:,}원 (불타기 1스텝 {PYRAMID_ADD_AMOUNT_WON:,}원)")
