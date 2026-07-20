# -*- coding: utf-8 -*-
"""
strategy_core_slot15.py
=======================
[슬롯 구조 변형 · 실험용] '새 기준선'(60일선삭제+90일선이격도2.5%+ROE≥15%+거래대금30억
+저평가ROE×EPS+20일모멘텀+시장필터) 위에서, 매수/매도/불타기/부분익절 조건은 '전부 동일'하게
두고 슬롯 개수와 슬롯당 배분금액만 바꾼 버전.

    (새 기준선) 20슬롯 × 500만원 = 1억원
    (이 변형)   15슬롯 × 667만원 ≈ 1억원   ← 슬롯 개수만 줄이고 슬롯당 금액을 키움

★ 불타기 1회 추가금액은 원본 설계대로 '슬롯 금액과 동일'하게 맞춘다
  (PYRAMID_ADD_AMOUNT_WON = SLOT_AMOUNT_WON). 즉 슬롯을 키우면 불타기 1스텝 금액도 커진다.

★ strategy_core.py는 건드리지 않는다. 매수판정 등 로직은 원본을 그대로 재사용하고,
  이 파일은 '슬롯 사이징 3개 상수'만 재정의한다(백테스트 엔진이 이 값을 읽어 씀).
"""
from strategy_core import check_buy_conditions  # noqa: F401  (매수조건은 새 기준선 그대로)

# --- 슬롯 사이징 (이 변형의 유일한 변경점) ---
NUM_SLOTS = 15
SLOT_AMOUNT_WON = 6_670_000          # 1억 ÷ 15 ≈ 666.7만 → 반올림 667만
PYRAMID_ADD_AMOUNT_WON = 6_670_000   # 불타기 1스텝 금액 = 슬롯 금액과 동일


if __name__ == "__main__":
    assert NUM_SLOTS == 15
    assert abs(NUM_SLOTS * SLOT_AMOUNT_WON - 100_000_000) <= 1_000_000  # ≈1억
    assert PYRAMID_ADD_AMOUNT_WON == SLOT_AMOUNT_WON
    print(f"[PASS] strategy_core_slot15: {NUM_SLOTS}슬롯 × {SLOT_AMOUNT_WON:,}원 "
          f"= {NUM_SLOTS*SLOT_AMOUNT_WON:,}원 (불타기 1스텝 {PYRAMID_ADD_AMOUNT_WON:,}원)")
