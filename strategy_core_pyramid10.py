# -*- coding: utf-8 -*-
"""
strategy_core_pyramid10.py
==========================
[불타기 단계 상한 변형 · 실험용] '새 기준선'(60일선삭제+90일선이격도2.5%+ROE≥15%+거래대금30억
+저평가ROE×EPS+20일모멘텀+시장필터, 20슬롯×500만) 위에서, 무제한 불타기에 '단계 수 상한'만 도입.

    (새 기준선) 불타기 무제한: 진입가×1.03^n 도달할 때마다 500만원씩 계속 추가
    (이 변형)   최대 10단계까지만 허용 (진입가×1.03^10 ≈ 진입가 대비 +34% 지점까지)
                그 이후로 더 올라도 추가매수 중단. (단, 매도조건 도달시 매도는 그대로 작동)

목적: 효성티앤씨 97.5% / slot10 99.0% 같은 극단적 단일종목 쏠림의 직접 원인인
      '무제한 불타기'를 '비중%'가 아니라 '단계 수'로 제한해 쏠림을 완화할 수 있는지 검증.

★ strategy_core.py는 건드리지 않는다. 불타기 판정(check_pyramid)만 이 파일 것으로 교체하고,
  나머지(매수/매도/부분익절/불타기 실행 apply_pyramid·추가금액 500만)는 전부 원본 재사용.
"""
from strategy_core import (  # noqa: F401
    Position,
    PYRAMID_TRIGGER_PCT,
    PYRAMID_ADD_AMOUNT_WON,
)
from datetime import date

MAX_PYRAMID_STEPS = 10   # ★불타기 최대 단계 수 상한


def check_pyramid(position: Position, current_price: float, current_date: date,
                  is_bull: bool, available_cash: float) -> bool:
    """원본 check_pyramid와 동일 + '단계 수 상한(pyramid_count >= MAX)' 한 줄 추가."""
    if not is_bull:
        return False
    if position.last_pyramid_date == current_date:
        return False
    if available_cash < PYRAMID_ADD_AMOUNT_WON:
        return False
    if position.pyramid_count >= MAX_PYRAMID_STEPS:   # ★상한 도달시 추가매수 중단
        return False
    next_step = position.pyramid_count + 1
    trigger_price = position.entry_price * ((1 + PYRAMID_TRIGGER_PCT) ** next_step)
    return current_price >= trigger_price


if __name__ == "__main__":
    pos = Position(ticker="T", entry_price=10000, avg_price=15000, shares=1000,
                   pyramid_count=MAX_PYRAMID_STEPS, entry_date=date(2026, 1, 1))
    # 이미 상한(10단계)에 도달 → 아무리 올라도 False
    assert check_pyramid(pos, 999999, date(2026, 7, 9), True, 10_000_000) is False
    pos2 = Position(ticker="T", entry_price=10000, avg_price=10000, shares=500,
                    pyramid_count=9, entry_date=date(2026, 1, 1))
    # 9단계 → 10단계 트리거(10000*1.03^10≈13439) 도달시 True (마지막 1회 허용)
    assert check_pyramid(pos2, 14000, date(2026, 7, 9), True, 10_000_000) is True
    print(f"[PASS] strategy_core_pyramid10 (불타기 최대 {MAX_PYRAMID_STEPS}단계) 로직 점검 완료")
