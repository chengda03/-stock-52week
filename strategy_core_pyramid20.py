# -*- coding: utf-8 -*-
"""
strategy_core_pyramid20.py
==========================
[불타기 단계 상한 변형 · 실험용] '새 기준선' 위에서 불타기를 최대 20단계까지만 허용.
    진입가×1.03^20 ≈ 진입가 대비 +81% 지점까지 추가매수, 이후 중단(매도는 그대로 작동).

★ strategy_core.py는 건드리지 않는다. 불타기 판정(check_pyramid)만 교체, 나머지는 원본 재사용.
"""
from strategy_core import (  # noqa: F401
    Position,
    PYRAMID_TRIGGER_PCT,
    PYRAMID_ADD_AMOUNT_WON,
)
from datetime import date

MAX_PYRAMID_STEPS = 20


def check_pyramid(position: Position, current_price: float, current_date: date,
                  is_bull: bool, available_cash: float) -> bool:
    if not is_bull:
        return False
    if position.last_pyramid_date == current_date:
        return False
    if available_cash < PYRAMID_ADD_AMOUNT_WON:
        return False
    if position.pyramid_count >= MAX_PYRAMID_STEPS:
        return False
    next_step = position.pyramid_count + 1
    trigger_price = position.entry_price * ((1 + PYRAMID_TRIGGER_PCT) ** next_step)
    return current_price >= trigger_price


if __name__ == "__main__":
    pos = Position(ticker="T", entry_price=10000, avg_price=15000, shares=1000,
                   pyramid_count=MAX_PYRAMID_STEPS, entry_date=date(2026, 1, 1))
    assert check_pyramid(pos, 999999, date(2026, 7, 9), True, 10_000_000) is False
    print(f"[PASS] strategy_core_pyramid20 (불타기 최대 {MAX_PYRAMID_STEPS}단계) 로직 점검 완료")
