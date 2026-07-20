# -*- coding: utf-8 -*-
"""
strategy_core_pp6.py
====================
[부분익절 변형 · 실험용] 원본 strategy_core.py에서 '부분익절 트리거'만
  · +8% 도달 → +6% 도달
로 낮춘 버전. 그 외는 전부 원본과 동일.

    · 매수조건 : base(60일선+90일선 포함 7조건) 그대로
    · 매도조건 : 90일선 이탈 / -12% 하드손절 그대로
    · 불타기   : +3% 복리 그대로
    · 부분익절 : 진입가 대비 +6% 도달시 '보유수량의 50%' 1회 매도  ← ★여기만 변경
                (1회성 플래그·나머지 50% 계속 보유·평단가/진입가 불변은 원본과 동일)

★ 원본은 건드리지 않는다. 이 파일은 부분익절 판정 함수(check_partial_exit)와
  실행 함수(apply_partial_exit)만 제공하고, 나머지 규칙은 백테스트 엔진이
  원본 strategy_core.py 것을 그대로 쓴다.
"""
from strategy_core import Position  # noqa: F401  (타입 참고 + 실행함수 재사용)

# --- 이 변형의 유일한 변경점 ---
PARTIAL_EXIT_TRIGGER_PCT = 0.06   # 진입가 대비 +6% 도달 시 트리거 (원본은 0.08)
PARTIAL_EXIT_RATIO = 0.5          # 그때 보유수량의 50% 매도 (원본과 동일)


def check_partial_exit(position: Position, current_price: float) -> bool:
    """
    부분익절(+6%에서 50%) 실행 여부 판정. 원본과 로직 동일, 트리거 비율만 6%.
      - 아직 부분익절을 안 했고 (position.partial_exit_done == False)
      - 현재가가 최초 진입가 대비 +6% 이상
    기준가격은 평단가가 아니라 '최초 진입가(entry_price)'. (원본과 동일)
    """
    if position.partial_exit_done:
        return False
    return current_price >= position.entry_price * (1 + PARTIAL_EXIT_TRIGGER_PCT)


def apply_partial_exit(position: Position, current_price: float) -> float:
    """부분익절 실행 시 상태 갱신. 원본과 완전히 동일(50% 매도, 평단가/진입가 불변)."""
    sell_shares = position.shares * PARTIAL_EXIT_RATIO
    position.shares -= sell_shares
    position.partial_exit_done = True
    return sell_shares


if __name__ == "__main__":
    from datetime import date
    pos = Position(ticker="T", entry_price=10000, avg_price=10000, shares=200,
                   entry_date=date(2026, 1, 1))
    assert check_partial_exit(pos, 10500) is False   # +5% → 아직
    assert check_partial_exit(pos, 10600) is True     # +6% 도달
    sold = apply_partial_exit(pos, 10600)
    assert abs(sold - 100) < 1e-9 and abs(pos.shares - 100) < 1e-9
    assert pos.partial_exit_done is True
    assert check_partial_exit(pos, 12000) is False    # 이미 했으면 중복 실행 안 됨
    print(f"[PASS] strategy_core_pp6 (부분익절 +{PARTIAL_EXIT_TRIGGER_PCT*100:.0f}%@{PARTIAL_EXIT_RATIO*100:.0f}%) 로직 점검 완료")
