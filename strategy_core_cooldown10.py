# -*- coding: utf-8 -*-
"""
strategy_core_cooldown10.py
===========================
[재진입 쿨다운 변형 · 실험용] 확정 조건 위에 '쿨다운' 규칙만 추가.
    전량매도(90일선이탈 또는 -12% 하드손절) 발생 후 10거래일 동안 같은 종목 재매수 금지.
    (부분익절 50%만 매도된 경우는 대상 아님 — 전량매도된 경우에만 적용)
나머지는 전부 새 기준선 그대로. strategy_core.py는 건드리지 않는다(드라이버에서 쿨다운 적용).
"""
COOLDOWN_DAYS = 10

if __name__ == "__main__":
    assert COOLDOWN_DAYS == 10
    print(f"[PASS] strategy_core_cooldown10 (재진입 금지 {COOLDOWN_DAYS}거래일) 선언 확인")
