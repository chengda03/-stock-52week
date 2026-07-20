# -*- coding: utf-8 -*-
"""
strategy_core_cooldown5.py
==========================
[재진입 쿨다운 변형 · 실험용] 확정 조건 위에 '쿨다운' 규칙만 추가.
    전량매도(90일선이탈 또는 -12% 하드손절) 발생 후 5거래일 동안 같은 종목 재매수 금지.
    (부분익절로 50%만 매도된 경우는 쿨다운 대상 아님 — 전량매도된 경우에만 적용)
나머지(매수/매도/불타기/부분익절/시장필터200일선 등)는 전부 새 기준선 그대로.

★ strategy_core.py는 건드리지 않는다. 쿨다운은 '종목별 마지막 전량매도일' 상태가 필요하므로
  백테스트 드라이버의 신규매수 루프에서 적용한다(check_buy_conditions는 무상태라 그대로 재사용).
  이 모듈은 쿨다운 길이(COOLDOWN_DAYS)만 선언한다.
"""
COOLDOWN_DAYS = 5

if __name__ == "__main__":
    assert COOLDOWN_DAYS == 5
    print(f"[PASS] strategy_core_cooldown5 (재진입 금지 {COOLDOWN_DAYS}거래일) 선언 확인")
