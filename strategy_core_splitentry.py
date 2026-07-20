# -*- coding: utf-8 -*-
"""
strategy_core_splitentry.py
===========================
[노출 방식 변형 B · 실험용] 신규 진입을 '분할 매수'로.
    · 최초 진입: 슬롯 금액 500만 중 250만원만 투입(절반).
    · 이후 SECOND_TRANCHE_DAYS(=5)거래일 동안 매도조건(90일선이탈/-12%손절)에 안 걸리고
      '생존'하면, 그 다음(6일째)에 나머지 250만원을 추가 투입.
      (이 추가는 불타기 +3% 트리거와 무관 — 조건 없이 '생존'만 확인해서 넣음)
    · 5거래일 이내에 매도조건에 걸리면 250만원 상태 그대로 손절(추가투입 없이 종료).
불타기/매도/부분익절/시장필터/슬롯수 등 나머지는 새 기준선 그대로.
★ strategy_core.py는 건드리지 않는다. 분할진입은 포지션별 '진입경과일/2차투입여부' 상태가 필요하므로
  드라이버에서 구현한다. 이 모듈은 분할 파라미터만 선언한다.
"""
SPLIT_ENTRY = True
INITIAL_TRANCHE_WON = 2_500_000     # 최초 250만
SECOND_TRANCHE_WON = 2_500_000      # 생존시 추가 250만
SECOND_TRANCHE_DAYS = 5             # 진입 후 5거래일 생존 확인 → 다음날 추가

if __name__ == "__main__":
    assert INITIAL_TRANCHE_WON + SECOND_TRANCHE_WON == 5_000_000
    print(f"[PASS] strategy_core_splitentry (최초 {INITIAL_TRANCHE_WON:,}원 → {SECOND_TRANCHE_DAYS}일 생존시 +{SECOND_TRANCHE_WON:,}원) 선언 확인")
