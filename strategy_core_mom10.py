# -*- coding: utf-8 -*-
"""
strategy_core_mom10.py
======================
[모멘텀 lookback 변형 · 실험용] '새 기준선' 위에서 모멘텀 기간을 20일 → 10일로 교체.
    매수조건 ④  "20일 수익률 > 0"  →  "10일 수익률 > 0"
    매수우선순위 "20일 모멘텀 높은 순" → "10일 모멘텀 높은 순"
나머지 매수조건(저평가 ROE×EPS, ROE≥15%, 90일선 이격도 2.5%, 거래대금 30억, 시장필터)은 새 기준선 그대로.

★ strategy_core.py는 건드리지 않는다. 모멘텀 lookback만 바꾼다.
  구현상 모멘텀 값은 StockSnapshot.momentum_20d 필드에 담겨 흐르므로(백테스트 드라이버가
  '10일 수익률' 패널을 계산해 이 필드에 주입), 매수판정/정렬 로직 자체는 원본과 동일하게 재사용한다.
  이 모듈은 lookback 기간(MOMENTUM_DAYS)만 선언하고 원본 함수를 그대로 노출한다.
"""
import strategy_core as sc

MOMENTUM_DAYS = 10

check_buy_conditions = sc.check_buy_conditions   # snap.momentum_20d(=드라이버가 넣은 10일 수익률) > 0
rank_by_momentum = sc.rank_by_momentum           # snap.momentum_20d 높은 순


if __name__ == "__main__":
    assert MOMENTUM_DAYS == 10
    print(f"[PASS] strategy_core_mom10 (모멘텀 lookback {MOMENTUM_DAYS}일) 선언 확인")
