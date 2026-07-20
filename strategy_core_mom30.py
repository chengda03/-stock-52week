# -*- coding: utf-8 -*-
"""
strategy_core_mom30.py
======================
[모멘텀 lookback 변형 · 실험용] '새 기준선' 위에서 모멘텀 기간을 20일 → 30일로 교체.
    매수조건 ④ "30일 수익률 > 0" + 매수우선순위 "30일 모멘텀 높은 순".
나머지 매수조건은 새 기준선 그대로. strategy_core.py는 건드리지 않는다.
(모멘텀 값은 드라이버가 '30일 수익률' 패널을 계산해 snapshot.momentum_20d에 주입 → 원본 판정/정렬 재사용)
"""
import strategy_core as sc

MOMENTUM_DAYS = 30

check_buy_conditions = sc.check_buy_conditions
rank_by_momentum = sc.rank_by_momentum


if __name__ == "__main__":
    assert MOMENTUM_DAYS == 30
    print(f"[PASS] strategy_core_mom30 (모멘텀 lookback {MOMENTUM_DAYS}일) 선언 확인")
