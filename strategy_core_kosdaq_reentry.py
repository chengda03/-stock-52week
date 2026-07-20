# -*- coding: utf-8 -*-
"""
strategy_core_kosdaq_reentry.py
===============================
[시장필터 변형 · 실험용] 기본은 dualindex(KOSPI>200 AND KOSDAQ>200)와 동일하되,
KOSDAQ이 200일선을 '새로 상향 돌파'한 직후 REENTRY_WINDOW 거래일 동안은 코스닥 조건을
완화(=KOSPI만 강세면 허용)하여, 돌파 직후 눌림(200선 재이탈 whipsaw)에서도 매수를 이어가
회복 초입을 놓치는 문제를 완화하려는 시도.
★ 신규매수·불타기를 함께 게이팅. 원본 strategy_core.py는 미수정.
"""
REENTRY_WINDOW = 20


def is_bull_market(kospi_close: float, kospi_ma200: float,
                   kosdaq_above200: bool, in_reentry_window: bool) -> bool:
    if not (kospi_close > kospi_ma200):
        return False
    return bool(kosdaq_above200) or bool(in_reentry_window)
