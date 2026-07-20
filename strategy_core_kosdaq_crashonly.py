# -*- coding: utf-8 -*-
"""
strategy_core_kosdaq_crashonly.py
=================================
[시장필터 변형 · 실험용] 시장필터 =
  KOSPI종가>KOSPI200일선 AND KOSDAQ 60일수익률 > -15%
단순 200일선 위/아래가 아니라 '급락 속도'로 판단: 코스닥이 최근 60거래일에 15% 넘게
빠지는 급락 국면에서만 신규매수 차단(그 외에는 코스닥 수준 무관 허용).
★ 신규매수·불타기를 함께 게이팅. 원본 strategy_core.py는 미수정.
"""
KOSDAQ_RET_DAYS = 60
CRASH_THRESHOLD = -0.15


def is_bull_market(kospi_close: float, kospi_ma200: float,
                   kosdaq_ret60: float) -> bool:
    if not (kospi_close > kospi_ma200):
        return False
    return kosdaq_ret60 > CRASH_THRESHOLD
