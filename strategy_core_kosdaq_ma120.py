# -*- coding: utf-8 -*-
"""
strategy_core_kosdaq_ma120.py
=============================
[시장필터 변형 · 실험용] 시장필터 = KOSPI종가>KOSPI200일선 AND KOSDAQ종가>KOSDAQ '120'일선.
코스닥 기준선을 200→120일로 짧게(덜 후행) 해서 2023년 회복 초입 진입을 덜 늦게 막으려는 시도.
★ 신규매수·불타기를 함께 게이팅(원본 시장필터와 동일 성격). 원본 strategy_core.py는 미수정.
"""
KOSDAQ_MA_DAYS = 120


def is_bull_market(kospi_close: float, kospi_ma200: float,
                   kosdaq_close: float, kosdaq_ma120: float) -> bool:
    return (kospi_close > kospi_ma200) and (kosdaq_close > kosdaq_ma120)
