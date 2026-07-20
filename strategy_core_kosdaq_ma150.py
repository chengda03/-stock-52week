# -*- coding: utf-8 -*-
"""
strategy_core_kosdaq_ma150.py
=============================
[시장필터 변형 · 실험용] 시장필터 = KOSPI종가>KOSPI200일선 AND KOSDAQ종가>KOSDAQ '150'일선.
★ 신규매수·불타기를 함께 게이팅. 원본 strategy_core.py는 미수정.
"""
KOSDAQ_MA_DAYS = 150


def is_bull_market(kospi_close: float, kospi_ma200: float,
                   kosdaq_close: float, kosdaq_ma150: float) -> bool:
    return (kospi_close > kospi_ma200) and (kosdaq_close > kosdaq_ma150)
