# -*- coding: utf-8 -*-
"""
strategy_core_both_slope.py
===========================
[시장필터 변형 · 실험용] KOSPI와 KOSDAQ 둘 다 '위치+방향' 로직 적용:
  (KOSPI>200일선 OR KOSPI 20일모멘텀>0) AND (KOSDAQ>200일선 OR KOSDAQ 20일모멘텀>0)
★ 신규매수·불타기를 함께 게이팅. 원본 strategy_core.py는 미수정.
"""
KOSPI_MOM_DAYS = 20
KOSDAQ_MOM_DAYS = 20


def is_bull_market(kospi_close: float, kospi_ma200: float, kospi_mom20: float,
                   kosdaq_close: float, kosdaq_ma200: float, kosdaq_mom20: float) -> bool:
    kospi_ok = (kospi_close > kospi_ma200) or (kospi_mom20 > 0.0)
    kosdaq_ok = (kosdaq_close > kosdaq_ma200) or (kosdaq_mom20 > 0.0)
    return kospi_ok and kosdaq_ok
