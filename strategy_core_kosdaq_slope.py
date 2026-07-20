# -*- coding: utf-8 -*-
"""
strategy_core_kosdaq_slope.py
=============================
[시장필터 변형 · 실험용] 시장필터 =
  KOSPI종가>KOSPI200일선 AND (KOSDAQ종가>KOSDAQ200일선 OR KOSDAQ 20일수익률>0)
코스닥이 200일선 아래여도 최근 20일 모멘텀이 양수(=반등 초입)면 허용 → 급락중만 차단.
★ 신규매수·불타기를 함께 게이팅. 원본 strategy_core.py는 미수정.
"""
KOSDAQ_MOM_DAYS = 20


def is_bull_market(kospi_close: float, kospi_ma200: float,
                   kosdaq_close: float, kosdaq_ma200: float,
                   kosdaq_mom20: float) -> bool:
    if not (kospi_close > kospi_ma200):
        return False
    return (kosdaq_close > kosdaq_ma200) or (kosdaq_mom20 > 0.0)
