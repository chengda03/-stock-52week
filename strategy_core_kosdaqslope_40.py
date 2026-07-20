# -*- coding: utf-8 -*-
"""[견고성검증] kosdaq_slope의 코스닥 모멘텀 lookback = 40일 버전.
시장필터 = KOSPI>200일선 AND (KOSDAQ>200일선 OR KOSDAQ 40일모멘텀>0). 원본 미수정."""
KOSDAQ_MOM_DAYS = 40


def is_bull_market(kospi_close, kospi_ma200, kosdaq_close, kosdaq_ma200, kosdaq_mom):
    if not (kospi_close > kospi_ma200):
        return False
    return (kosdaq_close > kosdaq_ma200) or (kosdaq_mom > 0.0)
