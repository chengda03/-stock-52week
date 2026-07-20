# -*- coding: utf-8 -*-
"""
strategy_core_dualindex.py
==========================
[시장필터 변형 · 실험용] 현재 확정 기준선(strategy_core.py)의 시장필터를
  기존:  KOSPI종가 > KOSPI 200일선            (강세장)
  변형:  KOSPI종가 > KOSPI200 AND KOSDAQ종가 > KOSDAQ200  (둘 다 만족해야 강세장)
로 교체한 버전.

★ 시장필터는 원본에서 '신규매수'와 '불타기'를 함께 게이팅한다. 이 변형도 동일하게
  둘 다에 적용된다(=시장필터 자체를 교체하는 것이므로 자연스러움).
★ 원본 strategy_core.py는 건드리지 않는다. 그 외 규칙(저평가/ROE/이격도/모멘텀/거래대금/
  매도/불타기/부분익절/사이징/비용)은 전부 원본을 재사용한다.
"""
from strategy_core import is_bull_market as _kospi_bull  # noqa: F401


def is_bull_market_dual(kospi_close: float, kospi_ma200: float,
                        kosdaq_close: float, kosdaq_ma200: float) -> bool:
    """KOSPI와 KOSDAQ이 '모두' 200일선 위일 때만 강세장(신규매수·불타기 허용)."""
    return (kospi_close > kospi_ma200) and (kosdaq_close > kosdaq_ma200)
