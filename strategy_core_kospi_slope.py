# -*- coding: utf-8 -*-
"""
strategy_core_kospi_slope.py
============================
[시장필터 변형 · 실험용] KOSPI 시장필터 본체에 '위치+방향' 로직 적용:
  기존:  KOSPI종가 > KOSPI200일선
  변형:  KOSPI종가 > KOSPI200일선  OR  KOSPI 20일모멘텀 > 0
즉 KOSPI가 200일선 아래여도 최근 20일 우상향(모멘텀 양수)이면 강세장으로 보고 매수 허용.
(코스닥 조건은 넣지 않음 — 오직 KOSPI 로직만 바꾼 버전)
★ 신규매수·불타기를 함께 게이팅. 원본 strategy_core.py는 미수정.
"""
KOSPI_MOM_DAYS = 20


def is_bull_market(kospi_close: float, kospi_ma200: float, kospi_mom20: float) -> bool:
    return (kospi_close > kospi_ma200) or (kospi_mom20 > 0.0)
