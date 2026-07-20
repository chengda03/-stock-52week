# -*- coding: utf-8 -*-
"""
strategy_core_kosdaq_strict.py
==============================
[대안 · 단순 위치기준] 시장필터 = KOSPI>200일선 AND KOSDAQ>200일선.
방향(모멘텀) 로직 없음 = "코스닥 200일선 이하는 매입하지 않는다"는 단순 규칙.
(내용상 앞서 테스트한 dualindex와 동일. slope가 과최적화로 판명될 경우의 안전한 대안.)
★ 신규매수·불타기를 함께 게이팅. 원본 strategy_core.py는 미수정.
"""
def is_bull_market(kospi_close, kospi_ma200, kosdaq_close, kosdaq_ma200):
    return (kospi_close > kospi_ma200) and (kosdaq_close > kosdaq_ma200)
