# -*- coding: utf-8 -*-
"""
strategy_core_ma240.py
======================
[시장필터 MA기간 변형 · 실험용] 확정 조건 위에서 시장필터의 이동평균 기간만 200일 → 240일로 교체.
    시장필터: "KOSPI 종가 > KOSPI 200일 이동평균"  →  "KOSPI 종가 > KOSPI 240일 이동평균"
나머지(매수/매도/부분익절/불타기/슬롯/거래대금/이격도/ROE/모멘텀)는 전부 기존과 동일.

★ strategy_core.py는 건드리지 않는다. 시장필터 MA 기간만 바꾼다.
  is_bull_market 판정 로직(종가 > 이동평균) 자체는 원본과 동일하며, '어떤 기간의 MA를 넣느냐'만 달라진다.
  → 백테스트 드라이버가 KOSPI 240일선 패널을 계산해 is_bull_market에 주입한다.
"""
import strategy_core as sc

MARKET_FILTER_MA_DAYS = 240

# 판정 로직은 원본과 완전히 동일(종가>이동평균). 넣는 MA만 240일짜리로 바뀔 뿐.
is_bull_market = sc.is_bull_market


if __name__ == "__main__":
    assert MARKET_FILTER_MA_DAYS == 240
    assert is_bull_market(100.0, 90.0) is True    # 종가>MA → 강세장(매수 허용)
    assert is_bull_market(90.0, 100.0) is False   # 종가<MA → 약세장(매수 차단)
    print(f"[PASS] strategy_core_ma240 (시장필터 {MARKET_FILTER_MA_DAYS}일선) 선언 확인")
