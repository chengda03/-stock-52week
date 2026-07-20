# -*- coding: utf-8 -*-
"""
strategy_core_slope0.py
=======================
[시장필터 기울기 변형 · 실험용] 확정 조건 위에서 시장필터에 '200일선 기울기' 조건만 추가.
    기존:  KOSPI 종가 > KOSPI 200일선            (위치만 봄)
    변형:  KOSPI 종가 > 200일선  AND  200일선 기울기(20일) > 0%   (방향도 요구)

    200일선 기울기(%) = (오늘 200일선 - 20거래일전 200일선) / 20거래일전 200일선 × 100
    → slope > 0% : 200일선이 최소한 우상향(하락 아님)일 때만 신규매수/불타기 허용.

나머지(매수/매도/불타기/부분익절/이격도/ROE/모멘텀/거래대금/슬롯)는 전부 새 기준선 그대로.
★ strategy_core.py는 건드리지 않는다. 기울기 패널은 드라이버가 계산해 시장필터에 주입한다.
"""
SLOPE_LOOKBACK = 20
SLOPE_MIN_PCT = 0.0


def is_bull_market_slope(kospi_close: float, kospi_ma200: float, slope_pct: float) -> bool:
    """종가>200일선 AND 200일선 기울기>임계값 → 강세장(매수/불타기 허용)."""
    return (kospi_close > kospi_ma200) and (slope_pct > SLOPE_MIN_PCT)


if __name__ == "__main__":
    assert is_bull_market_slope(100, 90, 0.5) is True    # 위 + 우상향 → 허용
    assert is_bull_market_slope(100, 90, -0.5) is False   # 위지만 하락중 → 차단
    assert is_bull_market_slope(80, 90, 3.0) is False     # 200선 아래 → 차단
    print(f"[PASS] strategy_core_slope0 (200일선 기울기>{SLOPE_MIN_PCT}%, {SLOPE_LOOKBACK}일) 선언 확인")
