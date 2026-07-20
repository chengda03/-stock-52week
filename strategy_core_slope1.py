# -*- coding: utf-8 -*-
"""
strategy_core_slope1.py
=======================
[시장필터 기울기 변형 · 실험용] 시장필터 = 종가>200일선 AND 200일선기울기(20일) > 1%.
나머지는 새 기준선 그대로. strategy_core.py는 건드리지 않는다(기울기 패널은 드라이버가 주입).
"""
SLOPE_LOOKBACK = 20
SLOPE_MIN_PCT = 1.0


def is_bull_market_slope(kospi_close: float, kospi_ma200: float, slope_pct: float) -> bool:
    return (kospi_close > kospi_ma200) and (slope_pct > SLOPE_MIN_PCT)


if __name__ == "__main__":
    assert is_bull_market_slope(100, 90, 1.5) is True
    assert is_bull_market_slope(100, 90, 0.5) is False   # 0.5%는 1% 미달 → 차단
    print(f"[PASS] strategy_core_slope1 (200일선 기울기>{SLOPE_MIN_PCT}%, {SLOPE_LOOKBACK}일) 선언 확인")
