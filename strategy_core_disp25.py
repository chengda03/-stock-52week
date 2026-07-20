# -*- coding: utf-8 -*-
"""
strategy_core_disp25.py
=======================
[이격도 변형 · 실험용] 원본 strategy_core.py의 매수조건에서
  · "현재가 > 60일선" (조건 3)
  · "현재가 > 90일선" (조건 3-b)
두 개를 '완전히 제거'하고, 대신 90일선 '이격도 ≥ 2.5%' 하나로 교체한 버전.

    이격도(%) = (현재가 - 90일선) / 90일선 × 100

지난 실험(disparity_filter_compare)에서 2%와 3%가 사실상 노이즈 수준으로 동등했기에,
그 중간값 2.5%를 '견고 구간 대표'로 잡아 정배열 실험의 비교 기준으로 쓴다.

★ 원본은 건드리지 않는다. 저평가/ROE/모멘텀/거래대금/시장필터는 그대로,
  매도·불타기·부분익절·사이징·비용 규칙도 전부 원본 strategy_core.py를 재사용한다.
"""
from strategy_core import (  # noqa: F401
    StockSnapshot,
    ROE_MIN_PCT,
    TRADING_VALUE_MIN_WON,
)

DISP_MIN_PCT = 2.5   # 90일선 이격도 최소 요구치(%)


def disparity_pct(price: float, ma90: float) -> float:
    if not ma90 or ma90 <= 0:
        return float("-inf")
    return (price - ma90) / ma90 * 100.0


def check_buy_conditions(stock: StockSnapshot, is_bull: bool) -> tuple[bool, list[str]]:
    fail_reasons: list[str] = []

    if not is_bull:
        fail_reasons.append("시장필터(KOSPI 200일선 하회, 약세장)")

    fair_price = stock.roe_pct * stock.eps
    if not (stock.price < fair_price):
        fail_reasons.append(f"저평가 아님 (현재가 {stock.price:,.0f} ≥ 적정가 {fair_price:,.0f})")

    if not (stock.roe_pct >= ROE_MIN_PCT):
        fail_reasons.append(f"ROE 미달 ({stock.roe_pct:.1f}% < {ROE_MIN_PCT}%)")

    disp = disparity_pct(stock.price, stock.ma_sell)
    if not (disp >= DISP_MIN_PCT):
        fail_reasons.append(f"90일선 이격도 미달 ({disp:.2f}% < {DISP_MIN_PCT:.1f}%)")

    if not (stock.momentum_20d > 0):
        fail_reasons.append(f"모멘텀 음수 ({stock.momentum_20d*100:.1f}%)")

    if not (stock.trading_value_20d_avg >= TRADING_VALUE_MIN_WON):
        fail_reasons.append(
            f"거래대금 부족 ({stock.trading_value_20d_avg/1e8:.1f}억 < {TRADING_VALUE_MIN_WON/1e8:.0f}억)"
        )

    return (len(fail_reasons) == 0), fail_reasons


if __name__ == "__main__":
    from datetime import date
    good = StockSnapshot(ticker="T", date=date(2026, 7, 9), price=9000,
                         ma_buy=99999, ma_sell=8000, roe_pct=18.0, eps=600,
                         momentum_20d=0.05, trading_value_20d_avg=5_000_000_000)
    assert check_buy_conditions(good, True)[0]
    # 이격도 2% (8160/8000) → disp25(≥2.5%)에서 탈락
    edge = StockSnapshot(ticker="T", date=date(2026, 7, 9), price=8160,
                         ma_buy=1, ma_sell=8000, roe_pct=18.0, eps=600,
                         momentum_20d=0.05, trading_value_20d_avg=5_000_000_000)
    assert not check_buy_conditions(edge, True)[0]
    print(f"[PASS] strategy_core_disp25 (이격도 ≥ {DISP_MIN_PCT:.1f}%) 로직 점검 완료")
