# -*- coding: utf-8 -*-
"""
strategy_core_combined_cap.py
=============================
[과열방지 상한 · 실험용] 모멘텀 상한 30% + 이격도 상한 30%를 '동시' 적용한 결합 버전.
  변형:  0 < 20일모멘텀 ≤ 30%   그리고   2.5% ≤ 이격도 ≤ 30%
★ 신규매수 판정(check_buy_conditions)에만 적용. 불타기(check_pyramid)는 원본 그대로.
★ 원본 strategy_core.py는 건드리지 않는다.
"""
from strategy_core import (  # noqa: F401
    StockSnapshot, ROE_MIN_PCT, TRADING_VALUE_MIN_WON, BUY_DISPARITY_MIN_PCT,
)

MOM_CAP = 0.30
DISP_CAP = 30.0


def check_buy_conditions(stock: StockSnapshot, is_bull: bool) -> tuple[bool, list[str]]:
    fail_reasons: list[str] = []
    if not is_bull:
        fail_reasons.append("시장필터(KOSPI 200일선 하회, 약세장)")

    fair_price = stock.roe_pct * stock.eps
    if not (stock.price < fair_price):
        fail_reasons.append(f"저평가 아님 (현재가 {stock.price:,.0f} ≥ 적정가 {fair_price:,.0f})")

    if not (stock.roe_pct >= ROE_MIN_PCT):
        fail_reasons.append(f"ROE 미달 ({stock.roe_pct:.1f}% < {ROE_MIN_PCT}%)")

    if stock.ma_sell and stock.ma_sell > 0:
        disparity = (stock.price - stock.ma_sell) / stock.ma_sell * 100.0
    else:
        disparity = float("-inf")
    if not (disparity >= BUY_DISPARITY_MIN_PCT):
        fail_reasons.append(f"90일선 이격도 미달 ({disparity:.2f}% < {BUY_DISPARITY_MIN_PCT:.1f}%)")
    elif not (disparity <= DISP_CAP):
        fail_reasons.append(f"90일선 이격도 과열 ({disparity:.2f}% > {DISP_CAP:.0f}%)")

    if not (stock.momentum_20d > 0):
        fail_reasons.append(f"모멘텀 음수 ({stock.momentum_20d*100:.1f}%)")
    elif not (stock.momentum_20d <= MOM_CAP):
        fail_reasons.append(f"모멘텀 과열 ({stock.momentum_20d*100:.1f}% > {MOM_CAP*100:.0f}%)")

    if not (stock.trading_value_20d_avg >= TRADING_VALUE_MIN_WON):
        fail_reasons.append(
            f"거래대금 부족 ({stock.trading_value_20d_avg/1e8:.1f}억 < {TRADING_VALUE_MIN_WON/1e8:.0f}억)"
        )

    return (len(fail_reasons) == 0), fail_reasons
