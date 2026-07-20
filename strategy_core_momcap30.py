# -*- coding: utf-8 -*-
"""
strategy_core_momcap30.py
=========================
[과열방지 상한 · 실험용] 현재 확정 기준선(strategy_core.py) 위에 '모멘텀 상한'을 추가.
  기존:  20일모멘텀 > 0
  변형:  0 < 20일모멘텀 ≤ 30%   (너무 낮으면(≤0) 원래대로 탈락, 너무 높으면(>30%) 새로 탈락)

★ 이 상한은 '신규매수 판정(check_buy_conditions)'에만 적용된다.
  불타기(check_pyramid)는 원본 strategy_core 그대로라 상한의 영향을 받지 않는다.
  → 즉 "과열된 종목을 새로 사지는 않지만, 이미 보유한 종목이 과열되며 오르는 건 계속 불타기 허용".
★ 원본 strategy_core.py는 건드리지 않는다. 그 외 규칙(저평가/ROE/이격도/거래대금/시장필터/
  매도/불타기/부분익절/사이징/비용)은 전부 원본을 재사용한다.
"""
from strategy_core import (  # noqa: F401
    StockSnapshot, ROE_MIN_PCT, TRADING_VALUE_MIN_WON, BUY_DISPARITY_MIN_PCT,
)

MOM_CAP = 0.30   # 20일모멘텀 상한(초과 시 신규매수 제외)


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

    # ★ 모멘텀: 하한(>0) + 상한(≤ MOM_CAP)
    if not (stock.momentum_20d > 0):
        fail_reasons.append(f"모멘텀 음수 ({stock.momentum_20d*100:.1f}%)")
    elif not (stock.momentum_20d <= MOM_CAP):
        fail_reasons.append(f"모멘텀 과열 ({stock.momentum_20d*100:.1f}% > {MOM_CAP*100:.0f}%)")

    if not (stock.trading_value_20d_avg >= TRADING_VALUE_MIN_WON):
        fail_reasons.append(
            f"거래대금 부족 ({stock.trading_value_20d_avg/1e8:.1f}억 < {TRADING_VALUE_MIN_WON/1e8:.0f}억)"
        )

    return (len(fail_reasons) == 0), fail_reasons
