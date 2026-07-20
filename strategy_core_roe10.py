# -*- coding: utf-8 -*-
"""
strategy_core_roe10.py
======================
[ROE 기준 변형 · 실험용] '새 기준선'(2026-07-14 반영: 60일선 삭제 + 90일선 이격도 2.5%,
거래대금 30억 유지) 위에서, 퀄리티 필터인 ROE 문턱만 완화한 버전.

    (새 기준선) ROE ≥ 15%
    (이 변형)   ROE ≥ 10%   ← ★여기만 변경

나머지 조건(저평가 ROE×EPS, 90일선 이격도≥2.5%, 20일모멘텀>0, 거래대금≥30억, 시장필터 KOSPI단독)은
새 기준선과 동일. 매도/불타기/부분익절/사이징/비용도 전부 원본 strategy_core.py 재사용.

★ strategy_core.py는 건드리지 않는다. 여기선 매수조건의 ROE 문턱만 10%로 완화.
"""
from strategy_core import (  # noqa: F401
    StockSnapshot,
    TRADING_VALUE_MIN_WON,
    BUY_DISPARITY_MIN_PCT,
)

ROE_MIN_PCT = 10.0   # ★완화된 ROE 문턱 (새 기준선은 15.0)


def disparity_pct(price: float, ma90: float) -> float:
    if not ma90 or ma90 <= 0:
        return float("-inf")
    return (price - ma90) / ma90 * 100.0


def check_buy_conditions(stock: StockSnapshot, is_bull: bool) -> tuple[bool, list[str]]:
    """새 기준선과 동일하되 ROE 문턱만 10%로 완화."""
    fail_reasons: list[str] = []

    if not is_bull:
        fail_reasons.append("시장필터(KOSPI 200일선 하회, 약세장)")

    # 1) 저평가 (적정가 = ROE(%)×EPS)  — 공식은 그대로(실제 ROE 사용)
    fair_price = stock.roe_pct * stock.eps
    if not (stock.price < fair_price):
        fail_reasons.append(f"저평가 아님 (현재가 {stock.price:,.0f} ≥ 적정가 {fair_price:,.0f})")

    # 2) ROE ≥ 10%  ← 완화된 문턱
    if not (stock.roe_pct >= ROE_MIN_PCT):
        fail_reasons.append(f"ROE 미달 ({stock.roe_pct:.1f}% < {ROE_MIN_PCT}%)")

    # 3) 90일선 이격도 ≥ 2.5%
    disp = disparity_pct(stock.price, stock.ma_sell)
    if not (disp >= BUY_DISPARITY_MIN_PCT):
        fail_reasons.append(f"90일선 이격도 미달 ({disp:.2f}% < {BUY_DISPARITY_MIN_PCT:.1f}%)")

    # 4) 20일 모멘텀 > 0
    if not (stock.momentum_20d > 0):
        fail_reasons.append(f"모멘텀 음수 ({stock.momentum_20d*100:.1f}%)")

    # 5) 20일평균거래대금 ≥ 30억
    if not (stock.trading_value_20d_avg >= TRADING_VALUE_MIN_WON):
        fail_reasons.append(
            f"거래대금 부족 ({stock.trading_value_20d_avg/1e8:.1f}억 < {TRADING_VALUE_MIN_WON/1e8:.0f}억)"
        )

    return (len(fail_reasons) == 0), fail_reasons


if __name__ == "__main__":
    from datetime import date
    # ROE 12% → 15% 기준선에선 탈락하지만 10% 기준선에선 통과 (적정가=12*900=10800 > 9000)
    s = StockSnapshot(ticker="T", date=date(2026, 7, 9), price=9000,
                      ma_buy=1, ma_sell=8000, roe_pct=12.0, eps=900,
                      momentum_20d=0.05, trading_value_20d_avg=5_000_000_000)
    assert check_buy_conditions(s, True)[0]
    # ROE 8% → 10% 기준선에서도 탈락
    s2 = StockSnapshot(ticker="T", date=date(2026, 7, 9), price=9000,
                       ma_buy=1, ma_sell=8000, roe_pct=8.0, eps=2000,
                       momentum_20d=0.05, trading_value_20d_avg=5_000_000_000)
    assert not check_buy_conditions(s2, True)[0]
    print(f"[PASS] strategy_core_roe10 (ROE ≥ {ROE_MIN_PCT:.0f}%) 로직 점검 완료")
