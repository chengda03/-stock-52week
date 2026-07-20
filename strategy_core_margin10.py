# -*- coding: utf-8 -*-
"""
strategy_core_margin10.py
=========================
[저평가 마진 변형 · 실험용] '새 기준선'(2026-07-14 반영: 60일선 삭제 + 90일선 이격도 2.5%)
위에, 저평가 조건에 '안전마진'을 추가한 버전.

    (기존)   저평가: 현재가 < 적정가            (마진 0% = 적정가가 현재가보다 조금이라도 높으면 통과)
    (이 변형) 저평가 마진 ≥ 10%                  ← 적정가가 현재가보다 최소 10% 이상 높아야 통과
      마진(%) = (적정가 - 현재가) / 현재가 × 100,   적정가 = ROE(%) × EPS

새 기준선의 나머지 조건(ROE≥15%, 90일선 이격도≥2.5%, 20일모멘텀>0, 거래대금≥30억,
시장필터 KOSPI단독)은 그대로 유지. 매도/불타기/부분익절/사이징/비용도 원본 재사용.

★ strategy_core.py는 건드리지 않는다(이미 새 기준선으로 반영 완료). 여기선 매수조건만 교체.
"""
from strategy_core import (  # noqa: F401
    StockSnapshot,
    ROE_MIN_PCT,
    TRADING_VALUE_MIN_WON,
    BUY_DISPARITY_MIN_PCT,   # 2.5 — 새 기준선의 90일선 이격도 기준
)

MARGIN_MIN_PCT = 10.0   # 저평가 안전마진 최소 요구치(%)


def valuation_margin_pct(price: float, fair_price: float) -> float:
    """마진(%) = (적정가 - 현재가) / 현재가 × 100. 현재가가 0 이하면 -inf(탈락)."""
    if not price or price <= 0:
        return float("-inf")
    return (fair_price - price) / price * 100.0


def disparity_pct(price: float, ma90: float) -> float:
    if not ma90 or ma90 <= 0:
        return float("-inf")
    return (price - ma90) / ma90 * 100.0


def check_buy_conditions(stock: StockSnapshot, is_bull: bool) -> tuple[bool, list[str]]:
    fail_reasons: list[str] = []

    if not is_bull:
        fail_reasons.append("시장필터(KOSPI 200일선 하회, 약세장)")

    # 1) 저평가 '마진' ≥ MARGIN_MIN_PCT  (적정가 = ROE(%)×EPS)
    fair_price = stock.roe_pct * stock.eps
    margin = valuation_margin_pct(stock.price, fair_price)
    if not (margin >= MARGIN_MIN_PCT):
        fail_reasons.append(
            f"저평가 마진 미달 ({margin:.1f}% < {MARGIN_MIN_PCT:.0f}%)"
        )

    # 2) ROE ≥ 15%
    if not (stock.roe_pct >= ROE_MIN_PCT):
        fail_reasons.append(f"ROE 미달 ({stock.roe_pct:.1f}% < {ROE_MIN_PCT}%)")

    # 3) 90일선 이격도 ≥ 2.5% (새 기준선)
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
    # 적정가 10800(=18*600), 현재가 9000 → 마진 20% ≥ 10% 통과, 이격도 12.5% 통과
    good = StockSnapshot(ticker="T", date=date(2026, 7, 9), price=9000,
                         ma_buy=1, ma_sell=8000, roe_pct=18.0, eps=600,
                         momentum_20d=0.05, trading_value_20d_avg=5_000_000_000)
    assert check_buy_conditions(good, True)[0]
    # 현재가 10200 → 마진 (10800-10200)/10200=5.9% < 10% → 탈락
    edge = StockSnapshot(ticker="T", date=date(2026, 7, 9), price=10200,
                         ma_buy=1, ma_sell=9000, roe_pct=18.0, eps=600,
                         momentum_20d=0.05, trading_value_20d_avg=5_000_000_000)
    assert not check_buy_conditions(edge, True)[0]
    print(f"[PASS] strategy_core_margin10 (저평가 마진 ≥ {MARGIN_MIN_PCT:.0f}%) 로직 점검 완료")
