# -*- coding: utf-8 -*-
"""
strategy_core_margin30.py
=========================
[저평가 마진 변형 · 실험용] '새 기준선'(60일선 삭제 + 90일선 이격도 2.5%) 위에
저평가 안전마진 ≥ 30% 를 요구하는 (가장 엄격한) 버전.

    마진(%) = (적정가 - 현재가) / 현재가 × 100,   적정가 = ROE(%) × EPS
    "마진 ≥ 30%" = 적정가가 현재가보다 최소 30% 이상 높아야 통과.

나머지 조건(ROE≥15%, 이격도≥2.5%, 모멘텀>0, 거래대금≥30억, KOSPI단독필터)은 새 기준선과 동일.
★ strategy_core.py는 건드리지 않는다. 매수조건만 교체.
"""
from strategy_core import (  # noqa: F401
    StockSnapshot,
    ROE_MIN_PCT,
    TRADING_VALUE_MIN_WON,
    BUY_DISPARITY_MIN_PCT,
)

MARGIN_MIN_PCT = 30.0


def valuation_margin_pct(price: float, fair_price: float) -> float:
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

    fair_price = stock.roe_pct * stock.eps
    margin = valuation_margin_pct(stock.price, fair_price)
    if not (margin >= MARGIN_MIN_PCT):
        fail_reasons.append(f"저평가 마진 미달 ({margin:.1f}% < {MARGIN_MIN_PCT:.0f}%)")

    if not (stock.roe_pct >= ROE_MIN_PCT):
        fail_reasons.append(f"ROE 미달 ({stock.roe_pct:.1f}% < {ROE_MIN_PCT}%)")

    disp = disparity_pct(stock.price, stock.ma_sell)
    if not (disp >= BUY_DISPARITY_MIN_PCT):
        fail_reasons.append(f"90일선 이격도 미달 ({disp:.2f}% < {BUY_DISPARITY_MIN_PCT:.1f}%)")

    if not (stock.momentum_20d > 0):
        fail_reasons.append(f"모멘텀 음수 ({stock.momentum_20d*100:.1f}%)")

    if not (stock.trading_value_20d_avg >= TRADING_VALUE_MIN_WON):
        fail_reasons.append(
            f"거래대금 부족 ({stock.trading_value_20d_avg/1e8:.1f}억 < {TRADING_VALUE_MIN_WON/1e8:.0f}억)"
        )

    return (len(fail_reasons) == 0), fail_reasons


if __name__ == "__main__":
    from datetime import date
    # 적정가 12000(=20*600), 현재가 9000 → 마진 33.3% ≥ 30% 통과, 이격도 12.5% 통과
    good = StockSnapshot(ticker="T", date=date(2026, 7, 9), price=9000,
                         ma_buy=1, ma_sell=8000, roe_pct=20.0, eps=600,
                         momentum_20d=0.05, trading_value_20d_avg=5_000_000_000)
    assert check_buy_conditions(good, True)[0]
    # 적정가 10800, 현재가 9000 → 마진 20% < 30% → 탈락
    edge = StockSnapshot(ticker="T", date=date(2026, 7, 9), price=9000,
                         ma_buy=1, ma_sell=8000, roe_pct=18.0, eps=600,
                         momentum_20d=0.05, trading_value_20d_avg=5_000_000_000)
    assert not check_buy_conditions(edge, True)[0]
    print(f"[PASS] strategy_core_margin30 (저평가 마진 ≥ {MARGIN_MIN_PCT:.0f}%) 로직 점검 완료")
