# -*- coding: utf-8 -*-
"""
strategy_core_align2.py
=======================
[정배열 변형 · 실험용] align1의 3단 정배열에 '현재가'까지 얹은 4단 정배열.

    정배열(4단): 현재가 > 20일선 > 60일선 > 90일선
    (이동평균 배열이 정렬돼 있을 뿐 아니라, 현재가도 가장 짧은 선 위에 있어야 함)

align1과의 유일한 차이는 맨 앞의 "현재가 > 20일선" 한 조각뿐이다. 상승추세라면
현재가가 20일선 위인 경우가 잦으므로 align1과 거의 같을 수 있다(그 검증이 목적).

★ 원본은 건드리지 않는다. 저평가/ROE/모멘텀/거래대금/시장필터는 그대로,
  매도·불타기·부분익절·사이징·비용 규칙도 전부 원본 strategy_core.py를 재사용한다.
  20일 이동평균(ma20)은 엔진이 스냅샷에 부착해 넘겨준다.
"""
from strategy_core import (  # noqa: F401
    StockSnapshot,
    ROE_MIN_PCT,
    TRADING_VALUE_MIN_WON,
)


def check_buy_conditions(stock: StockSnapshot, is_bull: bool) -> tuple[bool, list[str]]:
    fail_reasons: list[str] = []

    if not is_bull:
        fail_reasons.append("시장필터(KOSPI 200일선 하회, 약세장)")

    fair_price = stock.roe_pct * stock.eps
    if not (stock.price < fair_price):
        fail_reasons.append(f"저평가 아님 (현재가 {stock.price:,.0f} ≥ 적정가 {fair_price:,.0f})")

    if not (stock.roe_pct >= ROE_MIN_PCT):
        fail_reasons.append(f"ROE 미달 ({stock.roe_pct:.1f}% < {ROE_MIN_PCT}%)")

    # 3) 4단 정배열: 현재가 > 20일선 > 60일선 > 90일선
    ma20 = getattr(stock, "ma20", float("nan"))
    if not (stock.price > ma20 > stock.ma_buy > stock.ma_sell):
        fail_reasons.append(
            f"4단 정배열 아님 (현재가 {stock.price:,.0f} > 20선 {ma20:,.0f} > "
            f"60선 {stock.ma_buy:,.0f} > 90선 {stock.ma_sell:,.0f} 불성립)"
        )

    if not (stock.momentum_20d > 0):
        fail_reasons.append(f"모멘텀 음수 ({stock.momentum_20d*100:.1f}%)")

    if not (stock.trading_value_20d_avg >= TRADING_VALUE_MIN_WON):
        fail_reasons.append(
            f"거래대금 부족 ({stock.trading_value_20d_avg/1e8:.1f}억 < {TRADING_VALUE_MIN_WON/1e8:.0f}억)"
        )

    return (len(fail_reasons) == 0), fail_reasons


if __name__ == "__main__":
    from datetime import date
    # 4단 성립: 현재가 9000 > 20선 8800 > 60선 8500 > 90선 8000
    good = StockSnapshot(ticker="T", date=date(2026, 7, 9), price=9000,
                         ma_buy=8500, ma_sell=8000, roe_pct=18.0, eps=600,
                         momentum_20d=0.05, trading_value_20d_avg=5_000_000_000)
    good.ma20 = 8800
    assert check_buy_conditions(good, True)[0]
    # 이동평균 배열은 맞지만 현재가가 20선 아래(8700 < 8800) → 탈락
    bad = StockSnapshot(ticker="T", date=date(2026, 7, 9), price=8700,
                        ma_buy=8500, ma_sell=8000, roe_pct=18.0, eps=600,
                        momentum_20d=0.05, trading_value_20d_avg=5_000_000_000)
    bad.ma20 = 8800
    assert not check_buy_conditions(bad, True)[0]
    print("[PASS] strategy_core_align2 (정배열 현재가>20>60>90) 로직 점검 완료")
