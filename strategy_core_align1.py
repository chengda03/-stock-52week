# -*- coding: utf-8 -*-
"""
strategy_core_align1.py
=======================
[정배열 변형 · 실험용] 원본 strategy_core.py의 매수조건에서
  · "현재가 > 60일선" (조건 3)
  · "현재가 > 90일선" (조건 3-b)
두 개를 '완전히 제거'하고, 대신 '진짜 정배열' 조건 하나로 교체한 버전.

    정배열(3단): 20일선 > 60일선 > 90일선
    (짧은 이동평균이 항상 긴 이동평균보다 위 → 추세가 방향성 있게 정렬된 상태)

기존 방식("현재가 vs 각 선"의 독립 문턱)은 이동평균선들 '끼리의 순서'를 요구하지
않아 논리적 일관성이 약하다는 지적에 따른 실험. 여기서는 이동평균의 '배열'만 보고
현재가 위치는 조건에 넣지 않는다(그건 align2가 담당).

★ 이 변형은 20일 이동평균(ma20)이 필요하다. 원본 StockSnapshot에는 60/90일선만
  있으므로, 백테스트 엔진이 스냅샷에 stock.ma20 속성을 부착해 넘겨준다.
  (dataclass 동적 속성 — bt_valuation_engine의 fin2 부착과 동일한 방식)

★ 원본은 건드리지 않는다. 저평가/ROE/모멘텀/거래대금/시장필터는 그대로,
  매도·불타기·부분익절·사이징·비용 규칙도 전부 원본 strategy_core.py를 재사용한다.
"""
from strategy_core import (  # noqa: F401
    StockSnapshot,
    ROE_MIN_PCT,
    TRADING_VALUE_MIN_WON,
)


def check_buy_conditions(stock: StockSnapshot, is_bull: bool) -> tuple[bool, list[str]]:
    """
    매수조건 (정배열 3단 변형):
      1) 저평가: 현재가 < ROE(%) × EPS         (원본과 동일)
      2) ROE ≥ 15%                              (원본과 동일)
      3) 정배열: 20일선 > 60일선 > 90일선       ← ★60/90 개별 문턱을 배열조건으로 교체
      4) 20일 모멘텀 > 0                         (원본과 동일)
      5) 20일평균거래대금 ≥ 30억원               (원본과 동일)
      6) 시장필터: KOSPI 강세장일 때만           (원본과 동일)
    """
    fail_reasons: list[str] = []

    if not is_bull:
        fail_reasons.append("시장필터(KOSPI 200일선 하회, 약세장)")

    fair_price = stock.roe_pct * stock.eps
    if not (stock.price < fair_price):
        fail_reasons.append(f"저평가 아님 (현재가 {stock.price:,.0f} ≥ 적정가 {fair_price:,.0f})")

    if not (stock.roe_pct >= ROE_MIN_PCT):
        fail_reasons.append(f"ROE 미달 ({stock.roe_pct:.1f}% < {ROE_MIN_PCT}%)")

    # 3) 정배열: 20일선 > 60일선 > 90일선  (결측이면 chained-compare가 False → 자동 탈락)
    ma20 = getattr(stock, "ma20", float("nan"))
    if not (ma20 > stock.ma_buy > stock.ma_sell):
        fail_reasons.append(
            f"정배열 아님 (20선 {ma20:,.0f} > 60선 {stock.ma_buy:,.0f} > 90선 {stock.ma_sell:,.0f} 불성립)"
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
    # 정배열 성립: 20선 8800 > 60선 8500 > 90선 8000
    good = StockSnapshot(ticker="T", date=date(2026, 7, 9), price=9000,
                         ma_buy=8500, ma_sell=8000, roe_pct=18.0, eps=600,
                         momentum_20d=0.05, trading_value_20d_avg=5_000_000_000)
    good.ma20 = 8800
    assert check_buy_conditions(good, True)[0]
    # 정배열 깨짐: 20선(7900) < 60선(8500) → 탈락
    bad = StockSnapshot(ticker="T", date=date(2026, 7, 9), price=9000,
                        ma_buy=8500, ma_sell=8000, roe_pct=18.0, eps=600,
                        momentum_20d=0.05, trading_value_20d_avg=5_000_000_000)
    bad.ma20 = 7900
    assert not check_buy_conditions(bad, True)[0]
    print("[PASS] strategy_core_align1 (정배열 20>60>90) 로직 점검 완료")
