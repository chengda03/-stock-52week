# -*- coding: utf-8 -*-
"""
strategy_core_disp0.py
======================
[이격도 변형 · 실험용] 원본 strategy_core.py의 매수조건에서
  · "현재가 > 60일선" (조건 3)
  · "현재가 > 90일선" (조건 3-b)
두 개를 '완전히 제거'하고, 대신 90일선 '이격도' 조건 하나로 교체한 버전.

    이격도(%) = (현재가 - 90일선) / 90일선 × 100

이 파일(disp0)은 이격도 ≥ 0% 를 요구한다.
  → 현재가가 90일선보다 높기만 하면 됨. 기존 "현재가 > 90일선"과 사실상 동일하되
    (경계값 등호 차이만 있음) 60일선 조건은 뺀 버전이다.
  → 따라서 "60일선 조건이 실제로 무슨 역할을 했는가"를 base와의 차이로 보여준다.

★ 원본은 건드리지 않는다. 저평가/ROE/모멘텀/거래대금/시장필터는 그대로,
  매도·불타기·부분익절·사이징·비용 규칙도 전부 원본 strategy_core.py를 재사용한다.
  (엔진이 매수조건 함수만 이 파일 것으로 갈아끼운다)
"""
from strategy_core import (  # noqa: F401  (StockSnapshot는 타입 참고용)
    StockSnapshot,
    ROE_MIN_PCT,
    TRADING_VALUE_MIN_WON,
)

# --- 이 변형의 유일한 자유 파라미터 ---
DISP_MIN_PCT = 0.0   # 90일선 이격도 최소 요구치(%)


def disparity_pct(price: float, ma90: float) -> float:
    """이격도(%) = (현재가 - 90일선) / 90일선 × 100. 90일선이 없거나 0 이하면 -inf(탈락)."""
    if not ma90 or ma90 <= 0:
        return float("-inf")
    return (price - ma90) / ma90 * 100.0


def check_buy_conditions(stock: StockSnapshot, is_bull: bool) -> tuple[bool, list[str]]:
    """
    매수조건 (이격도 변형):
      1) 저평가: 현재가 < ROE(%) × EPS        (원본과 동일)
      2) ROE ≥ 15%                             (원본과 동일)
      3) 90일선 이격도 ≥ DISP_MIN_PCT(%)       ← ★60일선·90일선 위/아래 조건을 이걸로 교체
      4) 20일 모멘텀 > 0                        (원본과 동일)
      5) 20일평균거래대금 ≥ 30억원              (원본과 동일)
      6) 시장필터: KOSPI 강세장일 때만          (원본과 동일)
    """
    fail_reasons: list[str] = []

    if not is_bull:
        fail_reasons.append("시장필터(KOSPI 200일선 하회, 약세장)")

    # 1) 저평가
    fair_price = stock.roe_pct * stock.eps
    if not (stock.price < fair_price):
        fail_reasons.append(f"저평가 아님 (현재가 {stock.price:,.0f} ≥ 적정가 {fair_price:,.0f})")

    # 2) ROE ≥ 15%
    if not (stock.roe_pct >= ROE_MIN_PCT):
        fail_reasons.append(f"ROE 미달 ({stock.roe_pct:.1f}% < {ROE_MIN_PCT}%)")

    # 3) 90일선 이격도 ≥ DISP_MIN_PCT (60일선/90일선 위·아래 조건 대체)
    disp = disparity_pct(stock.price, stock.ma_sell)
    if not (disp >= DISP_MIN_PCT):
        fail_reasons.append(f"90일선 이격도 미달 ({disp:.2f}% < {DISP_MIN_PCT:.1f}%)")

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
    # 이격도 12.5% 종목 (price 9000, ma90 8000) → 모든 임계값 통과
    good = StockSnapshot(ticker="T", date=date(2026, 7, 9), price=9000,
                         ma_buy=99999, ma_sell=8000, roe_pct=18.0, eps=600,
                         momentum_20d=0.05, trading_value_20d_avg=5_000_000_000)
    ok, why = check_buy_conditions(good, is_bull=True)
    assert ok, why
    # 이격도 -1% (price 7920, ma90 8000) → disp0(≥0%)에서 탈락
    bad = StockSnapshot(ticker="T", date=date(2026, 7, 9), price=7920,
                        ma_buy=1, ma_sell=8000, roe_pct=18.0, eps=600,
                        momentum_20d=0.05, trading_value_20d_avg=5_000_000_000)
    ok2, _ = check_buy_conditions(bad, is_bull=True)
    assert not ok2
    print(f"[PASS] strategy_core_disp0 (이격도 ≥ {DISP_MIN_PCT:.1f}%) 로직 점검 완료")
