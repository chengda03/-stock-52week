# -*- coding: utf-8 -*-
"""
strategy_core_v3_combined.py
============================
[통합 변형 · 실험용] 원본 strategy_core.py에 아래 4가지를 '동시에' 적용한 통합본.

  1) 60일선 조건 삭제      : 매수조건 "현재가 > 60일선" 제거
  2) 90일선 이격도 조건     : "현재가 > 90일선" → 이격도 ≥ 2.5%
                             (이격도(%) = (현재가 - 90일선)/90일선 × 100)
  3) 코스피+코스닥 동시필터 : 신규매수는 'KOSPI>200선 AND KOSDAQ>200선'일 때만 허용
                             (★불타기·매도는 이 필터와 무관 — 기존 KOSPI 단독 로직대로)
  4) 부분익절 +8% → +6%     : 진입가 대비 +6% 도달시 50% 매도(1회성)

나머지(저평가 ROE×EPS, ROE≥15%, 20일모멘텀>0, 거래대금≥30억, 매도 90일선이탈/-12%하드손절,
불타기 +3% 복리)는 전부 원본과 동일.

★ 원본 strategy_core.py는 건드리지 않는다. 이 파일은 판정 함수만 제공하고,
  나머지 규칙과 실행은 백테스트 엔진(backtest_v3_combined.py)이 원본 것을 그대로 쓴다.

★ 시장필터 적용 위치(매우 중요):
  - is_bull_market_dual() : 신규매수 게이트용(코스피+코스닥 동시)
  - 불타기 게이트는 원본 is_bull_market()(KOSPI 단독)을 그대로 사용 → 엔진이 처리
  check_buy_conditions(stock, is_bull)의 is_bull 인자에는 엔진이 '동시필터 결과'를 넣어준다.
"""
from strategy_core import (  # noqa: F401
    StockSnapshot, Position,
    ROE_MIN_PCT, TRADING_VALUE_MIN_WON,
    is_bull_market,   # KOSPI 단독(불타기 게이트에 재사용)
)

DISP_MIN_PCT = 2.5              # (2) 90일선 이격도 최소 요구치(%)
PARTIAL_EXIT_TRIGGER_PCT = 0.06  # (4) 부분익절 트리거 +6%
PARTIAL_EXIT_RATIO = 0.5


# =========================================================================
# 시장필터 (3): 코스피 + 코스닥 동시 강세 — 신규매수 게이트용
# =========================================================================
def is_bull_market_dual(kospi_close: float, kospi_ma200: float,
                        kosdaq_close: float, kosdaq_ma200: float) -> bool:
    """KOSPI·KOSDAQ '둘 다' 200일선 위일 때만 True. 하나라도 아래면 신규매수 금지."""
    return (kospi_close > kospi_ma200) and (kosdaq_close > kosdaq_ma200)


# =========================================================================
# 매수조건 (1)+(2): 60일선 제거, 90일선을 이격도 2.5%로 교체
# =========================================================================
def disparity_pct(price: float, ma90: float) -> float:
    if not ma90 or ma90 <= 0:
        return float("-inf")
    return (price - ma90) / ma90 * 100.0


def check_buy_conditions(stock: StockSnapshot, is_bull: bool) -> tuple[bool, list[str]]:
    """
    매수조건 (통합 변형):
      1) 저평가: 현재가 < ROE(%) × EPS          (원본과 동일)
      2) ROE ≥ 15%                               (원본과 동일)
      3) 90일선 이격도 ≥ 2.5%                     ← ★60일선·90일선 위/아래 조건 대체
      4) 20일 모멘텀 > 0                          (원본과 동일)
      5) 20일평균거래대금 ≥ 30억원                (원본과 동일)
      6) 시장필터: 코스피+코스닥 동시 강세일 때만  ← ★엔진이 is_bull에 동시필터 결과 주입
    """
    fail_reasons: list[str] = []

    if not is_bull:
        fail_reasons.append("시장필터(코스피+코스닥 동시강세 아님)")

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


# =========================================================================
# 부분익절 (4): 트리거 +6% (원본 로직 동일, 비율만 변경)
# =========================================================================
def check_partial_exit(position: Position, current_price: float) -> bool:
    if position.partial_exit_done:
        return False
    return current_price >= position.entry_price * (1 + PARTIAL_EXIT_TRIGGER_PCT)


def apply_partial_exit(position: Position, current_price: float) -> float:
    sell_shares = position.shares * PARTIAL_EXIT_RATIO
    position.shares -= sell_shares
    position.partial_exit_done = True
    return sell_shares


if __name__ == "__main__":
    from datetime import date
    # 동시 시장필터
    assert is_bull_market_dual(2600, 2500, 900, 850) is True
    assert is_bull_market_dual(2600, 2500, 800, 850) is False   # 코스닥이 200선 아래
    assert is_bull_market_dual(2400, 2500, 900, 850) is False   # 코스피가 200선 아래
    # 매수조건: 이격도 12.5%(9000/8000), 60일선 무시
    good = StockSnapshot(ticker="T", date=date(2026, 7, 9), price=9000,
                         ma_buy=99999, ma_sell=8000, roe_pct=18.0, eps=600,
                         momentum_20d=0.05, trading_value_20d_avg=5_000_000_000)
    assert check_buy_conditions(good, True)[0]
    # 이격도 2%(8160/8000) < 2.5% → 탈락
    edge = StockSnapshot(ticker="T", date=date(2026, 7, 9), price=8160,
                         ma_buy=1, ma_sell=8000, roe_pct=18.0, eps=600,
                         momentum_20d=0.05, trading_value_20d_avg=5_000_000_000)
    assert not check_buy_conditions(edge, True)[0]
    # 부분익절 +6%
    pos = Position(ticker="T", entry_price=10000, avg_price=10000, shares=200,
                   entry_date=date(2026, 1, 1))
    assert check_partial_exit(pos, 10500) is False and check_partial_exit(pos, 10600) is True
    apply_partial_exit(pos, 10600)
    assert abs(pos.shares - 100) < 1e-9 and pos.partial_exit_done is True
    print("[PASS] strategy_core_v3_combined (60선삭제+이격2.5%+동시필터+익절6%) 로직 점검 완료")
