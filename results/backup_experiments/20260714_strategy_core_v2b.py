# -*- coding: utf-8 -*-
"""
strategy_core_v2b.py  (S1 v2 의 '90일선 매수필터' 변형)
=======================================================
★ 이 파일은 strategy_core.py 를 통째로 복사한 뒤, 매수조건 판정 함수
  check_buy_conditions() 에 **딱 한 줄짜리 조건**만 추가한 버전입니다.

    추가된 조건:  "현재가 > 90일 이동평균(ma_sell)"

왜 추가했나 (배경)
------------------
기존 S1 v2 는 매수조건에 60일선(ma_buy), 매도조건에 90일선(ma_sell)을
서로 다르게 씁니다. 그래서 현재가가 60일선과 90일선 '사이'에 낀 경우
  (예: 지엔씨에너지 60일선 27,840 < 현재가 28,950 < 90일선 29,796)
"매수조건 통과 + 매도조건(90일선 이탈) 동시 발동"이 나올 수 있었습니다.
→ 사자마자 매도 대상이 되는 whipsaw. 이걸 원천 차단하려고
   매수 단계에서 "현재가가 90일선 위에도 있어야 함"을 요구하는 변형입니다.

나머지(매도조건 2개, 불타기, 부분익절 8%@50%, 시장필터 등)는
strategy_core.py 와 완전히 동일합니다. 즉 '매수 진입 문턱만' 살짝 높인 버전.

⚠️ 이건 그리드서치성 실험(조건 추가 실험)이므로 과최적화 위험이 있습니다.
   결과가 좋아지든 나빠지든 있는 그대로 해석해야 합니다.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Optional


# =========================================================================
# 1. 설정값 (strategy_core.py 와 동일)
# =========================================================================

BUY_MA_DAYS = 60          # 매수조건 ③: 현재가가 이 이동평균선 위여야 함
SELL_MA_DAYS = 90         # 매도조건 ①: 종가가 이 이동평균선 아래면 매도
HARD_STOP_LOSS_PCT = 0.12 # 매도조건 ②: 평단가 대비 -12% 하드손절

PARTIAL_EXIT_TRIGGER_PCT = 0.08   # 진입가 대비 +8% 도달 시 트리거
PARTIAL_EXIT_RATIO = 0.5          # 그때 보유수량의 50% 매도

ROE_MIN_PCT = 15.0                     # 매수조건 ②: ROE(%) 최소값
MOMENTUM_DAYS = 20                      # 매수조건 ④: 20일 모멘텀
TRADING_VALUE_MIN_WON = 3_000_000_000   # 매수조건 ⑤: 20일평균거래대금 30억원 이상

MARKET_FILTER_MA_DAYS = 200   # KOSPI 종가가 이 이동평균 위일 때만 "강세장"

PYRAMID_TRIGGER_PCT = 0.03    # 진입가 대비 +3%씩 복리 상승 트리거
PYRAMID_ADD_AMOUNT_WON = 5_000_000
PYRAMID_DAILY_LIMIT_PER_STOCK = 1

NUM_SLOTS = 20
SLOT_AMOUNT_WON = 5_000_000


# =========================================================================
# 2. 입력 데이터 형태 정의 (strategy_core.py 와 동일)
# =========================================================================

@dataclass
class StockSnapshot:
    """어떤 종목의 '어느 날' 상태 한 장."""
    ticker: str
    date: date
    price: float
    ma_buy: float                 # 60일 이동평균 (매수조건용)
    ma_sell: float                # 90일 이동평균 (매도조건용, 이번 변형에서 매수에도 사용)
    roe_pct: float
    eps: float
    momentum_20d: float
    trading_value_20d_avg: float


@dataclass
class Position:
    """현재 보유중인 포지션 하나의 상태."""
    ticker: str
    entry_price: float
    avg_price: float
    shares: float
    pyramid_count: int = 0
    last_pyramid_date: Optional[date] = None
    entry_date: Optional[date] = None
    partial_exit_done: bool = False


# =========================================================================
# 3. 시장필터 (strategy_core.py 와 동일)
# =========================================================================

def is_bull_market(kospi_close: float, kospi_ma200: float) -> bool:
    """KOSPI 종가가 200일선 위면 강세장."""
    return kospi_close > kospi_ma200


# =========================================================================
# 4. 매수조건 판정  ★★★ 여기만 변경됨 ★★★
# =========================================================================

def check_buy_conditions(stock: StockSnapshot, is_bull: bool) -> tuple[bool, list[str]]:
    """
    매수조건을 확인합니다. strategy_core.py 의 6개 조건 그대로에,
    ★ 조건 3-b(신규): "현재가 > 90일선(ma_sell)" 한 가지를 추가했습니다.

    기존 6개:
      1) 저평가: 현재가 < ROE(%) × EPS
      2) ROE ≥ 15%
      3) 현재가 > 60일선
      4) 20일 모멘텀 > 0
      5) 20일평균거래대금 ≥ 30억원
      6) 시장필터: KOSPI 강세장
    추가 1개:
      3-b) 현재가 > 90일선   ← 매수 직후 90일선 이탈 매도(whipsaw) 방지

    반환값: (매수가능여부, 실패한 조건 이름 리스트)
    """
    fail_reasons = []

    # 조건 6: 시장필터
    if not is_bull:
        fail_reasons.append("시장필터(KOSPI 200일선 하회, 약세장)")

    # 조건 1: 저평가
    fair_price = stock.roe_pct * stock.eps
    if not (stock.price < fair_price):
        fail_reasons.append(f"저평가 아님 (현재가 {stock.price:,.0f} ≥ 적정가 {fair_price:,.0f})")

    # 조건 2: ROE ≥ 15%
    if not (stock.roe_pct >= ROE_MIN_PCT):
        fail_reasons.append(f"ROE 미달 ({stock.roe_pct:.1f}% < {ROE_MIN_PCT}%)")

    # 조건 3: 60일선 위
    if not (stock.price > stock.ma_buy):
        fail_reasons.append(f"60일선 아래 (현재가 {stock.price:,.0f} ≤ 60일선 {stock.ma_buy:,.0f})")

    # ★ 조건 3-b (이번 변형 신규): 90일선 위
    #    현재가가 90일선 아래면, 사는 즉시 매도조건(90일선 이탈)이 켜지므로 진입 자체를 막는다.
    if not (stock.price > stock.ma_sell):
        fail_reasons.append(f"90일선 아래 (현재가 {stock.price:,.0f} ≤ 90일선 {stock.ma_sell:,.0f})")

    # 조건 4: 20일 모멘텀 > 0
    if not (stock.momentum_20d > 0):
        fail_reasons.append(f"모멘텀 음수 ({stock.momentum_20d*100:.1f}%)")

    # 조건 5: 20일평균거래대금 ≥ 30억
    if not (stock.trading_value_20d_avg >= TRADING_VALUE_MIN_WON):
        fail_reasons.append(
            f"거래대금 부족 ({stock.trading_value_20d_avg/1e8:.1f}억 < {TRADING_VALUE_MIN_WON/1e8:.0f}억)"
        )

    can_buy = (len(fail_reasons) == 0)
    return can_buy, fail_reasons


def rank_by_momentum(candidates: list[StockSnapshot]) -> list[StockSnapshot]:
    """매수조건 통과 후보를 20일 모멘텀 높은 순 정렬 (strategy_core.py 와 동일)."""
    return sorted(candidates, key=lambda s: s.momentum_20d, reverse=True)


# =========================================================================
# 5. 매도조건 판정 (strategy_core.py 와 완전 동일)
# =========================================================================

def check_sell_condition(position: Position, stock: StockSnapshot) -> tuple[bool, Optional[str]]:
    """① 종가 < 90일선 → '90일선 이탈'  ② 현재가 ≤ 평단가×(1-0.12) → '하드손절'."""
    if stock.price < stock.ma_sell:
        return True, "90일선 이탈"
    stop_loss_price = position.avg_price * (1 - HARD_STOP_LOSS_PCT)
    if stock.price <= stop_loss_price:
        return True, f"하드손절(평단가 -{HARD_STOP_LOSS_PCT*100:.0f}%)"
    return False, None


def check_partial_exit(position: Position, current_price: float) -> bool:
    """부분익절(+8%에서 50%) 여부 (strategy_core.py 와 동일)."""
    if position.partial_exit_done:
        return False
    return current_price >= position.entry_price * (1 + PARTIAL_EXIT_TRIGGER_PCT)


def apply_partial_exit(position: Position, current_price: float) -> float:
    """부분익절 실행 시 포지션 갱신 (strategy_core.py 와 동일). 매도수량 반환."""
    sell_shares = position.shares * PARTIAL_EXIT_RATIO
    position.shares -= sell_shares
    position.partial_exit_done = True
    return sell_shares


# =========================================================================
# 6. 불타기(피라미딩) (strategy_core.py 와 완전 동일)
# =========================================================================

def _pyramid_trigger_price(entry_price: float, step: int) -> float:
    """n단계 불타기 트리거 가격 = 진입가 × 1.03^n (복리)."""
    return entry_price * ((1 + PYRAMID_TRIGGER_PCT) ** step)


def check_pyramid(position: Position, current_price: float, current_date: date,
                  is_bull: bool, available_cash: float) -> bool:
    """오늘 이 종목에 불타기 할지 판정 (strategy_core.py 와 동일)."""
    if not is_bull:
        return False
    if position.last_pyramid_date == current_date:
        return False
    if available_cash < PYRAMID_ADD_AMOUNT_WON:
        return False
    next_step = position.pyramid_count + 1
    trigger_price = _pyramid_trigger_price(position.entry_price, next_step)
    return current_price >= trigger_price


def apply_pyramid(position: Position, current_price: float, current_date: date) -> Position:
    """불타기 실행 시 포지션 갱신 (strategy_core.py 와 동일)."""
    add_shares = PYRAMID_ADD_AMOUNT_WON / current_price
    old_cost = position.avg_price * position.shares
    new_cost = old_cost + PYRAMID_ADD_AMOUNT_WON
    new_shares = position.shares + add_shares
    position.avg_price = new_cost / new_shares
    position.shares = new_shares
    position.pyramid_count += 1
    position.last_pyramid_date = current_date
    return position


# =========================================================================
# 7. 자체 점검 (90일선 매수필터가 실제로 걸리는지 확인)
# =========================================================================
if __name__ == "__main__":
    print("=== strategy_core_v2b.py 자체 점검 시작 ===\n")

    # 모든 조건 만족(90일선 위 포함) → 매수 가능
    good = StockSnapshot(ticker="T", date=date(2026, 7, 9), price=9000,
                         ma_buy=8500, ma_sell=8800,  # 현재가 9000 > 90일선 8800 → 통과
                         roe_pct=18.0, eps=600, momentum_20d=0.05,
                         trading_value_20d_avg=5_000_000_000)
    ok, reasons = check_buy_conditions(good, is_bull=True)
    assert ok is True, f"통과해야 하는데 실패: {reasons}"
    print("[PASS] 90일선 위 + 6조건 만족 → 매수 가능")

    # 60일선은 위지만 90일선 아래(경계상황) → 이번 변형에선 매수 차단돼야 함
    edge = StockSnapshot(ticker="T", date=date(2026, 7, 9), price=8600,
                         ma_buy=8500, ma_sell=8800,  # 8500 < 8600 < 8800
                         roe_pct=18.0, eps=600, momentum_20d=0.05,
                         trading_value_20d_avg=5_000_000_000)
    ok2, reasons2 = check_buy_conditions(edge, is_bull=True)
    assert ok2 is False and any("90일선 아래" in r for r in reasons2), reasons2
    print("[PASS] 60일선 위·90일선 아래(경계) → 매수 차단 (신규 필터 작동)")

    print("\n=== 통과: 90일선 매수필터가 정상 작동합니다 ===")
