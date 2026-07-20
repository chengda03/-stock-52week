# -*- coding: utf-8 -*-
"""
strategy_core.py
=================
"ROE×EPS 모멘텀 불타기 — 90일선 매도형" 전략의 판정 로직만 모아둔 파일입니다.

★★★ 매우 중요 ★★★
이 파일은 백테스트(backtest_v3.py)와 실운영 앱(app.py) 양쪽에서
"그대로 똑같이" import해서 씁니다. 즉, 이 파일에 있는 규칙이
"진짜 전략의 정의" 그 자체입니다.

    같은 로직 하나를 두 군데서 쓰는 이유
    -----------------------------------
    만약 백테스트용 코드와 실제 매매신호용 코드를 따로 짜면,
    나중에 "손절선을 -12%에서 -10%로 바꿔볼까?" 같은 수정을 할 때
    한쪽만 고치고 다른 쪽을 깜빡하는 사고가 반드시 일어납니다.
    그래서 판정 함수는 이 파일 하나에만 존재해야 합니다.

이 파일 자체는 "숫자 계산"만 합니다. 즉,
    - 인터넷에서 주가를 받아오는 코드 (X, 여긴 없음 → data_layer.py가 담당)
    - 화면에 표를 그리는 코드 (X, 여긴 없음 → app.py가 담당)
    - "지금 이 가격이 매수조건을 만족하는가?" 같은 순수 판정만 (O, 여기 있음)

이렇게 분리해두면, 이 파일만 따로 떼서 단위테스트(가짜 숫자로 검증)를
할 수 있어서 로직 자체의 정확성을 데이터 문제와 분리해서 확인할 수 있습니다.

출처: STRATEGY_FINAL.md (결정일 2026-06-10, 개정 2026-06-17)
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Optional


# =========================================================================
# 1. 설정값 (여기만 보면 전략의 모든 숫자를 한눈에 알 수 있게 모아둠)
# =========================================================================

# --- 매수조건에 쓰이는 이동평균 ---
BUY_MA_DAYS = 60          # (2026-07-14 개정) 매수조건에서 '60일선 위' 문턱은 삭제됨.
                          #   이 상수는 data_layer.py가 60일선을 계산/제공하는 파이프라인 호환용으로만
                          #   남겨둔다(매수판정에는 더 이상 쓰이지 않음). StockSnapshot.ma_buy도 마찬가지.

# --- 매도조건 ---
SELL_MA_DAYS = 90         # 매도조건 ①: 종가가 이 이동평균선 아래로 내려가면 매도
HARD_STOP_LOSS_PCT = 0.12 # 매도조건 ②: 평단가 대비 -12% 하드손절 (2026-06-17 신규 추가)

# --- 매수조건: 90일선 이격도 (2026-07-14 개정) ---
# 기존 "현재가 > 90일선" 단순 문턱을 '이격도(현재가가 90일선보다 얼마나 위인지 %)'로 교체.
#   이격도(%) = (현재가 - 90일선) / 90일선 × 100
# 90일선 바로 위에 겨우 걸친 종목(사자마자 90일선 이탈로 되팔릴 whipsaw 위험)을 걸러내기 위해
# 최소 여유폭 2.5%를 요구한다. 백테스트 근거: results/disparity_filter_compare.csv,
# results/align_filter_compare.csv (재손절 25.6%→18~19%로 감소, CAGR 소폭 개선/동등).
BUY_DISPARITY_MIN_PCT = 2.5

# --- 부분익절 (2026-07-09 확정: "+8% 도달시 50% 부분익절") ---
# 전량매도(위 90일선/하드손절)와는 별개 규칙. 이기는 종목의 절반만 실현하고
# 나머지 절반은 계속 보유해 큰 상승을 붙잡는다. 종목당 1회만 실행.
# 백테스트 근거: 승률 12.7%→33.5%, CAGR +75.5%→+78.0%, MDD -30.9%→-27.6% (모두 개선)
PARTIAL_EXIT_TRIGGER_PCT = 0.08   # 진입가 대비 +8% 도달 시 트리거
PARTIAL_EXIT_RATIO = 0.5          # 그때 보유수량의 50% 매도

# --- 매수조건 나머지 ---
ROE_MIN_PCT = 15.0                     # 매수조건 ②: ROE(%) 최소값
MOMENTUM_DAYS = 20                      # 매수조건 ④: 20일 모멘텀
TRADING_VALUE_MIN_WON = 3_000_000_000   # 매수조건 ⑤: 20일평균거래대금 30억원 이상

# --- 시장필터 (KOSPI 기준, 코스닥 아님! — STRATEGY_FINAL.md §1에서 확정) ---
MARKET_FILTER_MA_DAYS = 200   # KOSPI 종가가 이 이동평균 위일 때만 "강세장"

# --- 불타기(피라미딩) ---
PYRAMID_TRIGGER_PCT = 0.03    # 진입가 대비 +3%씩 "복리로" 상승할 때마다 트리거
                              # 예: 진입가 10,000원 → 10,300 / 10,609 / 10,927 ... (10000*1.03^n)
PYRAMID_ADD_AMOUNT_WON = 5_000_000   # 트리거마다 추가하는 고정 금액 (슬롯 금액과 동일)
PYRAMID_DAILY_LIMIT_PER_STOCK = 1    # ★2026-06-17 개정: 종목당 "하루 1회"로 제한
                                      #   (예전엔 하루 5회였다가, 실전성 문제로 1회로 축소 → 오히려 성과 개선됨)

# --- 운영 규모 ---
NUM_SLOTS = 20                       # 동시 보유 종목 수
SLOT_AMOUNT_WON = 5_000_000          # 슬롯(=신규매수 1건)당 금액 = 1억 / 20


# =========================================================================
# 2. 입력 데이터 형태 정의
# =========================================================================
# 아래 두 클래스는 "이 함수들이 어떤 모양의 데이터를 기대하는지"를
# 명확하게 보여주기 위한 것입니다. data_layer.py가 이 모양대로
# 데이터를 만들어서 넘겨주면 됩니다.

@dataclass
class StockSnapshot:
    """어떤 종목의 '어느 날' 상태 한 장. 매수/매도 판정에 필요한 값들만 담음."""
    ticker: str
    date: date
    price: float                 # 그날 종가
    ma_buy: float                # 60일 이동평균 (매수조건용)
    ma_sell: float                # 90일 이동평균 (매도조건용)
    roe_pct: float                # ROE (%), 예: 18.5
    eps: float                   # 주당순이익 (원)
    momentum_20d: float           # 20일 수익률, 예: 0.05 = +5%
    trading_value_20d_avg: float  # 20일평균거래대금 (원)


@dataclass
class Position:
    """현재 보유중인 포지션(종목) 하나의 상태."""
    ticker: str
    entry_price: float            # 최초 진입가 (불타기 트리거 계산 기준 — 절대 안 바뀜)
    avg_price: float               # 현재 평단가 (불타기 할 때마다 바뀜, 하드손절 기준)
    shares: float                  # 보유 수량
    pyramid_count: int = 0          # 지금까지 불타기 몇 단계까지 진행됐는지
    last_pyramid_date: Optional[date] = None  # 마지막으로 불타기한 날짜 (하루 1회 제한 체크용)
    entry_date: Optional[date] = None
    partial_exit_done: bool = False  # 부분익절(+8%에서 50%)을 이미 실행했는지 (종목당 1회 제한용)


# =========================================================================
# 3. 시장필터 (약세장/강세장 판정)
# =========================================================================

def is_bull_market(kospi_close: float, kospi_ma200: float) -> bool:
    """
    KOSPI 종가가 200일선 위에 있으면 '강세장' → True.

    이 값이 False(약세장)이면:
        - 신규매수 전면 중단
        - 불타기 전면 중단
        - 매도(90일선/하드손절)는 시장 상태와 무관하게 항상 정상 작동
    이 세 가지는 이 함수 자체가 결정하는 게 아니라, 이 함수의 결과를
    check_buy_conditions()와 check_pyramid()에 넘겨줘서 반영합니다.
    """
    return kospi_close > kospi_ma200


# =========================================================================
# 4. 매수조건 판정
# =========================================================================

def check_buy_conditions(stock: StockSnapshot, is_bull: bool) -> tuple[bool, list[str]]:
    """
    매수조건 5개를 모두 확인합니다. STRATEGY_FINAL.md §1 기준 + 90일선 이격도 필터:
      1) 저평가: 현재가 < ROE(%) × EPS  (적정주가 대비 상승여력)
      2) ROE ≥ 15%
      3) 90일선 이격도 ≥ 2.5%   ← 2026-07-14 개정. 아래 참고.
      4) 20일 모멘텀 > 0
      5) 20일평균거래대금 ≥ 30억원
      6) 시장필터: KOSPI가 강세장일 때만

    ★ 조건 3 개정 배경 (2026-07-14) — '60일선 삭제 + 90일선을 이격도로 교체':
      (구) 매수조건은 "현재가 > 60일선" AND "현재가 > 90일선" 두 문턱을 각각 걸었다.
      그런데 (a) 60일선 문턱은 90일선 문턱이 있으면 거의 잉여였고(제거해도 종목군
      Jaccard 98%로 성과 동일), (b) 90일선 '바로 위'에 겨우 걸친 종목은 사자마자
      90일선 이탈 매도로 되팔리는 whipsaw를 냈다.
      → 60일선 문턱을 삭제하고, 90일선은 단순 '위/아래'가 아니라 '이격도 ≥ 2.5%'
        (현재가가 90일선보다 최소 2.5% 위)로 바꿔 최소 여유폭을 요구한다.
      (이격도의 90일선 값 stock.ma_sell 은 매도조건과 '동일 값'을 재사용 — 이중계산 없음)
      백테스트 근거: results/disparity_filter_compare.csv, results/align_filter_compare.csv
      (매수 후 5일내 재손절 25.6%→18~19%로 대폭 감소, CAGR 소폭 개선/동등).
      단 단일 경로(약 6.5년) 백테스트라 과최적화 위험은 여전히 존재한다.

    ★ 참고: StockSnapshot.ma_buy(60일선)는 더 이상 매수판정에 쓰이지 않는다
      (데이터 파이프라인 호환을 위해 필드/계산 자체는 남겨둠).

    반환값: (매수가능여부, 실패한 조건 이름 리스트)
        실패 이유를 함께 반환하는 건 디버깅/신호화면 표시용입니다.
        예: "왜 이 종목은 매수 안 됐지?"를 앱 화면에서 바로 보여줄 수 있게.
    """
    fail_reasons = []

    # 조건 6: 시장필터부터 확인 (약세장이면 나머지 볼 필요도 없이 바로 탈락)
    if not is_bull:
        fail_reasons.append("시장필터(KOSPI 200일선 하회, 약세장)")
        # 그래도 나머지 조건 결과도 참고용으로 계속 계산은 해줍니다.

    # 조건 1: 저평가 (적정주가 = ROE(%) × EPS)
    fair_price = stock.roe_pct * stock.eps
    if not (stock.price < fair_price):
        fail_reasons.append(f"저평가 아님 (현재가 {stock.price:,.0f} ≥ 적정가 {fair_price:,.0f})")

    # 조건 2: ROE ≥ 15%
    if not (stock.roe_pct >= ROE_MIN_PCT):
        fail_reasons.append(f"ROE 미달 ({stock.roe_pct:.1f}% < {ROE_MIN_PCT}%)")

    # 조건 3 (2026-07-14 개정): 90일선 이격도 ≥ 2.5%
    #   (구 "현재가 > 60일선" 삭제 + "현재가 > 90일선"을 이격도 여유폭으로 교체)
    #   이격도(%) = (현재가 - 90일선) / 90일선 × 100. 90일선(stock.ma_sell)이 없거나
    #   0 이하면 정의 불가 → 자동 탈락.
    if stock.ma_sell and stock.ma_sell > 0:
        disparity = (stock.price - stock.ma_sell) / stock.ma_sell * 100.0
    else:
        disparity = float("-inf")
    if not (disparity >= BUY_DISPARITY_MIN_PCT):
        fail_reasons.append(
            f"90일선 이격도 미달 ({disparity:.2f}% < {BUY_DISPARITY_MIN_PCT:.1f}%)"
        )

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
    """
    매수조건을 통과한 후보들을 20일 모멘텀이 높은 순으로 정렬합니다.
    빈 슬롯 개수만큼 위에서부터 채우면 됩니다 (이 함수는 정렬만 하고,
    슬롯을 몇 개 채울지는 app.py / backtest_v3.py 쪽에서 결정합니다).
    """
    return sorted(candidates, key=lambda s: s.momentum_20d, reverse=True)


# =========================================================================
# 5. 매도조건 판정
# =========================================================================

def check_sell_condition(position: Position, stock: StockSnapshot) -> tuple[bool, Optional[str]]:
    """
    매도조건 2개 중 "먼저 닿는 쪽"을 확인합니다. 시장 상태와 무관하게 항상 작동합니다.
      ① 종가 < 90일선  → '90일선 이탈'
      ② 현재가 ≤ 평단가 × (1 - 0.12)  → '하드손절(-12%)'

    두 조건을 동시에 만족해도 사유는 하나만 반환합니다(먼저 검사한 쪽 우선).
    실전에서는 사유 표시 목적이라 순서가 성과에 영향을 주진 않습니다.

    반환값: (매도여부, 매도사유 또는 None)
    """
    # ① 90일선 이탈
    if stock.price < stock.ma_sell:
        return True, "90일선 이탈"

    # ② 평단가 -12% 하드손절
    stop_loss_price = position.avg_price * (1 - HARD_STOP_LOSS_PCT)
    if stock.price <= stop_loss_price:
        return True, f"하드손절(평단가 -{HARD_STOP_LOSS_PCT*100:.0f}%)"

    return False, None


def check_partial_exit(position: Position, current_price: float) -> bool:
    """
    부분익절(+8%에서 50%) 실행 여부 판정. 전량매도와는 '별개' 규칙입니다.

    조건:
      - 아직 부분익절을 안 했고 (position.partial_exit_done == False)
      - 현재가가 최초 진입가 대비 +PARTIAL_EXIT_TRIGGER_PCT(=+8%) 이상

    ★ 호출 순서 규칙 (매우 중요):
      매일 루프에서 먼저 check_sell_condition()으로 '전량매도' 여부를 확인하고,
      전량매도가 아닌 포지션에 대해서만 이 함수를 확인하세요.
      (전량매도로 이미 청산될 포지션에 부분익절을 또 적용하면 안 됩니다.)

    기준가격은 '평단가'가 아니라 '최초 진입가(entry_price)'입니다.
    (불타기로 평단가가 올라가도, 부분익절 트리거는 최초 진입가 기준 +8% 고정)
    """
    if position.partial_exit_done:
        return False
    return current_price >= position.entry_price * (1 + PARTIAL_EXIT_TRIGGER_PCT)


def apply_partial_exit(position: Position, current_price: float) -> float:
    """
    부분익절을 '실제로 실행'했을 때 포지션 상태를 갱신합니다.
    check_partial_exit()가 True를 반환한 경우에만 호출하세요.

    - 보유수량의 PARTIAL_EXIT_RATIO(=50%)만큼을 매도수량으로 계산해 반환합니다.
    - position.shares를 그만큼 차감하고, position.partial_exit_done = True로 바꿉니다.
    - 평단가(avg_price)와 최초 진입가(entry_price)는 바뀌지 않습니다
      (수량만 줄어들 뿐, 남은 50%의 원가 구조는 동일).

    반환값: 매도한 수량 (이 수량 × current_price 가 매도대금(수수료 차감 전))
    """
    sell_shares = position.shares * PARTIAL_EXIT_RATIO
    position.shares -= sell_shares
    position.partial_exit_done = True
    return sell_shares


# =========================================================================
# 6. 불타기(피라미딩) 판정
# =========================================================================

def _pyramid_trigger_price(entry_price: float, step: int) -> float:
    """
    n단계 불타기 트리거 가격 = 진입가 × (1.03)^n   (복리 계산, 단리 아님!)
    예: 진입가 10,000원
        1단계: 10,000 × 1.03^1 = 10,300원
        2단계: 10,000 × 1.03^2 = 10,609원
        3단계: 10,000 × 1.03^3 = 10,927.27원
    """
    return entry_price * ((1 + PYRAMID_TRIGGER_PCT) ** step)


def check_pyramid(
    position: Position,
    current_price: float,
    current_date: date,
    is_bull: bool,
    available_cash: float,
) -> bool:
    """
    이 종목에 오늘 불타기(추가매수)를 할지 판정합니다.

    조건 (STRATEGY_FINAL.md §1 "불타기" 표 기준):
      - 약세장이면 무조건 중단 (False)
      - 오늘 이미 이 종목에 불타기를 했으면 중단 (하루 1회 제한)
      - 현재가가 "다음 단계" 트리거 가격 이상이어야 함
      - 현금이 500만원 이상 있어야 함

    ★ 주의: 이 함수는 "해도 되는지 여부"만 판정합니다.
      실제로 몇 단계까지 건너뛰어 올라갔는지는 보지 않고,
      "다음 한 단계"만 확인합니다. 이렇게 만든 이유는
      STRATEGY_FINAL.md의 "확인형 분할 가산" 설명 때문입니다:
      하루에 급등해서 여러 단계를 건너뛰어도 하루 1회만 실행하고,
      나머지 단계는 다음날 가격이 그 트리거 위에 "그대로" 남아있어야
      한 단계씩 이어서 실행됩니다. (한번에 몰아서 여러 단계 처리 안 함)
    """
    # 약세장이면 불타기 중단
    if not is_bull:
        return False

    # 오늘 이미 이 종목에 불타기했으면 중단 (종목당 하루 1회)
    if position.last_pyramid_date == current_date:
        return False

    # 현금 부족하면 중단
    if available_cash < PYRAMID_ADD_AMOUNT_WON:
        return False

    # 다음 단계(현재 pyramid_count + 1) 트리거 가격에 도달했는지 확인
    next_step = position.pyramid_count + 1
    trigger_price = _pyramid_trigger_price(position.entry_price, next_step)

    return current_price >= trigger_price


def apply_pyramid(position: Position, current_price: float, current_date: date) -> Position:
    """
    불타기를 '실제로 실행'했을 때 포지션 상태를 갱신합니다.
    check_pyramid()가 True를 반환한 경우에만 이 함수를 호출하세요.

    - 체결가는 트리거 가격이 아니라 '그날 종가'(current_price) 기준입니다.
      → STRATEGY_FINAL.md: "체결가는 모두 그날 종가(불리한 현재가) 기준"
    - 평단가는 (기존 보유금액 + 추가매수금액) / (기존수량 + 추가수량) 으로 재계산됩니다.
      → 평단가가 바뀌면 하드손절선(-12%)도 자동으로 같이 움직입니다.
    """
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
# 7. 간단한 자체 점검 (실제 데이터 없이, 이 파일 로직만 맞는지 확인)
# =========================================================================
# python strategy_core.py 라고 그냥 실행하면 아래 테스트가 돌아갑니다.
# 데이터 연동 전에 "로직 자체가 STRATEGY_FINAL.md대로 동작하는지"를
# 먼저 확인하는 용도입니다. (data_layer.py가 없어도 됩니다)

if __name__ == "__main__":
    print("=== strategy_core.py 자체 점검 시작 ===\n")

    # --- 1) 시장필터 테스트 ---
    assert is_bull_market(2600, 2500) is True    # 종가가 200일선 위 → 강세장
    assert is_bull_market(2400, 2500) is False   # 종가가 200일선 아래 → 약세장
    print("[PASS] 시장필터 판정")

    # --- 2) 매수조건 테스트: 모든 조건을 만족하는 가짜 종목 ---
    good_stock = StockSnapshot(
        ticker="TEST001", date=date(2026, 7, 9),
        price=9000,                 # 적정가(=18*600=10800)보다 낮음 → 저평가 O
        ma_buy=8500,                # 60일선은 이제 매수판정에 쓰이지 않음(호환용 필드)
        ma_sell=8000,               # 이격도 = (9000-8000)/8000 = 12.5% ≥ 2.5% → O
        roe_pct=18.0,                # ROE 18% ≥ 15% → O
        eps=600,
        momentum_20d=0.05,           # +5% > 0 → O
        trading_value_20d_avg=5_000_000_000,  # 50억 ≥ 30억 → O
    )
    can_buy, reasons = check_buy_conditions(good_stock, is_bull=True)
    assert can_buy is True, f"모든 조건 만족했는데 실패: {reasons}"
    print("[PASS] 매수조건 - 정상 통과 케이스")

    # --- 2-b) 매수조건 테스트: 90일선 이격도 미달(2% < 2.5%)이면 탈락 ---
    #   현재가 8160, 90일선 8000 → 이격도 2.0% → 조건 3에서 탈락해야 함
    disp_edge = StockSnapshot(
        ticker="TEST001B", date=date(2026, 7, 9),
        price=8160, ma_buy=1, ma_sell=8000,   # 이격도 2.0% < 2.5%
        roe_pct=18.0, eps=600, momentum_20d=0.05,
        trading_value_20d_avg=5_000_000_000,
    )
    can_buy_edge, reasons_edge = check_buy_conditions(disp_edge, is_bull=True)
    assert can_buy_edge is False
    assert any("이격도" in r for r in reasons_edge)
    print("[PASS] 매수조건 - 90일선 이격도 2.5% 미달 차단")

    # --- 3) 매수조건 테스트: 약세장이면 무조건 탈락 ---
    can_buy_bear, reasons_bear = check_buy_conditions(good_stock, is_bull=False)
    assert can_buy_bear is False
    assert "시장필터" in reasons_bear[0]
    print("[PASS] 매수조건 - 약세장 차단")

    # --- 4) 매도조건 테스트: 90일선 이탈 ---
    pos = Position(ticker="TEST001", entry_price=10000, avg_price=10000, shares=500,
                    entry_date=date(2026, 1, 1))
    stock_below_ma90 = StockSnapshot(
        ticker="TEST001", date=date(2026, 7, 9),
        price=9500, ma_buy=9000, ma_sell=9800,   # 현재가(9500) < 90일선(9800) → 매도!
        roe_pct=18.0, eps=600, momentum_20d=0.02, trading_value_20d_avg=5_000_000_000,
    )
    should_sell, reason = check_sell_condition(pos, stock_below_ma90)
    assert should_sell is True and reason == "90일선 이탈"
    print("[PASS] 매도조건 - 90일선 이탈")

    # --- 5) 매도조건 테스트: 하드손절 -12% ---
    stock_crashed = StockSnapshot(
        ticker="TEST001", date=date(2026, 7, 9),
        price=8700,   # 평단가 10000 * 0.88 = 8800 → 8700은 그 밑 → 하드손절
        ma_buy=9000, ma_sell=8500,  # 90일선(8500)은 안 깨졌음 (8700 > 8500)
        roe_pct=18.0, eps=600, momentum_20d=0.02, trading_value_20d_avg=5_000_000_000,
    )
    should_sell2, reason2 = check_sell_condition(pos, stock_crashed)
    assert should_sell2 is True and "하드손절" in reason2
    print("[PASS] 매도조건 - 하드손절(-12%)")

    # --- 5-2) 부분익절 테스트: +8% 도달 → 50% 매도 → 중복실행 방지 ---
    pos_pe = Position(ticker="TEST003", entry_price=10000, avg_price=10000, shares=200,
                      entry_date=date(2026, 1, 1))
    # +8% 미만이면 아직 아님
    assert check_partial_exit(pos_pe, 10700) is False   # +7% → 아직
    # +8%(10800) 도달 → True
    assert check_partial_exit(pos_pe, 10800) is True
    sold = apply_partial_exit(pos_pe, 10800)
    assert abs(sold - 100) < 1e-9                        # 200의 50% = 100주 매도
    assert abs(pos_pe.shares - 100) < 1e-9               # 잔량 100주
    assert pos_pe.partial_exit_done is True              # 완료 플래그
    assert abs(pos_pe.avg_price - 10000) < 1e-9          # 평단가 불변
    # 이미 했으면 더 올라도(11000, +10%) 중복 실행 안 됨
    assert check_partial_exit(pos_pe, 11000) is False
    print("[PASS] 부분익절 - +8% 도달시 50% 매도 & 중복실행 방지")

    # --- 6) 불타기 트리거 가격 테스트 (복리 계산 확인) ---
    p1 = _pyramid_trigger_price(10000, 1)
    p2 = _pyramid_trigger_price(10000, 2)
    p3 = _pyramid_trigger_price(10000, 3)
    assert abs(p1 - 10300) < 0.01
    assert abs(p2 - 10609) < 0.01
    assert abs(p3 - 10927.27) < 0.01
    print(f"[PASS] 불타기 트리거 가격 (복리): 1단계={p1:.0f} 2단계={p2:.0f} 3단계={p3:.0f}")

    # --- 7) 불타기 판정: 하루 1회 제한 테스트 ---
    pos2 = Position(ticker="TEST002", entry_price=10000, avg_price=10000, shares=500,
                     entry_date=date(2026, 1, 1))
    today = date(2026, 7, 9)

    # 트리거 가격(10300) 이상, 현금 충분, 오늘 아직 안 했음 → True
    ok1 = check_pyramid(pos2, current_price=10500, current_date=today,
                         is_bull=True, available_cash=10_000_000)
    assert ok1 is True
    print("[PASS] 불타기 - 트리거 도달시 실행 가능")

    # 실행했다고 가정하고 상태 갱신
    pos2 = apply_pyramid(pos2, current_price=10500, current_date=today)
    assert pos2.pyramid_count == 1
    assert abs(pos2.avg_price - ((10000*500 + 5_000_000)/(500 + 5_000_000/10500))) < 0.01

    # 같은 날 또 트리거 가격을 넘어도(예: 급등) 하루 1회 제한 때문에 False여야 함
    ok2 = check_pyramid(pos2, current_price=11000, current_date=today,
                         is_bull=True, available_cash=10_000_000)
    assert ok2 is False
    print("[PASS] 불타기 - 종목당 하루 1회 제한 작동")

    # 다음날, 다음 단계(2단계=10609) 가격에 아직 못 미치면 False
    tomorrow = date(2026, 7, 10)
    ok3 = check_pyramid(pos2, current_price=10500, current_date=tomorrow,
                         is_bull=True, available_cash=10_000_000)
    assert ok3 is False  # 10500 < 2단계 트리거 10609.27
    print("[PASS] 불타기 - 다음 단계 미도달시 대기")

    # 다음날, 2단계 트리거 가격에 도달하면 True
    ok4 = check_pyramid(pos2, current_price=10700, current_date=tomorrow,
                         is_bull=True, available_cash=10_000_000)
    assert ok4 is True
    print("[PASS] 불타기 - 다음 단계 도달시 재실행 가능")

    # --- 8) 불타기 판정: 약세장이면 무조건 중단 ---
    ok5 = check_pyramid(pos2, current_price=99999, current_date=tomorrow,
                         is_bull=False, available_cash=10_000_000)
    assert ok5 is False
    print("[PASS] 불타기 - 약세장 차단")

    # --- 9) 불타기 판정: 현금 부족시 중단 ---
    ok6 = check_pyramid(pos2, current_price=10700, current_date=tomorrow,
                         is_bull=True, available_cash=1_000_000)  # 100만원뿐
    assert ok6 is False
    print("[PASS] 불타기 - 현금 부족시 중단")

    print("\n=== 전체 통과: strategy_core.py 로직이 STRATEGY_FINAL.md 스펙대로 동작합니다 ===")
