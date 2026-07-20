# -*- coding: utf-8 -*-
"""
strategy_core_pbrvalue.py
=========================
[완전히 독립적인 신규 전략] "저PBR 가치투자전략".
SF투자법(strategy_core.py)과는 무관하며, strategy_core.py를 import/참조하지 않는다.
매수판정·매도·부분익절·불타기 로직을 이 파일에서 처음부터 독립적으로 작성한다.

전략 개요
---------
유니버스: 코스피+코스닥 합산 시총 상위 500 (관리종목·스팩·우선주·리츠 제외)
          — data_layer의 유니버스 정의를 '데이터로서' 재사용(로직 재사용 아님).

매수조건(단 하나):
    현재 PBR ≤ 최근 3년 평균 PBR × 0.7
      · PBR = 현재가 / BPS,  BPS = 자기자본/주식수 = EPS × 100 / ROE(%)  (수학적 항등식)
      · '3년 평균 PBR'은 PIT(룩어헤드 없음): 매 거래일의 일별 PBR(그날까지 이미 공시된 재무로 계산)을
        과거 방향으로 3년(≈756거래일) 롤링 평균. 미래 데이터 일절 미사용.
    ROE·모멘텀·거래대금·이동평균·시장필터 등 다른 매수조건은 전혀 쓰지 않는다.

매수우선순위: 3년평균 대비 할인율이 큰(= PBR/3년평균 이 낮은) 순.

매도/운영(틀은 통상적 규칙을 독립 구현):
    · 매도: 현재가<90일선  또는  현재가 ≤ 평단가×(1-0.12)  (먼저 닿는 쪽)
    · 부분익절: 진입가 대비 +8% 도달시 보유수량 50% 매도(종목당 1회)
    · 불타기: 진입가×1.03^n 도달마다 500만 추가, 무제한 단계, 종목당 하루 1회, 현금 500만 이상일 때만
              (이 전략은 시장필터가 없으므로 불타기에도 시장조건을 걸지 않는다)
    · 운영: 1억, 20종목, 슬롯당 500만
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

# --- 운영 파라미터 ---
NUM_SLOTS = 20
SLOT_AMOUNT_WON = 5_000_000
PYRAMID_ADD_AMOUNT_WON = 5_000_000
PYRAMID_TRIGGER_PCT = 0.03

# --- 매수조건(저PBR) 파라미터 ---
PBR_AVG_TRADING_DAYS = 756     # 3년 ≈ 756 거래일 (롤링 평균 창)
PBR_AVG_MIN_DAYS = 120         # 최소 관측일(데이터 시작 한계로 초기엔 3년 미만일 수 있음)
PBR_DISCOUNT = 0.70            # 현재PBR ≤ 3년평균PBR × 0.7

# --- 매도/부분익절 파라미터 ---
SELL_MA_DAYS = 90
HARD_STOP_LOSS_PCT = 0.12
PARTIAL_EXIT_TRIGGER_PCT = 0.08
PARTIAL_EXIT_RATIO = 0.50


@dataclass
class PBRPosition:
    ticker: str
    entry_price: float
    avg_price: float
    shares: float
    pyramid_count: int = 0
    last_pyramid_date: Optional[date] = None
    entry_date: Optional[date] = None
    partial_exit_done: bool = False


def bps_from_roe_eps(roe_pct: float, eps: float) -> float:
    """BPS = 자기자본/주식수 = EPS / (ROE/100) = EPS×100/ROE(%). ROE>0·EPS>0에서만 유효."""
    if roe_pct is None or eps is None:
        return float("nan")
    if not (roe_pct > 0) or not (eps > 0):
        return float("nan")
    return eps * 100.0 / roe_pct


def check_buy_pbr(current_pbr: float, avg3y_pbr: float) -> bool:
    """매수조건: 현재 PBR ≤ 3년평균 PBR × 0.7. (둘 다 유효한 양수여야 함)"""
    if not (current_pbr == current_pbr) or not (avg3y_pbr == avg3y_pbr):  # nan 체크
        return False
    if current_pbr <= 0 or avg3y_pbr <= 0:
        return False
    return current_pbr <= avg3y_pbr * PBR_DISCOUNT


def discount_ratio(current_pbr: float, avg3y_pbr: float) -> float:
    """할인율 정렬 키: 현재PBR/3년평균PBR (낮을수록 더 크게 할인 → 우선)."""
    if avg3y_pbr and avg3y_pbr > 0:
        return current_pbr / avg3y_pbr
    return float("inf")


def check_sell(current_price: float, avg_price: float, ma90: float) -> tuple[bool, Optional[str]]:
    """① 종가<90일선  또는  ② 현재가≤평단가×(1-0.12). 먼저 닿는 쪽."""
    if ma90 == ma90 and current_price < ma90:   # ma90 유효 + 이탈
        return True, "90일선 이탈"
    if current_price <= avg_price * (1 - HARD_STOP_LOSS_PCT):
        return True, f"하드손절(-{HARD_STOP_LOSS_PCT*100:.0f}%)"
    return False, None


def check_partial_exit(pos: PBRPosition, current_price: float) -> bool:
    if pos.partial_exit_done:
        return False
    return current_price >= pos.entry_price * (1 + PARTIAL_EXIT_TRIGGER_PCT)


def apply_partial_exit(pos: PBRPosition, current_price: float) -> float:
    sell_shares = pos.shares * PARTIAL_EXIT_RATIO
    pos.shares -= sell_shares
    pos.partial_exit_done = True
    return sell_shares


def check_pyramid(pos: PBRPosition, current_price: float, current_date: date,
                  available_cash: float) -> bool:
    """진입가×1.03^(count+1) 도달 + 하루 1회 + 현금 500만 이상. (시장필터 없음)"""
    if pos.last_pyramid_date == current_date:
        return False
    if available_cash < PYRAMID_ADD_AMOUNT_WON:
        return False
    next_step = pos.pyramid_count + 1
    trigger = pos.entry_price * ((1 + PYRAMID_TRIGGER_PCT) ** next_step)
    return current_price >= trigger


def apply_pyramid(pos: PBRPosition, current_price: float, current_date: date) -> None:
    add_shares = PYRAMID_ADD_AMOUNT_WON / current_price
    old_cost = pos.avg_price * pos.shares
    new_shares = pos.shares + add_shares
    pos.avg_price = (old_cost + PYRAMID_ADD_AMOUNT_WON) / new_shares
    pos.shares = new_shares
    pos.pyramid_count += 1
    pos.last_pyramid_date = current_date


if __name__ == "__main__":
    # BPS 항등식: ROE 20%, EPS 1000 → BPS = 1000*100/20 = 5000
    assert abs(bps_from_roe_eps(20.0, 1000.0) - 5000.0) < 1e-6
    # PBR 10000/5000=2.0, 3년평균 4.0 → 2.0 ≤ 4.0*0.7=2.8 → 매수 O
    assert check_buy_pbr(2.0, 4.0) is True
    # 2.0 ≤ 2.5*0.7=1.75? 아니오 → 매수 X
    assert check_buy_pbr(2.0, 2.5) is False
    assert check_sell(89, 100, 90)[0] is True    # 90일선 이탈
    assert check_sell(87, 100, 80)[0] is True     # -13% 하드손절
    assert check_sell(95, 100, 90)[0] is False
    print("[PASS] strategy_core_pbrvalue 독립 로직 점검 완료")
