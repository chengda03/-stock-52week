# -*- coding: utf-8 -*-
"""
portfolio_state.py
==================
실운영 앱에서 "지금 어떤 종목을 얼마에 몇 개 들고 있는지"를 파일(JSON)에
저장하고 불러오는 기능만 담당합니다. 전략 판정(strategy_core)과는 완전히 무관.

포트폴리오 JSON 구조
{
  "cash": 23000000,
  "positions": {
     "005930": {"entry_price":72000,"avg_price":73500,"shares":138,
                "entry_date":"2026-05-10","pyramid_count":2,
                "last_pyramid_date":"2026-07-08"}
  },
  "trade_log": [
     {"date":"2026-07-09","ticker":"005930","action":"불타기",
      "price":74100,"amount":5000000,"reason":"+6.09% 트리거"}
  ]
}
"""
from __future__ import annotations

import json
import os
from datetime import date, datetime
from typing import Optional

from strategy_core import Position, SLOT_AMOUNT_WON, NUM_SLOTS

# 저장 위치 (없으면 자동 생성)
DATA_DIR = "data"
PORTFOLIO_PATH = os.path.join(DATA_DIR, "portfolio.json")

# 초기 운용자금 = 1억 (슬롯 20 × 500만)
INITIAL_CASH = SLOT_AMOUNT_WON * NUM_SLOTS


def _empty_portfolio() -> dict:
    """빈 포트폴리오(현금 1억, 보유 없음)."""
    return {"cash": float(INITIAL_CASH), "positions": {}, "trade_log": []}


def load_portfolio() -> dict:
    """
    data/portfolio.json 을 읽어서 반환.
    파일이 없으면 빈 포트폴리오(현금 1억)를 만들어 반환합니다.
    """
    if not os.path.exists(PORTFOLIO_PATH):
        return _empty_portfolio()
    try:
        with open(PORTFOLIO_PATH, "r", encoding="utf-8") as f:
            state = json.load(f)
    except Exception:
        # 파일이 깨졌으면 안전하게 빈 포트폴리오로 시작
        return _empty_portfolio()

    # 필수 키 보정
    state.setdefault("cash", float(INITIAL_CASH))
    state.setdefault("positions", {})
    state.setdefault("trade_log", [])
    return state


def save_portfolio(state: dict) -> None:
    """data/portfolio.json 에 저장 (폴더 없으면 생성)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(PORTFOLIO_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _today_str() -> str:
    return date.today().isoformat()


def _to_iso(d) -> Optional[str]:
    if d is None:
        return None
    if isinstance(d, (date, datetime)):
        return d.isoformat()[:10]
    return str(d)


def record_trade(state: dict, ticker: str, action: str, price: float,
                 shares: float, date_=None, reason: str = "",
                 amount: Optional[float] = None) -> dict:
    """
    거래내역 로그를 추가하고, 포지션/현금도 함께 갱신합니다.

    action: "신규매수" | "불타기" | "매도"
      - 신규매수: 새 포지션 생성 (entry_price=avg_price=price)
      - 불타기  : 기존 포지션에 추가, 평단가 재계산, pyramid_count +1
      - 매도    : 포지션 전량 청산 (shares 인자는 보유수량 그대로여도 OK)

    price   : 체결가
    shares  : 체결 수량 (매도는 보유 전량)
    amount  : 투입/회수 금액(선택). 없으면 price*shares로 계산.
    """
    d = _to_iso(date_) or _today_str()
    positions = state["positions"]

    if amount is None:
        amount = float(price) * float(shares)

    if action == "신규매수":
        positions[ticker] = {
            "entry_price": float(price),
            "avg_price": float(price),
            "shares": float(shares),
            "entry_date": d,
            "pyramid_count": 0,
            "last_pyramid_date": None,
            "partial_exit_done": False,
        }
        state["cash"] -= amount

    elif action == "불타기":
        pos = positions.get(ticker)
        if pos is None:
            raise ValueError(f"불타기 대상 종목({ticker})이 보유목록에 없습니다.")
        old_cost = pos["avg_price"] * pos["shares"]
        new_shares = pos["shares"] + float(shares)
        pos["avg_price"] = (old_cost + amount) / new_shares
        pos["shares"] = new_shares
        pos["pyramid_count"] = pos.get("pyramid_count", 0) + 1
        pos["last_pyramid_date"] = d
        state["cash"] -= amount

    elif action == "부분익절":
        # +8% 도달시 보유수량의 일부(50%)만 매도. 포지션은 잔량 유지.
        pos = positions.get(ticker)
        if pos is None:
            raise ValueError(f"부분익절 대상 종목({ticker})이 보유목록에 없습니다.")
        pos["shares"] = max(0.0, pos["shares"] - float(shares))
        pos["partial_exit_done"] = True
        state["cash"] += amount
        if pos["shares"] <= 0:            # 혹시 전량이 됐으면 정리
            positions.pop(ticker, None)

    elif action == "매도":
        pos = positions.pop(ticker, None)
        if pos is not None:
            # 매도금액은 인자 amount(=회수금액) 우선
            state["cash"] += amount
    else:
        raise ValueError(f"알 수 없는 action: {action}")

    state["trade_log"].append({
        "date": d, "ticker": ticker, "action": action,
        "price": float(price), "shares": float(shares),
        "amount": float(amount), "reason": reason,
    })
    return state


# ------------------------------------------------------------------
# strategy_core.Position <-> JSON dict 변환 도우미
# (app.py / 백테스트에서 strategy_core 함수에 넘길 때 편하게 쓰라고 제공)
# ------------------------------------------------------------------
def dict_to_position(ticker: str, pos: dict) -> Position:
    """JSON 포지션 dict → strategy_core.Position 객체."""
    def _pdate(s):
        return date.fromisoformat(s) if s else None
    return Position(
        ticker=ticker,
        entry_price=float(pos["entry_price"]),
        avg_price=float(pos["avg_price"]),
        shares=float(pos["shares"]),
        pyramid_count=int(pos.get("pyramid_count", 0)),
        last_pyramid_date=_pdate(pos.get("last_pyramid_date")),
        entry_date=_pdate(pos.get("entry_date")),
        partial_exit_done=bool(pos.get("partial_exit_done", False)),
    )


if __name__ == "__main__":
    # 간단 점검: 빈 포트폴리오 → 매수 → 불타기 → 매도 흐름
    st = _empty_portfolio()
    record_trade(st, "005930", "신규매수", price=70000, shares=71, reason="6조건 통과")
    assert st["positions"]["005930"]["shares"] == 71
    record_trade(st, "005930", "불타기", price=72100, shares=69,
                 amount=5_000_000, reason="+3% 트리거")
    assert st["positions"]["005930"]["pyramid_count"] == 1
    p = dict_to_position("005930", st["positions"]["005930"])
    assert p.pyramid_count == 1 and p.shares == 140
    record_trade(st, "005930", "매도", price=80000, shares=140,
                 amount=80000 * 140, reason="90일선 이탈")
    assert "005930" not in st["positions"]
    assert len(st["trade_log"]) == 3
    print("[PASS] portfolio_state 매수/불타기/매도 흐름 정상")
    print(f"  최종 현금: {st['cash']:,.0f}원, 거래로그 {len(st['trade_log'])}건")
