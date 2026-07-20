# -*- coding: utf-8 -*-
"""
strategy_core_sl10.py
=====================
[하드손절 임계값 변형 · 실험용] '새 기준선' 위에서 하드손절만 -12% → -10%로 교체.
매수/부분익절/불타기/90일선이탈은 새 기준선 그대로. strategy_core.py는 건드리지 않는다.
매도판정은 "먼저 닿는 쪽"(① 90일선 이탈 우선, ② 하드손절) 순서 유지.
"""
from typing import Optional
from strategy_core import Position, StockSnapshot

HARD_STOP_LOSS_PCT = 0.10


def check_sell_condition(position: Position, stock: StockSnapshot) -> tuple[bool, Optional[str]]:
    if stock.price < stock.ma_sell:
        return True, "90일선 이탈"
    stop_loss_price = position.avg_price * (1 - HARD_STOP_LOSS_PCT)
    if stock.price <= stop_loss_price:
        return True, f"하드손절(평단가 -{HARD_STOP_LOSS_PCT*100:.0f}%)"
    return False, None


if __name__ == "__main__":
    pos = Position(ticker="T", entry_price=10000, avg_price=10000, shares=100)
    ok, why = check_sell_condition(pos, StockSnapshot(ticker="T", date=None, price=8900,
        ma_buy=1, ma_sell=8000, roe_pct=18, eps=600, momentum_20d=0.02, trading_value_20d_avg=5e9))
    assert ok and "하드손절" in why, (ok, why)
    print(f"[PASS] strategy_core_sl10 (하드손절 -{HARD_STOP_LOSS_PCT*100:.0f}%) 로직 점검 완료")
