# -*- coding: utf-8 -*-
"""
backtest_base.py
================
[변형 1 · 비교 기준선] 기존 strategy_core.py 그대로(ROE×EPS 저평가 + 90일선 필터).
공용 엔진(bt_valuation_engine)에 strategy_core_base.check_buy_conditions만 꽂아 실행.
단독 실행하면 이 변형 하나의 리포트를 출력한다.
"""
import strategy_core_base as sv
from bt_valuation_engine import run_variant
from backtest_roe_eps_event import fpct

TAG = "1) base(ROE×EPS)"
CHECK = sv.check_buy_conditions


def run(store=None) -> dict:
    return run_variant(CHECK, TAG, store=store)


def print_report(m: dict) -> None:
    print("=" * 78)
    print(f" {m['tag']}  ·  {m['start']} ~ {m['end']}  ·  1억 · 거래비용 반영")
    print("=" * 78)
    print(f"  CAGR {fpct(m['CAGR'])}  MDD {fpct(m['MDD'])}  Sharpe {m['Sharpe']:.2f}  "
          f"Calmar {m['Calmar']:.2f}  손익비 {m['PL']:.2f}  승률 {m['win_rate']*100:.1f}%")
    print(f"  거래 {m['n_trades']} (신규 {m['n_buy']}/불타기 {m['n_add']}/부분익절 {m['n_partial']}"
          f"/전량매도 {m['n_final']})  매수즉시매도 {m['immediate_sell']}")
    print(f"  통과종목 평균 {m['avg_pass_cnt']:.1f}개 (유니버스 대비 {m['pass_rate']*100:.1f}%)  "
          f"최대단일비중 {m['max_weight']*100:.1f}%")


if __name__ == "__main__":
    print_report(run())
