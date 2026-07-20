# -*- coding: utf-8 -*-
"""
strategy_core_align_disp.py
===========================
[정배열 + 이격도 결합 변형 · 실험용] align1(3단 정배열)에, 지난 실험에서 견고했던
'90일선 이격도 ≥ 2.5%'를 추가로 결합한 버전.

    조건 = 정배열(20일선 > 60일선 > 90일선)  AND  이격도(현재가 vs 90일선) ≥ 2.5%
    이격도(%) = (현재가 - 90일선) / 90일선 × 100

의도: '이동평균 배열이 정렬됨'(추세의 질)과 '현재가가 90일선에서 충분히 떨어짐'
      (진입 시점의 여유폭)을 동시에 요구해 whipsaw를 더 강하게 막을 수 있는지 확인.
      다만 조건이 겹쳐 통과 종목이 지나치게 줄어들 위험도 함께 본다.

★ 원본은 건드리지 않는다. 저평가/ROE/모멘텀/거래대금/시장필터는 그대로,
  매도·불타기·부분익절·사이징·비용 규칙도 전부 원본 strategy_core.py를 재사용한다.
  20일 이동평균(ma20)은 엔진이 스냅샷에 부착해 넘겨준다.
"""
from strategy_core import (  # noqa: F401
    StockSnapshot,
    ROE_MIN_PCT,
    TRADING_VALUE_MIN_WON,
)

DISP_MIN_PCT = 2.5   # 90일선 이격도 최소 요구치(%)


def disparity_pct(price: float, ma90: float) -> float:
    if not ma90 or ma90 <= 0:
        return float("-inf")
    return (price - ma90) / ma90 * 100.0


def check_buy_conditions(stock: StockSnapshot, is_bull: bool) -> tuple[bool, list[str]]:
    fail_reasons: list[str] = []

    if not is_bull:
        fail_reasons.append("시장필터(KOSPI 200일선 하회, 약세장)")

    fair_price = stock.roe_pct * stock.eps
    if not (stock.price < fair_price):
        fail_reasons.append(f"저평가 아님 (현재가 {stock.price:,.0f} ≥ 적정가 {fair_price:,.0f})")

    if not (stock.roe_pct >= ROE_MIN_PCT):
        fail_reasons.append(f"ROE 미달 ({stock.roe_pct:.1f}% < {ROE_MIN_PCT}%)")

    # 3-a) 정배열: 20일선 > 60일선 > 90일선
    ma20 = getattr(stock, "ma20", float("nan"))
    if not (ma20 > stock.ma_buy > stock.ma_sell):
        fail_reasons.append(
            f"정배열 아님 (20선 {ma20:,.0f} > 60선 {stock.ma_buy:,.0f} > 90선 {stock.ma_sell:,.0f} 불성립)"
        )

    # 3-b) 90일선 이격도 ≥ 2.5%
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


if __name__ == "__main__":
    from datetime import date
    # 정배열 성립 + 이격도 12.5%(9000/8000) → 통과
    good = StockSnapshot(ticker="T", date=date(2026, 7, 9), price=9000,
                         ma_buy=8500, ma_sell=8000, roe_pct=18.0, eps=600,
                         momentum_20d=0.05, trading_value_20d_avg=5_000_000_000)
    good.ma20 = 8800
    assert check_buy_conditions(good, True)[0]
    # 정배열은 맞지만 이격도 1.25%(8100/8000) < 2.5% → 탈락
    edge = StockSnapshot(ticker="T", date=date(2026, 7, 9), price=8100,
                         ma_buy=7900, ma_sell=7800, roe_pct=18.0, eps=600,
                         momentum_20d=0.05, trading_value_20d_avg=5_000_000_000)
    edge.ma20 = 8000  # 8000 > 7900 > 7800 정배열 OK, but disp=(8100-7800)/7800=3.85%... adjust
    # 이격도 미달 케이스 재구성: ma_sell 8050 → disp=(8100-8050)/8050=0.62% < 2.5%
    edge2 = StockSnapshot(ticker="T", date=date(2026, 7, 9), price=8100,
                          ma_buy=8060, ma_sell=8050, roe_pct=18.0, eps=600,
                          momentum_20d=0.05, trading_value_20d_avg=5_000_000_000)
    edge2.ma20 = 8070  # 8070 > 8060 > 8050 정배열 OK, 이격도 0.62% → 탈락
    assert not check_buy_conditions(edge2, True)[0]
    print(f"[PASS] strategy_core_align_disp (정배열+이격도≥{DISP_MIN_PCT:.1f}%) 로직 점검 완료")
