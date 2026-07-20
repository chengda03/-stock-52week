# -*- coding: utf-8 -*-
"""
strategy_qv.py — "가치우량주 정배열"(전략2) 매수/매도 판정 로직
============================================================================
S1 의 strategy_core.py 에 대응하는, 전략2만의 '단일 진실소스'입니다.
데이터 조달(무엇을 어떻게 불러오는가)은 data_layer_qv.py 가 담당하고,
여기서는 오직 "이 종목을 살까/팔까"만 판정합니다.

매수조건(§2, 8개 전부 동시 충족):
  1) ROE 전년比 증가
  2) EPS 전년比 증가
  3) 영업이익 전년比 증가
  4) PER ≤ 10
  5) PBR ≤ 1.5
  6) 부채비율 ≤ 100%   (★금융업은 이 조건 면제)
  7) 정배열: 20일선 > 60일선 > 120일선 (이동평균선끼리 비교)
  8) 20일 평균거래대금 ≥ 20억원

매도조건(§5, 셋 중 하나라도 → 매도. 매월 첫 거래일 점검):
  1) 현재가 < 20일선
  2) 현재가 ≤ 진입가 × 0.90 (-10% 손절)
  3) [분기 첫달만 추가] 매수조건 8개 재점검 → 하나라도 탈락 시 매도
"""
from __future__ import annotations

# ===== 전략 설정값 (여기만 바꾸면 됨) =====
PER_MAX = 10
PBR_MAX = 1.5
DEBT_RATIO_MAX = 1.00  # 100%
MA_SHORT, MA_MID, MA_LONG = 20, 60, 120
TRADING_VALUE_MIN = 2_000_000_000  # 20억원, 20일 평균 기준
STOP_LOSS_PCT = -0.10  # -10%
NUM_STOCKS = 20
WEIGHT_PER_STOCK = 0.05  # 5%
INITIAL_CAPITAL = 100_000_000
# ==========================================


def compute_valuation(price: float, fin_cur: dict) -> dict:
    """현재가 + 현재 FY 재무 → PER/PBR/부채비율 계산.
    분모가 0 이하이거나 값이 없으면 None(→ 해당 조건 자동 탈락)."""
    ni = fin_cur.get("ni")
    eq = fin_cur.get("eq")
    liab = fin_cur.get("liab")
    sh = fin_cur.get("sh")
    marcap = price * sh if (sh and sh > 0) else None
    per = (marcap / ni) if (marcap is not None and ni and ni > 0) else None
    pbr = (marcap / eq) if (marcap is not None and eq and eq > 0) else None
    debt = (liab / eq) if (liab is not None and eq and eq > 0) else None
    return {"marcap": marcap, "per": per, "pbr": pbr, "debt_ratio": debt}


def check_buy_conditions(pf: dict, fin: dict | None, is_financial: bool) -> tuple[bool, dict]:
    """
    매수조건 8개를 판정. 반환 (전부충족?, 상세dict).
    상세dict: 각 조건 통과여부(c1~c8) + 계산값(per/pbr/debt/roe...) + 사용한 FY 연도.

    pf   : data_layer_qv.get_price_features 결과 (price/ma20/ma60/ma120/tv20)
    fin  : data_layer_qv.get_financials 결과 (year/cur/prev/estimated) 또는 None
    is_financial : 금융업 여부(True면 부채비율 조건 §2-6 면제)
    """
    info: dict = {"per": None, "pbr": None, "debt": None, "year": None,
                  "estimated": None}
    if fin is None:
        info["fail"] = "재무없음(공시전)"
        return False, info
    cur = fin["cur"]
    prev = fin["prev"]
    info["year"] = fin["year"]
    info["estimated"] = fin["estimated"]

    val = compute_valuation(pf["price"], cur)
    info["per"] = val["per"]
    info["pbr"] = val["pbr"]
    info["debt"] = val["debt_ratio"]

    # 전년비 증가 3종 — 현재/전년 값이 모두 있어야 판정 가능
    def increased(key):
        if prev is None:
            return False
        a, b = cur.get(key), prev.get(key)
        return (a is not None and b is not None and a > b)

    c1 = increased("roe")                                   # ROE↑
    c2 = increased("eps")                                   # EPS↑
    c3 = increased("op")                                    # 영업이익↑
    c4 = val["per"] is not None and val["per"] <= PER_MAX   # PER≤10
    c5 = val["pbr"] is not None and val["pbr"] <= PBR_MAX   # PBR≤1.5
    # 부채비율≤100% (금융업 면제). 비금융업인데 부채값 없으면 판정불가 → 탈락.
    if is_financial:
        c6 = True
    else:
        c6 = val["debt_ratio"] is not None and val["debt_ratio"] <= DEBT_RATIO_MAX
    # 정배열 20>60>120
    ma20, ma60, ma120 = pf["ma20"], pf["ma60"], pf["ma120"]
    c7 = (ma20 == ma20 and ma60 == ma60 and ma120 == ma120  # NaN 아님
          and ma20 > ma60 > ma120)
    c8 = pf["tv20"] == pf["tv20"] and pf["tv20"] >= TRADING_VALUE_MIN  # 거래대금

    info.update(c1=c1, c2=c2, c3=c3, c4=c4, c5=c5, c6=c6, c7=c7, c8=c8)
    ok = all((c1, c2, c3, c4, c5, c6, c7, c8))
    return ok, info


def check_sell_basic(entry_price: float, price: float, ma20: float) -> tuple[bool, str]:
    """매도조건 1,2번(상시 점검): 20일선 이탈 / -10% 손절."""
    if price <= entry_price * (1.0 + STOP_LOSS_PCT):
        return True, "손절(-10%)"
    if ma20 == ma20 and price < ma20:   # ma20 유효 & 현재가 < 20일선
        return True, "20일선이탈"
    return False, ""


def sort_key_per(info: dict) -> float:
    """매수 우선순위(§4): PER 낮은 순(저평가 우선).
    ※ 확정 규칙 아님 — 결과 보고 사용자와 재논의 예정."""
    per = info.get("per")
    return per if per is not None else float("inf")
