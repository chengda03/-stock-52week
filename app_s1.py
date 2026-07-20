# -*- coding: utf-8 -*-
"""
app_s1.py — "ROE×EPS 모멘텀 불타기(S1 v2)" 실운영 신호 앱
=====================================================
자동매매가 아니라 "오늘 뭘 사고 팔지" 알려주는 신호 앱입니다.
실제 체결은 사용자가 증권사 앱에서 직접 하고, 끝나면 여기서 '확정' 버튼을 눌러
포트폴리오(JSON)에 반영합니다. 화면은
      [살 후보] → 매입확정 → [보유종목] → 매도확정 → [팔 후보]
흐름을 그대로 보여줍니다.

계좌 2개:
  · 계좌 A(성장)     = portfolio_A.json  (인출 없음)
  · 계좌 B(현금흐름) = portfolio_B.json  (연말 인출)
화면 최상단 라디오로 계좌를 고르면 상단 카드+3단 리스트가 통째로 그 계좌로 전환됩니다.
(계좌 합산/비교 화면은 만들지 않습니다.)

탭:
  탭1 오늘의 매매    : 위 3단 흐름(오늘의신호+보유포트폴리오 통합 — 이번에 새로 만든 화면)
  탭2 백테스트 리포트: backtest_v3 결과 (기존 그대로, 손대지 않음)

판정 로직은 strategy_core.py 를 그대로 import 해서 사용합니다(수정 없음).
정직성 원칙: 룩어헤드 없음, 생존편향 존재 명시, 백테스트는 참고용.
"""
from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data_layer
import portfolio_state as ps
from strategy_core import (
    is_bull_market, check_buy_conditions, rank_by_momentum,
    check_sell_condition, check_pyramid, check_partial_exit,
    NUM_SLOTS, SLOT_AMOUNT_WON,
    HARD_STOP_LOSS_PCT, SELL_MA_DAYS,
    PARTIAL_EXIT_TRIGGER_PCT, PARTIAL_EXIT_RATIO,
    PYRAMID_TRIGGER_PCT, PYRAMID_ADD_AMOUNT_WON,
)

BUY_COST = 0.00115     # 매수 수수료+슬리피지(0.115%)
SELL_COST = 0.00295    # 매도 수수료+세금+슬리피지(0.295%)
V3_RESULT_DIR = os.path.join("data", "cache_v3", "results")

st.set_page_config(page_title="SF 투자법", page_icon="📈", layout="wide")

# ---------------------------------------------------------------------------
# 남색(네이비) 다크 테마 색상 팔레트
#   - 배경만 남색 계열로 바꾸고, 텍스트는 밝게 유지해 가독성 확보.
#   - 카드/패널 구조는 그대로 두고 색만 이 톤에 맞춤.
# ---------------------------------------------------------------------------
NAVY_BG = "#0B1F3A"        # 전체 배경(요청 예시 톤)
NAVY_SIDEBAR = "#0A1A30"   # 사이드바(살짝 더 진하게)
PANEL_BG = "#13294B"       # 카드/패널 기본 배경
PANEL_BG_FLAG = "#3A1622"  # 매도신호 걸린 보유카드(붉은 남색 톤)
PANEL_BORDER = "#22436E"   # 패널 테두리
TEXT_MAIN = "#E6EDF5"      # 주 텍스트(밝은 회백색)
TEXT_MUTED = "#9FB3C8"     # 보조 텍스트
POS_COLOR = "#51CF66"      # 수익(초록)
NEG_COLOR = "#FF6B6B"      # 손실(빨강)
TAG_PARTIAL = "#F4B400"    # 부분익절 태그(노랑 계열)
TAG_FULL = "#E03131"       # 전량매도 태그(빨강 계열)


def inject_theme() -> None:
    """전체 배경을 남색 다크로 바꾸는 전역 CSS를 주입(배경/텍스트 색만 조정)."""
    st.markdown(
        f"""
        <style>
        /* 전체 배경 남색 + 밝은 텍스트 (헤더/툴바/메인컨테이너까지 끊김없이) */
        .stApp {{ background-color: {NAVY_BG}; }}
        [data-testid="stAppViewContainer"] {{ background-color: {NAVY_BG}; }}
        [data-testid="stMain"] {{ background-color: {NAVY_BG}; }}
        [data-testid="stMainBlockContainer"], .block-container {{ background-color: {NAVY_BG}; }}
        header[data-testid="stHeader"] {{ background-color: {NAVY_BG}; }}
        [data-testid="stToolbar"] {{ background: transparent; }}
        [data-testid="stDecoration"] {{ background: {NAVY_BG}; }}
        .stApp, .stApp p, .stApp span, .stApp label,
        .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4 {{ color: {TEXT_MAIN}; }}
        /* 사이드바 */
        section[data-testid="stSidebar"] {{ background-color: {NAVY_SIDEBAR}; }}
        section[data-testid="stSidebar"] * {{ color: {TEXT_MAIN}; }}
        /* 지표(metric) 카드 */
        div[data-testid="stMetric"] {{
            background-color: {PANEL_BG};
            border: 1px solid {PANEL_BORDER};
            border-radius: 10px;
            padding: 10px 12px;
            margin-bottom: 8px;
        }}
        div[data-testid="stMetricLabel"] {{ color: {TEXT_MUTED}; }}
        div[data-testid="stMetricValue"] {{ color: {TEXT_MAIN}; }}
        /* 테두리 컨테이너(st.container(border=True)) → 남색 패널 */
        div[data-testid="stVerticalBlockBorderWrapper"] {{
            background-color: {PANEL_BG};
            border-radius: 10px;
        }}
        /* 탭 라벨 */
        button[data-baseweb="tab"] {{ color: {TEXT_MUTED}; }}
        button[data-baseweb="tab"][aria-selected="true"] {{ color: {TEXT_MAIN}; }}
        /* 데이터프레임 배경 살짝 조정 */
        div[data-testid="stDataFrame"] {{ background-color: {PANEL_BG}; }}
        /* 입력 위젯(날짜 입력·숫자 입력 등) 배경 남색 통일 */
        [data-testid="stDateInput"] div[data-baseweb="input"],
        [data-testid="stDateInput"] div[data-baseweb="base-input"],
        [data-testid="stNumberInput"] div[data-baseweb="input"],
        [data-testid="stNumberInput"] div[data-baseweb="base-input"],
        div[data-baseweb="input"], div[data-baseweb="base-input"] {{
            background-color: {PANEL_BG} !important;
            border-color: {PANEL_BORDER} !important;
        }}
        [data-testid="stDateInput"] input,
        [data-testid="stNumberInput"] input,
        div[data-baseweb="input"] input {{
            background-color: {PANEL_BG} !important;
            color: {TEXT_MAIN} !important;
            -webkit-text-fill-color: {TEXT_MAIN} !important;
        }}
        /* 날짜 선택 달력 팝오버 */
        div[data-baseweb="calendar"] {{ background-color: {PANEL_BG} !important; }}
        div[data-baseweb="calendar"] * {{ color: {TEXT_MAIN} !important; }}
        /* number_input 증감 버튼 */
        [data-testid="stNumberInput"] button {{ background-color: {PANEL_BG} !important; }}
        /* 셀렉트박스/드롭다운 */
        div[data-baseweb="select"] > div {{
            background-color: {PANEL_BG} !important; border-color: {PANEL_BORDER} !important;
        }}
        div[data-baseweb="popover"], ul[data-baseweb="menu"], div[data-baseweb="menu"] {{
            background-color: {PANEL_BG} !important;
        }}
        div[data-baseweb="popover"] *, ul[data-baseweb="menu"] * {{ color: {TEXT_MAIN} !important; }}
        /* expander 헤더/본문 */
        details[data-testid="stExpander"], [data-testid="stExpander"] > details {{
            background-color: {PANEL_BG} !important; border-color: {PANEL_BORDER} !important;
        }}
        [data-testid="stExpander"] summary {{ background-color: {PANEL_BG} !important; color: {TEXT_MAIN} !important; }}
        /* 모달(dialog) */
        div[data-testid="stDialog"] div[role="dialog"] {{
            background-color: {NAVY_SIDEBAR} !important; color: {TEXT_MAIN} !important;
        }}
        div[data-testid="stDialog"] div[role="dialog"] * {{ color: {TEXT_MAIN} !important; }}
        /* dataframe(글라이드 그리드) 컨테이너 */
        div[data-testid="stDataFrame"], [data-testid="stDataFrameResizable"] {{
            background-color: {PANEL_BG} !important;
        }}
        /* 기본 버튼(비강조) 배경도 패널색으로 */
        div[data-testid="stButton"] > button {{
            background-color: {PANEL_BG}; color: {TEXT_MAIN}; border-color: {PANEL_BORDER};
        }}
        div[data-testid="stButton"] > button:disabled {{ color: {TEXT_MUTED}; opacity: 0.5; }}
        /* 알림박스(info/success/error 등) 텍스트 밝게 + 크게(안내 메시지 ≥16px) */
        div[data-testid="stAlert"] {{ color: {TEXT_MAIN}; }}
        div[data-testid="stAlert"] p {{ color: {TEXT_MAIN} !important; font-size: 16px !important; }}
        /* 섹션 제목(st.subheader=h3, st.header=h2 등) 크게·굵게 — 메인 영역만 */
        [data-testid="stMain"] h1 {{ font-size: 30px !important; font-weight: 700 !important; }}
        [data-testid="stMain"] h2 {{ font-size: 26px !important; font-weight: 700 !important; }}
        [data-testid="stMain"] h3 {{ font-size: 22px !important; font-weight: 700 !important; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# 공통 유틸
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="데이터 로딩 중… (최초 1회, 30초 내외)")
def load_store():
    """무거운 데이터 로딩은 1회만 (Streamlit 캐시)."""
    return data_layer.get_store()


def won(x) -> str:
    """천단위 콤마 원화 표기."""
    try:
        return f"{float(x):,.0f}원"
    except Exception:
        return "-"


def eok(x) -> str:
    """억 단위 간략 표기."""
    try:
        return f"{float(x)/1e8:,.2f}억"
    except Exception:
        return "-"


def pct(x) -> str:
    try:
        return f"{float(x)*100:+.1f}%"
    except Exception:
        return "-"


def next_yearend_trading_day(today: date) -> date:
    """
    올해(또는 이미 지났으면 내년) '12월 마지막 거래일'의 근사값.
    한국거래소는 통상 12/31이 연말 폐장일(휴장)이라 그 직전 영업일이 마지막 거래일.
    → 12/30 부터 뒤로 가며 주말(토/일)을 피한 날을 사용(공휴일까진 미반영, 표시용).
    """
    def _last(year: int) -> date:
        d = date(year, 12, 30)
        while d.weekday() >= 5:   # 5=토, 6=일
            d -= timedelta(days=1)
        return d
    target = _last(today.year)
    if today > target:
        target = _last(today.year + 1)
    return target


# ---------------------------------------------------------------------------
# 판정 → 화면용 데이터(좌/중/우 3리스트 + 상단지표) 조립
#   전체 순서: strategy_core 로 오늘 판정 → portfolio 보유현황과 조합 → 3리스트 생성
# ---------------------------------------------------------------------------
def evaluate_holdings(store, state: dict, as_of):
    """보유종목 각각을 평가하고 매도조건/부분익절 여부까지 판정해 행 리스트로 반환."""
    rows = []
    for ticker, pos_d in state["positions"].items():
        snap = store.get_price_snapshot(ticker, as_of)
        cur = snap.price if snap else pos_d["avg_price"]
        val = cur * pos_d["shares"]
        pos = ps.dict_to_position(ticker, pos_d)

        # 매도조건(전량) 먼저, 아니면 부분익절(+8%) 확인 — strategy_core 호출 순서 규칙 준수
        sell_full, sell_reason = check_sell_condition(pos, snap) if snap else (False, None)
        partial = (not sell_full) and snap is not None and check_partial_exit(pos, snap.price)

        entry_d = pos_d.get("entry_date")
        days = None
        if entry_d:
            try:
                days = (pd.Timestamp(as_of).date() - date.fromisoformat(entry_d)).days
            except Exception:
                days = None
        rows.append({
            "ticker": ticker,
            "name": store.name_map.get(ticker, ticker),
            "shares": pos_d["shares"],
            "avg_price": pos_d["avg_price"],
            "entry_price": pos_d["entry_price"],
            "entry_date": entry_d,
            "cur": cur,
            "val": val,
            "pnl": (cur - pos_d["avg_price"]) * pos_d["shares"],
            "pnl_pct": (cur / pos_d["avg_price"] - 1.0) if pos_d["avg_price"] else 0.0,
            "days": days,
            "ma_sell": snap.ma_sell if snap else float("nan"),
            "sell_full": sell_full,
            "sell_reason": sell_reason,
            "partial": partial,
            "partial_qty": pos_d["shares"] * PARTIAL_EXIT_RATIO if partial else 0.0,
            "has_price": snap is not None,
        })
    return rows


def build_buy_candidates(store, state: dict, as_of, is_bull: bool):
    """오늘 매수조건을 통과한 (미보유) 종목을 20일 모멘텀 높은 순으로 반환."""
    if not is_bull:
        return []   # 약세장이면 매수조건(시장필터)에서 전원 탈락
    positions = state["positions"]
    cands = []
    for t in store.get_universe(as_of):
        if t in positions:
            continue
        snap = store.get_price_snapshot(t, as_of)
        if snap is None:
            continue
        ok, _ = check_buy_conditions(snap, is_bull)
        if ok:
            cands.append(snap)
    ranked = rank_by_momentum(cands)
    out = []
    for snap in ranked:
        sh = int(SLOT_AMOUNT_WON // snap.price) if snap.price > 0 else 0
        out.append({
            "ticker": snap.ticker,
            "name": store.name_map.get(snap.ticker, snap.ticker),
            "price": snap.price,
            "momentum": snap.momentum_20d,
            "est_shares": sh,                 # 슬롯금액(500만) // 현재가 → 정수 내림
            "est_amount": sh * snap.price,    # 실제 투입금액(수량×현재가)
        })
    return out


def build_pyramid_candidates(store, state: dict, as_of, is_bull: bool):
    """
    불타기(추가매수) 조건을 만족한 '기존 보유종목'을 반환.
    strategy_core.check_pyramid() 판정을 그대로 사용한다(진입가×1.03^n 도달).
    각 항목: 현재 몇 차 불타기인지 · 트리거가격 · 추가매수 수량/금액.
    """
    if not is_bull:
        return []   # 약세장이면 불타기 전면 중단(전략 규칙)
    cash = float(state["cash"])
    as_of_d = pd.Timestamp(as_of).date()
    out = []
    for ticker, pos_d in state["positions"].items():
        snap = store.get_price_snapshot(ticker, as_of)
        if snap is None:
            continue
        pos = ps.dict_to_position(ticker, pos_d)
        if not check_pyramid(pos, snap.price, as_of_d, is_bull, cash):
            continue
        next_step = pos.pyramid_count + 1   # 이번에 진행될 불타기 차수
        trigger = pos.entry_price * ((1 + PYRAMID_TRIGGER_PCT) ** next_step)
        add_sh = int(PYRAMID_ADD_AMOUNT_WON // snap.price) if snap.price > 0 else 0
        out.append({
            "ticker": ticker,
            "name": store.name_map.get(ticker, ticker),
            "price": snap.price,
            "step": next_step,
            "trigger": trigger,
            "add_shares": add_sh,
            "add_amount": add_sh * snap.price,
        })
    # 트리거가 높은(=더 많이 오른) 종목부터 위로
    out.sort(key=lambda x: x["step"], reverse=True)
    return out


def summary_metrics(state: dict, holdings_rows: list) -> dict:
    """상단 카드용 지표: 투자원금 / 총자산 / 평가손익 / 현금."""
    holdings_val = sum(r["val"] for r in holdings_rows)
    cash = float(state["cash"])
    total = cash + holdings_val
    withdrawn = float(state.get("withdrawn", 0.0))
    # 투자원금 = 누적 투입 총액. 인출계좌(B)는 인출된 금액을 제외한 순수 투입액.
    principal = ps.INITIAL_CASH - (withdrawn if state.get("account") == "B" else 0.0)
    pnl = total - principal
    return {
        "principal": principal, "total": total, "cash": cash,
        "holdings_val": holdings_val, "pnl": pnl,
        "pnl_pct": (pnl / principal) if principal else 0.0,
    }


# ---------------------------------------------------------------------------
# 확정 팝업 (Streamlit 1.59 → st.dialog 사용)
# ---------------------------------------------------------------------------
@st.dialog("매입 확정")
def buy_dialog(account: str, as_of, ticker: str, name: str,
               default_price: float, default_shares: int):
    """매입가/수량을 입력받아 portfolio_<account>.json 에 신규매수로 반영."""
    st.write(f"**{name}** ({ticker})")
    price = st.number_input("매입가(원)", min_value=0.0,
                            value=float(round(default_price)), step=10.0)
    shares = st.number_input("수량(주)", min_value=1,
                             value=int(max(default_shares, 1)), step=1)
    amount = price * shares * (1.0 + BUY_COST)
    st.caption(f"예상 투입금액(수수료 {BUY_COST*100:.3f}% 포함): {won(amount)}")
    if st.button("확인 — 보유종목에 추가", type="primary", width="stretch"):
        state = ps.load_portfolio(account)
        ps.record_trade(state, ticker, "신규매수", price=float(price),
                        shares=float(shares),
                        date_=pd.Timestamp(as_of).date(),
                        reason="매입확정(사용자 입력)", amount=amount)
        ps.save_portfolio(state, account)
        st.rerun()


@st.dialog("불타기(추가매수) 확정")
def pyramid_dialog(account: str, as_of, ticker: str, name: str,
                   step: int, default_price: float, default_shares: int):
    """추가매수가/수량을 입력받아 '불타기'로 반영(평단가 가중평균 재계산)."""
    st.write(f"**{name}** ({ticker}) — {step}차 불타기")
    price = st.number_input("추가매수가(원)", min_value=0.0,
                            value=float(round(default_price)), step=10.0)
    shares = st.number_input("추가수량(주)", min_value=1,
                             value=int(max(default_shares, 1)), step=1)
    amount = price * shares * (1.0 + BUY_COST)
    st.caption(f"예상 투입금액(수수료 {BUY_COST*100:.3f}% 포함): {won(amount)}")
    st.caption("확정하면 기존 보유수량·평단가가 가중평균으로 재계산됩니다.")
    if st.button("확인 — 불타기 반영", type="primary", width="stretch"):
        state = ps.load_portfolio(account)
        ps.record_trade(state, ticker, "불타기", price=float(price),
                        shares=float(shares),
                        date_=pd.Timestamp(as_of).date(),
                        reason=f"{step}차 불타기(사용자 입력)", amount=amount)
        ps.save_portfolio(state, account)
        st.rerun()


@st.dialog("매도 확정")
def sell_dialog(account: str, as_of, ticker: str, name: str,
                default_price: float, mode: str, qty: float):
    """매도가를 입력받아 반영. mode='full'이면 전량매도, 'partial'이면 50%만 매도."""
    st.write(f"**{name}** ({ticker})")
    label = "전량매도" if mode == "full" else f"부분익절 — {qty:.0f}주(50%)만 매도"
    st.caption(f"매도 유형: {label}")
    price = st.number_input("매도가(원)", min_value=0.0,
                            value=float(round(default_price)), step=10.0)
    sell_sh = qty
    amount = price * sell_sh * (1.0 - SELL_COST)
    st.caption(f"예상 회수금액(수수료·세금 {SELL_COST*100:.3f}% 차감): {won(amount)}")
    if st.button("확인 — 매도 반영", type="primary", width="stretch"):
        state = ps.load_portfolio(account)
        action = "매도" if mode == "full" else "부분익절"
        ps.record_trade(state, ticker, action, price=float(price),
                        shares=float(sell_sh),
                        date_=pd.Timestamp(as_of).date(),
                        reason="매도확정(사용자 입력)", amount=amount)
        ps.save_portfolio(state, account)
        st.rerun()


# ---------------------------------------------------------------------------
# 렌더링 함수 (역할별로 작게 분리)
# ---------------------------------------------------------------------------
def render_summary_cards(store, state: dict, holdings_rows: list, is_bull: bool):
    """사이드바 요약 카드(세로 스택): 투자원금/총자산/평가손익/현금 (+B계좌면 '다음 인출까지').
    계좌 선택 라디오 바로 아래에 세로로 쌓이도록 st.columns 없이 순서대로 렌더한다."""
    m = summary_metrics(state, holdings_rows)
    account = state.get("account", "A")

    st.metric("투자원금", eok(m["principal"]),
              help="누적 투입 총액(계좌 B는 인출금 제외한 순수 투입액)")
    st.metric("총자산", eok(m["total"]), help="보유종목 평가액 + 현금")
    # 평가손익: 금액 + % (플러스/마이너스 색상은 st.metric delta가 자동 처리)
    st.metric("평가손익", won(m["pnl"]), delta=pct(m["pnl_pct"]))
    st.metric("현금 잔고", eok(m["cash"]), help="매수 대기 중인 현금")

    if account == "B":
        target = next_yearend_trading_day(date.today())
        dday = (target - date.today()).days
        st.metric("다음 인출까지", f"D-{dday}",
                  help=f"{target.isoformat()} (12월 마지막 거래일 근사) · 실제 인출 실행은 미구현")


def render_market_chart(store, as_of):
    """메인 최상단: 최근 1년 KOSPI 종가 + 200일선 라인차트(plotly).
    교차점(x 마커)·현재 종가(원 마커)를 강조하고, 강세장/약세장을 색/제목으로 표시."""
    di = store._di(as_of)
    lo = max(0, di - 251)                    # 약 1년(≈252 거래일)
    idx = store.trading_days[lo:di + 1]
    close = store.kospi_close_v[lo:di + 1]
    ma200 = store.kospi_ma200_v[lo:di + 1]
    if len(idx) == 0:
        st.info("KOSPI 차트를 그릴 데이터가 없습니다.")
        return
    # 60일선·120일선은 전체 종가에서 계산 후 표시구간만 슬라이스(min_periods=1로 통일)
    close_full = pd.Series(store.kospi_close_v)
    ma60 = close_full.rolling(60, min_periods=1).mean().to_numpy()[lo:di + 1]
    ma120 = close_full.rolling(120, min_periods=1).mean().to_numpy()[lo:di + 1]

    is_bull = close[-1] > ma200[-1]
    state_txt = "강세장" if is_bull else "약세장"
    state_color = POS_COLOR if is_bull else NEG_COLOR

    # 200일선 상/하 교차점 탐지(부호 변화 지점)
    diff = close - ma200
    cross_x, cross_y = [], []
    for i in range(1, len(diff)):
        if (diff[i - 1] <= 0 < diff[i]) or (diff[i - 1] >= 0 > diff[i]):
            cross_x.append(idx[i])
            cross_y.append(close[i])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx, y=close, name="KOSPI",
                             line=dict(color="#4DABF7", width=2)))
    fig.add_trace(go.Scatter(x=idx, y=ma60, name="60일선",
                             line=dict(color="#51CF66", width=1.5)))
    fig.add_trace(go.Scatter(x=idx, y=ma120, name="120일선",
                             line=dict(color="#FF922B", width=1.5)))
    fig.add_trace(go.Scatter(x=idx, y=ma200, name="200일선",
                             line=dict(color=TAG_PARTIAL, width=1.5, dash="dash")))
    if cross_x:
        fig.add_trace(go.Scatter(x=cross_x, y=cross_y, mode="markers", name="교차",
                                 marker=dict(color="#FF6B6B", size=9, symbol="x")))
    # 현재 지점(가장 최근 종가) 마커
    fig.add_trace(go.Scatter(
        x=[idx[-1]], y=[close[-1]], mode="markers", name="현재",
        marker=dict(color=state_color, size=13, symbol="circle",
                    line=dict(color="#FFFFFF", width=1)),
    ))
    # 강세장/약세장 텍스트를 차트 위에 함께 표시(사이드바)
    st.markdown(
        f"<div style='font-size:0.95rem;font-weight:700;color:{state_color};margin:2px 0'>"
        f"KOSPI {state_txt} <span style='color:{TEXT_MUTED};font-weight:400;font-size:0.8rem'>"
        f"(종가 {close[-1]:,.0f} / 200일선 {ma200[-1]:,.0f})</span></div>",
        unsafe_allow_html=True,
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=NAVY_SIDEBAR, plot_bgcolor=PANEL_BG,
        height=230, margin=dict(l=4, r=4, t=6, b=4),
        showlegend=True,
        legend=dict(orientation="h", y=1.16, x=0, font=dict(color=TEXT_MAIN, size=9),
                    bgcolor="rgba(0,0,0,0)"),
        font=dict(color=TEXT_MAIN, size=10),
    )
    fig.update_xaxes(gridcolor=PANEL_BORDER, zeroline=False, tickfont=dict(size=9))
    fig.update_yaxes(gridcolor=PANEL_BORDER, zeroline=False, tickfont=dict(size=9))
    st.plotly_chart(fig, width="stretch", theme=None)
    st.caption("60일선(초록) · 120일선(주황) · 200일선(노랑 점선) · 원=현재 · X=200일선 교차")


def _cell(col, text: str, *, bold=False, color=None, size="0.82rem", align="left"):
    """표 셀 하나를 렌더하는 도우미(작은 폰트·색상 지원)."""
    c = color or TEXT_MAIN
    w = "700" if bold else "400"
    col.markdown(
        f"<div style='font-size:{size};color:{c};font-weight:{w};"
        f"text-align:{align};line-height:1.35;word-break:break-all'>{text}</div>",
        unsafe_allow_html=True,
    )


def render_buy_candidates(account: str, as_of, buy_cands: list, free_slots: int,
                          is_bull: bool):
    """좌측 상단: '신규매수 후보'를 표 형태(종목·현재가·수량·금액·확정)로."""
    st.subheader("🟠 신규매수 후보")
    st.markdown(
        f"<div style='font-size:18px;font-weight:700;color:{TEXT_MAIN};margin:2px 0 6px'>"
        f"현재 빈 슬롯: {free_slots}개 "
        f"<span style='color:{TEXT_MUTED};font-weight:400'>/ {NUM_SLOTS}</span></div>",
        unsafe_allow_html=True,
    )
    if not is_bull:
        st.info("약세장이라 신규매수 후보가 없습니다.")
        return
    if not buy_cands:
        st.info("매수조건(6개)을 통과한 종목이 없습니다.")
        return
    W = [0.5, 2.2, 1.5, 1.0, 1.7, 1.3]   # 연번 / 종목 / 현재가 / 수량 / 금액 / 버튼
    hdr = st.columns(W, vertical_alignment="center")
    for col, t in zip(hdr, ["#", "종목", "현재가", "수량", "예정금액", ""]):
        _cell(col, t, bold=True, color=TEXT_MUTED, size="18px",
              align=("center" if t == "#" else "left"))
    # buy_cands는 20일 모멘텀 높은 순으로 이미 정렬됨 → 그 순서대로 1번부터 연번 부여.
    for i, c in enumerate(buy_cands):
        row = st.columns(W, vertical_alignment="center")
        _cell(row[0], f"{i+1}", bold=True, color=TEXT_MUTED, size="18px", align="center")
        _cell(row[1], f"{c['name']}<br><span style='color:{TEXT_MUTED};font-size:14px'>{c['ticker']}</span>",
              size="18px")
        _cell(row[2], f"{c['price']:,.0f}", size="18px", align="right")
        _cell(row[3], f"{c['est_shares']}주", size="18px", align="right")
        _cell(row[4], f"{c['est_amount']:,.0f}", bold=True, size="18px", align="right")
        if row[5].button("매입확정", key=f"buy_{account}_{c['ticker']}_{i}",
                         width="stretch"):
            buy_dialog(account, as_of, c["ticker"], c["name"],
                       c["price"], c["est_shares"])


def render_pyramid_signals(account: str, as_of, pyr_cands: list, is_bull: bool):
    """전체폭 '불타기해야 하는 종목' 섹션. 진입가×1.03^n 도달 보유종목을
    N차·트리거가격·추가매수 수량/금액과 함께 표시, 클릭시 매입확정(불타기)."""
    st.subheader("🔥 불타기해야 하는 종목")
    if not is_bull:
        st.info("약세장이라 불타기 신호가 없습니다.")
        return
    if not pyr_cands:
        st.info("불타기 트리거(진입가×1.03ⁿ)에 도달한 보유종목이 없습니다.")
        return
    for i, c in enumerate(pyr_cands):
        with st.container(border=True):
            cols = st.columns([4, 1.3], vertical_alignment="center")
            cols[0].markdown(
                f"<div style='font-size:19px;font-weight:700;color:{TEXT_MAIN}'>"
                f"{c['name']} <span style='background:{TAG_PARTIAL};color:#1a1a1a;padding:2px 10px;"
                f"border-radius:6px;font-size:15px;font-weight:700'>{c['step']}차 불타기</span> "
                f"<span style='color:{TEXT_MUTED};font-weight:400;font-size:15px'>({c['ticker']})</span></div>"
                f"<div style='font-size:16px;color:{TEXT_MUTED};margin:4px 0'>"
                f"트리거가격 {c['trigger']:,.0f}원 · 현재가 {c['price']:,.0f}원</div>"
                f"<div style='font-size:17px;color:{TEXT_MAIN}'>"
                f"추가매수 <b>{c['add_shares']}주</b> · <b>{c['add_amount']:,.0f}원</b> (500만원 기준)</div>",
                unsafe_allow_html=True,
            )
            if cols[1].button("매입확정", key=f"pyr_{account}_{c['ticker']}_{i}",
                              type="primary", width="stretch"):
                pyramid_dialog(account, as_of, c["ticker"], c["name"],
                               c["step"], c["price"], c["add_shares"])


def render_holdings(account: str, as_of, holdings_rows: list):
    """중앙: 보유종목을 표 형태로만 표시(신호 배지 없음 — 매도/불타기 확정은 우측 신호섹션에서).
    컬럼: 종목·현재가·평단가·수량·매입일·경과·총금액·평가손익(금액+%)."""
    st.subheader(f"💼 보유종목 ({len(holdings_rows)}개/{NUM_SLOTS})")
    if not holdings_rows:
        st.info("보유 종목이 없습니다. (현금 100%)")
        return

    W = [0.4, 1.8, 1.15, 1.15, 0.85, 1.65, 0.7, 1.35, 1.45]
    DATA = "18px"          # 표 본문(숫자·종목명) 폰트 ≥18px
    HEAD = "18px"          # 표 헤더 폰트 ≥18px(굵게)
    hdr = st.columns(W, vertical_alignment="center")
    for col, t in zip(hdr, ["#", "종목", "현재가", "평단가", "수량", "매입일", "경과",
                            "총금액", "평가손익"]):
        _cell(col, t, bold=True, color=TEXT_MUTED, size=HEAD,
              align=("center" if t == "#"
                     else "right" if t in ("현재가", "평단가", "수량", "경과", "총금액", "평가손익")
                     else "left"))

    # 지금 표시 순서(총금액 큰 순) 그대로 1번부터 연번 부여.
    for i, r in enumerate(sorted(holdings_rows, key=lambda x: x["val"], reverse=True)):
        t = r["ticker"]
        pnl_color = POS_COLOR if r["pnl"] >= 0 else NEG_COLOR
        days_txt = f"{r['days']}" if r["days"] is not None else "-"
        entry_txt = r.get("entry_date") or "-"
        row = st.columns(W, vertical_alignment="center")
        _cell(row[0], f"{i+1}", bold=True, color=TEXT_MUTED, size=DATA, align="center")
        _cell(row[1], f"{r['name']}<br><span style='color:{TEXT_MUTED};font-size:14px'>{t}</span>",
              size=DATA)
        _cell(row[2], f"{r['cur']:,.0f}", size=DATA, align="right")
        _cell(row[3], f"{r['avg_price']:,.0f}", size=DATA, align="right")
        _cell(row[4], f"{r['shares']:.0f}", size=DATA, align="right")
        _cell(row[5], f"{entry_txt}", size=DATA)
        _cell(row[6], days_txt, size=DATA, align="right")
        _cell(row[7], f"{r['val']:,.0f}", size=DATA, align="right")
        _cell(row[8],
              f"{r['pnl']:,.0f}<br><span style='font-size:16px'>({r['pnl_pct']*100:+.1f}%)</span>",
              bold=True, color=pnl_color, size=DATA, align="right")


def render_sell_signals(account: str, as_of, holdings_rows: list):
    """전체폭 '매도해야 하는 종목' 섹션.
    매도조건(90일선 이탈/-12%손절/+8%부분익절) 걸린 보유종목을 사유와 함께 표시, 클릭시 매도확정."""
    st.subheader("🔴 매도해야 하는 종목")
    sell_rows = [r for r in holdings_rows if r["sell_full"] or r["partial"]]
    if not sell_rows:
        st.info("오늘 매도 신호가 없습니다.")
        return
    for i, r in enumerate(sell_rows):
        with st.container(border=True):
            cols = st.columns([4, 1.3], vertical_alignment="center")
            if r["sell_full"]:
                tag = (f"<span style='background:{TAG_FULL};color:#fff;padding:2px 10px;"
                       f"border-radius:6px;font-size:15px;font-weight:700'>전량매도</span>")
                reason = f"{r['sell_reason']} — 전량매도"
                detail = f"현재가 {r['cur']:,.0f}원 · 보유 {r['shares']:.0f}주 전량"
                mode, qty = "full", r["shares"]
            else:
                tag = (f"<span style='background:{TAG_PARTIAL};color:#1a1a1a;padding:2px 10px;"
                       f"border-radius:6px;font-size:15px;font-weight:700'>부분익절</span>")
                reason = f"부분익절 +{PARTIAL_EXIT_TRIGGER_PCT*100:.0f}% (50% 매도)"
                detail = f"현재가 {r['cur']:,.0f}원 · {r['partial_qty']:.0f}주(50%)만 매도"
                mode, qty = "partial", r["partial_qty"]
            cols[0].markdown(
                f"<div style='font-size:19px;font-weight:700;color:{TEXT_MAIN}'>"
                f"{r['name']} <span style='color:{TEXT_MUTED};font-weight:400;font-size:15px'>"
                f"({r['ticker']})</span> {tag}</div>"
                f"<div style='font-size:17px;color:{TEXT_MAIN};margin:4px 0'>사유: <b>{reason}</b></div>"
                f"<div style='font-size:16px;color:{TEXT_MUTED}'>{detail}</div>",
                unsafe_allow_html=True,
            )
            if cols[1].button("매도확정", key=f"sell_{account}_{r['ticker']}_{i}",
                              type="primary", width="stretch"):
                sell_dialog(account, as_of, r["ticker"], r["name"],
                            r["cur"], mode, qty)


def render_main_tab(store, as_of, account, state, is_bull,
                    holdings_rows, buy_cands, pyr_cands):
    """탭1 레이아웃(좌우 2열 · 2단 구성):
      · 1단 좌우 2칸: [신규매수 후보] | [보유종목 표]
      · 2단 좌우 2칸: [매도해야 하는 종목] | [불타기해야 하는 종목]
    (KOSPI 차트는 사이드바.)"""
    free_slots = NUM_SLOTS - len(state["positions"])

    # 1단: 좌 신규매수 후보(40%) / 우 보유종목(60%)
    col_buy, col_hold = st.columns([4, 6])
    with col_buy:
        render_buy_candidates(account, as_of, buy_cands, free_slots, is_bull)
    with col_hold:
        render_holdings(account, as_of, holdings_rows)

    st.divider()

    # 2단: 좌 매도해야 하는 종목 / 우 불타기해야 하는 종목
    col_sell, col_pyr = st.columns(2)
    with col_sell:
        render_sell_signals(account, as_of, holdings_rows)
    with col_pyr:
        render_pyramid_signals(account, as_of, pyr_cands, is_bull)

    with st.expander("최근 거래내역"):
        log = state.get("trade_log", [])
        if log:
            st.dataframe(pd.DataFrame(log[::-1]), width="stretch", hide_index=True)
        else:
            st.write("거래내역 없음")


def render_backtest_tab():
    """탭2: 백테스트 리포트 (기존 그대로 — 손대지 않음)."""
    st.subheader("백테스트 리포트 (backtest_v3.py 결과)")
    eq_path = os.path.join(V3_RESULT_DIR, "equity.csv")
    yr_path = os.path.join(V3_RESULT_DIR, "yearly.csv")
    sm_path = os.path.join(V3_RESULT_DIR, "summary.json")
    if not os.path.exists(sm_path):
        st.warning("백테스트 결과가 없습니다. 터미널에서 `python backtest_v3.py`를 먼저 실행하세요.")
        return
    with open(sm_path, encoding="utf-8") as f:
        sm = json.load(f)
    c = st.columns(4)
    c[0].metric("누적수익률", pct(sm["total_ret"]))
    c[1].metric("CAGR", pct(sm["CAGR"]))
    c[2].metric("MDD", pct(sm["MDD"]))
    c[3].metric("Sharpe", f"{sm['Sharpe']:.2f}")
    c = st.columns(4)
    c[0].metric("Calmar", f"{sm['Calmar']:.2f}")
    c[1].metric("손익비", f"{sm['PL']:.2f}")
    c[2].metric("승률", pct(sm["win_rate"]))
    c[3].metric("거래(신규/불타기/매도)", f"{sm['n_buy']}/{sm['n_add']}/{sm['n_sell']}")
    if os.path.exists(eq_path):
        eq = pd.read_csv(eq_path, index_col=0, parse_dates=True)
        st.markdown("**자산곡선**")
        st.line_chart(eq["total"])
    if os.path.exists(yr_path):
        yr = pd.read_csv(yr_path, index_col=0)
        st.markdown("**연도별 수익률**")
        st.dataframe((yr * 100).round(1).astype(str) + "%", width="stretch")


# ---------------------------------------------------------------------------
# 화면 진입점
# ---------------------------------------------------------------------------
def main():
    inject_theme()   # 전체 배경 남색 다크 테마

    # 오늘까지 최신 데이터 자동 갱신 (하루 1회, 세션당 1회)
    if not st.session_state.get("data_refreshed"):
        with st.spinner("최신 시장데이터 확인/갱신 중…"):
            try:
                changed = data_layer.ensure_fresh()
            except Exception as e:
                changed = False
                st.warning(f"자동 데이터 갱신 실패 — 기존 캐시로 표시합니다. ({e})")
        if changed:
            load_store.clear()
        st.session_state["data_refreshed"] = True

    store = load_store()
    last_day = store.trading_days[-1]

    # ---- 사이드바: 기준일 → 계좌 선택 → (그 아래 세로로) 요약 카드 → 시장상태 ----
    with st.sidebar:
        st.header("📈 SF 투자법")
        as_of = st.date_input("기준일", value=last_day.date(),
                              min_value=store.trading_days[0].date(),
                              max_value=last_day.date())
        as_of = pd.Timestamp(as_of)

        # 계좌 선택 라디오 (토글하면 화면 전체가 그 계좌 데이터로 전환)
        acc_label = st.radio(
            "계좌 선택",
            ["계좌 A (성장)", "계좌 B (현금흐름)"],
            key="account_sel",
        )
        account = "A" if acc_label.startswith("계좌 A") else "B"

        # 계좌별 데이터 조립 (사이드바 카드 + 메인 3단 리스트에서 공용)
        state = ps.load_portfolio(account)
        kc, km = store.get_kospi_index(as_of)
        is_bull = is_bull_market(kc, km)
        holdings_rows = evaluate_holdings(store, state, as_of)
        buy_cands = build_buy_candidates(store, state, as_of, is_bull)
        pyr_cands = build_pyramid_candidates(store, state, as_of, is_bull)

        # 요약 카드 (계좌 라디오 바로 아래 세로 스택)
        render_summary_cards(store, state, holdings_rows, is_bull)

        # KOSPI vs 200일선 차트(사이드바로 이동, 강세장/약세장 텍스트 포함)
        st.divider()
        render_market_chart(store, as_of)

        st.caption(f"최신데이터 {last_day.date()} · 운영자금 기준 {eok(ps.INITIAL_CASH)}")
        st.caption("자동매매 아님 — 신호 확인 후 증권사 앱에서 직접 체결하고 '확정' 버튼을 누르세요.")

    tab_main, tab_bt = st.tabs(["📋 오늘의 매매", "📊 백테스트 리포트"])
    with tab_main:
        render_main_tab(store, as_of, account, state, is_bull,
                        holdings_rows, buy_cands, pyr_cands)
    with tab_bt:
        render_backtest_tab()

    # 하단 고정 면책 (정직성 §7)
    st.divider()
    st.caption(
        "⚠️ 데이터 한계: 유니버스가 '현재 상장사' 기준이라 상장폐지 종목이 빠지는 "
        "**생존편향**이 있습니다. 백테스트 수치는 **참고용**이며 실거래 시 슬리피지·세금으로 낮아질 수 있습니다. "
        "모든 재무는 공시 시점 이후에만 사용해 룩어헤드를 제거했습니다."
    )


if __name__ == "__main__":
    main()
