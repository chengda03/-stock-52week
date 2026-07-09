# -*- coding: utf-8 -*-
"""
app_s1.py — "ROE×EPS 모멘텀 불타기(S1)" 실운영 신호 앱
=====================================================
strategy_core.py(전략 판정) + data_layer.py(데이터) + portfolio_state.py(보유현황)
+ backtest_v3.py(백테스트 결과)를 조합한 Streamlit 앱.

실행:  streamlit run app_s1.py

※ 기존 52주신고가 앱(app.py)과는 별개 파일입니다(전략이 다름). 서로 영향 없음.

3개 탭:
  탭1 오늘의 신호     : 오늘자 데이터로 매도/불타기/신규매수 신호 + 체결완료 버튼
  탭2 보유 포트폴리오 : 보유종목 현황(평단가/손절선/90일선/수익률/리밸런싱 알림)
  탭3 백테스트 리포트 : backtest_v3 결과(자산곡선/연도별 수익률)

정직성 원칙(지시문 §7): 룩어헤드 없음, 생존편향 존재 명시, 백테스트는 참고용.
"""
from __future__ import annotations

import json
import os
from datetime import date, datetime

import pandas as pd
import streamlit as st

import data_layer
import portfolio_state as ps
from strategy_core import (
    Position,
    is_bull_market, check_buy_conditions, rank_by_momentum,
    check_sell_condition, check_pyramid, check_partial_exit,
    NUM_SLOTS, SLOT_AMOUNT_WON, PYRAMID_ADD_AMOUNT_WON,
    HARD_STOP_LOSS_PCT, SELL_MA_DAYS,
    PARTIAL_EXIT_TRIGGER_PCT, PARTIAL_EXIT_RATIO,
)

BUY_COST = 0.00115
V3_RESULT_DIR = os.path.join("data", "cache_v3", "results")

st.set_page_config(page_title="S1 불타기 전략", page_icon="📈", layout="wide")


# ---------------------------------------------------------------------------
# 데이터 로딩 (무거우므로 1회만 — Streamlit 캐시)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="데이터 로딩 중… (최초 1회, 30초 내외)")
def load_store():
    return data_layer.get_store()


def won(x) -> str:
    """천단위 콤마 원화 표기."""
    try:
        return f"{float(x):,.0f}원"
    except Exception:
        return "-"


def pct(x) -> str:
    try:
        return f"{float(x)*100:+.1f}%"
    except Exception:
        return "-"


# ---------------------------------------------------------------------------
# 평가/신호 계산
# ---------------------------------------------------------------------------
def evaluate_portfolio(store, state: dict, as_of):
    """보유종목 평가금액 합계와 종목별 현재가를 계산."""
    holdings_val = 0.0
    rows = []
    for ticker, pos in state["positions"].items():
        snap = store.get_price_snapshot(ticker, as_of)
        cur = snap.price if snap else pos["avg_price"]
        val = cur * pos["shares"]
        holdings_val += val
        rows.append((ticker, pos, snap, cur, val))
    return holdings_val, rows


def build_signals(store, state: dict, as_of, is_bull: bool):
    """오늘자 매도/불타기/신규매수 신호 리스트 생성."""
    signals = []
    cash = state["cash"]
    positions = state["positions"]

    # 1) 매도 신호 (항상) — 전량매도 먼저, 아니면 부분익절 확인 (호출 순서 규칙 준수)
    for ticker, pos_d in positions.items():
        snap = store.get_price_snapshot(ticker, as_of)
        if snap is None:
            continue
        pos = ps.dict_to_position(ticker, pos_d)
        should_sell, reason = check_sell_condition(pos, snap)
        if should_sell:
            signals.append({
                "구분": "매도", "종목코드": ticker,
                "종목명": store.name_map.get(ticker, ticker),
                "현재가": snap.price, "수량": pos.shares,
                "금액": snap.price * pos.shares * (1 - 0.00295),
                "사유": reason,
            })
        elif check_partial_exit(pos, snap.price):
            # 전량매도가 아닌 종목만 부분익절 신호 (보유수량의 50%)
            sell_sh = pos.shares * PARTIAL_EXIT_RATIO
            signals.append({
                "구분": "부분익절", "종목코드": ticker,
                "종목명": store.name_map.get(ticker, ticker),
                "현재가": snap.price, "수량": sell_sh,
                "금액": snap.price * sell_sh * (1 - 0.00295),
                "사유": f"+{PARTIAL_EXIT_TRIGGER_PCT*100:.0f}% 부분익절 트리거",
            })

    # 2) 불타기 신호 (강세장, 수익률 높은 순)
    if is_bull:
        def cur_ret(t):
            s = store.get_price_snapshot(t, as_of)
            return (s.price / positions[t]["avg_price"]) if s else -1.0
        for ticker in sorted(positions.keys(), key=cur_ret, reverse=True):
            snap = store.get_price_snapshot(ticker, as_of)
            if snap is None:
                continue
            pos = ps.dict_to_position(ticker, positions[ticker])
            if check_pyramid(pos, snap.price, as_of if isinstance(as_of, date) else as_of.date(),
                             is_bull, cash):
                signals.append({
                    "구분": "불타기", "종목코드": ticker,
                    "종목명": store.name_map.get(ticker, ticker),
                    "현재가": snap.price,
                    "수량": PYRAMID_ADD_AMOUNT_WON / snap.price,
                    "금액": PYRAMID_ADD_AMOUNT_WON * (1 + BUY_COST),
                    "사유": f"{pos.pyramid_count + 1}단계 +3% 트리거 도달",
                })
                cash -= PYRAMID_ADD_AMOUNT_WON * (1 + BUY_COST)

    # 3) 신규매수 신호 (강세장, 빈 슬롯만큼)
    free = NUM_SLOTS - len(positions)
    if is_bull and free > 0:
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
        for snap in rank_by_momentum(cands)[:free]:
            sh = int(SLOT_AMOUNT_WON // snap.price)
            if sh <= 0:
                continue
            signals.append({
                "구분": "신규매수", "종목코드": snap.ticker,
                "종목명": store.name_map.get(snap.ticker, snap.ticker),
                "현재가": snap.price, "수량": float(sh),
                "금액": sh * snap.price * (1 + BUY_COST),
                "사유": f"6조건 통과 (모멘텀 {snap.momentum_20d*100:.1f}%)",
            })
    return signals


def apply_signal(state, sig, as_of):
    """'체결완료' — 신호 하나를 포트폴리오에 반영."""
    d = as_of if isinstance(as_of, date) else as_of.date()
    ps.record_trade(state, sig["종목코드"], sig["구분"], price=sig["현재가"],
                    shares=sig["수량"], date_=d, reason=sig["사유"],
                    amount=sig["금액"])
    ps.save_portfolio(state)


# ---------------------------------------------------------------------------
# 화면
# ---------------------------------------------------------------------------
def main():
    # 오늘까지 최신 데이터 자동 갱신 (하루 1회, 세션당 1회) — 없으면 증분 다운로드
    if not st.session_state.get("data_refreshed"):
        with st.spinner("최신 시장데이터 확인/갱신 중… (오늘 처음이면 증분 다운로드로 수 분 소요될 수 있음)"):
            try:
                changed = data_layer.ensure_fresh()
            except Exception as e:
                changed = False
                st.warning(f"자동 데이터 갱신 실패 — 기존 캐시로 표시합니다. ({e})")
        if changed:
            load_store.clear()   # 새 데이터 반영 위해 캐시된 store 무효화
        st.session_state["data_refreshed"] = True

    store = load_store()
    last_day = store.trading_days[-1]

    # 사이드바 -------------------------------------------------------------
    with st.sidebar:
        st.header("📈 S1 불타기 전략")
        as_of = st.date_input("기준일 (데이터 보유 범위 내)", value=last_day.date(),
                              min_value=store.trading_days[0].date(),
                              max_value=last_day.date())
        as_of = pd.Timestamp(as_of)

        state = ps.load_portfolio()
        kc, km = store.get_kospi_index(as_of)
        is_bull = is_bull_market(kc, km)

        holdings_val, hold_rows = evaluate_portfolio(store, state, as_of)
        total = state["cash"] + holdings_val
        profit = total - ps.INITIAL_CASH

        st.divider()
        st.metric("총자산", won(total), pct(total / ps.INITIAL_CASH - 1))
        st.metric("평가액(보유)", won(holdings_val))
        st.metric("보유 현금", won(state["cash"]))
        st.metric("누적수익", won(profit))

        # 최대 집중(단일종목 최대비중) — 쏠림 수동관리용, 크게 표시
        if hold_rows and total > 0:
            top_t, _tp, _ts, _tc, top_val = max(hold_rows, key=lambda r: r[4])
            top_w = top_val / total
            top_name = store.name_map.get(top_t, top_t)
            if top_w >= 0.50:
                bg, fg, icon = "#ffe3e3", "#c92a2a", "🔴"
            elif top_w >= 0.30:
                bg, fg, icon = "#fff3bf", "#e67700", "🟡"
            else:
                bg, fg, icon = "#ebfbee", "#2b8a3e", "🟢"
            st.markdown(
                f"<div style='background:{bg};border-radius:10px;padding:12px 14px;margin-top:6px'>"
                f"<div style='font-size:0.85rem;color:{fg};font-weight:600'>{icon} 최대 집중</div>"
                f"<div style='font-size:1.5rem;color:{fg};font-weight:800;line-height:1.3'>"
                f"{top_name} {top_w*100:.1f}%</div>"
                f"<div style='font-size:0.75rem;color:{fg}'>{won(top_val)}</div></div>",
                unsafe_allow_html=True,
            )
        st.divider()
        if is_bull:
            st.success(f"🟢 강세장 (KOSPI {kc:,.0f} > 200일선 {km:,.0f})")
        else:
            st.error(f"🔴 약세장 (KOSPI {kc:,.0f} ≤ 200일선 {km:,.0f}) — 신규매수·불타기 중단")
        st.caption(f"운영자금 기준 {won(ps.INITIAL_CASH)} · 최신데이터 {last_day.date()}")

    tab1, tab2, tab3 = st.tabs(["🔔 오늘의 신호", "💼 보유 포트폴리오", "📊 백테스트 리포트"])

    # 탭1: 오늘의 신호 -----------------------------------------------------
    with tab1:
        st.subheader(f"{as_of.date()} 신호")
        if not is_bull:
            st.warning("약세장이라 신규매수·불타기는 나오지 않습니다. 매도 신호만 표시됩니다.")
        signals = build_signals(store, state, as_of, is_bull)
        if not signals:
            st.info("오늘 발생한 신호가 없습니다.")
        else:
            color = {"매도": "🔵", "부분익절": "🟢", "불타기": "🔴", "신규매수": "🟠"}
            for i, sig in enumerate(signals):
                c1, c2, c3, c4 = st.columns([1.2, 3, 3, 1.4])
                c1.markdown(f"{color.get(sig['구분'],'')} **{sig['구분']}**")
                c2.write(f"{sig['종목명']} ({sig['종목코드']})")
                c3.write(f"{won(sig['현재가'])} × {sig['수량']:.0f}주 · {sig['사유']}")
                if c4.button("체결완료", key=f"exec_{i}_{sig['종목코드']}"):
                    apply_signal(state, sig, as_of)
                    st.success(f"{sig['종목명']} {sig['구분']} 체결 기록됨")
                    st.rerun()
        st.caption("※ 체결가는 '기준일 종가'로 기록됩니다. 실거래 슬리피지/세금은 별도.")

    # 탭2: 보유 포트폴리오 -------------------------------------------------
    with tab2:
        st.subheader("보유 종목 현황")
        holdings_val, rows = evaluate_portfolio(store, state, as_of)
        if not rows:
            st.info("보유 종목이 없습니다. (현금 100%)")
        else:
            total_assets = state["cash"] + holdings_val
            # 비중(평가액/총자산) 높은 순으로 정렬
            rows_sorted = sorted(rows, key=lambda r: r[4], reverse=True)
            table = []
            for ticker, pos, snap, cur, val in rows_sorted:
                stop_line = pos["avg_price"] * (1 - HARD_STOP_LOSS_PCT)
                ma_sell = snap.ma_sell if snap else float("nan")
                entry_d = pos.get("entry_date")
                # 리밸런싱 알림: 진입 후 1개월 경과
                rebal = ""
                if entry_d:
                    days = (as_of.date() - date.fromisoformat(entry_d)).days
                    if days >= 30:
                        rebal = f"🔔 {days}일 경과"
                # 비중 + 쏠림 경고 마커
                weight = val / total_assets if total_assets > 0 else 0.0
                mark = "🔴" if weight >= 0.50 else ("🟡" if weight >= 0.30 else "")
                weight_str = f"{mark} {weight*100:.1f}%".strip()
                table.append({
                    "종목": f"{store.name_map.get(ticker, ticker)} ({ticker})",
                    "비중": weight_str,
                    "진입일": entry_d or "-",
                    "진입가": f"{pos['entry_price']:,.0f}",
                    "현재가": f"{cur:,.0f}",
                    "평단가": f"{pos['avg_price']:,.0f}",
                    "불타기": pos.get("pyramid_count", 0),
                    "부분익절여부": "✅ 완료" if pos.get("partial_exit_done", False) else "⏳ 대기",
                    "수익률": pct(cur / pos["avg_price"] - 1),
                    "손절선(-12%)": f"{stop_line:,.0f}",
                    f"{SELL_MA_DAYS}일선": f"{ma_sell:,.0f}" if pd.notna(ma_sell) else "-",
                    "평가액": f"{val:,.0f}",
                    "알림": rebal,
                })
            st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)
            st.caption("비중 = 현재가 × 보유수량 ÷ 총자산(현금+평가액). 🟡 30% 이상 / 🔴 50% 이상은 단일종목 쏠림 경고. "
                       "이 전략은 종목당 비중 상한이 없는 집중투자형이므로, 쏠림이 부담되면 이 화면을 보고 직접 판단해 수동 매도하세요. "
                       "손절선 = 평단가 × 0.88 (하드손절 -12%). 90일선 이탈 또는 손절선 도달 시 전량매도. "
                       "부분익절 = 진입가 +8% 도달 시 보유수량 50% 매도(종목당 1회, 나머지 50% 보유).")

        with st.expander("최근 거래내역"):
            log = state.get("trade_log", [])
            if log:
                st.dataframe(pd.DataFrame(log[::-1]), use_container_width=True, hide_index=True)
            else:
                st.write("거래내역 없음")

    # 탭3: 백테스트 리포트 -------------------------------------------------
    with tab3:
        st.subheader("백테스트 리포트 (backtest_v3.py 결과)")
        eq_path = os.path.join(V3_RESULT_DIR, "equity.csv")
        yr_path = os.path.join(V3_RESULT_DIR, "yearly.csv")
        sm_path = os.path.join(V3_RESULT_DIR, "summary.json")
        if not os.path.exists(sm_path):
            st.warning("백테스트 결과가 없습니다. 터미널에서 `python backtest_v3.py`를 먼저 실행하세요.")
        else:
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
                st.markdown("**자산곡선** (로그 스케일)")
                st.line_chart(eq["total"])
            if os.path.exists(yr_path):
                yr = pd.read_csv(yr_path, index_col=0)
                st.markdown("**연도별 수익률**")
                st.dataframe((yr * 100).round(1).astype(str) + "%",
                             use_container_width=True)

    # 하단 고정 면책 (지시문 §7) -------------------------------------------
    st.divider()
    st.caption(
        "⚠️ 데이터 한계: 유니버스가 '현재 상장사' 기준이라 상장폐지 종목이 빠지는 "
        "**생존편향**이 있습니다. 발행주식수는 2024 사업보고서 단일 시점(시총 근사). "
        "백테스트 수치는 **참고용**이며 실거래 시 슬리피지·세금·심리적 실행오차로 낮아질 수 있습니다. "
        "모든 재무는 공시 시점 이후에만 사용해 룩어헤드를 제거했습니다."
    )


if __name__ == "__main__":
    main()
