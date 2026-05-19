import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import base64
from datetime import datetime, date
import FinanceDataReader as fdr
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 기본 설정
# ============================================================
st.set_page_config(
    page_title="52주 신고가 전략",
    page_icon="📈",
    layout="wide"
)

GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
GITHUB_REPO  = st.secrets["GITHUB_REPO"]
USERS        = dict(st.secrets["users"])

TOP_N      = 300
HIGH_RATIO = 0.90
WEEKS_52   = 252
HOLD_COUNT = 30

# ============================================================
# GitHub 데이터 저장/불러오기
# ============================================================
def github_get(filename):
    """GitHub에서 파일 불러오기"""
    url     = f"https://api.github.com/repos/{GITHUB_REPO}/contents/data/{filename}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    res     = requests.get(url, headers=headers)
    if res.status_code == 200:
        content = res.json()["content"]
        data    = json.loads(base64.b64decode(content).decode("utf-8"))
        return data, res.json()["sha"]
    return {}, None

def github_save(filename, data, sha=None):
    """GitHub에 파일 저장"""
    url     = f"https://api.github.com/repos/{GITHUB_REPO}/contents/data/{filename}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    content = base64.b64encode(json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")).decode("utf-8")
    body    = {
        "message": f"Update {filename}",
        "content": content
    }
    if sha:
        body["sha"] = sha
    requests.put(url, headers=headers, json=body)

def load_user_data(user_id):
    """사용자 데이터 불러오기"""
    data, sha = github_get(f"{user_id}.json")
    if not data:
        data = {
            "user_id"       : user_id,
            "initial_capital": 0,
            "holdings"      : [],   # 보유 종목
            "trades"        : []    # 거래 내역
        }
    return data, sha

def save_user_data(user_id, data, sha):
    """사용자 데이터 저장"""
    github_save(f"{user_id}.json", data, sha)

# ============================================================
# 30종목 자동 선정
# ============================================================
@st.cache_data(ttl=3600)  # 1시간 캐시
def get_top30():
    """오늘 기준 52주 신고가 상위 30종목"""
    try:
        kospi  = fdr.StockListing('KOSPI')
        kosdaq = fdr.StockListing('KOSDAQ')
        combined = pd.concat([kospi, kosdaq], ignore_index=True)
        combined = combined[combined['Code'].str.match(r'^\d{6}$')]
        combined = combined[combined['Code'].str.endswith('0')]
        combined = combined.drop_duplicates(subset='Code')

        cap_col = None
        for col in ['Marcap', 'MarCap', 'marcap', 'market_cap', 'MarketCap', 'Cap']:
            if col in combined.columns:
                cap_col = col
                break

        if cap_col:
            combined = combined[combined[cap_col] > 0]
            combined = combined.sort_values(cap_col, ascending=False).head(TOP_N)

        ticker_list = combined['Code'].tolist()
        name_dict   = dict(zip(combined['Code'], combined['Name']))

        end_date   = datetime.today().strftime('%Y-%m-%d')
        start_date = str(int(end_date[:4]) - 1) + end_date[4:]

        scores = {}
        prices = {}
        high52 = {}
        for code in ticker_list:
            try:
                df = fdr.DataReader(code, start_date, end_date)
                if df is None or len(df) < 60:
                    continue
                current  = df['Close'].iloc[-1]
                h52w     = df['Close'].tail(WEEKS_52).max()
                if h52w <= 0 or current <= 0:
                    continue
                ratio = current / h52w
                if ratio >= HIGH_RATIO:
                    scores[code] = ratio
                    prices[code] = current
                    high52[code] = h52w
            except:
                continue

        top30 = sorted(scores, key=scores.get, reverse=True)[:HOLD_COUNT]

        result = []
        for rank, code in enumerate(top30, 1):
            result.append({
                'rank'   : rank,
                'code'   : code,
                'name'   : name_dict.get(code, code),
                'price'  : prices[code],
                'high52' : high52[code],
                'ratio'  : round(scores[code] * 100, 1)
            })
        return result
    except Exception as e:
        st.error(f"종목 조회 오류: {e}")
        return []

# ============================================================
# 수익률 계산
# ============================================================
def calc_profit(data):
    """수익률 계산"""
    initial   = data.get('initial_capital', 0)
    holdings  = data.get('holdings', [])
    trades    = data.get('trades', [])

    # 실현 손익
    realized = sum(t.get('profit', 0) for t in trades if t.get('type') == 'sell')

    # 평가 손익 (현재가 기준)
    unrealized = 0
    if holdings:
        codes = [h['code'] for h in holdings]
        for h in holdings:
            try:
                df      = fdr.DataReader(h['code'],
                          datetime.today().strftime('%Y-%m-%d'),
                          datetime.today().strftime('%Y-%m-%d'))
                if df is not None and len(df) > 0:
                    current_price = df['Close'].iloc[-1]
                else:
                    current_price = h['buy_price']
                unrealized += (current_price - h['buy_price']) * h['quantity']
                h['current_price'] = current_price
                h['eval_profit']   = round((current_price - h['buy_price']) * h['quantity'], 0)
                h['return_pct']    = round((current_price / h['buy_price'] - 1) * 100, 1)
            except:
                h['current_price'] = h['buy_price']
                h['eval_profit']   = 0
                h['return_pct']    = 0.0

    total_profit = realized + unrealized
    return_pct   = (total_profit / initial * 100) if initial > 0 else 0

    # 이번 달 수익률
    this_month = datetime.today().strftime('%Y-%m')
    month_sells = [t for t in trades
                   if t.get('type') == 'sell' and t.get('date', '').startswith(this_month)]
    month_profit = sum(t.get('profit', 0) for t in month_sells)
    month_pct    = (month_profit / initial * 100) if initial > 0 else 0

    return {
        'initial'     : initial,
        'realized'    : round(realized, 0),
        'unrealized'  : round(unrealized, 0),
        'total_profit': round(total_profit, 0),
        'return_pct'  : round(return_pct, 1),
        'month_pct'   : round(month_pct, 1),
        'month_profit': round(month_profit, 0),
        'holdings'    : holdings
    }

# ============================================================
# 로그인 화면
# ============================================================
def login_page():
    st.title("📈 52주 신고가 전략")
    st.subheader("로그인")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        user_id  = st.text_input("아이디")
        password = st.text_input("비밀번호", type="password")

        if st.button("로그인", use_container_width=True):
            if user_id in USERS and USERS[user_id] == password:
                st.session_state['logged_in'] = True
                st.session_state['user_id']   = user_id
                st.rerun()
            else:
                st.error("아이디 또는 비밀번호가 틀립니다.")

# ============================================================
# 메인 화면
# ============================================================
def main_page():
    user_id = st.session_state['user_id']

    # 사이드바
    with st.sidebar:
        st.title("📈 52주 신고가")
        st.write(f"👤 {user_id}")
        st.divider()
        menu = st.radio("메뉴", [
            "📊 대시보드",
            "🏆 이번 달 30종목",
            "💼 매수 입력",
            "💰 매도 입력",
            "📋 거래 내역",
            "⚙️ 초기 투자금 설정"
        ])
        st.divider()
        if st.button("로그아웃"):
            st.session_state.clear()
            st.rerun()

    # 사용자 데이터 불러오기
    data, sha = load_user_data(user_id)

    # ==========================================
    if menu == "📊 대시보드":
        st.title("📊 대시보드")

        profit = calc_profit(data)

        if profit['initial'] == 0:
            st.warning("⚙️ 메뉴에서 초기 투자금을 먼저 설정해주세요.")
        else:
            # 수익률 박스
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                color = "🟢" if profit['return_pct'] >= 0 else "🔴"
                st.metric("누적 수익률", f"{color} {profit['return_pct']}%")
            with col2:
                st.metric("이번 달 수익률", f"{profit['month_pct']}%",
                          f"{profit['month_profit']:+,.0f}원")
            with col3:
                st.metric("누적 손익",
                          f"{profit['total_profit']:+,.0f}원")
            with col4:
                st.metric("실현 손익",
                          f"{profit['realized']:+,.0f}원")

            st.divider()

            # 보유 종목
            st.subheader("💼 보유 종목")
            holdings = profit['holdings']
            if holdings:
                df_h = pd.DataFrame(holdings)
                df_h = df_h[['name', 'code', 'buy_price', 'quantity',
                              'current_price', 'eval_profit', 'return_pct']]
                df_h.columns = ['종목명', '코드', '매수가', '수량',
                                 '현재가', '평가손익', '수익률(%)']
                df_h['매수가']   = df_h['매수가'].apply(lambda x: f"{x:,.0f}")
                df_h['현재가']   = df_h['현재가'].apply(lambda x: f"{x:,.0f}")
                df_h['평가손익'] = df_h['평가손익'].apply(lambda x: f"{x:+,.0f}")
                df_h['수익률(%)'] = df_h['수익률(%)'].apply(
                    lambda x: f"{'🟢' if x >= 0 else '🔴'} {x}%")
                st.dataframe(df_h, use_container_width=True, hide_index=True)
            else:
                st.info("보유 종목이 없습니다.")

    # ==========================================
    elif menu == "🏆 이번 달 30종목":
        st.title("🏆 이번 달 30종목")
        st.caption(f"기준일: {datetime.today().strftime('%Y년 %m월 %d일')} | "
                   f"KOSPI+KOSDAQ 시총 상위 {TOP_N}종목 중 52주 신고가 90% 이상")

        with st.spinner("종목 계산 중... (약 10~20분 소요)"):
            top30 = get_top30()

        if top30:
            df_top = pd.DataFrame(top30)
            df_top.columns = ['순위', '코드', '종목명', '현재가', '52주최고가', '비율(%)']
            df_top['현재가']    = df_top['현재가'].apply(lambda x: f"{x:,.0f}")
            df_top['52주최고가'] = df_top['52주최고가'].apply(lambda x: f"{x:,.0f}")
            df_top['비율(%)']   = df_top['비율(%)'].apply(lambda x: f"{x}%")
            st.dataframe(df_top, use_container_width=True, hide_index=True)
            st.caption("※ 매월 첫 거래일 기준으로 매수하세요.")
        else:
            st.error("종목 조회 실패. 잠시 후 다시 시도해주세요.")

    # ==========================================
    elif menu == "💼 매수 입력":
        st.title("💼 매수 입력")

        col1, col2 = st.columns(2)
        with col1:
            code      = st.text_input("종목코드 (예: 005930)")
            name      = st.text_input("종목명 (예: 삼성전자)")
            buy_price = st.number_input("매수가 (원)", min_value=0, step=1)
        with col2:
            quantity  = st.number_input("수량 (주)", min_value=0, step=1)
            buy_date  = st.date_input("매수일", value=date.today())
            buy_amount = buy_price * quantity
            st.metric("매수 금액", f"{buy_amount:,.0f}원")

        if st.button("매수 저장", use_container_width=True):
            if code and name and buy_price > 0 and quantity > 0:
                new_holding = {
                    'code'     : code,
                    'name'     : name,
                    'buy_price': buy_price,
                    'quantity' : quantity,
                    'buy_date' : str(buy_date),
                    'buy_amount': buy_amount
                }
                data['holdings'].append(new_holding)
                data['trades'].append({
                    'type'  : 'buy',
                    'code'  : code,
                    'name'  : name,
                    'price' : buy_price,
                    'quantity': quantity,
                    'amount': buy_amount,
                    'date'  : str(buy_date)
                })
                save_user_data(user_id, data, sha)
                st.success(f"✅ {name} 매수 저장 완료!")
            else:
                st.error("모든 항목을 입력해주세요.")

    # ==========================================
    elif menu == "💰 매도 입력":
        st.title("💰 매도 입력")

        holdings = data.get('holdings', [])
        if not holdings:
            st.info("보유 종목이 없습니다.")
        else:
            holding_names = [f"{h['name']} ({h['code']})" for h in holdings]
            selected = st.selectbox("매도 종목 선택", holding_names)
            sel_idx  = holding_names.index(selected)
            sel_hold = holdings[sel_idx]

            col1, col2 = st.columns(2)
            with col1:
                st.info(f"매수가: {sel_hold['buy_price']:,.0f}원 | "
                        f"수량: {sel_hold['quantity']:,}주")
                sell_price = st.number_input("매도가 (원)", min_value=0, step=1)
            with col2:
                sell_date  = st.date_input("매도일", value=date.today())
                if sell_price > 0:
                    profit     = (sell_price - sel_hold['buy_price']) * sel_hold['quantity']
                    return_pct = (sell_price / sel_hold['buy_price'] - 1) * 100
                    color      = "🟢" if profit >= 0 else "🔴"
                    st.metric("손익", f"{color} {profit:+,.0f}원",
                              f"{return_pct:+.1f}%")

            if st.button("매도 저장", use_container_width=True):
                if sell_price > 0:
                    profit = (sell_price - sel_hold['buy_price']) * sel_hold['quantity']
                    data['trades'].append({
                        'type'      : 'sell',
                        'code'      : sel_hold['code'],
                        'name'      : sel_hold['name'],
                        'buy_price' : sel_hold['buy_price'],
                        'sell_price': sell_price,
                        'quantity'  : sel_hold['quantity'],
                        'profit'    : profit,
                        'date'      : str(sell_date)
                    })
                    data['holdings'].pop(sel_idx)
                    save_user_data(user_id, data, sha)
                    st.success(f"✅ {sel_hold['name']} 매도 저장 완료! "
                               f"손익: {profit:+,.0f}원")
                else:
                    st.error("매도가를 입력해주세요.")

    # ==========================================
    elif menu == "📋 거래 내역":
        st.title("📋 거래 내역")

        trades = data.get('trades', [])
        if not trades:
            st.info("거래 내역이 없습니다.")
        else:
            df_t = pd.DataFrame(trades)
            df_t = df_t.fillna('')

            buy_df  = df_t[df_t['type'] == 'buy']
            sell_df = df_t[df_t['type'] == 'sell']

            st.subheader("매수 내역")
            if len(buy_df) > 0:
                show_buy = buy_df[['date', 'name', 'code', 'price', 'quantity', 'amount']].copy()
                show_buy.columns = ['날짜', '종목명', '코드', '매수가', '수량', '금액']
                show_buy['매수가'] = show_buy['매수가'].apply(lambda x: f"{x:,.0f}")
                show_buy['금액']   = show_buy['금액'].apply(lambda x: f"{x:,.0f}")
                st.dataframe(show_buy, use_container_width=True, hide_index=True)

            st.subheader("매도 내역")
            if len(sell_df) > 0:
                show_sell = sell_df[['date', 'name', 'code',
                                     'buy_price', 'sell_price', 'quantity', 'profit']].copy()
                show_sell.columns = ['날짜', '종목명', '코드', '매수가', '매도가', '수량', '손익']
                show_sell['매수가'] = show_sell['매수가'].apply(lambda x: f"{x:,.0f}")
                show_sell['매도가'] = show_sell['매도가'].apply(lambda x: f"{x:,.0f}")
                show_sell['손익']   = show_sell['손익'].apply(
                    lambda x: f"{'🟢' if x >= 0 else '🔴'} {x:+,.0f}")
                st.dataframe(show_sell, use_container_width=True, hide_index=True)

    # ==========================================
    elif menu == "⚙️ 초기 투자금 설정":
        st.title("⚙️ 초기 투자금 설정")

        current = data.get('initial_capital', 0)
        st.info(f"현재 설정된 초기 투자금: {current:,.0f}원")

        new_capital = st.number_input(
            "초기 투자금 (원)", min_value=0, step=1_000_000,
            value=current
        )

        if st.button("저장", use_container_width=True):
            data['initial_capital'] = new_capital
            save_user_data(user_id, data, sha)
            st.success(f"✅ 초기 투자금 {new_capital:,.0f}원 저장 완료!")

# ============================================================
# 실행
# ============================================================
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    login_page()
else:
    main_page()
