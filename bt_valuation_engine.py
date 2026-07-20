# -*- coding: utf-8 -*-
"""
bt_valuation_engine.py
======================
저평가 필터 교체 실험(base / PER / debt / per_debt)을 '공정하게' 비교하기 위한
공용 백테스트 엔진.

★ 왜 공용 엔진인가?
  backtest_v3.py를 4번 복사하면 엔진 코드가 4벌로 갈라져(오타·미세수정) 비교가
  오염될 수 있다. 그래서 엔진은 이 파일 하나로 고정하고, 변형은 '매수조건 판정
  함수(check_buy_conditions)만' 갈아끼운다. 매도/불타기/부분익절/사이징/비용은
  전부 strategy_core.py(원본)와 동일한 규칙을 그대로 쓴다.
  → 4개 결과의 차이는 오로지 '저평가 필터' 차이에서만 나온다.

엔진 규칙은 backtest_v3.py와 100% 동일:
  - 기간 2020-01-02 ~ 데이터최신, 1억, 매수 0.115% / 매도 0.295%
  - 강세장(KOSPI 200일선 위)에서만 신규매수/불타기
  - 전량매도(90일선/-12%) → 부분익절(+8%@50%) → 불타기(+3%복리) → 신규매수(빈 슬롯)
  - 슬롯 20개, 슬롯당 500만원(정수주)

추가 계측(비교표용, 매매에는 영향 없음):
  - immediate_sell : 매수 당일 종가 기준 매도조건이 이미 켜진 건수
  - max_weight     : 일별 (최대 단일종목 평가액 / 총자산)의 최댓값
  - pass_rate      : 강세장일(day)마다 '유니버스 중 매수조건 통과 종목 %'의 평균
  - avg_pass_cnt   : 위와 같은 날들의 평균 통과 종목 '개수'
  - bought_set     : 이 변형이 '신규매수'한 종목 집합(변형 간 겹침 분석용)
"""
from __future__ import annotations

import math
import os
import numpy as np
import pandas as pd

import data_layer
# 매도/불타기/부분익절/사이징 등 '매수 외' 규칙은 원본 strategy_core를 공용으로 사용
import strategy_core as sc_base
from strategy_core import StockSnapshot, Position

INITIAL_CASH = 100_000_000

# --- fin2 부가필드(영업이익률/부채비율/금융업) ------------------------------
#   debt_opmargin.parquet(fetch_debt_opmargin.py 수집분)을 (corp_code, year)로 인덱싱.
#   opmargin/debt2 변형만 이 값을 읽는다(base/PER는 무시). '그날 적용 연도'는
#   ROE/EPS와 동일한 store.day_fin_year 를 써서 룩어헤드를 맞춘다.
_FIN2_PATH = os.path.join("data", "cache_52w_bt", "debt_opmargin.parquet")
_FIN2: dict | None = None
# 금융업 판정 키워드(부채비율 조건 면제용) — data_layer_qv 와 동일 기준
_FIN_KEYWORDS = ("은행", "증권", "보험", "카드", "캐피탈", "저축은행",
                 "금융", "선물", "자산운용", "신용", "종금", "금고")


def _load_fin2() -> dict:
    """(corp_code, year) -> (debt_ratio%, op_margin%). 1회 로드 후 캐시."""
    global _FIN2
    if _FIN2 is None:
        d: dict = {}
        if os.path.exists(_FIN2_PATH):
            df = pd.read_parquet(_FIN2_PATH)
            for r in df.itertuples(index=False):
                d[(str(r.corp_code), int(r.year))] = (
                    float(r.debt_ratio) if pd.notna(r.debt_ratio) else float("nan"),
                    float(r.op_margin) if pd.notna(r.op_margin) else float("nan"),
                )
        _FIN2 = d
    return _FIN2


def _is_financial(name: str) -> bool:
    return any(k in (name or "") for k in _FIN_KEYWORDS)


def _attach_fin2(snap, store, ticker: str, di: int) -> None:
    """스냅샷에 op_margin/debt_ratio/is_financial 부착(비슬롯 dataclass라 동적 속성 가능)."""
    cc = store.code_to_corp.get(ticker)
    yr = int(store.day_fin_year[di])
    dr, om = _load_fin2().get((cc, yr), (float("nan"), float("nan")))
    snap.debt_ratio = dr
    snap.op_margin = om
    snap.is_financial = _is_financial(store.name_map.get(ticker, ""))
BUY_COST = 0.00115
SELL_COST = 0.00295
TRADE_START = pd.Timestamp("2020-01-01")


def _build_snap(store, col, di) -> StockSnapshot:
    """store 배열에서 di일·col종목의 StockSnapshot 조립 (backtest_v3._build_snap와 동일)."""
    yr = int(store.day_fin_year[di])
    roe_a = store.roe_by_year.get(yr)
    eps_a = store.eps_by_year.get(yr)
    roe = float(roe_a[col]) if roe_a is not None else float("nan")
    eps = float(eps_a[col]) if eps_a is not None else float("nan")
    return StockSnapshot(
        ticker=store.valid_codes[col],
        date=store.trading_days[di].date(),
        price=float(store.close_v[di, col]),
        ma_buy=float(store.ma_buy_v[di, col]),
        ma_sell=float(store.ma_sell_v[di, col]),
        roe_pct=roe, eps=eps,
        momentum_20d=float(store.mom_v[di, col]),
        trading_value_20d_avg=float(store.tv_v[di, col]),
    )


def _metrics(daily_total: np.ndarray, td: pd.DatetimeIndex) -> dict:
    s = pd.Series(daily_total, index=td)
    s = s[s.index >= TRADE_START]
    final = float(s.iloc[-1])
    years = (s.index[-1] - s.index[0]).days / 365.25
    cagr = (final / INITIAL_CASH) ** (1.0 / max(years, 1e-9)) - 1.0
    peak = s.cummax()
    mdd = float(((s - peak) / peak).min())
    rets = s.pct_change().fillna(0)
    sharpe = float((rets.mean() / (rets.std() + 1e-12)) * math.sqrt(252))
    calmar = cagr / abs(mdd) if mdd != 0 else float("inf")
    return {"final": final, "total_ret": final / INITIAL_CASH - 1.0,
            "CAGR": cagr, "MDD": mdd, "Sharpe": sharpe, "Calmar": calmar,
            "start": str(s.index[0].date()), "end": str(s.index[-1].date())}


def run_variant(check_buy_fn, tag: str, store=None) -> dict:
    """
    하나의 변형을 백테스트한다.
      check_buy_fn : (StockSnapshot, is_bull) -> (bool, list)  ← 변형별 매수조건 함수
      tag          : 결과 라벨
      store        : 공유 DataStore (None이면 get_store로 로드 — 같은 프로세스면 캐시 재사용)
    반환: 지표 dict
    """
    if store is None:
        store = data_layer.get_store()
    td = store.trading_days
    close_v = store.close_v
    close_ff = store.close_ff
    c2c = store.code_to_col
    start_di = store.start_di

    cash = float(INITIAL_CASH)
    positions: dict[str, Position] = {}
    cost: dict[str, float] = {}
    daily_total = np.empty(len(td))

    n_buy = n_add = n_partial = n_final = 0
    wins = 0
    gross_w = gross_l = 0.0
    immediate_sell = 0
    max_weight = 0.0
    pass_pct_sum = 0.0
    pass_cnt_sum = 0
    pass_days = 0
    bought_set: set[str] = set()

    for di in range(len(td)):
        today = td[di].date()
        if di < start_di:
            daily_total[di] = INITIAL_CASH
            continue

        is_bull = sc_base.is_bull_market(store.kospi_close_v[di], store.kospi_ma200_v[di])
        sold_today: set[str] = set()

        # 1) 전량매도
        for t in list(positions.keys()):
            col = c2c[t]; px = close_v[di, col]
            if not np.isfinite(px):
                continue
            pos = positions[t]; snap = _build_snap(store, col, di)
            s, _ = sc_base.check_sell_condition(pos, snap)
            if s:
                proceeds = px * pos.shares * (1.0 - SELL_COST)
                cash += proceeds
                pnl = proceeds - cost[t]
                if pnl > 0: wins += 1; gross_w += pnl
                else: gross_l += pnl
                n_final += 1
                del positions[t]; del cost[t]; sold_today.add(t)

        # 1-2) 부분익절 (+8%@50%)
        for t in list(positions.keys()):
            col = c2c[t]; px = close_v[di, col]
            if not np.isfinite(px) or px <= 0:
                continue
            pos = positions[t]
            if sc_base.check_partial_exit(pos, float(px)):
                before = pos.shares
                sell_sh = sc_base.apply_partial_exit(pos, float(px))
                ratio = sell_sh / before
                proceeds = px * sell_sh * (1.0 - SELL_COST)
                cost_sold = cost[t] * ratio
                pnl = proceeds - cost_sold
                cash += proceeds; cost[t] -= cost_sold
                n_partial += 1
                if pnl > 0: wins += 1; gross_w += pnl
                else: gross_l += pnl

        # 2) 불타기 (강세장, 수익률 높은 순)
        if is_bull and positions:
            def cur_ret(t):
                col = c2c[t]; px = close_v[di, col]
                return px / positions[t].avg_price if np.isfinite(px) else -1.0
            for t in sorted(positions.keys(), key=cur_ret, reverse=True):
                col = c2c[t]; px = close_v[di, col]
                if not np.isfinite(px) or px <= 0:
                    continue
                pos = positions[t]
                if sc_base.check_pyramid(pos, float(px), today, is_bull, cash):
                    spent = sc_base.PYRAMID_ADD_AMOUNT_WON * (1.0 + BUY_COST)
                    if cash < spent:
                        continue
                    sc_base.apply_pyramid(pos, float(px), today)
                    cash -= spent; cost[t] += spent; n_add += 1

        # 3) 신규매수 (강세장, 빈 슬롯) + 통과율 계측
        free = sc_base.NUM_SLOTS - len(positions)
        if is_bull:
            uni = store.get_universe(td[di])
            day_uni = 0; day_pass = 0
            cands: list[StockSnapshot] = []
            for t in uni:
                col = c2c[t]; px = close_v[di, col]
                if not np.isfinite(px) or px <= 0:
                    continue
                day_uni += 1
                snap = _build_snap(store, col, di)
                _attach_fin2(snap, store, t, di)   # opmargin/debt2 변형용 부가필드 부착(base/PER는 무시)
                ok, _ = check_buy_fn(snap, is_bull)
                if ok:
                    day_pass += 1
                    if t not in positions and t not in sold_today:
                        cands.append(snap)
            if day_uni > 0:
                pass_pct_sum += day_pass / day_uni
                pass_cnt_sum += day_pass
                pass_days += 1
            for snap in sc_base.rank_by_momentum(cands):
                if free <= 0:
                    break
                px = snap.price
                sh = int(sc_base.SLOT_AMOUNT_WON // px)
                if sh <= 0:
                    continue
                spent = sh * px * (1.0 + BUY_COST)
                if spent > cash:
                    continue
                new_pos = Position(ticker=snap.ticker, entry_price=px, avg_price=px,
                                   shares=float(sh), entry_date=today)
                imm, _ = sc_base.check_sell_condition(new_pos, snap)
                if imm:
                    immediate_sell += 1
                positions[snap.ticker] = new_pos
                cost[snap.ticker] = spent
                cash -= spent; n_buy += 1; free -= 1
                bought_set.add(snap.ticker)

        # 4) 일별 평가 + 최대 단일종목 비중
        hv = 0.0; vmax = 0.0
        for t, pos in positions.items():
            col = c2c[t]; p = close_v[di, col]
            if not np.isfinite(p):
                p = close_ff[di, col]
                if not np.isfinite(p):
                    p = pos.avg_price
            v = p * pos.shares; hv += v
            if v > vmax:
                vmax = v
        total = cash + hv
        daily_total[di] = total
        if td[di] >= TRADE_START and total > 0:
            w = vmax / total
            if w > max_weight:
                max_weight = w

    m = _metrics(daily_total, td)
    n_sell = n_partial + n_final
    m.update({
        "tag": tag,
        "PL": abs(gross_w / gross_l) if gross_l != 0 else float("inf"),
        "win_rate": wins / n_sell if n_sell else 0.0,
        "n_buy": n_buy, "n_add": n_add, "n_partial": n_partial, "n_final": n_final,
        "n_trades": n_buy + n_add + n_partial + n_final,
        "immediate_sell": immediate_sell,
        "max_weight": max_weight,
        "pass_rate": pass_pct_sum / pass_days if pass_days else 0.0,
        "avg_pass_cnt": pass_cnt_sum / pass_days if pass_days else 0.0,
        "bought_set": bought_set,
    })
    return m
