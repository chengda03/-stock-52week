"""
ROE×EPS 저평가 + 추세/모멘텀 이벤트드리븐 백테스트
============================================================================

유니버스
  • KOSPI + KOSDAQ 합산, 매 거래일 시가총액(종가×발행주식수) 상위 500위까지

매수 조건 (3가지 모두 충족)
  ② 현재가 < ROE(%) × EPS          ← 저평가 (적정주가 = ROE% × EPS)
  ③ 현재가 > 60일 이동평균          ← 상승 추세
  ④ 40일 수익률 > 0%               ← 단기 모멘텀

매도 조건 (먼저 발동하는 것 적용)
  ① 매수가 대비 -15%               ← 손절
  ② 현재가 < 60일선                ← 추세 이탈
  ③ 현재가 > ROE(%) × EPS          ← 목표가(적정주가) 달성

운용
  → 20종목 / 종목당 500만원 (총 1억원)
  → 매도 즉시 빈 슬롯 재매수 (매일 신호 점검)

지표 정의
  ROE(%) = 당기순이익 / 자본총계 × 100
  EPS    = 당기순이익 / 발행주식수
  적정주가(FV) = ROE(%) × EPS = 100 × 당기순이익² / (자본총계 × 발행주식수)
  → 당기순이익 > 0, 자본총계 > 0 인 종목만 밸류에이션 유효(아니면 매수 불가)

데이터
  data/cache_52w_bt/prices/<stock>.parquet           (OHLCV 일봉)
  data/cache_52w_bt/shares/<corp>_2024_11011.json    (발행주식수, 2024 기준 상수)
  data/cache_52w_bt/financials_full/<corp>_<yr>.json (net_income, equity ...)
  data/cache_52w_bt/corp_cls/<corp>.json             (Y=KOSPI, K=KOSDAQ)

펀더멘털 시점 (룩어헤드 방지)
  날짜 d 가 5월 이후면 (d.year-1) 사업보고서, 4월 이전이면 (d.year-2) 사업보고서 사용
  (사업보고서 통상 3~4월 공시 → 5월부터 반영)

거래비용/세금: 0 (프로젝트 기존 백테스트 관례와 동일, 상대비교 목적)
"""
from __future__ import annotations

import json
import math
import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

APP_DIR = os.path.dirname(os.path.abspath(__file__))
BT_CACHE = os.path.join(APP_DIR, "data", "cache_52w_bt")
PRICE_DIR = os.path.join(BT_CACHE, "prices")
SHARES_DIR = os.path.join(BT_CACHE, "shares")
CORP_CLS_DIR = os.path.join(BT_CACHE, "corp_cls")
FIN_FULL_DIR = os.path.join(BT_CACHE, "financials_full")
RESULT_DIR = os.path.join(APP_DIR, "data", "cache_roe_eps_event", "results")
os.makedirs(RESULT_DIR, exist_ok=True)

# ---- DART OpenAPI 키 로딩 -------------------------------------------------
# 프로젝트 루트의 .env 파일에서 DART_API_KEY를 읽어온다(하드코딩 금지).
#   .env 예) DART_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# python-dotenv가 없거나 .env가 없어도 죽지 않고, 이미 설정된 OS 환경변수를 쓴다.
try:
    from dotenv import load_dotenv
    # utf-8-sig: 메모장 등으로 저장해 BOM이 붙은 .env도 키 이름이 깨지지 않게 처리
    load_dotenv(os.path.join(APP_DIR, ".env"), encoding="utf-8-sig")
except Exception:
    pass

DART_API_KEY = os.environ.get("DART_API_KEY", "")

# FinanceDataReader(urllib 기반)가 이 PC에서 SSL 인증서 검증 실패
# (CERTIFICATE_VERIFY_FAILED)로 죽는 문제 방지: certifi CA 번들을 urllib에 알려준다.
# (requests는 자체 certifi를 쓰므로 영향 없음.) 이미 지정돼 있으면 존중.
if not os.environ.get("SSL_CERT_FILE"):
    try:
        import certifi
        os.environ["SSL_CERT_FILE"] = certifi.where()
    except Exception:
        pass


def get_dart_api_key() -> str:
    """.env(또는 OS 환경변수)에서 읽은 DART OpenAPI 키를 반환.

    키가 없으면 명확한 안내와 함께 예외를 던진다(빈 문자열로 조용히 실패 방지).
    """
    key = os.environ.get("DART_API_KEY", "") or DART_API_KEY
    if not key:
        raise RuntimeError(
            "DART_API_KEY가 설정되어 있지 않습니다. 프로젝트 루트의 .env 파일에 "
            "'DART_API_KEY=발급받은키' 형식으로 추가하세요."
        )
    return key

# ---- 전략 파라미터 -------------------------------------------------------
START_DATE = pd.Timestamp("2020-01-01")
END_DATE = pd.Timestamp("2026-12-31")
INITIAL_CASH = 100_000_000      # 1억원
SLOT_COUNT = 20                 # 보유 종목 수
SLOT_AMOUNT = 5_000_000         # 종목당 투입 금액
TOP_MARCAP = 500                # 시총 상위 유니버스
MA_WINDOW = 60                  # 추세 이동평균
MOM_WINDOW = 40                 # 모멘텀 룩백 (40일 유지)
STOP_LOSS = 0.15                # 손절 -15%

SHARES_REFERENCE_YEAR = 2024
SHARES_REPRT_CODE = "11011"
FIN_YEARS = list(range(2018, 2026))   # 사용 가능한 사업보고서 연도 (2018~2025)


def log(msg: str) -> None:
    print(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 데이터 로딩
# ---------------------------------------------------------------------------
def load_dart_corp() -> pd.DataFrame:
    return pd.read_parquet(os.path.join(BT_CACHE, "dart_corp.parquet"))


def load_corp_cls(corp_codes: List[str]) -> Dict[str, str]:
    out = {}
    for cc in corp_codes:
        p = os.path.join(CORP_CLS_DIR, f"{cc}.json")
        if not os.path.exists(p):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                cls = json.load(f).get("corp_cls")
            if cls:
                out[cc] = cls
        except Exception:
            continue
    return out


def load_shares(corp_codes: List[str]) -> Dict[str, int]:
    out = {}
    for cc in corp_codes:
        p = os.path.join(SHARES_DIR,
                         f"{cc}_{SHARES_REFERENCE_YEAR}_{SHARES_REPRT_CODE}.json")
        if not os.path.exists(p):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                v = json.load(f).get("shares_outstanding")
            if v and 0 < int(v) < 10_000_000_000:
                out[cc] = int(v)
        except Exception:
            continue
    return out


def is_excluded_name(name: str) -> bool:
    n = (name or "").strip()
    if not n:
        return True
    if "스팩" in n or "리츠" in n or "기업인수목적" in n:
        return True
    if n.endswith("우") or n.endswith("우B") or n.endswith("우C") \
            or "우선주" in n or n.endswith("(전환)") or n.endswith("(신형)"):
        return True
    return False


def load_fin(corp_code: str, year: int) -> Dict[str, Optional[float]]:
    p = os.path.join(FIN_FULL_DIR, f"{corp_code}_{year}.json")
    if not os.path.exists(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_price_panel(codes: List[str]) -> Dict[str, pd.DataFrame]:
    panel = {}
    for c in codes:
        p = os.path.join(PRICE_DIR, f"{c}.parquet")
        if not os.path.exists(p):
            continue
        try:
            df = pd.read_parquet(p)
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            if "Close" not in df.columns:
                continue
            panel[c] = df[["Open", "High", "Low", "Close", "Volume"]].dropna(how="all")
        except Exception:
            continue
    return panel


def load_index(name: str) -> pd.Series:
    df = pd.read_parquet(os.path.join(BT_CACHE, f"{name}.parquet"))
    df.index = pd.to_datetime(df.index)
    return df["Close"].sort_index()


def fin_year_for_date(d: pd.Timestamp) -> int:
    """룩어헤드 방지: 5월 이후 → 전년도, 4월 이전 → 전전년도 사업보고서."""
    return d.year - 1 if d.month >= 5 else d.year - 2


# ---------------------------------------------------------------------------
# 백테스트
# ---------------------------------------------------------------------------
def fpct(x, d=2):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "-"
    return f"{x*100:+.{d}f}%"


def main():
    log("===== ROE×EPS 저평가 + 추세/모멘텀 이벤트드리븐 백테스트 =====")
    dart = load_dart_corp()
    corp_codes = dart["corp_code"].astype(str).tolist()
    stock_codes = dart["stock_code"].astype(str).str.zfill(6).tolist()
    names = dart["corp_name"].astype(str).tolist()
    name_map = dict(zip(stock_codes, names))
    code_to_corp = dict(zip(stock_codes, corp_codes))

    cls_map = load_corp_cls(corp_codes)
    shares_by_corp = load_shares(corp_codes)

    # 유니버스 후보: KOSPI(Y)/KOSDAQ(K) + 발행주식수 + 비우선주
    keep = []
    for sc, cc, nm in zip(stock_codes, corp_codes, names):
        if cc not in shares_by_corp:
            continue
        if cls_map.get(cc) not in ("Y", "K"):
            continue
        if is_excluded_name(nm):
            continue
        keep.append(sc)
    log(f"유니버스 후보(KOSPI+KOSDAQ, 비우선주, 주식수有): {len(keep)}")

    price_panel = load_price_panel(keep)
    valid_codes = sorted(price_panel.keys())
    log(f"가격 패널 확보: {len(valid_codes)}/{len(keep)}")

    shares_by_code = {sc: shares_by_corp[code_to_corp[sc]] for sc in valid_codes}

    # 거래일 인덱스
    s = set()
    for df in price_panel.values():
        s.update(df.index.tolist())
    trading_days = pd.DatetimeIndex(sorted(s))
    data_end = trading_days.max()
    end = min(END_DATE, data_end)
    trading_days = trading_days[(trading_days >= pd.Timestamp("2019-06-01"))
                                & (trading_days <= end)]
    log(f"거래일: {len(trading_days)} ({trading_days[0].date()} ~ {trading_days[-1].date()})")

    N = len(valid_codes)
    code_to_col = {c: i for i, c in enumerate(valid_codes)}

    close = pd.DataFrame({c: price_panel[c]["Close"] for c in valid_codes}).reindex(trading_days)
    close_v = close.to_numpy(dtype=float)                  # (T, N)
    shares_arr = np.array([shares_by_code[c] for c in valid_codes], dtype=float)

    # 지표 사전계산
    ma = close.rolling(MA_WINDOW, min_periods=MA_WINDOW).mean().to_numpy(dtype=float)
    ret_mom = (close / close.shift(MOM_WINDOW) - 1.0).to_numpy(dtype=float)
    marcap_v = close_v * shares_arr[None, :]               # (T, N)
    close_ff = close.ffill().to_numpy(dtype=float)         # 평가용(보유 종목 결측 대비)

    # 펀더멘털: 연도별 적정주가(FV) 사전계산
    #   FV = ROE(%) × EPS = (ni/eq*100) × (ni/shares)
    #   유효 조건: ni>0, eq>0, shares>0
    fv_by_year: Dict[int, np.ndarray] = {}
    roe_by_year: Dict[int, np.ndarray] = {}
    eps_by_year: Dict[int, np.ndarray] = {}
    fin_cnt = {}
    for yr in FIN_YEARS:
        fv = np.full(N, np.nan)
        roe_a = np.full(N, np.nan)
        eps_a = np.full(N, np.nan)
        c = 0
        for sc in valid_codes:
            cc = code_to_corp[sc]
            fin = load_fin(cc, yr)
            if not fin:
                continue
            ni = fin.get("net_income")
            eq = fin.get("equity")
            sh = shares_by_code[sc]
            if ni is None or eq is None or eq <= 0 or ni <= 0 or sh <= 0:
                continue
            roe_pct = ni / eq * 100.0
            eps = ni / sh
            i = code_to_col[sc]
            fv[i] = roe_pct * eps
            roe_a[i] = roe_pct
            eps_a[i] = eps
            c += 1
        fv_by_year[yr] = fv
        roe_by_year[yr] = roe_a
        eps_by_year[yr] = eps_a
        fin_cnt[yr] = c
    log(f"적정주가 산출 가능 종목수(연도별): "
        + ", ".join(f"{y}:{fin_cnt[y]}" for y in FIN_YEARS))

    # 날짜별 적정주가 매핑
    day_fin_year = np.array([fin_year_for_date(d) for d in trading_days])

    # ---- 시뮬레이션 -------------------------------------------------------
    cash = float(INITIAL_CASH)
    # portfolio[col] = dict(shares, avg_price, buy_di)
    portfolio: Dict[int, Dict] = {}
    trade_log: List[Dict] = []
    daily: List[Dict] = []

    start_di = int(np.searchsorted(trading_days.values, START_DATE.to_datetime64()))

    for di in range(len(trading_days)):
        d = trading_days[di]
        row_close = close_v[di]
        row_close_ff = close_ff[di]
        row_ma = ma[di]
        row_mom = ret_mom[di]
        yr = int(day_fin_year[di])
        fv_today = fv_by_year.get(yr)
        roe_today = roe_by_year.get(yr)
        eps_today = eps_by_year.get(yr)

        if di < start_di:
            continue

        # ---------------- 1) 매도 점검 (보유 종목) ----------------
        if fv_today is not None:
            for col in list(portfolio.keys()):
                pos = portfolio[col]
                px = row_close[col]
                if not np.isfinite(px):
                    continue  # 거래정지 등 → 평가만, 매매 보류
                reason = None
                # ① 손절
                if px <= pos["avg_price"] * (1.0 - STOP_LOSS):
                    reason = "STOP_-15%"
                # ② 추세 이탈
                elif np.isfinite(row_ma[col]) and px < row_ma[col]:
                    reason = "BELOW_60MA"
                # ③ 목표가(적정주가) 달성
                elif np.isfinite(fv_today[col]) and px > fv_today[col]:
                    reason = "TARGET_FV"
                if reason is not None:
                    proceeds = px * pos["shares"]
                    profit = (px - pos["avg_price"]) * pos["shares"]
                    cash += proceeds
                    trade_log.append({
                        "date": d, "code": valid_codes[col],
                        "name": name_map.get(valid_codes[col], valid_codes[col]),
                        "type": "SELL", "reason": reason, "price": float(px),
                        "shares": pos["shares"], "avg_price": pos["avg_price"],
                        "profit": float(profit),
                        "ret": float(px / pos["avg_price"] - 1.0),
                        "hold_days": di - pos["buy_di"],
                    })
                    del portfolio[col]

        # ---------------- 2) 매수 (빈 슬롯 채우기, 즉시 재매수) ----------------
        free_slots = SLOT_COUNT - len(portfolio)
        if free_slots > 0 and fv_today is not None:
            # 시총 상위 500 유니버스
            mc = marcap_v[di]
            valid_mc = np.isfinite(mc) & (mc > 0)
            if valid_mc.any():
                v_idx = np.where(valid_mc)[0]
                topN = v_idx[np.argsort(-mc[v_idx])][:TOP_MARCAP]
                in_top = np.zeros(N, dtype=bool)
                in_top[topN] = True

                px = row_close
                cond_uni = in_top
                cond_val = np.isfinite(fv_today) & np.isfinite(px) & (px < fv_today)
                cond_ma = np.isfinite(row_ma) & np.isfinite(px) & (px > row_ma)
                cond_mom = np.isfinite(row_mom) & (row_mom > 0.0)
                eligible = cond_uni & cond_val & cond_ma & cond_mom
                # 이미 보유중인 종목 제외
                for col in portfolio:
                    eligible[col] = False

                cand = np.where(eligible)[0]
                if len(cand) > 0:
                    # 저평가 매력도(적정주가/현재가) 높은 순으로 채움
                    upside = fv_today[cand] / px[cand]
                    order = cand[np.argsort(-upside)]
                    for col in order:
                        if free_slots <= 0:
                            break
                        p = row_close[col]
                        if not np.isfinite(p) or p <= 0:
                            continue
                        invest = min(SLOT_AMOUNT, cash)
                        if invest < p:   # 1주도 못 사면 패스
                            continue
                        sh = int(SLOT_AMOUNT // p)
                        if sh <= 0:
                            continue
                        cost = sh * p
                        if cost > cash:
                            sh = int(cash // p)
                            cost = sh * p
                        if sh <= 0:
                            continue
                        portfolio[col] = {"shares": sh, "avg_price": float(p),
                                          "buy_di": di}
                        cash -= cost
                        trade_log.append({
                            "date": d, "code": valid_codes[col],
                            "name": name_map.get(valid_codes[col], valid_codes[col]),
                            "type": "BUY", "reason": "ENTRY", "price": float(p),
                            "shares": sh, "avg_price": float(p),
                            "profit": 0.0, "ret": 0.0, "hold_days": 0,
                            "roe": float(roe_today[col]) if np.isfinite(roe_today[col]) else None,
                            "eps": float(eps_today[col]) if np.isfinite(eps_today[col]) else None,
                            "fv": float(fv_today[col]),
                        })
                        free_slots -= 1

        # ---------------- 3) 일별 평가 ----------------
        holdings_val = 0.0
        for col, pos in portfolio.items():
            p = row_close[col]
            if not np.isfinite(p):
                p = row_close_ff[col]
                if not np.isfinite(p):
                    p = pos["avg_price"]
            holdings_val += p * pos["shares"]
        total = cash + holdings_val
        daily.append({"date": d, "cash": cash, "holdings": holdings_val,
                      "total": total, "n_holdings": len(portfolio)})

    dv = pd.DataFrame(daily).set_index("date")
    tl = pd.DataFrame(trade_log)

    # ---- 성과지표 ---------------------------------------------------------
    final = float(dv["total"].iloc[-1])
    years = (dv.index[-1] - dv.index[0]).days / 365.25
    total_ret = final / INITIAL_CASH - 1.0
    cagr = (final / INITIAL_CASH) ** (1.0 / max(years, 1e-9)) - 1.0
    peak = dv["total"].cummax()
    mdd = float(((dv["total"] - peak) / peak).min())
    rets = dv["total"].pct_change().fillna(0)
    sharpe = float((rets.mean() / (rets.std() + 1e-12)) * math.sqrt(252))

    sells = tl[tl["type"] == "SELL"] if len(tl) else pd.DataFrame()
    buys = tl[tl["type"] == "BUY"] if len(tl) else pd.DataFrame()
    win_rate = float((sells["profit"] > 0).mean()) if len(sells) else 0.0
    avg_hold = float(sells["hold_days"].mean()) if len(sells) else 0.0
    gross_win = float(sells[sells["profit"] > 0]["profit"].sum()) if len(sells) else 0.0
    gross_loss = float(sells[sells["profit"] < 0]["profit"].sum()) if len(sells) else 0.0
    pl_ratio = abs(gross_win / gross_loss) if gross_loss != 0 else float("inf")

    # ---- 벤치마크 ---------------------------------------------------------
    bench = {}
    for nm, idxname in [("KOSPI", "kospi_index"), ("KOSDAQ", "kosdaq_index")]:
        try:
            ser = load_index(idxname).reindex(dv.index).ffill().bfill()
            bench[nm] = ser
        except Exception:
            pass

    # ---- 출력 -------------------------------------------------------------
    print()
    print("=" * 100)
    print(" ROE×EPS 저평가 + 60일선 추세 + 40일 모멘텀  /  20종목 × 500만원  /  매도 즉시 재매수")
    print(f" 기간 {dv.index[0].date()} ~ {dv.index[-1].date()}  |  시작 {INITIAL_CASH:,.0f}원  |  거래비용 0")
    print("=" * 100)
    print(f"  최종 평가금액 : {final:,.0f} 원")
    print(f"  누적 수익률   : {fpct(total_ret)}")
    print(f"  CAGR          : {fpct(cagr)}")
    print(f"  MDD           : {fpct(mdd)}")
    print(f"  Sharpe        : {sharpe:.2f}")
    print(f"  매수 횟수     : {len(buys)}   매도 횟수: {len(sells)}")
    print(f"  승률          : {fpct(win_rate)}   손익비: {pl_ratio:.2f}")
    print(f"  평균 보유일   : {avg_hold:.1f} 거래일")
    if len(sells):
        rc = sells["reason"].value_counts()
        print("  매도 사유     : " + ", ".join(f"{k} {v}건" for k, v in rc.items()))

    # 벤치마크 비교
    print()
    print("  [ 벤치마크 (단순보유, 동기간) ]")
    for nm, ser in bench.items():
        b_ret = ser.iloc[-1] / ser.iloc[0] - 1.0
        b_cagr = (ser.iloc[-1] / ser.iloc[0]) ** (1.0 / max(years, 1e-9)) - 1.0
        b_peak = ser.cummax()
        b_mdd = float(((ser - b_peak) / b_peak).min())
        print(f"    {nm:7s} 누적 {fpct(b_ret):>9s}  CAGR {fpct(b_cagr):>8s}  MDD {fpct(b_mdd):>8s}")

    # 연도별 수익률
    print()
    print("  [ 연도별 수익률 ]")
    yr_strat = dv["total"].groupby(dv.index.year).apply(lambda g: g.iloc[-1] / g.iloc[0] - 1)
    ytab = {"전략": yr_strat}
    for nm, ser in bench.items():
        ytab[nm] = ser.groupby(ser.index.year).apply(lambda g: g.iloc[-1] / g.iloc[0] - 1)
    ydf = pd.DataFrame(ytab)
    print(ydf.map(lambda v: fpct(v) if pd.notna(v) else "-").to_string())

    # ---- 저장 -------------------------------------------------------------
    dv.to_csv(os.path.join(RESULT_DIR, "daily.csv"), encoding="utf-8-sig")
    if len(tl):
        tl.to_csv(os.path.join(RESULT_DIR, "trades.csv"), index=False, encoding="utf-8-sig")
    ydf.to_csv(os.path.join(RESULT_DIR, "yearly.csv"), encoding="utf-8-sig")
    summary = {
        "final": final, "total_ret": total_ret, "CAGR": cagr, "MDD": mdd,
        "Sharpe": sharpe, "n_buy": len(buys), "n_sell": len(sells),
        "win_rate": win_rate, "pl_ratio": pl_ratio, "avg_hold_days": avg_hold,
    }
    with open(os.path.join(RESULT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 그래프
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(dv.index, dv["total"], label="ROE×EPS 전략", lw=1.6, color="#3182F6")
        for nm, ser in bench.items():
            norm = ser / ser.iloc[0] * INITIAL_CASH
            ax.plot(norm.index, norm.values, label=nm, lw=1.0, linestyle="--")
        ax.set_title("ROE×EPS Value + 60MA Trend + 40d Momentum (20 slots × 5M KRW)")
        ax.legend(loc="upper left", fontsize=9)
        ax.set_ylabel("Equity (KRW)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e8:.2f}억"))
        ax.grid(alpha=0.3)
        plt.tight_layout()
        png = os.path.join(RESULT_DIR, "equity_curve.png")
        plt.savefig(png, dpi=150)
        plt.close(fig)
        print(f"\n  그래프 : {png}")
    except Exception as e:
        print(f"\n  그래프 실패: {e}")

    print(f"  결과 저장 : {RESULT_DIR}")


if __name__ == "__main__":
    main()
