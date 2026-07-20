# -*- coding: utf-8 -*-
"""
data_layer.py
=============
데이터 조달 계층. "인터넷/파일에서 데이터를 받아와서, strategy_core.py가
바로 먹을 수 있는 모양(StockSnapshot)으로 바꿔주는" 역할만 합니다.

★ 설계 원칙
  - 판정 로직(매수/매도/불타기)은 여기 없습니다. 그건 strategy_core.py 담당.
  - 데이터 파이프라인은 새로 짜지 않고, 이미 검증된 기존 백테스트 로더
    (backtest_roe_eps_event.py)의 함수들을 그대로 재사용합니다.
    → 이렇게 해야 backtest_v3.py가 STRATEGY_FINAL.md 수치를 재현할 수 있습니다.

★ 데이터 소스 (기존 저장소 관례와 동일)
  - 가격/거래량/지수 : FinanceDataReader로 미리 받아 data/cache_52w_bt 에 저장된 parquet
  - 재무(ROE/EPS)   : DART OpenAPI로 미리 받아 저장된 json (financials_full)
  - pykrx 는 방화벽 차단으로 사용하지 않음
  - 발행주식수: 연도별 사업보고서 값(2018~2025)을 shares/<corp>_<year>_11011.json 에서 로드.
    결측 연도는 가장 가까운 연도 값으로 fallback. 시총·EPS 모두 '그날 적용 연도' 주식수 사용.
  - (한계) 유니버스가 '현재 상장사' 기준이라 상장폐지 종목이 빠지는 생존편향 존재

핵심 클래스: DataStore
  - 무거운 로딩/전처리를 딱 한 번만 하고, 이후엔 배열에서 빠르게 꺼내 씁니다.
  - 제공 함수:
        get_universe(as_of_date)        → 시총 상위 500 티커 리스트
        get_price_snapshot(t, as_of)    → StockSnapshot (가격/MA/모멘텀/거래대금/ROE/EPS)
        get_fundamentals(t, as_of)      → (roe_pct, eps)
        get_kospi_index(as_of)          → (KOSPI 종가, KOSPI 200일선)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategy_core import (
    StockSnapshot,
    BUY_MA_DAYS,       # 60
    SELL_MA_DAYS,      # 90
    MOMENTUM_DAYS,     # 20
    MARKET_FILTER_MA_DAYS,  # 200
)

# 기존 저장소의 데이터 로더를 그대로 재사용 (새로 짜지 않음)
import backtest_roe_eps_event as ev
from backtest_roe_eps_event import (
    load_dart_corp, load_corp_cls, load_shares, is_excluded_name,
    load_fin, load_price_panel, load_index, fin_year_for_date, log,
)

# --- 설정값 (STRATEGY_FINAL.md / build_pre 와 동일하게 맞춤) ------------------
TOP_MARCAP = 500                       # 시총 상위 유니버스 크기
FIN_YEARS = list(range(2018, 2026))    # 사용 가능한 사업보고서 연도
TRADING_VALUE_WIN = 20                 # 20거래일 평균 거래대금
WARMUP_START = pd.Timestamp("2019-06-01")  # 지표(MA/모멘텀) 계산용 준비기간 시작
TRADE_START = pd.Timestamp("2020-01-01")   # 실제 매매 시작일

# 관리종목(administrative issue) 캐시 경로 (FDR로 받아 저장 → 오프라인 재현용)
import os as _os
import glob as _glob
import json as _json
import time as _time
from concurrent.futures import ThreadPoolExecutor, as_completed

ADMIN_CACHE_PATH = _os.path.join(ev.BT_CACHE, "admin_issues.csv")
PRICE_DIR = ev.PRICE_DIR
REFRESH_MARKER = _os.path.join(ev.BT_CACHE, "last_refresh.json")


def load_admin_issues(refresh: bool = False) -> dict[str, pd.Timestamp]:
    """
    관리종목 목록을 {종목코드(6자리): 지정일(Timestamp)} 형태로 반환.

    소스: FinanceDataReader 의 StockListing('KRX-ADMINISTRATIVE')
      - 컬럼: Symbol, Name, DesignationDate, Reason
      - is_excluded_name()은 종목명 패턴만 걸러서 '관리종목 지정'은 못 거르므로,
        여기서 별도로 받아와 유니버스에서 제외하는 데 씁니다.

    캐시: 한 번 받으면 data/cache_52w_bt/admin_issues.csv 에 저장해 재사용
          (네트워크 불가 시 캐시로 동작, 캐시도 없으면 빈 dict + 경고).

    ★ 한계(정직성):
      - 이 목록은 '현재 시점' 관리종목 스냅샷입니다. 지정일(DesignationDate)이
        있어 '지정일 이후'만 제외하도록 하여 룩어헤드를 줄였지만,
        '지정 해제일'은 알 수 없어 한 번 지정되면 이후 기간 계속 제외됩니다.
      - 과거에 관리종목이었다가 현재는 해제/상장폐지된 종목은 목록에 없어
        완벽한 PIT는 아닙니다(기존 유니버스의 생존편향과 동일한 성격).
    """
    if not refresh and _os.path.exists(ADMIN_CACHE_PATH):
        try:
            df = pd.read_csv(ADMIN_CACHE_PATH, dtype={"code": str})
            return {r["code"].zfill(6): pd.Timestamp(r["designation_date"])
                    for _, r in df.iterrows() if pd.notna(r["designation_date"])}
        except Exception:
            pass

    # FDR에서 조회 시도
    try:
        import FinanceDataReader as fdr
        raw = fdr.StockListing("KRX-ADMINISTRATIVE")
        out = {}
        recs = []
        for _, r in raw.iterrows():
            code = str(r.get("Symbol", "")).strip().zfill(6)
            dd = pd.to_datetime(r.get("DesignationDate"), errors="coerce")
            if len(code) != 6 or pd.isna(dd):
                continue
            out[code] = dd
            recs.append({"code": code, "name": r.get("Name"),
                         "designation_date": dd.date().isoformat(),
                         "reason": r.get("Reason")})
        # 캐시 저장
        try:
            pd.DataFrame(recs).to_csv(ADMIN_CACHE_PATH, index=False, encoding="utf-8-sig")
        except Exception:
            pass
        log(f"[data_layer] 관리종목 {len(out)}건 (FDR 조회)")
        return out
    except Exception as e:
        log(f"[data_layer] ⚠ 관리종목 조회 실패({e}) — 관리종목 제외 없이 진행")
        return {}


# ===========================================================================
# 증분(incremental) 데이터 갱신 — FDR 가격/거래량/지수 + DART 재무
#   · 전체 재다운로드 대신 '캐시 마지막 날짜 이후'만 받아 parquet에 이어붙임
#   · app_s1.py 가 실행 시 자동 호출(ensure_fresh)해 오늘까지 최신화
# ===========================================================================
def _read_marker() -> dict:
    try:
        with open(REFRESH_MARKER, encoding="utf-8") as f:
            return _json.load(f)
    except Exception:
        return {}


def _write_marker(info: dict) -> None:
    try:
        with open(REFRESH_MARKER, "w", encoding="utf-8") as f:
            _json.dump(info, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _fdr():
    import FinanceDataReader as fdr
    return fdr


def update_index(name: str, sym: str, label: str) -> tuple[pd.Timestamp | None, int]:
    """지수 parquet을 증분 갱신. 반환 (최신일, 신규행수)."""
    path = _os.path.join(ev.BT_CACHE, f"{name}.parquet")
    fdr = _fdr()
    existing = None
    start = WARMUP_START.strftime("%Y-%m-%d")
    if _os.path.exists(path):
        try:
            existing = pd.read_parquet(path)
            existing.index = pd.to_datetime(existing.index)
            existing = existing.sort_index()
            # 마지막 5일은 정정 가능성 대비 겹쳐 다시 받음
            start = (existing.index.max() - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        except Exception:
            existing = None
    end = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")

    df = None
    for s in (sym, f"^{sym.lstrip('^')}"):
        try:
            df = fdr.DataReader(s, start, end)
            if df is not None and not df.empty:
                break
        except Exception:
            df = None
    if df is None or df.empty or "Close" not in df.columns:
        mx = existing.index.max() if existing is not None else None
        log(f"[data_layer] {label} 지수 신규분 없음(또는 조회실패)")
        return mx, 0

    new = df[["Close"]].copy()
    new.index = pd.to_datetime(new.index)
    combined = new if existing is None else pd.concat([existing, new])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    n_new = len(combined) - (0 if existing is None else len(existing))
    if n_new > 0 or existing is None:
        combined.to_parquet(path)
    log(f"[data_layer] {label} 지수 → {combined.index.max().date()} (신규 {max(n_new,0)}일)")
    return combined.index.max(), max(n_new, 0)


def _incr_price_one(code: str, target: pd.Timestamp) -> tuple[str, int]:
    """개별종목 일봉 parquet을 target까지 증분 갱신. 반환 (코드, 신규행수)."""
    path = _os.path.join(PRICE_DIR, f"{code}.parquet")
    fdr = _fdr()
    need = ["Open", "High", "Low", "Close", "Volume"]
    existing = None
    start = WARMUP_START.strftime("%Y-%m-%d")
    if _os.path.exists(path):
        try:
            existing = pd.read_parquet(path)
            existing.index = pd.to_datetime(existing.index)
            existing = existing.sort_index()
            if existing.index.max() >= target:
                return code, 0
            start = (existing.index.max() - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        except Exception:
            existing = None
    end = target.strftime("%Y-%m-%d")
    try:
        df = fdr.DataReader(code, start, end)
    except Exception:
        return code, 0
    if df is None or df.empty or not all(c in df.columns for c in need):
        return code, 0
    df = df[need].dropna(how="all")
    df.index = pd.to_datetime(df.index)
    combined = df if existing is None else pd.concat([existing, df])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    n_new = len(combined) - (0 if existing is None else len(existing))
    if n_new > 0:
        try:
            combined.to_parquet(path)
        except Exception:
            return code, 0
    return code, max(n_new, 0)


def _existing_price_codes() -> list[str]:
    """이미 캐시(parquet)가 있는 종목코드 목록 (증분 갱신 대상)."""
    return sorted(_os.path.splitext(_os.path.basename(p))[0]
                  for p in _glob.glob(_os.path.join(PRICE_DIR, "*.parquet")))


def update_prices(codes: list[str], target: pd.Timestamp, force: bool = False) -> int:
    """target 거래일까지 개별종목 일봉을 증분 갱신(병렬). 반환 총 신규행수."""
    todo = []
    for c in codes:
        if force:
            todo.append(c)
            continue
        p = _os.path.join(PRICE_DIR, f"{c}.parquet")
        try:
            mx = pd.to_datetime(pd.read_parquet(p, columns=["Close"]).index).max()
        except Exception:
            todo.append(c)
            continue
        if mx < target:
            todo.append(c)
    log(f"[data_layer] 가격 증분 대상: {len(todo)}/{len(codes)} 종목 (목표일 {target.date()})")
    if not todo:
        return 0
    total_new = done = 0
    t0 = _time.time()
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(_incr_price_one, c, target): c for c in todo}
        for fut in as_completed(futs):
            try:
                _, n = fut.result()
            except Exception:
                n = 0
            total_new += n
            done += 1
            if done % 300 == 0 or done == len(todo):
                el = _time.time() - t0
                log(f"  진행 {done}/{len(todo)} (신규행 {total_new}, {done/max(el,1e-9):.1f}/s)")
    log(f"[data_layer] 가격 증분 완료: 신규행 {total_new}")
    return total_new


def update_financials() -> int:
    """
    최신 사업보고서(연간) 재무가 캐시에 있는지 확인하고, 없으면 DART로 수집.
    - 연간보고서 FY는 다음해 3~4월 공시 → today.month>=4면 (year-1), 아니면 (year-2).
    - 이미 충분히 확보돼 있으면 no-op(네트워크 호출 없음).
    반환: 새로 받은 파일 수(추정).
    """
    today = pd.Timestamp.today()
    latest_fy = today.year - 1 if today.month >= 4 else today.year - 2
    if latest_fy < min(FIN_YEARS):
        return 0
    if latest_fy > max(FIN_YEARS):
        log(f"[data_layer] 재무: FY{latest_fy} 공시가능하나 FIN_YEARS 최대={max(FIN_YEARS)} "
            f"→ data_layer.FIN_YEARS 확장 후 재수집 필요")
        return 0
    n = len(_glob.glob(_os.path.join(ev.FIN_FULL_DIR, f"*_{latest_fy}.json")))
    if n >= 500:
        log(f"[data_layer] 재무: 최신 사업보고서 FY{latest_fy} 이미 {n}건 확보 — 갱신 불필요")
        return 0
    log(f"[data_layer] 재무: FY{latest_fy} {n}건뿐 → DART 추가 수집 시도")
    try:
        import fetch_dart_full_financials as fd
        dart = load_dart_corp()
        corp_codes = dart["corp_code"].astype(str).tolist()
        names = dart["corp_name"].astype(str).tolist()
        cls = load_corp_cls(corp_codes)
        shares = load_shares(corp_codes)
        keep = [cc for cc, nm in zip(corp_codes, names)
                if cc in shares and cls.get(cc) in ("Y", "K") and not is_excluded_name(nm)]
        fd.bulk(keep, [latest_fy], max_workers=12)
        n2 = len(_glob.glob(_os.path.join(ev.FIN_FULL_DIR, f"*_{latest_fy}.json")))
        log(f"[data_layer] 재무: FY{latest_fy} 수집 후 {n2}건")
        return max(n2 - n, 0)
    except Exception as e:
        log(f"[data_layer] ⚠ 재무 수집 실패({e}) — 기존 재무로 진행")
        return 0


def update_market_data(force: bool = False, include_financials: bool = True) -> bool:
    """
    가격/거래량/지수(증분) + 재무(필요시)를 오늘까지 갱신.
    반환: parquet이 하나라도 변경됐으면 True (호출측에서 store 캐시 무효화용).
    """
    t0 = _time.time()
    log("[data_layer] ===== 시장데이터 증분 갱신 시작 =====")
    changed = False

    ks_max, ks_n = update_index("kospi_index", "KS11", "KOSPI")
    _, kq_n = update_index("kosdaq_index", "KQ11", "KOSDAQ")
    changed = changed or ks_n > 0 or kq_n > 0

    target = ks_max  # 지수 최신일 = 실제 최신 거래일의 기준
    if target is None:
        log("[data_layer] ⚠ 지수 최신일 확인 실패 — 가격 증분 생략(기존 캐시 사용)")
    else:
        codes = _existing_price_codes()
        rep = _os.path.join(PRICE_DIR, "005930.parquet")
        rep_max = None
        if _os.path.exists(rep):
            try:
                rep_max = pd.to_datetime(pd.read_parquet(rep, columns=["Close"]).index).max()
            except Exception:
                rep_max = None
        if not force and rep_max is not None and rep_max >= target:
            log(f"[data_layer] 가격 캐시 최신({rep_max.date()}) = 지수 최신일 → 종목 증분 생략")
        else:
            n_upd = update_prices(codes, target, force=force)
            changed = changed or n_upd > 0

    if include_financials:
        update_financials()

    _write_marker({"date": pd.Timestamp.today().normalize().isoformat()[:10],
                   "index_max": str(target.date()) if target is not None else None,
                   "changed": bool(changed)})
    log(f"[data_layer] ===== 갱신 완료 ({_time.time()-t0:.0f}s, 변경={changed}) =====")
    return changed


def ensure_fresh(force: bool = False, include_financials: bool = True) -> bool:
    """
    '오늘 아직 갱신 안 했으면' 최신 데이터를 받아온다(하루 1회).
    - 마커(last_refresh.json)의 날짜가 오늘이면 즉시 skip → 앱 재실행이 빨라짐.
    - 네트워크 오류 등은 삼켜서 앱이 죽지 않게 함(기존 캐시로 진행).
    반환: 데이터가 실제로 갱신됐으면 True.
    """
    today = pd.Timestamp.today().normalize().isoformat()[:10]
    m = _read_marker()
    if not force and m.get("date") == today:
        return False
    try:
        return update_market_data(force=force, include_financials=include_financials)
    except Exception as e:
        log(f"[data_layer] ⚠ 자동 갱신 실패({e}) — 기존 캐시로 진행")
        return False


class DataStore:
    """모든 데이터를 1회 로드/전처리해서 배열로 들고 있는 저장소."""

    def __init__(self, end_date: pd.Timestamp | None = None):
        self._build(end_date)

    # -------------------------------------------------------------------
    # 내부: 무거운 로딩/전처리 (build_pre 와 동일한 절차)
    # -------------------------------------------------------------------
    def _build(self, end_date):
        log("[data_layer] 데이터 로딩 시작")
        dart = load_dart_corp()
        corp_codes = dart["corp_code"].astype(str).tolist()
        stock_codes = dart["stock_code"].astype(str).str.zfill(6).tolist()
        names = dart["corp_name"].astype(str).tolist()
        self.name_map = dict(zip(stock_codes, names))
        code_to_corp = dict(zip(stock_codes, corp_codes))

        cls_map = load_corp_cls(corp_codes)
        shares_by_corp = load_shares(corp_codes)

        # 유니버스 후보: KOSPI(Y)/KOSDAQ(K), 발행주식수 있음, 우선주·스팩·리츠 제외
        keep = []
        for sc, cc, nm in zip(stock_codes, corp_codes, names):
            if cc not in shares_by_corp:
                continue
            if cls_map.get(cc) not in ("Y", "K"):
                continue
            if is_excluded_name(nm):
                continue
            keep.append(sc)
        log(f"[data_layer] 유니버스 후보: {len(keep)}")

        panel = load_price_panel(keep)
        valid_codes = sorted(panel.keys())
        self.valid_codes = valid_codes
        self.code_to_col = {c: i for i, c in enumerate(valid_codes)}
        self.code_to_corp = {c: code_to_corp[c] for c in valid_codes}
        log(f"[data_layer] 가격 패널 확보: {len(valid_codes)}")

        # 연도별 발행주식수 로드 + 결측 연도는 '가장 가까운 연도'로 fallback.
        #   - 후보 종목은 모두 2024 주식수를 가짐(keep 조건) → fallback은 항상 성공.
        #   - shares_year_arr[yr] = 유니버스 순서(valid_codes)의 (N,) 주식수 배열.
        shares_yr = _load_shares_by_year(corp_codes)
        self._sc_year_sh = {}           # sc -> {yr: shares}
        fb_pairs = 0                    # fallback 적용된 (종목,연도) 건수
        fb_corps = set()                # fallback이 한 번이라도 적용된 종목
        for sc in valid_codes:
            cc = code_to_corp[sc]
            avail = shares_yr.get(cc, {})
            ymap = {}
            for yr in FIN_YEARS:
                if yr in avail:
                    ymap[yr] = avail[yr]
                elif avail:
                    # 가장 가까운 연도(동률이면 더 최근 연도 우선)
                    near = sorted(avail.keys(), key=lambda y: (abs(y - yr), -y))[0]
                    ymap[yr] = avail[near]
                    fb_pairs += 1
                    fb_corps.add(sc)
                else:
                    # 연도별 파일이 전혀 없으면 2024 단일값으로
                    ymap[yr] = shares_by_corp[cc]
                    fb_pairs += 1
                    fb_corps.add(sc)
            self._sc_year_sh[sc] = ymap
        self.shares_year_arr = {
            yr: np.array([self._sc_year_sh[sc][yr] for sc in valid_codes], dtype=float)
            for yr in FIN_YEARS
        }
        self._fallback_pairs = fb_pairs
        self._fallback_corps = len(fb_corps)
        log(f"[data_layer] 연도별 주식수: fallback {fb_pairs}건 / {len(fb_corps)}종목 "
            f"(전체 {len(valid_codes)}종목 × {len(FIN_YEARS)}년)")

        # 거래일 인덱스: 준비기간(2019-06) ~ 데이터 마지막(또는 end_date)
        s = set()
        for df in panel.values():
            s.update(df.index.tolist())
        trading_days = pd.DatetimeIndex(sorted(s))
        data_end = trading_days.max()
        end = data_end if end_date is None else min(pd.Timestamp(end_date), data_end)
        trading_days = trading_days[(trading_days >= WARMUP_START) & (trading_days <= end)]
        self.trading_days = trading_days
        self._day_index = {d.normalize(): i for i, d in enumerate(trading_days)}
        log(f"[data_layer] 거래일: {len(trading_days)} "
            f"({trading_days[0].date()} ~ {trading_days[-1].date()})")

        # 룩어헤드 방지: 날짜 → 사용할 사업보고서 연도 (5월 이후=전년, 4월 이전=전전년)
        # 시총 계산에 '그날 적용 연도'의 주식수를 쓰기 위해 여기서 먼저 계산.
        self.day_fin_year = np.array([fin_year_for_date(d) for d in trading_days])

        N = len(valid_codes)
        close = pd.DataFrame({c: panel[c]["Close"] for c in valid_codes}).reindex(trading_days)
        vol = pd.DataFrame({c: panel[c]["Volume"] for c in valid_codes}).reindex(trading_days)

        self.close_v = close.to_numpy(dtype=float)          # (T, N) 종가
        self.close_ff = close.ffill().to_numpy(dtype=float)  # 결측 보정(평가용)

        # 시가총액 = 종가 × '그날 적용 사업보고서연도'의 발행주식수 (연도별 주식수 반영)
        T = len(trading_days)
        shares_day = np.empty((T, N), dtype=float)
        for yr in FIN_YEARS:
            mask = (self.day_fin_year == yr)
            if mask.any():
                shares_day[mask, :] = self.shares_year_arr[yr][None, :]
        # 혹시 FIN_YEARS 밖의 연도가 있으면(이론상 없음) 2024 값으로 채움
        oob = ~np.isin(self.day_fin_year, FIN_YEARS)
        if oob.any():
            shares_day[oob, :] = self.shares_year_arr[2024][None, :]
        self.marcap_v = self.close_v * shares_day

        # 20일 평균 거래대금 = (종가 × 거래량)의 20일 이동평균
        self.tv_v = (close * vol).rolling(TRADING_VALUE_WIN,
                                          min_periods=TRADING_VALUE_WIN).mean().to_numpy(dtype=float)

        # 이동평균/모멘텀 (strategy_core 상수와 동일 창)
        self.ma_buy_v = close.rolling(BUY_MA_DAYS, min_periods=BUY_MA_DAYS).mean().to_numpy(dtype=float)
        self.ma_sell_v = close.rolling(SELL_MA_DAYS, min_periods=SELL_MA_DAYS).mean().to_numpy(dtype=float)
        self.mom_v = (close / close.shift(MOMENTUM_DAYS) - 1.0).to_numpy(dtype=float)

        # 연도별 ROE(%)/EPS/적정주가(FV) 사전계산
        #   ROE(%) = 당기순이익/자본 × 100,  EPS = 당기순이익/주식수
        #   유효 조건: ni>0, eq>0, shares>0  (아니면 nan → 매수조건에서 자동 탈락)
        self.roe_by_year = {}
        self.eps_by_year = {}
        cnt = {}
        for yr in FIN_YEARS:
            roe_a = np.full(N, np.nan)
            eps_a = np.full(N, np.nan)
            c = 0
            for sc in valid_codes:
                fin = load_fin(code_to_corp[sc], yr)
                if not fin:
                    continue
                ni = fin.get("net_income")
                eq = fin.get("equity")
                sh = self._sc_year_sh[sc][yr]   # 해당 연도 발행주식수 (EPS 정합성)
                if ni is None or eq is None or eq <= 0 or ni <= 0 or sh <= 0:
                    continue
                i = self.code_to_col[sc]
                roe_a[i] = ni / eq * 100.0
                eps_a[i] = ni / sh
                c += 1
            self.roe_by_year[yr] = roe_a
            self.eps_by_year[yr] = eps_a
            cnt[yr] = c
        log("[data_layer] 재무 종목수: " + ", ".join(f"{y}:{cnt[y]}" for y in FIN_YEARS))

        # 관리종목 지정일: 종목별 datetime64 배열(비관리종목은 NaT).
        # get_universe에서 'as_of ≥ 지정일'인 종목을 유니버스에서 제외하는 데 사용.
        admin = load_admin_issues()
        desig = np.full(N, np.datetime64("NaT"), dtype="datetime64[ns]")
        n_admin_in_uni = 0
        for code, dd in admin.items():
            col = self.code_to_col.get(code)
            if col is not None:
                desig[col] = np.datetime64(pd.Timestamp(dd))
                n_admin_in_uni += 1
        self.admin_desig_v = desig
        log(f"[data_layer] 관리종목 중 유니버스 후보 겹침: {n_admin_in_uni}건 (지정일 이후 제외)")

        # KOSPI 지수와 200일선 (시장필터용). build_pre와 동일하게 min_periods=1.
        ks = load_index("kospi_index").reindex(trading_days).ffill().bfill()
        self.kospi_close_v = ks.to_numpy(dtype=float)
        self.kospi_ma200_v = ks.rolling(MARKET_FILTER_MA_DAYS, min_periods=1).mean().to_numpy(dtype=float)

        # 매매 시작 인덱스
        self.start_di = int(np.searchsorted(trading_days.values, TRADE_START.to_datetime64()))
        log("[data_layer] 전처리 완료")

    # -------------------------------------------------------------------
    # 날짜 → 거래일 인덱스 (정확히 일치하거나, 그 날짜 이하의 가장 최근 거래일)
    # -------------------------------------------------------------------
    def _di(self, as_of_date) -> int:
        ts = pd.Timestamp(as_of_date).normalize()
        di = self._day_index.get(ts)
        if di is not None:
            return di
        # 정확한 거래일이 아니면, 그 날짜 이하 가장 최근 거래일 사용
        pos = int(np.searchsorted(self.trading_days.values, ts.to_datetime64(), side="right")) - 1
        if pos < 0:
            raise ValueError(f"{ts.date()} 이전 거래일 데이터가 없습니다.")
        return pos

    # -------------------------------------------------------------------
    # 공개 API
    # -------------------------------------------------------------------
    def get_universe(self, as_of_date) -> list[str]:
        """
        as_of_date 시점 코스피+코스닥 합산 시총 상위 500 티커.
        - 시총 = 종가 × 상장주식수('그날 적용 사업보고서연도' 값 → 룩어헤드 안전)
        - 관리종목/스팩/우선주/리츠는 후보 구성 단계에서 이미 제외됨
        - PIT: as_of_date 종가만 사용 (미래 데이터 없음)
        """
        di = self._di(as_of_date)
        mc = self.marcap_v[di]
        valid = np.isfinite(mc) & (mc > 0)
        # 관리종목 제외: as_of 시점에 '이미 지정된'(지정일 ≤ as_of) 종목은 후보에서 빼고
        # 그만큼 다음 순위 종목으로 상위 500을 채운다.
        as_of64 = np.datetime64(pd.Timestamp(as_of_date).normalize())
        admin_active = (~np.isnat(self.admin_desig_v)) & (self.admin_desig_v <= as_of64)
        valid &= ~admin_active
        idx = np.where(valid)[0]
        if len(idx) == 0:
            return []
        top = idx[np.argsort(-mc[idx])][:TOP_MARCAP]
        return [self.valid_codes[i] for i in top]

    def get_fundamentals(self, ticker: str, as_of_date) -> tuple[float, float]:
        """
        as_of_date 시점에 '이미 공시된' 최근 사업보고서의 ROE(%), EPS.
        - 5월 이후면 전년도, 4월 이전이면 전전년도 재무 사용 (룩어헤드 방지)
        - 데이터 없으면 (nan, nan) → 매수조건에서 자동 탈락
        """
        col = self.code_to_col.get(ticker)
        if col is None:
            return float("nan"), float("nan")
        di = self._di(as_of_date)
        yr = int(self.day_fin_year[di])
        roe_a = self.roe_by_year.get(yr)
        eps_a = self.eps_by_year.get(yr)
        if roe_a is None:
            return float("nan"), float("nan")
        return float(roe_a[col]), float(eps_a[col])

    def get_price_snapshot(self, ticker: str, as_of_date) -> StockSnapshot | None:
        """
        strategy_core.StockSnapshot 한 장 생성.
        - 가격/60일선/90일선/20일모멘텀/20일평균거래대금 + ROE/EPS 포함
        - 종가가 없으면(상장 전/거래정지) None 반환
        """
        col = self.code_to_col.get(ticker)
        if col is None:
            return None
        di = self._di(as_of_date)
        price = self.close_v[di, col]
        if not np.isfinite(price) or price <= 0:
            return None
        roe_pct, eps = self.get_fundamentals(ticker, as_of_date)
        return StockSnapshot(
            ticker=ticker,
            date=self.trading_days[di].date(),
            price=float(price),
            ma_buy=float(self.ma_buy_v[di, col]),
            ma_sell=float(self.ma_sell_v[di, col]),
            roe_pct=roe_pct,
            eps=eps,
            momentum_20d=float(self.mom_v[di, col]),
            trading_value_20d_avg=float(self.tv_v[di, col]),
        )

    def get_kospi_index(self, as_of_date) -> tuple[float, float]:
        """KOSPI 종가와 200일 이동평균 (시장필터용)."""
        di = self._di(as_of_date)
        return float(self.kospi_close_v[di]), float(self.kospi_ma200_v[di])

    def is_admin_issue(self, ticker: str, as_of_date) -> bool:
        """as_of 시점 기준 해당 종목이 (이미 지정된) 관리종목인지."""
        col = self.code_to_col.get(ticker)
        if col is None:
            return False
        dd = self.admin_desig_v[col]
        if np.isnat(dd):
            return False
        return dd <= np.datetime64(pd.Timestamp(as_of_date).normalize())


def _load_shares_by_year(corp_codes: list[str]) -> dict[str, dict[int, int]]:
    """corp_code -> {year: 발행주식수}. 파일 shares/<corp>_<year>_11011.json 에서 로드.
    (없는 연도는 dict 에서 빠짐 → 호출측에서 가장 가까운 연도로 fallback)"""
    out: dict[str, dict[int, int]] = {}
    for cc in corp_codes:
        ymap: dict[int, int] = {}
        for yr in FIN_YEARS:
            p = _os.path.join(ev.SHARES_DIR, f"{cc}_{yr}_{ev.SHARES_REPRT_CODE}.json")
            if not _os.path.exists(p):
                continue
            try:
                with open(p, encoding="utf-8") as f:
                    v = _json.load(f).get("shares_outstanding")
                if v and 0 < int(v) < 10_000_000_000:
                    ymap[yr] = int(v)
            except Exception:
                continue
        if ymap:
            out[cc] = ymap
    return out


# 모듈 전역 싱글턴 (여러 번 로딩하지 않도록 재사용)
_STORE: DataStore | None = None


def get_store(end_date: pd.Timestamp | None = None) -> DataStore:
    """DataStore를 한 번만 만들어 재사용."""
    global _STORE
    if _STORE is None:
        _STORE = DataStore(end_date=end_date)
    return _STORE


# --- 앱/스크립트에서 함수형으로도 쓸 수 있게 얇은 래퍼 제공 -------------------
def get_universe(as_of_date) -> list[str]:
    return get_store().get_universe(as_of_date)


def get_price_snapshot(ticker: str, as_of_date) -> StockSnapshot | None:
    return get_store().get_price_snapshot(ticker, as_of_date)


def get_fundamentals(ticker: str, as_of_date) -> tuple[float, float]:
    return get_store().get_fundamentals(ticker, as_of_date)


def get_kospi_index(as_of_date) -> tuple[float, float]:
    return get_store().get_kospi_index(as_of_date)


if __name__ == "__main__":
    import sys as _sys
    # 사용법:
    #   python data_layer.py            → 최신 거래일 기준 점검(갱신 안 함)
    #   python data_layer.py --refresh  → 오늘까지 증분 갱신 후 점검
    #   python data_layer.py --refresh --force  → 마커 무시하고 강제 갱신
    if "--refresh" in _sys.argv:
        ensure_fresh(force=("--force" in _sys.argv))

    # 간단 점검: 최신 거래일 기준 유니버스/스냅샷/지수 확인
    store = get_store()
    last_day = store.trading_days[-1]
    uni = store.get_universe(last_day)
    print(f"\n[점검] {last_day.date()} 유니버스 상위 500 중 앞 5개: {uni[:5]}")
    if uni:
        snap = store.get_price_snapshot(uni[0], last_day)
        print(f"[점검] {uni[0]} 스냅샷: price={snap.price:,.0f} ma60={snap.ma_buy:,.0f} "
              f"ma90={snap.ma_sell:,.0f} mom20={snap.momentum_20d*100:.1f}% "
              f"거래대금={snap.trading_value_20d_avg/1e8:.1f}억 ROE={snap.roe_pct:.1f}% EPS={snap.eps:,.0f}")
    kc, km = store.get_kospi_index(last_day)
    print(f"[점검] KOSPI 종가={kc:,.1f} 200일선={km:,.1f} → {'강세장' if kc>km else '약세장'}")
