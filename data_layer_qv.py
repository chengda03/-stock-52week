# -*- coding: utf-8 -*-
"""
data_layer_qv.py — "가치우량주 정배열"(전략2) 전용 데이터 조달 계층
============================================================================
S1 의 data_layer.py 에 대응하지만, 전략2가 요구하는 추가 데이터를 다룹니다.
  · 재무 부가필드: 영업이익 / 부채총계 / 공시접수일(rcept_dt)  ← financials_qv
  · 업종(금융업 여부): 부채비율 조건 면제 판정용                ← FDR StockListing
  · 이동평균 20/60/120, 20일 평균거래대금, PER/PBR/ROE/EPS

★★★ 핵심: PIT(발표시점) 처리 — get_financials(ticker, as_of_date) ★★★
  "그 시점(as_of_date)에 이미 공시돼 있던 가장 최신 사업보고서"만 돌려줍니다.
  각 사업보고서(FY)의 공시접수일(rcept_dt)을 기준으로,
        rcept_dt ≤ as_of_date 인 FY 중 가장 최근 FY
  를 '현재 실적'으로, 그 직전 FY 를 '전년 동기'로 사용합니다.
  - rcept_dt 가 수집돼 있으면 실제 접수일 사용.
  - 없으면 사업보고서 통상 시한(사업연도末 + 90일 = 익년 3/31)으로 '추정'하고,
    이 종목/연도는 estimated=True 로 표시(백테스트 로그에서 '추정 공시일'로 구분).

★ 데이터 소스 (S1 캐시 재사용)
  - 가격/거래량/지수 : data/cache_52w_bt 의 parquet (load_price_panel/load_index)
  - 재무 부가필드    : data/cache_52w_bt/financials_qv/<corp>_<year>.json
                       (없으면 financials_full 의 net_income/equity 로 부분 대체)
  - 발행주식수       : data_layer._load_shares_by_year (연도별)
  - 업종             : FDR StockListing('KRX') → krx_sector.csv 로 캐시

★ 한계(정직성): 유니버스가 '현재 상장사' 기준이라 상장폐지 종목 누락(생존편향).
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

import backtest_roe_eps_event as ev
from backtest_roe_eps_event import (
    load_dart_corp, load_corp_cls, load_fin, load_price_panel, load_index,
    is_excluded_name, log,
)
import data_layer as dl  # _load_shares_by_year, load_admin_issues 재사용

# --- 설정값 --------------------------------------------------------------
FIN_YEARS = list(range(2018, 2026))
WARMUP_START = pd.Timestamp("2019-06-01")   # 지표 계산용 준비기간
MA_LONG_WIN = 120                            # 가장 긴 이동평균(정배열 §2-7)
TRADING_VALUE_WIN = 20                       # 20일 평균 거래대금
FIN_QV_DIR = os.path.join(ev.BT_CACHE, "financials_qv")
SECTOR_CACHE = os.path.join(ev.BT_CACHE, "krx_sector.csv")

# 금융업 판정 키워드(업종/종목명에 포함되면 금융업 → 부채비율 조건 면제)
_FIN_KEYWORDS = ("은행", "증권", "보험", "카드", "캐피탈", "저축은행",
                 "금융", "선물", "자산운용", "신용", "종금", "금고")


# ===========================================================================
# 업종(금융업 여부) — FDR StockListing 으로 매핑해 CSV 캐시
# ===========================================================================
def _build_sector_map(name_by_code: dict[str, str], refresh: bool = False
                      ) -> dict[str, bool]:
    """{종목코드: 금융업여부(bool)}. FDR 업종 + 종목명 키워드 병행 판정."""
    if not refresh and os.path.exists(SECTOR_CACHE):
        try:
            df = pd.read_csv(SECTOR_CACHE, dtype={"code": str})
            return {r["code"].zfill(6): bool(r["is_financial"])
                    for _, r in df.iterrows()}
        except Exception:
            pass

    sector_by_code: dict[str, str] = {}
    try:
        import FinanceDataReader as fdr
        listing = fdr.StockListing("KRX")
        code_col = "Code" if "Code" in listing.columns else "Symbol"
        # 업종 컬럼 후보(FDR 버전에 따라 명칭이 다름)
        sec_col = next((c for c in ("Sector", "Industry", "SectorName",
                                    "업종", "Dept") if c in listing.columns), None)
        for _, row in listing.iterrows():
            code = str(row.get(code_col, "")).strip().zfill(6)
            sec = str(row.get(sec_col, "") if sec_col else "").strip()
            sector_by_code[code] = sec
        log(f"[qv] 업종 매핑: FDR StockListing (업종컬럼={sec_col})")
    except Exception as e:
        log(f"[qv] ⚠ FDR 업종 조회 실패({e}) — 종목명 키워드로만 금융업 판정")

    out: dict[str, bool] = {}
    recs = []
    for code, nm in name_by_code.items():
        sec = sector_by_code.get(code, "")
        text = f"{sec} {nm}"
        is_fin = any(k in text for k in _FIN_KEYWORDS)
        out[code] = is_fin
        recs.append({"code": code, "name": nm, "sector": sec,
                     "is_financial": int(is_fin)})
    try:
        pd.DataFrame(recs).to_csv(SECTOR_CACHE, index=False, encoding="utf-8-sig")
    except Exception:
        pass
    log(f"[qv] 금융업 종목 {sum(out.values())}건 / {len(out)} (부채비율 조건 면제)")
    return out


# ===========================================================================
# 재무 부가필드 로딩 (financials_qv 우선, 없으면 financials_full 부분대체)
# ===========================================================================
def _est_rcept(year: int) -> pd.Timestamp:
    """사업보고서 추정 공시일: 사업연도末(12/31) + 90일 ≈ 익년 3/31."""
    return pd.Timestamp(year + 1, 3, 31)


def _load_qv_fin(corp_code: str, year: int) -> dict | None:
    """financials_qv/<corp>_<year>.json → dict. 없으면 financials_full 로 부분대체."""
    p = os.path.join(FIN_QV_DIR, f"{corp_code}_{year}.json")
    if os.path.exists(p):
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            rcept = d.get("rcept_dt")
            d["_rcept_ts"] = (pd.Timestamp(rcept) if rcept else _est_rcept(year))
            d["_estimated"] = rcept is None
            return d
        except Exception:
            pass
    # QV 파일이 없으면 S1 재무로 부분대체(영업이익/부채 없음 → 해당 조건 판정 불가)
    base = load_fin(corp_code, year)
    if not base:
        return None
    base = dict(base)
    base.setdefault("op_income", None)
    base.setdefault("liabilities", None)
    base["_rcept_ts"] = _est_rcept(year)
    base["_estimated"] = True
    return base


class DataStoreQV:
    """전략2용 데이터 저장소. 무거운 로딩/전처리를 1회만 수행."""

    def __init__(self, end_date: pd.Timestamp | None = None):
        self._build(end_date)

    def _build(self, end_date):
        log("[qv] 데이터 로딩 시작")
        dart = load_dart_corp()
        corp_codes = dart["corp_code"].astype(str).tolist()
        stock_codes = dart["stock_code"].astype(str).str.zfill(6).tolist()
        names = dart["corp_name"].astype(str).tolist()
        self.name_map = dict(zip(stock_codes, names))
        code_to_corp = dict(zip(stock_codes, corp_codes))

        cls_map = load_corp_cls(corp_codes)
        shares_yr = dl._load_shares_by_year(corp_codes)

        # 유니버스 후보: KOSPI(Y)/KOSDAQ(K), 주식수 있음, 우선주·스팩·리츠 제외
        #   (전략2는 시총 상위 500 제한 없음 → 전체 후보. 거래대금 조건이 필터 역할)
        keep = []
        for sc, cc, nm in zip(stock_codes, corp_codes, names):
            if cc not in shares_yr:
                continue
            if cls_map.get(cc) not in ("Y", "K"):
                continue
            if is_excluded_name(nm):
                continue
            keep.append(sc)
        log(f"[qv] 유니버스 후보(전체): {len(keep)}")

        panel = load_price_panel(keep)
        valid_codes = sorted(panel.keys())
        self.valid_codes = valid_codes
        self.code_to_col = {c: i for i, c in enumerate(valid_codes)}
        self.code_to_corp = {c: code_to_corp[c] for c in valid_codes}
        N = len(valid_codes)
        log(f"[qv] 가격 패널 확보: {N}")

        # 연도별 주식수(결측은 가장 가까운 연도로 fallback)
        self._sc_year_sh: dict[str, dict[int, int]] = {}
        for sc in valid_codes:
            avail = shares_yr.get(code_to_corp[sc], {})
            ymap = {}
            for yr in FIN_YEARS:
                if yr in avail:
                    ymap[yr] = avail[yr]
                elif avail:
                    near = sorted(avail.keys(), key=lambda y: (abs(y - yr), -y))[0]
                    ymap[yr] = avail[near]
                else:
                    ymap[yr] = 0
            self._sc_year_sh[sc] = ymap

        # 업종(금융업) 매핑
        self.is_financial = _build_sector_map(
            {sc: self.name_map[sc] for sc in valid_codes})

        # 거래일 인덱스
        s = set()
        for df in panel.values():
            s.update(df.index.tolist())
        trading_days = pd.DatetimeIndex(sorted(s))
        data_end = trading_days.max()
        end = data_end if end_date is None else min(pd.Timestamp(end_date), data_end)
        trading_days = trading_days[(trading_days >= WARMUP_START) & (trading_days <= end)]
        self.trading_days = trading_days
        self._day_index = {d.normalize(): i for i, d in enumerate(trading_days)}
        log(f"[qv] 거래일: {len(trading_days)} "
            f"({trading_days[0].date()} ~ {trading_days[-1].date()})")

        # 가격/거래량 → 배열 + 지표
        close = pd.DataFrame({c: panel[c]["Close"] for c in valid_codes}).reindex(trading_days)
        vol = pd.DataFrame({c: panel[c]["Volume"] for c in valid_codes}).reindex(trading_days)
        self.close_v = close.to_numpy(dtype=float)
        self.close_ff = close.ffill().to_numpy(dtype=float)
        self.ma20_v = close.rolling(20, min_periods=20).mean().to_numpy(dtype=float)
        self.ma60_v = close.rolling(60, min_periods=60).mean().to_numpy(dtype=float)
        self.ma120_v = close.rolling(120, min_periods=120).mean().to_numpy(dtype=float)
        self.tv_v = (close * vol).rolling(
            TRADING_VALUE_WIN, min_periods=TRADING_VALUE_WIN).mean().to_numpy(dtype=float)

        # 종목별 재무 타임라인(공시접수일 순 정렬) — PIT 조회용
        #   fin_tl[sc] = [(rcept_ts, year, rec), ...]  (rcept_ts 오름차순)
        #   rec: {ni, eq, op, liab, sh, roe, eps, estimated}
        self.fin_tl: dict[str, list] = {}
        n_est = n_real = 0
        for sc in valid_codes:
            cc = code_to_corp[sc]
            tl = []
            for yr in FIN_YEARS:
                d = _load_qv_fin(cc, yr)
                if not d:
                    continue
                ni = d.get("net_income")
                eq = d.get("equity")
                op = d.get("op_income")
                liab = d.get("liabilities")
                sh = self._sc_year_sh[sc].get(yr, 0)
                rec = {
                    "ni": ni, "eq": eq, "op": op, "liab": liab, "sh": sh,
                    "roe": (ni / eq * 100.0) if (ni is not None and eq and eq > 0) else None,
                    "eps": (ni / sh) if (ni is not None and sh and sh > 0) else None,
                    "estimated": bool(d.get("_estimated")),
                }
                tl.append((d["_rcept_ts"], yr, rec))
                if d.get("_estimated"):
                    n_est += 1
                else:
                    n_real += 1
            tl.sort(key=lambda x: x[0])
            self.fin_tl[sc] = tl
        log(f"[qv] 재무 타임라인: 실접수일 {n_real}건 / 추정공시일 {n_est}건 "
            f"({N}종목 × 최대 {len(FIN_YEARS)}년)")

        # 관리종목 지정일(as_of ≥ 지정일 이면 제외) — S1 과 동일 소스 재사용
        admin = dl.load_admin_issues()
        desig = np.full(N, np.datetime64("NaT"), dtype="datetime64[ns]")
        for code, dd in admin.items():
            col = self.code_to_col.get(code)
            if col is not None:
                desig[col] = np.datetime64(pd.Timestamp(dd))
        self.admin_desig_v = desig

        # KOSPI 지수(벤치마크 Buy&Hold 용)
        ks = load_index("kospi_index").reindex(trading_days).ffill().bfill()
        self.kospi_close_v = ks.to_numpy(dtype=float)

        # QV 부가필드 실제 수집 여부(백테스트 실행가능성 점검용)
        self.has_qv_extras = os.path.isdir(FIN_QV_DIR) and \
            len(glob.glob(os.path.join(FIN_QV_DIR, "*.json"))) > 0
        log(f"[qv] 부가필드(영업이익/부채/접수일) 수집파일 존재: {self.has_qv_extras}")
        log("[qv] 전처리 완료")

    # ------------------------------------------------------------------
    def di(self, as_of_date) -> int:
        """날짜 → 거래일 인덱스(정확히 일치하거나 그 이하 가장 최근 거래일)."""
        ts = pd.Timestamp(as_of_date).normalize()
        d = self._day_index.get(ts)
        if d is not None:
            return d
        pos = int(np.searchsorted(self.trading_days.values,
                                  ts.to_datetime64(), side="right")) - 1
        if pos < 0:
            raise ValueError(f"{ts.date()} 이전 거래일 데이터 없음")
        return pos

    # ------------------------------------------------------------------
    def get_universe(self, as_of_date) -> list[str]:
        """as_of 시점 가격이 있는 전체 후보(관리종목 제외). 시총 상위 제한 없음."""
        di = self.di(as_of_date)
        px = self.close_v[di]
        valid = np.isfinite(px) & (px > 0)
        as64 = np.datetime64(pd.Timestamp(as_of_date).normalize())
        admin_active = (~np.isnat(self.admin_desig_v)) & (self.admin_desig_v <= as64)
        valid &= ~admin_active
        return [self.valid_codes[i] for i in np.where(valid)[0]]

    def get_price_features(self, ticker: str, as_of_date) -> dict | None:
        """가격/20·60·120일선/20일 평균거래대금."""
        col = self.code_to_col.get(ticker)
        if col is None:
            return None
        di = self.di(as_of_date)
        px = self.close_v[di, col]
        if not np.isfinite(px) or px <= 0:
            return None
        return {
            "price": float(px),
            "ma20": float(self.ma20_v[di, col]),
            "ma60": float(self.ma60_v[di, col]),
            "ma120": float(self.ma120_v[di, col]),
            "tv20": float(self.tv_v[di, col]),
        }

    def get_financials(self, ticker: str, as_of_date) -> dict | None:
        """
        ★ PIT 핵심 ★ as_of_date 에 '이미 공시돼 있던' 가장 최신 사업보고서와
        그 직전연도(전년 동기) 재무를 함께 반환.

        반환 dict:
          year         현재 사용 FY (사업연도)
          cur          {ni, eq, op, liab, sh, roe, eps}  현재 FY
          prev         {...}  직전 FY (없으면 None → 전년비 조건 판정 불가)
          estimated    현재 FY 공시일이 '추정'인지(True=추정 공시일 사용)
          rcept        현재 FY 공시(추정)일 Timestamp
        없으면 None (아직 어떤 사업보고서도 공시 전).
        """
        tl = self.fin_tl.get(ticker)
        if not tl:
            return None
        as_ts = pd.Timestamp(as_of_date).normalize()
        # rcept_ts ≤ as_of 인 것 중 가장 최근(타임라인은 접수일 오름차순 정렬)
        chosen = None
        for rcept_ts, yr, rec in tl:
            if rcept_ts <= as_ts:
                chosen = (rcept_ts, yr, rec)
            else:
                break
        if chosen is None:
            return None
        rcept_ts, yr, cur = chosen
        # 전년 동기: 직전 FY 레코드(연도 = yr-1)
        prev = next((r for (_, y, r) in tl if y == yr - 1), None)
        return {"year": yr, "cur": cur, "prev": prev,
                "estimated": cur["estimated"], "rcept": rcept_ts}


_STORE_QV: DataStoreQV | None = None


def get_store(end_date: pd.Timestamp | None = None) -> DataStoreQV:
    global _STORE_QV
    if _STORE_QV is None:
        _STORE_QV = DataStoreQV(end_date=end_date)
    return _STORE_QV


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    store = get_store()
    last = store.trading_days[-1]
    uni = store.get_universe(last)
    print(f"\n[점검] {last.date()} 유니버스 후보수: {len(uni)} (앞 5개 {uni[:5]})")
    if uni:
        t = uni[0]
        pf = store.get_price_features(t, last)
        fin = store.get_financials(t, last)
        print(f"[점검] {t} 가격지표: {pf}")
        print(f"[점검] {t} 금융업여부: {store.is_financial.get(t)}")
        if fin:
            print(f"[점검] {t} PIT 재무 FY{fin['year']} estimated={fin['estimated']} "
                  f"rcept={fin['rcept'].date()} cur={fin['cur']}")
