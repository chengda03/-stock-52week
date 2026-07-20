# -*- coding: utf-8 -*-
"""
fetch_dart_full_financials.py
=============================
DART OpenAPI 로 재무제표(ROE/EPS 계산용)와 부속 메타데이터를 수집해
data_layer.py / backtest_roe_eps_event.py 가 기대하는 캐시 파일을 생성한다.

★ API 키
  - 하드코딩 금지. backtest_roe_eps_event.get_dart_api_key() 로 .env 의
    DART_API_KEY 를 읽어 사용한다.

★ 생성하는 캐시 (경로/포맷은 기존 로더 인터페이스에 정확히 맞춤)
  data/cache_52w_bt/
    dart_corp.parquet                         컬럼: corp_code, stock_code, corp_name
    corp_cls/<corp_code>.json                 {"corp_cls": "Y"|"K", ...}  (Y=KOSPI, K=KOSDAQ)
    shares/<corp_code>_2024_11011.json        {"shares_outstanding": int}
    financials_full/<corp_code>_<year>.json   {"net_income": float, "equity": float, ...}
  (선택) --prices 옵션 시 FinanceDataReader 로:
    prices/<stock_code>.parquet               OHLCV 일봉
    kospi_index.parquet / kosdaq_index.parquet  Close

★ data_layer.update_financials() 가 호출하는 유일한 하드 인터페이스
      bulk(corp_codes: list[str], years: list[int], max_workers: int = 12) -> int
  → (corp_code, year) 조합마다 financials_full/<corp>_<year>.json 을 생성.
    이미 있으면 건너뜀(idempotent) → 중단/재개(예: 일일 한도 초과) 안전.

★ DART 일일 호출 한도
  - 기본 20,000건/일. 초과 시 응답 status "020". 이 스크립트는 020 을 만나면
    남은 작업을 멈추고 안내한다. 이미 받은 파일은 건너뛰므로 다음 날 이어서
    실행하면 나머지만 채운다.

CLI
  python fetch_dart_full_financials.py                 # 전체(재무 FIN_YEARS 전연도)
  python fetch_dart_full_financials.py --years 2024 2025
  python fetch_dart_full_financials.py --prices        # 가격/지수 부트스트랩도 함께
  python fetch_dart_full_financials.py --workers 8
  python fetch_dart_full_financials.py --workers 1     # IP 차단 회피(가장 안전)
  python fetch_dart_full_financials.py --workers 1 --sleep 0.5   # 요청당 0.5초 강제 대기

★ IP 차단(ConnectionReset) 회피
  - --workers 1 : 동시 요청을 1개로 (가장 안전, 대신 느림)
  - --sleep 0.5 : 각 요청을 보낸 뒤 워커가 무조건 0.5초 쉼 (요청 속도 물리 제한)
  - --delay 0.5 : 모든 호출 시작 사이 전역 최소 간격 0.5초
  세 옵션은 함께 적용된다. 차단이 반복되면 --sleep/--delay 값을 올리세요.
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
import zipfile
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Event, Lock
from typing import Dict, List, Optional

import pandas as pd
import requests

import backtest_roe_eps_event as ev
from backtest_roe_eps_event import (
    BT_CACHE, PRICE_DIR, SHARES_DIR, CORP_CLS_DIR, FIN_FULL_DIR,
    SHARES_REFERENCE_YEAR, SHARES_REPRT_CODE, FIN_YEARS,
    get_dart_api_key, is_excluded_name, log,
)

DART_BASE = "https://opendart.fss.or.kr/api"
DART_CORP_PARQUET = os.path.join(BT_CACHE, "dart_corp.parquet")

# 필요한 캐시 폴더 미리 생성
for _d in (BT_CACHE, PRICE_DIR, SHARES_DIR, CORP_CLS_DIR, FIN_FULL_DIR):
    os.makedirs(_d, exist_ok=True)


class QuotaExceeded(RuntimeError):
    """DART 일일 호출 한도(status 020) 초과."""


class ConnectionBlocked(RuntimeError):
    """DART 서버가 연결을 계속 리셋(ConnectionReset) — IP 차단/스로틀로 추정."""


# 중단 신호: 한도 초과/연결 차단 시 모든 워커가 즉시 멈추도록 공유 플래그
_STOP = Event()
_SESSION_LOCK = Lock()
_SESSIONS: dict[int, requests.Session] = {}

# 연결 리셋이 짧은 시간에 이 횟수 이상 누적되면 IP 차단으로 보고 전체 중단
_CONN_ERR_LIMIT = 20
_CONN_ERRORS = 0
_CONN_LOCK = Lock()

# ---- 전역 요청 속도 제한 (rate limiter) -----------------------------------
# 워커 수와 무관하게 '모든 DART 호출' 사이에 최소 간격을 강제한다.
# 워커를 1개로 줘도 요청이 너무 빠르면 IP 차단되므로, 실제 요청 시작 시각을
# 전역 락으로 직렬화해 간격을 벌린다. (요청당 대기이므로 처리량 = 1/간격 상한)
_MIN_INTERVAL = 0.4          # 초 단위, --delay 로 조정
_RATE_LOCK = Lock()
_LAST_CALL = 0.0

# ---- 요청당 강제 대기 (per-call sleep) -------------------------------------
# 위 전역 rate limiter 는 '요청 시작' 시각만 벌린다. 여기에 더해, 각 워커는
# 요청을 하나 보낸 뒤 반드시 이 시간만큼 쉰다(성공/실패 무관, finally 로 보장).
# → 워커 수를 낮추는 것과 별개로, 요청 자체의 속도를 물리적으로 제한한다.
_PER_CALL_SLEEP = 0.4        # 초 단위, --sleep 로 조정 (권장 0.3~0.5)


def set_min_interval(seconds: float) -> None:
    """DART 호출 사이 최소 대기시간(초)을 설정."""
    global _MIN_INTERVAL
    _MIN_INTERVAL = max(0.0, float(seconds))


def set_per_call_sleep(seconds: float) -> None:
    """각 요청 뒤 강제로 쉬는 시간(초)을 설정."""
    global _PER_CALL_SLEEP
    _PER_CALL_SLEEP = max(0.0, float(seconds))


def _throttle() -> None:
    """직전 호출로부터 _MIN_INTERVAL 초가 지나도록 대기(전역 직렬화)."""
    global _LAST_CALL
    if _MIN_INTERVAL <= 0:
        return
    with _RATE_LOCK:
        now = time.monotonic()
        wait = _MIN_INTERVAL - (now - _LAST_CALL)
        if wait > 0:
            time.sleep(wait)
        _LAST_CALL = time.monotonic()


def _session() -> requests.Session:
    """스레드별 requests.Session 재사용(커넥션 풀)."""
    import threading
    tid = threading.get_ident()
    with _SESSION_LOCK:
        s = _SESSIONS.get(tid)
        if s is None:
            s = requests.Session()
            _SESSIONS[tid] = s
        return s


def _note_conn_error() -> None:
    """연결 리셋 누적. 임계치 초과 시 IP 차단으로 판단해 전체 중단 신호."""
    global _CONN_ERRORS
    with _CONN_LOCK:
        _CONN_ERRORS += 1
        if _CONN_ERRORS >= _CONN_ERR_LIMIT:
            _STOP.set()


def _is_conn_reset(err: Exception) -> bool:
    s = str(err)
    return ("ConnectionReset" in s or "10054" in s
            or "aborted" in s or "RemoteDisconnected" in s)


def _get_json(endpoint: str, params: dict, timeout: int = 30,
              retries: int = 3) -> dict:
    """DART JSON 엔드포인트 호출.

    - status 020(한도초과) → QuotaExceeded
    - 연결 리셋이 반복 누적되면(IP 차단 추정) → ConnectionBlocked
    """
    p = dict(params)
    p["crtfc_key"] = get_dart_api_key()
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        if _STOP.is_set():
            raise ConnectionBlocked("중단 신호(한도 초과 또는 IP 차단)")
        _throttle()  # 호출 사이 최소 간격 강제(전역)
        try:
            r = _session().get(f"{DART_BASE}/{endpoint}", params=p, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            status = str(data.get("status", ""))
            if status == "020":
                _STOP.set()
                raise QuotaExceeded(data.get("message", "요청 제한 초과"))
            return data
        except QuotaExceeded:
            raise
        except Exception as e:  # 네트워크 일시 오류 → 재시도
            last_err = e
            if _is_conn_reset(e):
                _note_conn_error()
                if _STOP.is_set():
                    raise ConnectionBlocked(f"연결 리셋 반복: {e}")
            time.sleep(0.6 * (attempt + 1))
        finally:
            # 각 워커: 요청 하나 보낸 뒤 무조건 쉼(성공/실패/재시도 무관).
            # 요청 속도를 물리적으로 제한해 IP 차단(ConnectionReset)을 예방한다.
            if _PER_CALL_SLEEP > 0:
                time.sleep(_PER_CALL_SLEEP)
    raise RuntimeError(f"{endpoint} 호출 실패: {last_err}")


def _to_float(s) -> Optional[float]:
    """'363,677,865,000,000' / '-15,000' / '-' / '' → float | None."""
    if s is None:
        return None
    txt = str(s).strip().replace(",", "")
    if txt in ("", "-", "None"):
        return None
    try:
        return float(txt)
    except ValueError:
        return None


# ===========================================================================
# 1) 기업 고유번호(CORPCODE) → dart_corp.parquet
# ===========================================================================
def download_corp_codes() -> List[dict]:
    """DART corpCode.xml(zip) 다운로드/파싱 → 상장사(stock_code 有) 목록."""
    log("[dart] 기업 고유번호(CORPCODE) 다운로드")
    _throttle()
    r = _session().get(f"{DART_BASE}/corpCode.xml",
                        params={"crtfc_key": get_dart_api_key()}, timeout=60)
    r.raise_for_status()
    # 정상이면 zip, 오류면 xml(status 포함)
    if r.content[:2] != b"PK":
        try:
            status = ET.fromstring(r.content).findtext("status")
        except Exception:
            status = "?"
        if status == "020":
            raise QuotaExceeded("corpCode 요청 제한 초과")
        raise RuntimeError(f"corpCode 응답 이상(status={status})")
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".xml"))
        xml_bytes = zf.read(name)
    root = ET.fromstring(xml_bytes)
    out = []
    for node in root.iter("list"):
        stock = (node.findtext("stock_code") or "").strip()
        if not stock or stock == " ":
            continue  # 비상장(고유번호만 있는 회사) 제외
        out.append({
            "corp_code": (node.findtext("corp_code") or "").strip(),
            "corp_name": (node.findtext("corp_name") or "").strip(),
            "stock_code": stock.zfill(6),
        })
    log(f"[dart] 상장사 {len(out)}건 파싱")
    return out


def build_dart_corp(refresh: bool = False) -> pd.DataFrame:
    """dart_corp.parquet 생성/로드. 컬럼: corp_code, stock_code, corp_name."""
    if not refresh and os.path.exists(DART_CORP_PARQUET):
        return pd.read_parquet(DART_CORP_PARQUET)
    rows = download_corp_codes()
    df = pd.DataFrame(rows)[["corp_code", "stock_code", "corp_name"]]
    df = df.drop_duplicates("corp_code").reset_index(drop=True)
    df.to_parquet(DART_CORP_PARQUET)
    log(f"[dart] dart_corp.parquet 저장 ({len(df)}건)")
    return df


# ===========================================================================
# 2) 시장구분 corp_cls (Y=KOSPI, K=KOSDAQ)
#    - DART company.json(회사당 1콜) 대신 FDR StockListing 으로 매핑해 호출 절약
# ===========================================================================
def build_corp_cls(corp_df: pd.DataFrame, refresh: bool = False) -> Dict[str, str]:
    """corp_cls/<corp>.json 생성. 반환 {corp_code: 'Y'|'K'}."""
    log("[dart] 시장구분(corp_cls) 매핑: FinanceDataReader StockListing('KRX')")
    market_by_code: Dict[str, str] = {}
    try:
        import FinanceDataReader as fdr
        listing = fdr.StockListing("KRX")
        code_col = "Code" if "Code" in listing.columns else "Symbol"
        for _, row in listing.iterrows():
            code = str(row.get(code_col, "")).strip().zfill(6)
            mkt = str(row.get("Market", "")).strip().upper()
            if mkt == "KOSPI":
                market_by_code[code] = "Y"
            elif mkt == "KOSDAQ":
                market_by_code[code] = "K"
    except Exception as e:
        log(f"[dart] ⚠ FDR 시장구분 조회 실패({e}) — DART company.json 폴백 사용")
        return _build_corp_cls_via_dart(corp_df, refresh=refresh)

    out: Dict[str, str] = {}
    written = 0
    for _, r in corp_df.iterrows():
        cc = str(r["corp_code"])
        sc = str(r["stock_code"]).zfill(6)
        cls = market_by_code.get(sc)
        if cls not in ("Y", "K"):
            continue
        out[cc] = cls
        path = os.path.join(CORP_CLS_DIR, f"{cc}.json")
        if refresh or not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"corp_code": cc, "stock_code": sc, "corp_cls": cls},
                          f, ensure_ascii=False)
            written += 1
    log(f"[dart] corp_cls: KOSPI/KOSDAQ {len(out)}건 (신규 {written} 파일)")
    return out


def _build_corp_cls_via_dart(corp_df: pd.DataFrame, refresh: bool = False) -> Dict[str, str]:
    """폴백: DART company.json 으로 corp_cls 조회(회사당 1콜)."""
    out: Dict[str, str] = {}
    for _, r in corp_df.iterrows():
        cc = str(r["corp_code"])
        path = os.path.join(CORP_CLS_DIR, f"{cc}.json")
        if not refresh and os.path.exists(path):
            try:
                cls = json.load(open(path, encoding="utf-8")).get("corp_cls")
                if cls in ("Y", "K"):
                    out[cc] = cls
                continue
            except Exception:
                pass
        try:
            data = _get_json("company.json", {"corp_code": cc})
        except QuotaExceeded:
            log("[dart] ⚠ corp_cls 조회 중 일일 한도 초과 — 중단")
            break
        if str(data.get("status")) != "000":
            continue
        cls = data.get("corp_cls")
        if cls in ("Y", "K"):
            out[cc] = cls
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"corp_code": cc, "corp_cls": cls,
                       "stock_code": data.get("stock_code")}, f, ensure_ascii=False)
    return out


# ===========================================================================
# 3) 발행주식총수 shares/<corp>_2024_11011.json
# ===========================================================================
def fetch_shares_one(corp_code: str, year: int = SHARES_REFERENCE_YEAR,
                     reprt: str = SHARES_REPRT_CODE) -> Optional[int]:
    """보통주 발행주식의 총수(istc_totqy)를 반환."""
    data = _get_json("stockTotqySttus.json", {
        "corp_code": corp_code, "bsns_year": str(year), "reprt_code": reprt})
    if str(data.get("status")) != "000":
        return None
    common = None
    total = None
    for it in data.get("list", []):
        se = (it.get("se") or "").strip()
        qty = _to_float(it.get("istc_totqy"))
        if qty is None:
            continue
        if se.startswith("보통주"):
            common = qty
        elif se.startswith("합계"):
            total = qty
    val = common if common is not None else total
    if val is None or not (0 < val < 10_000_000_000):
        return None
    return int(val)


def _shares_path(corp_code: str) -> str:
    return os.path.join(
        SHARES_DIR, f"{corp_code}_{SHARES_REFERENCE_YEAR}_{SHARES_REPRT_CODE}.json")


def build_shares(corp_codes: List[str], max_workers: int = 12) -> int:
    """대상 기업들의 발행주식총수를 병렬 수집. 반환 신규 저장 건수."""
    todo = [cc for cc in corp_codes if not os.path.exists(_shares_path(cc))]
    log(f"[dart] 발행주식총수 수집 대상 {len(todo)}/{len(corp_codes)}")
    if not todo:
        return 0
    written = 0

    def _one(cc: str):
        if _STOP.is_set():
            return None
        v = fetch_shares_one(cc)
        if v is not None:
            with open(_shares_path(cc), "w", encoding="utf-8") as f:
                json.dump({"corp_code": cc, "bsns_year": SHARES_REFERENCE_YEAR,
                           "reprt_code": SHARES_REPRT_CODE,
                           "shares_outstanding": v}, f, ensure_ascii=False)
            return cc
        return None

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_one, cc): cc for cc in todo}
        done = 0
        for fut in as_completed(futs):
            try:
                if fut.result():
                    written += 1
            except QuotaExceeded:
                log("[dart] ⚠ 발행주식총수 수집 중 일일 한도(020) 초과 — 남은 작업 중단")
                break
            except ConnectionBlocked:
                log("[dart] ⛔ DART 연결이 반복 리셋됨(IP 차단/스로틀 추정) — 중단. "
                    "잠시(수십 분~수 시간) 뒤 --workers 2~4 로 재실행하세요.")
                break
            except Exception:
                pass
            done += 1
            if done % 300 == 0:
                log(f"  shares 진행 {done}/{len(todo)} (저장 {written})")
    log(f"[dart] 발행주식총수 저장 {written}건")
    return written


# ===========================================================================
# 4) 재무제표 financials_full/<corp>_<year>.json
# ===========================================================================
# 당기순이익(총액) 표준 계정코드 — 지배+비지배 합계
_NET_INCOME_IDS = ("ifrs-full_ProfitLoss", "ifrs_ProfitLoss")
# 지배기업 소유주 귀속 순이익 — 총액이 없을 때의 fallback
_NET_INCOME_PARENT_IDS = ("ifrs-full_ProfitLossAttributableToOwnersOfParent",
                          "ifrs_ProfitLossAttributableToOwnersOfParent")
# 자본총계 표준 계정코드 / 지배기업 소유주 귀속 자본(fallback)
_EQUITY_IDS = ("ifrs-full_Equity", "ifrs_Equity")
_EQUITY_PARENT_IDS = ("ifrs-full_EquityAttributableToOwnersOfParent",
                      "ifrs_EquityAttributableToOwnersOfParent")


def _is_total_ni_name(nm: str) -> bool:
    """'연결당기순이익(순손실)' 등 총액 당기순이익 명칭인지.
    지배/비지배 귀속분·포괄손익은 총액이 아니므로 제외."""
    if not any(k in nm for k in ("당기순이익", "당기순손익", "당기순손실")):
        return False
    if any(k in nm for k in ("지배", "비지배", "소유주", "주주지분", "포괄")):
        return False
    return True


def _is_parent_ni_name(nm: str) -> bool:
    """'지배기업 소유주에게 귀속되는 당기순이익' 계열(총액 없을 때 fallback)."""
    if not any(k in nm for k in ("당기순이익", "당기순손익", "당기순손실")):
        return False
    if "비지배" in nm:
        return False
    return ("지배기업" in nm or "소유주" in nm or "주주지분" in nm)


def _extract_fin(items: List[dict]) -> dict:
    """전체 재무제표 list 에서 net_income(당기순이익)/equity(자본총계) 추출.

    당기순이익 추출 우선순위(2018 등 비표준 태깅 대응):
      1) 표준 코드 ifrs-full_ProfitLoss (IS 우선, 없으면 CIS)
      2) 명칭 총액('당기순이익'/'연결당기순이익(순손실)' 등, 귀속분·포괄 제외)
      3) 지배기업 소유주 귀속 순이익 (코드 → 명칭)
    자본총계: 표준 코드 → 명칭('자본총계' 포함) → 지배주주 귀속 자본.
    """
    ni_id_is = ni_id_cis = None   # 표준코드 총액
    ni_nm = None                  # 명칭 총액
    ni_parent_id = ni_parent_nm = None  # 지배주주 귀속(fallback)
    eq_id = eq_nm = eq_parent = None
    for it in items:
        sj = (it.get("sj_div") or "").strip()
        aid = (it.get("account_id") or "").strip()
        nm = (it.get("account_nm") or "").strip()
        amt = _to_float(it.get("thstrm_amount"))
        if amt is None:
            continue
        # ---- 자본총계 (재무상태표) ----
        if sj == "BS":
            if aid in _EQUITY_IDS:
                if eq_id is None:
                    eq_id = amt
            elif aid in _EQUITY_PARENT_IDS:
                if eq_parent is None:
                    eq_parent = amt
            elif "자본총계" in nm and eq_nm is None:
                eq_nm = amt
        # ---- 당기순이익 (손익/포괄손익계산서) ----
        if sj in ("IS", "CIS"):
            if aid in _NET_INCOME_IDS:
                if sj == "IS":
                    if ni_id_is is None:
                        ni_id_is = amt
                elif ni_id_cis is None:
                    ni_id_cis = amt
            elif aid in _NET_INCOME_PARENT_IDS:
                if ni_parent_id is None:
                    ni_parent_id = amt
            elif _is_total_ni_name(nm):
                if ni_nm is None:
                    ni_nm = amt
            elif _is_parent_ni_name(nm) and ni_parent_nm is None:
                ni_parent_nm = amt
    net_income = (ni_id_is if ni_id_is is not None else
                  ni_id_cis if ni_id_cis is not None else
                  ni_nm if ni_nm is not None else
                  ni_parent_id if ni_parent_id is not None else
                  ni_parent_nm)
    equity = (eq_id if eq_id is not None else
              eq_nm if eq_nm is not None else eq_parent)
    return {"net_income": net_income, "equity": equity}


def fetch_fin_one(corp_code: str, year: int) -> Optional[dict]:
    """연결(CFS) 우선, 없으면 개별(OFS) 사업보고서 재무 수집."""
    for fs_div in ("CFS", "OFS"):
        data = _get_json("fnlttSinglAcntAll.json", {
            "corp_code": corp_code, "bsns_year": str(year),
            "reprt_code": "11011", "fs_div": fs_div})
        status = str(data.get("status"))
        if status == "013":  # 해당 fs_div 데이터 없음 → 다음 것 시도
            continue
        if status != "000":
            return None
        fin = _extract_fin(data.get("list", []))
        if fin["net_income"] is not None or fin["equity"] is not None:
            fin.update({"corp_code": corp_code, "year": year, "fs_div": fs_div})
            return fin
    return None


def _fin_path(corp_code: str, year: int) -> str:
    return os.path.join(FIN_FULL_DIR, f"{corp_code}_{year}.json")


def bulk(corp_codes: List[str], years: List[int], max_workers: int = 12) -> int:
    """
    data_layer.update_financials() 가 호출하는 진입점.
    (corp_code, year) 조합별로 financials_full/<corp>_<year>.json 생성.
    이미 존재하면 건너뜀. 반환: 신규 저장 파일 수.
    """
    tasks = [(cc, yr) for yr in years for cc in corp_codes
             if not os.path.exists(_fin_path(cc, yr))]
    log(f"[dart] 재무제표 수집 대상 {len(tasks)}건 "
        f"(기업 {len(corp_codes)} × 연도 {years})")
    if not tasks:
        return 0
    written = 0

    def _one(task):
        cc, yr = task
        if _STOP.is_set():
            return None
        fin = fetch_fin_one(cc, yr)
        if fin is not None:
            with open(_fin_path(cc, yr), "w", encoding="utf-8") as f:
                json.dump(fin, f, ensure_ascii=False)
            return task
        return None

    t0 = time.time()
    quota_hit = False
    blocked = False
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_one, t): t for t in tasks}
        done = 0
        for fut in as_completed(futs):
            try:
                if fut.result():
                    written += 1
            except QuotaExceeded:
                quota_hit = True
            except ConnectionBlocked:
                blocked = True
            except Exception:
                pass
            done += 1
            if done % 500 == 0 or done == len(tasks):
                el = time.time() - t0
                log(f"  재무 진행 {done}/{len(tasks)} "
                    f"(저장 {written}, {done/max(el,1e-9):.1f}/s)")
    if quota_hit:
        log("[dart] ⚠ 재무 수집 중 일일 한도(020) 초과 — 이미 받은 파일은 보존됨. "
            "내일 같은 명령을 다시 실행하면 나머지만 이어서 받습니다.")
    if blocked:
        log("[dart] ⛔ DART 연결이 반복 리셋됨(IP 차단/스로틀 추정) — 저장 0에 가까울 수 있음. "
            "잠시(수십 분~수 시간) 뒤 --workers 2~4 로 재실행하세요. "
            "이미 받은 파일은 건너뛰므로 안전하게 이어집니다.")
    log(f"[dart] 재무제표 저장 {written}건 ({time.time()-t0:.0f}s)")
    return written


# ===========================================================================
# 5) (선택) 가격/지수 부트스트랩 — FinanceDataReader (DART 아님)
#    data_layer.update_prices 는 '이미 존재하는' parquet 만 증분 갱신하므로,
#    최초 1회 전체 다운로드는 여기서 수행해야 한다.
# ===========================================================================
PRICE_START = "2019-06-01"


def bootstrap_indices() -> None:
    import FinanceDataReader as fdr
    end = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
    for name, sym in (("kospi_index", "KS11"), ("kosdaq_index", "KQ11")):
        path = os.path.join(BT_CACHE, f"{name}.parquet")
        if os.path.exists(path):
            continue
        try:
            df = fdr.DataReader(sym, PRICE_START, end)
            df[["Close"]].to_parquet(path)
            log(f"[price] {name} 저장 ({len(df)}행)")
        except Exception as e:
            log(f"[price] ⚠ {name} 실패({e})")


def bootstrap_prices(stock_codes: List[str], max_workers: int = 8) -> int:
    """개별종목 OHLCV 일봉 최초 다운로드(FDR). 반환 신규 저장 종목수."""
    import FinanceDataReader as fdr
    end = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
    need = ["Open", "High", "Low", "Close", "Volume"]
    todo = [c for c in stock_codes
            if not os.path.exists(os.path.join(PRICE_DIR, f"{c}.parquet"))]
    log(f"[price] 가격 부트스트랩 대상 {len(todo)}/{len(stock_codes)} 종목")
    if not todo:
        return 0
    saved = 0

    def _one(code: str):
        try:
            df = fdr.DataReader(code, PRICE_START, end)
            if df is None or df.empty or not all(c in df.columns for c in need):
                return None
            df[need].dropna(how="all").to_parquet(
                os.path.join(PRICE_DIR, f"{code}.parquet"))
            return code
        except Exception:
            return None

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_one, c): c for c in todo}
        done = 0
        for fut in as_completed(futs):
            if fut.result():
                saved += 1
            done += 1
            if done % 200 == 0 or done == len(todo):
                el = time.time() - t0
                log(f"  price 진행 {done}/{len(todo)} (저장 {saved}, {done/max(el,1e-9):.1f}/s)")
    log(f"[price] 가격 저장 {saved}종목 ({time.time()-t0:.0f}s)")
    return saved


# ===========================================================================
# 6) 전체 파이프라인
# ===========================================================================
def build_all(years: Optional[List[int]] = None, max_workers: int = 4,
              with_prices: bool = False, refresh_corp: bool = False,
              delay: Optional[float] = None,
              per_call_sleep: Optional[float] = None) -> None:
    years = years or list(FIN_YEARS)
    if delay is not None:
        set_min_interval(delay)
    if per_call_sleep is not None:
        set_per_call_sleep(per_call_sleep)
    t0 = time.time()
    log(f"[dart] ===== 전체 데이터 수집 시작 (워커 {max_workers}, "
        f"호출간격 {_MIN_INTERVAL:.2f}s, 요청당대기 {_PER_CALL_SLEEP:.2f}s) =====")

    corp_df = build_dart_corp(refresh=refresh_corp)
    cls_map = build_corp_cls(corp_df, refresh=refresh_corp)

    # 유니버스 후보: KOSPI/KOSDAQ + 비우선주/스팩/리츠 (data_layer.keep 필터와 동일)
    name_by_corp = dict(zip(corp_df["corp_code"].astype(str),
                            corp_df["corp_name"].astype(str)))
    keep = [cc for cc, cls in cls_map.items()
            if cls in ("Y", "K") and not is_excluded_name(name_by_corp.get(cc, ""))]
    log(f"[dart] 재무/주식수 수집 대상(유니버스 후보): {len(keep)}")

    build_shares(keep, max_workers=max_workers)
    # 주식수 확보된 기업만 재무 수집(시총·EPS 계산에 주식수 필요)
    with_shares = [cc for cc in keep if os.path.exists(_shares_path(cc))]
    log(f"[dart] 주식수 확보 {len(with_shares)}건 → 재무 수집")
    bulk(with_shares, years, max_workers=max_workers)

    if with_prices:
        stock_codes = corp_df.set_index("corp_code")["stock_code"].astype(str)
        codes = [stock_codes[cc].zfill(6) for cc in with_shares if cc in stock_codes.index]
        bootstrap_indices()
        bootstrap_prices(sorted(set(codes)))

    log(f"[dart] ===== 전체 수집 완료 ({time.time()-t0:.0f}s) =====")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="DART 재무/메타 캐시 수집기")
    ap.add_argument("--years", type=int, nargs="*", default=None,
                    help=f"수집 연도(기본 {list(FIN_YEARS)})")
    ap.add_argument("--workers", type=int, default=4,
                    help="DART 동시 요청 수(기본 4). 연결 리셋/차단 시 1~2로 낮추세요")
    ap.add_argument("--delay", type=float, default=_MIN_INTERVAL,
                    help=f"DART 호출 시작 사이 최소 간격(초, 전역, 기본 {_MIN_INTERVAL}). "
                         "차단이 잦으면 0.5~1.0 으로 올리세요")
    ap.add_argument("--sleep", type=float, default=_PER_CALL_SLEEP,
                    help=f"각 요청을 보낸 뒤 워커가 무조건 쉬는 시간(초, 기본 {_PER_CALL_SLEEP}). "
                         "IP 차단이 잦으면 0.5~1.0 으로 올리세요")
    ap.add_argument("--prices", action="store_true",
                    help="FDR 가격/지수 부트스트랩도 함께 수행")
    ap.add_argument("--refresh-corp", action="store_true",
                    help="dart_corp/corp_cls 재생성")
    args = ap.parse_args()
    build_all(years=args.years, max_workers=args.workers,
              with_prices=args.prices, refresh_corp=args.refresh_corp,
              delay=args.delay, per_call_sleep=args.sleep)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
