# -*- coding: utf-8 -*-
"""
fetch_qv_extras.py — "가치우량주 정배열"(전략2) 전용 재무 부가필드 수집기
============================================================================
전략2(backtest_qv.py)는 S1 이 쓰지 않던 다음 3가지가 추가로 필요합니다.
  · 영업이익 (op_income)      → 매수조건 §2-3 (영업이익 전년比 증가)
  · 부채총계 (liabilities)    → 매수조건 §2-6 (부채비율 ≤ 100%)
  · 공시 접수일자 (rcept_dt)  → §3 PIT(발표시점) 처리의 핵심

기존 캐시(financials_full/<corp>_<year>.json)에는 net_income/equity 만 있어서
위 3개를 얻으려면 DART fnlttSinglAcntAll.json 을 (기업,연도)별로 다시 한 번
호출해야 합니다. 결과는 S1 파일을 건드리지 않도록 '별도 폴더'에 저장합니다.
    data/cache_52w_bt/financials_qv/<corp>_<year>.json
        {net_income, equity, op_income, liabilities,
         rcept_no, rcept_dt("YYYY-MM-DD"), fs_div, corp_code, year}

★ 안전장치 (이 프로젝트는 과거 IP 차단을 여러 번 겪음)
  - 저수준 호출/스로틀/세션/중단로직은 fetch_dart_full_financials 의 것을 그대로 재사용.
  - 기본 workers=2, --sleep 0.4, --delay 0.4 (요청 속도 물리 제한).
  - 이미 받은 파일은 건너뜀(idempotent) → 일일 한도(020) 초과나 중단 후 재개 안전.

★ 수집 대상 최소화
  - "S1 이 이미 재무를 확보한 (기업,연도)"에 대해서만 수집합니다.
    (financials_full 폴더의 파일 목록을 그대로 대상으로 삼음 → 불필요한 호출 최소화)

CLI
  python fetch_qv_extras.py                      # 기본(workers 2, sleep 0.4)
  python fetch_qv_extras.py --workers 1 --sleep 0.5   # 가장 안전(느림)
  python fetch_qv_extras.py --limit 5000         # 이번 실행에서 최대 5000건만
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

# 저수준 DART 호출 기계장치를 그대로 재사용(스로틀/세션/중단신호/파서)
import fetch_dart_full_financials as fd
from backtest_roe_eps_event import BT_CACHE, FIN_FULL_DIR, log

FIN_QV_DIR = os.path.join(BT_CACHE, "financials_qv")
os.makedirs(FIN_QV_DIR, exist_ok=True)

# ---- 계정 식별 (표준 IFRS 코드 + 한글 명칭 병행 매칭) ----------------------
# 영업이익: DART 확장코드/표준코드 + 명칭 '영업이익' (비율/세전 등은 제외)
_OP_IDS = ("dart_OperatingIncomeLoss", "dart_OperatingIncome",
           "ifrs-full_ProfitLossFromOperatingActivities",
           "ifrs_ProfitLossFromOperatingActivities")
# 부채총계
_LIAB_IDS = ("ifrs-full_Liabilities", "ifrs_Liabilities")


def _is_op_name(nm: str) -> bool:
    """'영업이익'(또는 '영업이익(손실)') 명칭인지. '영업이익률'·'세전' 등은 제외."""
    if "영업이익" not in nm:
        return False
    if "률" in nm or "세전" in nm or "비율" in nm:
        return False
    return True


def _is_liab_name(nm: str) -> bool:
    return nm.replace(" ", "") == "부채총계"


def _extract_qv(items: list[dict]) -> dict:
    """재무제표 list 에서 net_income/equity/op_income/liabilities/rcept 추출.

    - net_income/equity 는 S1 과 100% 동일한 로직(fd._extract_fin) 재사용.
    - op_income: 손익/포괄손익(IS/CIS)에서 표준코드 또는 명칭('영업이익') 매칭.
    - liabilities: 재무상태표(BS)에서 표준코드 또는 명칭('부채총계') 매칭.
    - rcept_no: 응답 아이템 공통 접수번호(14자리). 앞 8자리 = 접수일(YYYYMMDD).
    """
    base = fd._extract_fin(items)          # {net_income, equity}
    op = liab = None
    rcept: Optional[str] = None
    for it in items:
        if rcept is None:
            rn = (it.get("rcept_no") or "").strip()
            if len(rn) >= 8 and rn[:8].isdigit():
                rcept = rn
        sj = (it.get("sj_div") or "").strip()
        aid = (it.get("account_id") or "").strip()
        nm = (it.get("account_nm") or "").strip()
        amt = fd._to_float(it.get("thstrm_amount"))
        if amt is None:
            continue
        if sj in ("IS", "CIS") and op is None:
            if aid in _OP_IDS or _is_op_name(nm):
                op = amt
        elif sj == "BS" and liab is None:
            if aid in _LIAB_IDS or _is_liab_name(nm):
                liab = amt
    dt = None
    if rcept:
        dt = f"{rcept[:4]}-{rcept[4:6]}-{rcept[6:8]}"
    base.update(op_income=op, liabilities=liab, rcept_no=rcept, rcept_dt=dt)
    return base


def fetch_qv_one(corp_code: str, year: int) -> Optional[dict]:
    """연결(CFS) 우선, 없으면 개별(OFS) 사업보고서에서 QV 부가필드 수집."""
    for fs_div in ("CFS", "OFS"):
        data = fd._get_json("fnlttSinglAcntAll.json", {
            "corp_code": corp_code, "bsns_year": str(year),
            "reprt_code": "11011", "fs_div": fs_div})
        status = str(data.get("status"))
        if status == "013":     # 해당 fs_div 데이터 없음 → 다음 것 시도
            continue
        if status != "000":
            return None
        qv = _extract_qv(data.get("list", []))
        # 최소 하나라도 건졌으면 저장(영업이익/부채/접수일 중 하나라도)
        if any(qv.get(k) is not None for k in
               ("net_income", "equity", "op_income", "liabilities", "rcept_dt")):
            qv.update({"corp_code": corp_code, "year": year, "fs_div": fs_div})
            return qv
    return None


def _qv_path(corp_code: str, year: int) -> str:
    return os.path.join(FIN_QV_DIR, f"{corp_code}_{year}.json")


def collect_targets() -> list[tuple[str, int]]:
    """수집 대상 (corp_code, year): S1이 이미 확보한 재무파일 기준, QV 미수집분만."""
    targets = []
    for p in glob.glob(os.path.join(FIN_FULL_DIR, "*.json")):
        base = os.path.splitext(os.path.basename(p))[0]
        try:
            cc, yr = base.rsplit("_", 1)
            year = int(yr)
        except ValueError:
            continue
        if not os.path.exists(_qv_path(cc, year)):
            targets.append((cc, year))
    return targets


def main():
    import argparse
    ap = argparse.ArgumentParser(description="전략2용 재무 부가필드(영업이익/부채/접수일) 수집기")
    ap.add_argument("--workers", type=int, default=2,
                    help="DART 동시 요청 수(기본 2). 차단 시 1로 낮추세요")
    ap.add_argument("--sleep", type=float, default=0.4,
                    help="각 요청 뒤 워커 강제 대기(초, 기본 0.4)")
    ap.add_argument("--delay", type=float, default=0.4,
                    help="DART 호출 시작 사이 전역 최소 간격(초, 기본 0.4)")
    ap.add_argument("--limit", type=int, default=None,
                    help="이번 실행에서 처리할 최대 건수(생략 시 전부)")
    args = ap.parse_args()

    fd.set_min_interval(args.delay)
    fd.set_per_call_sleep(args.sleep)

    targets = collect_targets()
    if args.limit:
        targets = targets[:args.limit]
    log(f"[qv] 수집 대상 {len(targets)}건 "
        f"(workers={args.workers}, sleep={args.sleep}s, delay={args.delay}s)")
    if not targets:
        log("[qv] 모두 수집 완료 — 할 일 없음")
        return

    written = done = 0
    quota_hit = blocked = False
    t0 = time.time()

    def _one(task):
        cc, yr = task
        if fd._STOP.is_set():
            return None
        qv = fetch_qv_one(cc, yr)
        if qv is not None:
            with open(_qv_path(cc, yr), "w", encoding="utf-8") as f:
                json.dump(qv, f, ensure_ascii=False)
            return task
        return None

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_one, t): t for t in targets}
        for fut in as_completed(futs):
            try:
                if fut.result():
                    written += 1
            except fd.QuotaExceeded:
                quota_hit = True
            except fd.ConnectionBlocked:
                blocked = True
            except Exception:
                pass
            done += 1
            if done % 500 == 0 or done == len(targets):
                el = time.time() - t0
                log(f"  qv 진행 {done}/{len(targets)} (저장 {written}, {done/max(el,1e-9):.1f}/s)")

    if quota_hit:
        log("[qv] ⚠ 일일 한도(020) 초과 — 이미 받은 파일은 보존. 내일 같은 명령으로 이어받으세요.")
    if blocked:
        log("[qv] ⛔ 연결 반복 리셋(IP 차단 추정) — 잠시 뒤 --workers 1 로 재실행하세요.")
    log(f"[qv] 완료: 저장 {written}건 ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
