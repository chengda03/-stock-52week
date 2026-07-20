# -*- coding: utf-8 -*-
"""
fetch_debt_opmargin.py — 부채비율·영업이익률 검증용 재무 부가필드 수집기
=======================================================================
strategy_core_debt2.py(부채비율 필터)·strategy_core_opmargin.py(영업이익률 필터)
백테스트에 필요한 다음 4개 재무 항목을 DART에서 수집한다(FY2018~FY2025):
  · 부채총계 (liabilities)  · 자본총계 (equity)   → 부채비율 = 부채총계/자본총계×100
  · 영업이익 (op_income)    · 매출액   (revenue)  → 영업이익률 = 영업이익/매출액×100
                                                    (+ 전년比 증가 판정에도 사용)

기존 ROE/EPS 캐시(financials_full)와 '같은 종목·같은 연도'를 대상으로 하되,
기존 캐시 파일은 절대 건드리지 않는다. 결과는 별도 위치에만 저장한다:
  · 원본(재개용, idempotent): data/cache_52w_bt/financials_fin2/<corp>_<year>.json
  · 집계 캐시(parquet)       : data/cache_52w_bt/debt_opmargin.parquet
    컬럼: corp_code, year, net_income, equity, liabilities, op_income, revenue,
          debt_ratio(%), op_margin(%), fs_div

★ 안전장치 (이 프로젝트는 과거 IP 차단을 여러 번 겪음 — 인계문서 원칙 준수)
  - 저수준 호출/스로틀/세션/중단로직은 fetch_dart_full_financials 의 것을 재사용.
  - 기본 workers=2, --sleep 0.4, --delay 0.4 (요청 속도 물리 제한).
  - 이미 받은 (corp,year) json 은 건너뜀 → 일일 한도 초과/중단 후 재개 안전.
  - 일일 한도(020) 초과: --auto-wait 면 다음날 00:30(KST)까지 자동 대기 후 재개,
    아니면 즉시 멈추고 안내(내일 같은 명령으로 이어받기).
  - IP 차단 징후(연속 연결리셋 누적)면 fd._STOP 이 서고 즉시 전체 중단 → 몇 시간씩
    재시도하지 않는다(요구사항).

CLI
  python fetch_debt_opmargin.py                       # 기본(workers 2, sleep/delay 0.4)
  python fetch_debt_opmargin.py --workers 1 --sleep 0.5   # 가장 안전(느림)
  python fetch_debt_opmargin.py --limit 5000          # 이번 실행 최대 5000건
  python fetch_debt_opmargin.py --auto-wait           # 한도 걸리면 다음날까지 자동 대기
  python fetch_debt_opmargin.py --build-only          # 수집 없이 json → parquet 집계만
  python fetch_debt_opmargin.py --dry-run             # 대상 건수/예상시간만 출력
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

# 저수준 DART 호출 기계장치 재사용(스로틀/세션/중단신호/파서/예외)
import fetch_dart_full_financials as fd
from backtest_roe_eps_event import BT_CACHE, FIN_FULL_DIR, log

FIN2_DIR = os.path.join(BT_CACHE, "financials_fin2")
os.makedirs(FIN2_DIR, exist_ok=True)
PARQUET_PATH = os.path.join(BT_CACHE, "debt_opmargin.parquet")

KST = timezone(timedelta(hours=9))

# ---- 계정 식별 (표준 IFRS 코드 + 한글 명칭 병행 매칭) ----------------------
_OP_IDS = ("dart_OperatingIncomeLoss", "dart_OperatingIncome",
           "ifrs-full_ProfitLossFromOperatingActivities",
           "ifrs_ProfitLossFromOperatingActivities")
_LIAB_IDS = ("ifrs-full_Liabilities", "ifrs_Liabilities")
_REV_IDS = ("ifrs-full_Revenue", "ifrs_Revenue", "dart_OperatingRevenue")
# 매출액 명칭(정규화 후 '완전일치'로만 — '매출총이익' 등 오매칭 방지)
_REV_NAMES = ("매출액", "수익(매출액)", "영업수익", "매출및지분법손익", "영업수익(매출액)")


def _is_op_name(nm: str) -> bool:
    if "영업이익" not in nm:
        return False
    return not ("률" in nm or "세전" in nm or "비율" in nm)


def _is_liab_name(nm: str) -> bool:
    return nm.replace(" ", "") == "부채총계"


def _is_rev_name(nm: str) -> bool:
    return nm.replace(" ", "") in {n.replace(" ", "") for n in _REV_NAMES}


def _extract(items: list[dict]) -> dict:
    """재무제표 list → {net_income, equity, op_income, liabilities, revenue, rcept_*}."""
    base = fd._extract_fin(items)          # {net_income, equity}
    op = liab = rev = None
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
        if sj in ("IS", "CIS"):
            if op is None and (aid in _OP_IDS or _is_op_name(nm)):
                op = amt
            if rev is None and (aid in _REV_IDS or _is_rev_name(nm)):
                rev = amt
        elif sj == "BS" and liab is None:
            if aid in _LIAB_IDS or _is_liab_name(nm):
                liab = amt
    dt = f"{rcept[:4]}-{rcept[4:6]}-{rcept[6:8]}" if rcept else None
    base.update(op_income=op, liabilities=liab, revenue=rev, rcept_no=rcept, rcept_dt=dt)
    return base


def fetch_one(corp_code: str, year: int) -> Optional[dict]:
    """연결(CFS) 우선, 없으면 개별(OFS) 사업보고서에서 부가필드 수집."""
    for fs_div in ("CFS", "OFS"):
        data = fd._get_json("fnlttSinglAcntAll.json", {
            "corp_code": corp_code, "bsns_year": str(year),
            "reprt_code": "11011", "fs_div": fs_div})
        status = str(data.get("status"))
        if status == "013":     # 해당 fs_div 데이터 없음 → 다음 것
            continue
        if status != "000":
            return None
        rec = _extract(data.get("list", []))
        if any(rec.get(k) is not None for k in
               ("net_income", "equity", "op_income", "liabilities", "revenue")):
            rec.update({"corp_code": corp_code, "year": year, "fs_div": fs_div})
            return rec
    return None


def _path(corp_code: str, year: int) -> str:
    return os.path.join(FIN2_DIR, f"{corp_code}_{year}.json")


def collect_targets() -> list[tuple[str, int]]:
    """대상 (corp,year): financials_full 기준(ROE/EPS와 동일), 이번 캐시 미수집분만."""
    targets = []
    for p in glob.glob(os.path.join(FIN_FULL_DIR, "*.json")):
        b = os.path.splitext(os.path.basename(p))[0]
        try:
            cc, yr = b.rsplit("_", 1)
            year = int(yr)
        except ValueError:
            continue
        if not os.path.exists(_path(cc, year)):
            targets.append((cc, year))
    return targets


def build_parquet() -> int:
    """financials_fin2/*.json 을 모아 debt_opmargin.parquet 로 집계."""
    rows = []
    for p in glob.glob(os.path.join(FIN2_DIR, "*.json")):
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        eq = d.get("equity"); liab = d.get("liabilities")
        op = d.get("op_income"); rev = d.get("revenue")
        debt_ratio = (liab / eq * 100.0) if (liab is not None and eq and eq > 0) else None
        op_margin = (op / rev * 100.0) if (op is not None and rev and rev > 0) else None
        rows.append({
            "corp_code": d.get("corp_code"), "year": d.get("year"),
            "net_income": d.get("net_income"), "equity": eq,
            "liabilities": liab, "op_income": op, "revenue": rev,
            "debt_ratio": debt_ratio, "op_margin": op_margin,
            "fs_div": d.get("fs_div"), "rcept_dt": d.get("rcept_dt"),
        })
    if not rows:
        log("[fin2] 집계할 json 이 없습니다.")
        return 0
    df = pd.DataFrame(rows).sort_values(["corp_code", "year"]).reset_index(drop=True)
    df.to_parquet(PARQUET_PATH)
    n_debt = int(df["debt_ratio"].notna().sum())
    n_opm = int(df["op_margin"].notna().sum())
    log(f"[fin2] parquet 저장: {PARQUET_PATH}  (행 {len(df)}, "
        f"부채비율 유효 {n_debt}, 영업이익률 유효 {n_opm})")
    return len(df)


def _seconds_to_next_kst_reset() -> float:
    """다음 KST 00:30 까지 남은 초(일일 한도 리셋 대기용)."""
    now = datetime.now(KST)
    nxt = (now + timedelta(days=1)).replace(hour=0, minute=30, second=0, microsecond=0)
    return max(60.0, (nxt - now).total_seconds())


def _estimate(n: int, delay: float, sleep: float) -> str:
    """콜 수 기준 예상 소요시간(전역 최소간격이 처리량을 지배)."""
    per = max(delay, sleep, 0.05)  # 전역 직렬화 간격이 병목
    sec = n * per
    h = sec / 3600.0
    return f"약 {sec/60:.0f}분(≈{h:.1f}시간), 처리량 ~{1/per:.1f}콜/s 가정"


def run_collection(workers: int, limit: Optional[int], auto_wait: bool) -> None:
    targets_all = collect_targets()
    if limit:
        targets_all = targets_all[:limit]
    if not targets_all:
        log("[fin2] 수집 대상 없음 — 모두 수집 완료 상태.")
        build_parquet()
        return

    log(f"[fin2] 수집 시작: 대상 {len(targets_all)}건 "
        f"(workers={workers}, sleep={fd._PER_CALL_SLEEP}s, delay={fd._MIN_INTERVAL}s)")

    remaining = list(targets_all)
    while remaining:
        fd._STOP.clear()            # 재개 시 중단신호 초기화
        fd._CONN_ERRORS = 0
        written = done = 0
        quota = blocked = False
        t0 = time.time()

        def _one(task):
            cc, yr = task
            if fd._STOP.is_set():
                return None
            rec = fetch_one(cc, yr)
            if rec is not None:
                with open(_path(cc, yr), "w", encoding="utf-8") as f:
                    json.dump(rec, f, ensure_ascii=False)
                return task
            return None

        batch = list(remaining)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_one, t): t for t in batch}
            for fut in as_completed(futs):
                try:
                    if fut.result():
                        written += 1
                except fd.QuotaExceeded:
                    quota = True
                except fd.ConnectionBlocked:
                    blocked = True
                except Exception:
                    pass
                done += 1
                if done % 500 == 0 or done == len(batch):
                    el = time.time() - t0
                    log(f"  [fin2] 진행 {done}/{len(batch)} "
                        f"(이번배치 저장 {written}, {done/max(el,1e-9):.1f}/s)")

        # 이번 배치 후 미수집분 재계산(성공분 제외)
        remaining = [t for t in remaining if not os.path.exists(_path(*t))]
        build_parquet()   # 중간 집계(진행상황 보존)

        if blocked:
            log("[fin2] ⛔ 연결 반복 리셋(IP 차단 추정) — 즉시 중단. "
                "몇 시간 뒤 --workers 1 로 재실행하세요. (재시도 안 함)")
            return
        if quota:
            if auto_wait and remaining:
                wait = _seconds_to_next_kst_reset()
                log(f"[fin2] ⚠ 일일 한도(020) 초과 — 남은 {len(remaining)}건. "
                    f"다음날 리셋까지 {wait/3600:.1f}시간 대기 후 자동 재개(--auto-wait).")
                time.sleep(wait)
                continue
            log(f"[fin2] ⚠ 일일 한도(020) 초과 — 남은 {len(remaining)}건. "
                f"내일 같은 명령으로 이어받으세요(이미 받은 파일은 건너뜀).")
            return
        # 정상 종료(배치 전부 처리)
        if not remaining:
            break

    log("[fin2] ✅ 전체 수집 완료.")
    build_parquet()


def main():
    ap = argparse.ArgumentParser(description="부채비율/영업이익률용 재무 부가필드 수집기")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--sleep", type=float, default=0.4)
    ap.add_argument("--delay", type=float, default=0.4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--auto-wait", action="store_true",
                    help="일일 한도 초과 시 다음날까지 자동 대기 후 재개")
    ap.add_argument("--build-only", action="store_true",
                    help="수집 없이 json → parquet 집계만")
    ap.add_argument("--dry-run", action="store_true",
                    help="대상 건수/예상시간만 출력하고 종료")
    args = ap.parse_args()

    fd.set_min_interval(args.delay)
    fd.set_per_call_sleep(args.sleep)

    if args.build_only:
        build_parquet()
        return

    targets = collect_targets()
    n = len(targets) if not args.limit else min(args.limit, len(targets))
    log(f"[fin2] 미수집 대상: {len(targets)}건 "
        f"(이번 실행 {n}건) · 예상: {_estimate(n, args.delay, args.sleep)}")
    if args.dry_run:
        return

    run_collection(args.workers, args.limit, args.auto_wait)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
