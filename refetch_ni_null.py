# -*- coding: utf-8 -*-
"""
refetch_ni_null.py
==================
net_income 이 null 인 기존 재무 파일(주로 2018년, 추출버그 피해분)만 정확히
타겟해서 DART 에서 다시 받아 개선된 _extract_fin 으로 재추출한다.

- file_missing(013)은 대상이 아니다(원래 데이터 없음). 이미 존재하는 파일 중
  net_income == None 인 것만 대상.
- 재수집 성공 시 파일을 덮어쓴다(백업 후). 실패(네트워크)면 기존 파일 보존.
- 전역 rate limiter + 요청당 강제 대기로 IP 차단을 예방한다.

사용:
  python refetch_ni_null.py                 # 전체 ni_null 재수집
  python refetch_ni_null.py --limit 30      # 앞 30건만(사전 검증용)
  python refetch_ni_null.py --workers 3 --delay 0.4 --sleep 0.4
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import fetch_dart_full_financials as fd
from backtest_roe_eps_event import FIN_FULL_DIR, log

BACKUP_DIR = os.path.join(FIN_FULL_DIR, "_ni_null_backup")


def collect_targets():
    """net_income == None 인 파일 목록 → [(corp_code, year, path), ...]."""
    targets = []
    for p in glob.glob(os.path.join(FIN_FULL_DIR, "*.json")):
        base = os.path.basename(p)
        if base.startswith("_"):
            continue
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        if d.get("net_income") is None:
            cc = d.get("corp_code")
            yr = d.get("year")
            if cc is None or yr is None:
                stem = os.path.splitext(base)[0]
                parts = stem.rsplit("_", 1)
                if len(parts) == 2:
                    cc = cc or parts[0]
                    try:
                        yr = yr or int(parts[1])
                    except ValueError:
                        continue
            if cc is not None and yr is not None:
                targets.append((str(cc), int(yr), p))
    return targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="앞 N건만(0=전체)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--delay", type=float, default=0.4)
    ap.add_argument("--sleep", type=float, default=0.4)
    args = ap.parse_args()

    fd.set_min_interval(args.delay)
    fd.set_per_call_sleep(args.sleep)

    targets = collect_targets()
    by_year = {}
    for _, yr, _ in targets:
        by_year[yr] = by_year.get(yr, 0) + 1
    log(f"[refetch] net_income=null 대상 {len(targets)}건 "
        f"(연도별 {dict(sorted(by_year.items()))})")
    if args.limit:
        targets = targets[:args.limit]
        log(f"[refetch] --limit {args.limit} → {len(targets)}건만 처리")
    if not targets:
        log("[refetch] 대상 없음 — 종료")
        return

    os.makedirs(BACKUP_DIR, exist_ok=True)

    filled = still_null = failed = 0
    quota = blocked = False
    t0 = time.time()

    def _one(task):
        cc, yr, path = task
        if fd._STOP.is_set():
            return ("stop", task)
        fin = fd.fetch_fin_one(cc, yr)   # 개선된 _extract_fin 사용
        if fin is None:
            return ("fail", task)
        # 백업(최초 1회) 후 덮어쓰기
        bpath = os.path.join(BACKUP_DIR, os.path.basename(path))
        if os.path.exists(path) and not os.path.exists(bpath):
            try:
                shutil.copy2(path, bpath)
            except Exception:
                pass
        with open(path, "w", encoding="utf-8") as f:
            json.dump(fin, f, ensure_ascii=False)
        return ("filled" if fin.get("net_income") is not None else "null", task)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_one, t): t for t in targets}
        done = 0
        for fut in as_completed(futs):
            try:
                status, _ = fut.result()
                if status == "filled":
                    filled += 1
                elif status == "null":
                    still_null += 1
                elif status == "fail":
                    failed += 1
            except fd.QuotaExceeded:
                quota = True
            except fd.ConnectionBlocked:
                blocked = True
            except Exception:
                failed += 1
            done += 1
            if done % 200 == 0 or done == len(targets):
                el = time.time() - t0
                log(f"  진행 {done}/{len(targets)} "
                    f"(복구 {filled}, 여전히null {still_null}, 실패 {failed}, "
                    f"{done/max(el,1e-9):.1f}/s)")

    if quota:
        log("[refetch] ⚠ 일일 한도(020) 초과 — 이미 복구분은 보존. 내일 재실행하면 이어서 진행.")
    if blocked:
        log("[refetch] ⛔ 연결 반복 리셋(IP 차단 추정) — 중단. --workers 1 로 잠시 뒤 재실행.")
    log(f"[refetch] 완료: 복구(net_income 채움) {filled} · 여전히 null {still_null} · "
        f"실패 {failed} · 소요 {time.time()-t0:.0f}s")
    log(f"[refetch] 백업 위치: {BACKUP_DIR}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
