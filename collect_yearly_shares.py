# -*- coding: utf-8 -*-
"""
collect_yearly_shares.py
========================
유니버스 후보(KOSPI/KOSDAQ·2024주식수 有·우선주/스팩/리츠 제외)의
연도별(2018~2025) 발행주식수(보통주 총수)를 DART 사업보고서에서 각각 수집한다.

- 저장: data/cache_52w_bt/shares/<corp>_<year>_11011.json
        {"corp_code","bsns_year","reprt_code","shares_outstanding"}
- 이미 있는 파일은 건너뜀(idempotent) → 중단/한도초과 시 재실행하면 이어받음.
- 상장 이전 연도(가격 데이터 시작 이후만 유효)는 건너뜀 → 불필요한 013 호출 절감.
- IP 차단 예방: workers 낮춤 + 요청당 강제 대기.

사용:
  python collect_yearly_shares.py --count            # 대상 건수만 계산(호출 안 함)
  python collect_yearly_shares.py --workers 2 --delay 0.4 --sleep 0.4
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

import fetch_dart_full_financials as fd
from backtest_roe_eps_event import (
    SHARES_DIR, SHARES_REPRT_CODE, PRICE_DIR, FIN_FULL_DIR, FIN_YEARS,
    load_dart_corp, load_corp_cls, load_shares, is_excluded_name, log)


def shares_path(cc: str, yr: int) -> str:
    return os.path.join(SHARES_DIR, f"{cc}_{yr}_{SHARES_REPRT_CODE}.json")


def build_tasks():
    """(corp, year) 대상 목록 + 진단정보 반환."""
    dart = load_dart_corp()
    corp_codes = dart["corp_code"].astype(str).tolist()
    stock_codes = dart["stock_code"].astype(str).str.zfill(6).tolist()
    names = dart["corp_name"].astype(str).tolist()
    cls = load_corp_cls(corp_codes)
    shares2024 = load_shares(corp_codes)  # 2024 단일시점(이미 수집됨)
    corp_to_stock = dict(zip(corp_codes, stock_codes))

    keep = [cc for cc, nm in zip(corp_codes, names)
            if cc in shares2024 and cls.get(cc) in ("Y", "K") and not is_excluded_name(nm)]
    keep_set = set(keep)

    # 재무제표 파일이 있는 연도 = 그 해 사업보고서를 낸(상장) 해 → 상장연도 하한 보정.
    # (가격 데이터는 2019-06부터라 상장연도 프록시로 부적합. 재무 파일로 2018 포함)
    earliest_fin = {}
    for p in glob.glob(os.path.join(FIN_FULL_DIR, "*_*.json")):
        base = os.path.splitext(os.path.basename(p))[0]
        if base.startswith("_"):
            continue
        parts = base.rsplit("_", 1)
        if len(parts) != 2:
            continue
        cc = parts[0]
        if cc not in keep_set:
            continue
        try:
            yr = int(parts[1])
        except ValueError:
            continue
        if cc not in earliest_fin or yr < earliest_fin[cc]:
            earliest_fin[cc] = yr

    # 종목별 상장연도 = min(가격시작연도, 재무최초연도) → 상장 이전 연도만 스킵
    listed_year = {}
    for cc in keep:
        sc = corp_to_stock.get(cc)
        cands = []
        p = os.path.join(PRICE_DIR, f"{sc}.parquet")
        if os.path.exists(p):
            try:
                idx = pd.read_parquet(p, columns=["Close"]).index
                cands.append(pd.to_datetime(idx.min()).year)
            except Exception:
                pass
        if cc in earliest_fin:
            cands.append(earliest_fin[cc])
        listed_year[cc] = min(cands) if cands else min(FIN_YEARS)

    years = list(FIN_YEARS)  # 2018~2025
    tasks = []
    skipped_exist = skipped_prelist = 0
    for cc in keep:
        ly = listed_year[cc]
        for yr in years:
            if yr < ly:            # 상장 이전 → 보고서 없음
                skipped_prelist += 1
                continue
            if os.path.exists(shares_path(cc, yr)):
                skipped_exist += 1
                continue
            tasks.append((cc, yr))
    return keep, tasks, skipped_exist, skipped_prelist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", action="store_true", help="대상 건수만 계산")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--delay", type=float, default=0.4)
    ap.add_argument("--sleep", type=float, default=0.4)
    args = ap.parse_args()

    keep, tasks, sk_exist, sk_pre = build_tasks()
    by_year = {}
    for _, yr in tasks:
        by_year[yr] = by_year.get(yr, 0) + 1
    log(f"[shares] 유니버스 후보 {len(keep)} · 신규수집 대상 {len(tasks)}건 "
        f"(연도별 {dict(sorted(by_year.items()))})")
    log(f"[shares] 스킵: 기존파일 {sk_exist} · 상장이전 {sk_pre}")
    if args.count or not tasks:
        est = len(tasks) / max(args.workers / (args.delay + args.sleep + 0.15), 1e-9)
        log(f"[shares] 예상 소요(대략): {est/60:.0f}분 "
            f"(workers={args.workers}, delay={args.delay}, sleep={args.sleep})")
        return

    fd.set_min_interval(args.delay)
    fd.set_per_call_sleep(args.sleep)

    written = 0
    none_cnt = 0
    quota = blocked = False
    t0 = time.time()

    def _one(task):
        cc, yr = task
        if fd._STOP.is_set():
            return None
        v = fd.fetch_shares_one(cc, yr)
        if v is not None:
            with open(shares_path(cc, yr), "w", encoding="utf-8") as f:
                json.dump({"corp_code": cc, "bsns_year": yr,
                           "reprt_code": SHARES_REPRT_CODE,
                           "shares_outstanding": int(v)}, f, ensure_ascii=False)
            return True
        return False

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_one, t): t for t in tasks}
        done = 0
        for fut in as_completed(futs):
            try:
                r = fut.result()
                if r is True:
                    written += 1
                elif r is False:
                    none_cnt += 1
            except fd.QuotaExceeded:
                quota = True
            except fd.ConnectionBlocked:
                blocked = True
            except Exception:
                pass
            done += 1
            if done % 500 == 0 or done == len(tasks):
                el = time.time() - t0
                log(f"  진행 {done}/{len(tasks)} (저장 {written}, 무데이터 {none_cnt}, "
                    f"{done/max(el,1e-9):.1f}/s)")

    if quota:
        log("[shares] ⚠ 일일 한도(020) 초과 — 이미 받은 파일 보존. 내일 재실행하면 이어서 수집.")
    if blocked:
        log("[shares] ⛔ 연결 반복 리셋(IP 차단 추정) — 중단. --workers 1 로 잠시 뒤 재실행.")
    log(f"[shares] 완료: 저장 {written} · 무데이터(013 등) {none_cnt} · 소요 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
