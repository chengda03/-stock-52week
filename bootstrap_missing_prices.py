# -*- coding: utf-8 -*-
"""
bootstrap_missing_prices.py
===========================
유니버스 후보(KOSPI/KOSDAQ · 발행주식수 有 · 우선주/스팩/리츠 제외) 중
가격(OHLCV) parquet 이 없는 종목만 정확히 타겟해 FinanceDataReader 로 받아
data/cache_52w_bt/prices/<code>.parquet 로 저장한다.

- 이미 있는 parquet 은 건너뜀(idempotent) → 중단/재개 안전.
- DART 가 아니라 FDR 소스라 IP 차단 이슈 없음.
- 지수(kospi/kosdaq)도 없으면 함께 부트스트랩.

사용:
  python bootstrap_missing_prices.py                # 빠진 것만 전체
  python bootstrap_missing_prices.py --workers 8
  python bootstrap_missing_prices.py --limit 30     # 앞 30건만(검증용)
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import backtest_roe_eps_event as ev
from backtest_roe_eps_event import (
    load_dart_corp, load_corp_cls, load_shares, is_excluded_name, PRICE_DIR, log)
import fetch_dart_full_financials as fd


def universe_stock_codes():
    """data_layer 와 동일한 유니버스 후보의 종목코드(6자리) 집합."""
    dart = load_dart_corp()
    corp_codes = dart["corp_code"].astype(str).tolist()
    stock_codes = dart["stock_code"].astype(str).str.zfill(6).tolist()
    names = dart["corp_name"].astype(str).tolist()
    cls = load_corp_cls(corp_codes)
    shares = load_shares(corp_codes)
    codes = []
    for sc, cc, nm in zip(stock_codes, corp_codes, names):
        if cc in shares and cls.get(cc) in ("Y", "K") and not is_excluded_name(nm):
            codes.append(sc)
    return sorted(set(codes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    codes = universe_stock_codes()
    have = set(os.path.splitext(os.path.basename(p))[0]
               for p in glob.glob(os.path.join(PRICE_DIR, "*.parquet")))
    missing = [c for c in codes if c not in have]
    log(f"[price] 유니버스 후보 {len(codes)} · 가격보유 {len(codes)-len(missing)} · "
        f"빠짐 {len(missing)}  (커버리지 {(len(codes)-len(missing))/max(len(codes),1)*100:.1f}%)")
    if args.limit:
        missing = missing[:args.limit]
        log(f"[price] --limit {args.limit} → {len(missing)}건만 처리")
    if not missing:
        log("[price] 빠진 종목 없음 — 지수만 확인")

    # 지수 먼저(없으면)
    fd.bootstrap_indices()
    # 빠진 종목 가격 부트스트랩(기존 parquet 은 내부에서 skip)
    saved = fd.bootstrap_prices(missing, max_workers=args.workers)

    have2 = set(os.path.splitext(os.path.basename(p))[0]
                for p in glob.glob(os.path.join(PRICE_DIR, "*.parquet")))
    cov = len([c for c in codes if c in have2]) / max(len(codes), 1) * 100
    log(f"[price] 저장 {saved}종목 · 유니버스 가격 커버리지 {cov:.1f}% "
        f"({len([c for c in codes if c in have2])}/{len(codes)})")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
