"""52투자법: 60일 모멘텀 + 공시시점 기준 연간 영업이익 흑자 필터.
입력: financials/annual_operating_profit.csv
필수 컬럼: code,period_end,disclosed_at,operating_profit
operating_profit은 원 단위, 연결재무제표 우선. 수정공시 시 행 추가.
"""
import pandas as pd, numpy as np
from pathlib import Path

F=Path("financials/annual_operating_profit.csv")
if not F.exists():
 raise SystemExit("MISSING FINANCIAL DATA: financials/annual_operating_profit.csv. 공시일별 실제 연간 영업이익 데이터가 필요합니다. 임의 수치로 백테스트하지 않습니다.")
fin=pd.read_csv(F,dtype={"code":str})
required={"code","period_end","disclosed_at","operating_profit"}
if not required.issubset(fin.columns):raise ValueError(f"Missing columns: {required-set(fin.columns)}")
fin["code"]=fin.code.str.zfill(6)
fin["period_end"]=pd.to_datetime(fin.period_end,errors="raise")
fin["disclosed_at"]=pd.to_datetime(fin.disclosed_at,errors="raise")
fin["operating_profit"]=pd.to_numeric(fin.operating_profit,errors="raise")
if fin[["code","period_end","disclosed_at","operating_profit"]].isna().any().any():raise ValueError("Null financial fields")
if (fin.disclosed_at<fin.period_end).any():raise ValueError("Disclosure precedes fiscal period end")
if fin.duplicated(["code","period_end","disclosed_at"]).any():raise ValueError("Duplicate disclosure records")
fin=fin.sort_values(["code","disclosed_at","period_end"])
# Each disclosure may revise a previous year; use latest known fiscal year and latest filed value at signal date.
frames=[]
for y in range(2019,2027):
 p=Path(f"marcap/data/marcap-{y}.parquet")
 if p.exists():
  x=pd.read_parquet(p).reset_index()
  if "Date" not in x:x=x.rename(columns={x.columns[0]:"Date"})
  frames.append(x)
if not frames:raise SystemExit("Missing marcap/data price parquet files")
df=pd.concat(frames,ignore_index=True)
df["Date"]=pd.to_datetime(df.Date)
df=df[(df.Date<="2026-10-09")&df.Market.isin(["KOSPI","KOSDAQ"])].copy()
df["Code"]=df.Code.astype(str).str.zfill(6)
df=df[~df.Name.astype(str).str.contains(r"스팩|SPAC|우B?$|우C$|우\\(",regex=True)]
for c in ["Open","High","Low","Close","Amount","Rank"]:df[c]=pd.to_numeric(df[c],errors="coerce")
df=df.sort_values(["Code","Date"]).drop_duplicates(["Code","Date"])
g=df.groupby("Code",sort=False)
for n in [20,90,200]:df[f"ma{n}"]=g.Close.transform(lambda x:x.rolling(n,min_periods=n).mean())
df["breakout"]=df.Close>g.High.transform(lambda x:x.shift(1).rolling(252,min_periods=252).max())
df["momentum60"]=df.Close/g.Close.shift(60)-1
df["gap20"]=df.Close/df.ma20-1
# Excluded anomaly must not alter market-wide breadth.
breadth=(df[(df.Market=="KOSPI")&df.ma90.notna()].assign(above=lambda x:x.Close>x.ma90).groupby("Date").above.mean().shift(1))
# Point-in-time: for each date use only disclosures already published by that date.
# Select the latest fiscal period_end, then latest revision for that period.
# Build per-code disclosure timelines to avoid forward filling future reports.
events={}
for code,x in fin.groupby("code"):
 events[code]=list(x[["disclosed_at","period_end","operating_profit"]].itertuples(index=False,name=None))
def eligible_financial(code,date):
 known=[(period,disclosed,profit) for disclosed,period,profit in events.get(code,[]) if disclosed<=date]
 if not known:return False
 period,disclosed,profit=max(known,key=lambda a:(a[0],a[1]))
 # Reject stale statements: last fiscal year end must be within 24 months.
 return (date-period).days<=730 and profit>0
df=df.sort_values(["Date","Code"])
daily={d:x.set_index("Code") for d,x in df.groupby("Date")}
dates=[d for d in sorted(daily) if d>=pd.Timestamp("2020-01-01")]
for filter_on in [False,True]:
 cash=1e8;positions={};sell=set();buy=[];last={};equity=[];trades=wins=0
 for d in dates:
  day=daily[d];last.update(day.Close.dropna().to_dict())
  for code in tuple(sell):
   if code in positions and code in day.index:
    px=float(day.at[code,"Open"])
    if np.isfinite(px) and px>0:
     qty,cost=positions.pop(code);cash+=qty*px*(1-.0025);trades+=1;wins+=px*(1-.0025)>cost*1.0002
  sell.clear()
  for code in buy:
   if len(positions)>=8 or code in positions or code not in day.index:continue
   px=float(day.at[code,"Open"])
   if not np.isfinite(px) or px<=0:continue
   nav=cash+sum(q*(float(day.at[k,"Open"]) if k in day.index and pd.notna(day.at[k,"Open"]) else last.get(k,c)) for k,(q,c) in positions.items())
   qty=int(min(cash/1.0002,nav/8)/px)
   if qty>0:cash-=qty*px*1.0002;positions[code]=(qty,px)
  buy=[]
  nav=cash+sum(q*last.get(k,c) for k,(q,c) in positions.items());equity.append(nav)
  for code in positions:
   if code in day.index and pd.notna(day.at[code,"ma20"]) and (day.at[code,"Close"]<day.at[code,"ma20"] or day.at[code,"Close"]<positions[code][1]*.90):sell.add(code)
  active=pd.notna(breadth.get(d,np.nan)) and breadth.get(d,0)>=.35
  if not active:sell.update(positions.keys())
  if active:
   c=day[(day.Rank<=500)&(day.Close>=2000)&(day.Amount>=3e9)&day.breakout&(day.Close>day.ma200)&(day.gap20<=.40)]
   c=c.drop(index="011930",errors="ignore")
   if filter_on:c=c[[eligible_financial(k,d) for k in c.index]]
   buy=[k for k in c.sort_values("momentum60",ascending=False).index if k not in positions][:8]
 e=np.array(equity);years=(dates[-1]-dates[0]).days/365.25
 print(f"RESULT financial_filter={filter_on} CAGR={((e[-1]/1e8)**(1/years)-1)*100:.2f}% MDD={(e/np.maximum.accumulate(e)-1).min()*100:.2f}% FINAL={e[-1]:.0f} TRADES={trades} WIN={wins/trades*100 if trades else 0:.2f}%",flush=True)
