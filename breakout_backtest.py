import pandas as pd, numpy as np
from pathlib import Path
from collections import defaultdict
BASE=Path("marcap/data")
frames=[]
for y in range(2019,2027):
 p=BASE/f"marcap-{y}.parquet"
 if p.exists():
  x=pd.read_parquet(p).reset_index()
  if "Date" not in x.columns: x=x.rename(columns={x.columns[0]:"Date"})
  frames.append(x)
if not frames: raise RuntimeError("No marcap data")
df=pd.concat(frames,ignore_index=True)
df["Date"]=pd.to_datetime(df["Date"])
df=df[(df.Date<="2026-10-09") & df.Market.isin(["KOSPI","KOSDAQ"])]
df=df[~df.Name.astype(str).str.contains("스팩|SPAC|우B?$|우C$|우\\(",regex=True)]
df=df.sort_values(["Code","Date"]).drop_duplicates(["Code","Date"])
for c in ["Open","High","Close","Amount","Marcap"]: df[c]=pd.to_numeric(df[c],errors="coerce")
g=df.groupby("Code",sort=False)
df["ma50"]=g.Close.transform(lambda s:s.rolling(50,min_periods=50).mean())
for n in [20,60,252]:
 df[f"break{n}"]=df.Close>g.High.transform(lambda s:s.shift(1).rolling(n,min_periods=n).max())
df["eligible"]=(df.Rank<=500)&(df.Amount>=3e9)&(df.Close>=2000)
df=df.sort_values(["Date","Code"])
days=sorted(df.Date.unique())
byday={d:z.set_index("Code") for d,z in df.groupby("Date")}
out=[]
for n in [20,60,252]:
 cash=100_000_000.; positions={}; pending_sell=set(); pending_buy=[]; equity=[]; trades=[]
 for d in days:
  day=byday[d]
  if d<pd.Timestamp("2020-01-01"): continue
  for code in list(pending_sell):
   if code in positions and code in day.index:
    p=day.loc[code]; px=float(p.Open)
    if px>0:
     qty,cost=positions.pop(code); cash+=qty*px*(1-.0025);trades.append((code,(px/cost-1)*100))
  pending_sell.clear()
  for code in pending_buy:
   if len(positions)>=40:break
   if code not in day.index or code in positions:continue
   p=day.loc[code];px=float(p.Open)
   if not np.isfinite(px) or px<=0:continue
   allocation=min(cash, max(0,(cash+sum(q*float(day.loc[k].Close) for k,(q,_) in positions.items() if k in day.index))/40))
   qty=int(allocation/px)
   if qty>0: cash-=qty*px*1.0002;positions[code]=(qty,px)
  pending_buy=[]
  value=cash
  for code,(qty,cost) in positions.items():
   if code in day.index:
    p=day.loc[code];value+=qty*float(p.Close)
    if pd.notna(p.ma50) and p.Close<p.ma50:pending_sell.add(code)
  equity.append((d,value))
  candidates=day[day.eligible & day[f"break{n}"] & day.ma50.notna()].sort_values("Amount",ascending=False)
  pending_buy=[k for k in candidates.index if k not in positions][:40]
 eq=pd.DataFrame(equity,columns=["Date","equity"])
 if len(eq)==0:continue
 years=(eq.Date.iloc[-1]-eq.Date.iloc[0]).days/365.25
 cagr=(eq.equity.iloc[-1]/1e8)**(1/years)-1
 mdd=(eq.equity/eq.equity.cummax()-1).min()
 wins=sum(x[1]>0 for x in trades)
 out.append(dict(strategy=f"{n}d",final_krw=round(eq.equity.iloc[-1]),cagr_pct=round(cagr*100,2),mdd_pct=round(mdd*100,2),trades=len(trades),win_pct=round(100*wins/len(trades),2) if trades else None))
 pd.DataFrame(equity,columns=["Date","equity"]).to_csv(f"equity_{n}.csv",index=False)
pd.DataFrame(out).to_csv("comparison.csv",index=False)
print(pd.DataFrame(out).to_string(index=False))
