import os
import numpy as np
import pandas as pd
YEARS=range(2009,2027)
parts=[]
for y in YEARS:
 p=f"marcap/data/marcap-{y}.parquet"
 if not os.path.exists(p): raise FileNotFoundError(p)
 d=pd.read_parquet(p)
 if "Date" not in d.columns:d=d.reset_index()
 d=d[["Date","Code","Market","Open","High","Low","Close"]]
 d=d[d.Market.isin(["KOSPI","KOSDAQ"])]
 parts.append(d)
 print("Loaded",y,len(d),flush=True)
df=pd.concat(parts,ignore_index=True);del parts
df.Date=pd.to_datetime(df.Date)
df=df[(df.Date>="2009-01-01")&(df.Date<="2026-10-27")].copy()
df.Code=df.Code.astype(str).str.zfill(6)
df=df[df.Code.str.fullmatch(r"\d{6}")]
for c in ["Open","High","Low","Close"]:df[c]=pd.to_numeric(df[c],errors="coerce")
df=df.dropna(subset=["Open","High","Low","Close"])
df=df[(df[["Open","High","Low","Close"]]>0).all(axis=1)]
df=df.sort_values(["Code","Date"]).drop_duplicates(["Code","Date"])
events=[]
for code,s in df.groupby("Code",sort=False):
 s=s.reset_index(drop=True)
 close=s.Close.to_numpy(dtype=float);op=s.Open.to_numpy(dtype=float)
 high=s.High.to_numpy(dtype=float);low=s.Low.to_numpy(dtype=float)
 ma20=pd.Series(close).rolling(20,min_periods=20).mean().to_numpy()
 ma5=pd.Series(close).rolling(5,min_periods=5).mean().to_numpy()
 lower=ma20*.8
 breached=False
 for i in range(20,len(s)-31):
  if close[i]<lower[i]:breached=True
  cross=close[i]>ma5[i] and close[i-1]<=ma5[i-1]
  recovered=close[i]>=lower[i]
  if not (breached and recovered and cross):continue
  breached=False
  if s.Date.iloc[i]<pd.Timestamp("2010-01-01"):continue
  entry=i+1;buy=op[entry]
  row={"date":str(s.Date.iloc[i].date()),"entry_date":str(s.Date.iloc[entry].date()),"code":code,"market":s.Market.iloc[i],"entry_open":buy}
  for days in [5,10,20,30]:
   hh=high[entry:entry+days];ll=low[entry:entry+days];oo=op[entry:entry+days]
   for target in [.03,.05,.08]:
    key=f"hit{int(target*100)}_{days}d"
    row[key]=int(bool(np.any(hh>=buy*(1+target))))
    first="none"
    for k in range(days):
     if oo[k]<=buy*.95:first="stop";break
     if oo[k]>=buy*(1+target):first="target";break
     if ll[k]<=buy*.95:first="stop";break
     if hh[k]>=buy*(1+target):first="target";break
    row[f"first{int(target*100)}_{days}d"]=int(first=="target")
  events.append(row)
out=pd.DataFrame(events)
os.makedirs("results/envelope20",exist_ok=True)
if out.empty:raise RuntimeError("No signals")
out.to_csv("results/envelope20/signals.csv",index=False)
summary=[]
for period,subset in [("2010-2022",out[out.date<"2023-01-01"]),("2023-2026",out[out.date>="2023-01-01"]),("all",out)]:
 for days in [5,10,20,30]:
  for target in [3,5,8]:
   summary.append({"period":period,"days":days,"target_pct":target,"signals":len(subset),"hit_pct":round(100*subset[f"hit{target}_{days}d"].mean(),2),"hit_before_5pct_stop_pct":round(100*subset[f"first{target}_{days}d"].mean(),2)})
pd.DataFrame(summary).to_csv("results/envelope20/summary.csv",index=False)
print(pd.DataFrame(summary).to_string(index=False),flush=True)
