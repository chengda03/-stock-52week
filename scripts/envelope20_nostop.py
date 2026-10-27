import os
import numpy as np
import pandas as pd
parts=[]
for y in range(2009,2027):
 d=pd.read_parquet(f"marcap/data/marcap-{y}.parquet")
 if "Date" not in d:d=d.reset_index()
 d=d[["Date","Code","Market","Open","High","Low","Close"]]
 parts.append(d[d.Market.isin(["KOSPI","KOSDAQ"])])
df=pd.concat(parts,ignore_index=True);del parts
df.Date=pd.to_datetime(df.Date)
df=df[df.Date<="2026-10-27"].copy()
df.Code=df.Code.astype(str).str.zfill(6)
df=df[df.Code.str.fullmatch(r"\d{6}")]
for c in ["Open","High","Low","Close"]:df[c]=pd.to_numeric(df[c],errors="coerce")
df=df.dropna(subset=["Open","High","Low","Close"])
df=df[(df[["Open","High","Low","Close"]]>0).all(axis=1)]
df=df.sort_values(["Code","Date"]).drop_duplicates(["Code","Date"])
configs=[(3,10),(3,20),(3,30),(5,30),(8,30)]
trades={c:[] for c in configs}
for code,s in df.groupby("Code",sort=False):
 s=s.reset_index(drop=True)
 close=s.Close.to_numpy(float);op=s.Open.to_numpy(float);hi=s.High.to_numpy(float)
 ma20=pd.Series(close).rolling(20).mean().to_numpy()
 ma5=pd.Series(close).rolling(5).mean().to_numpy()
 breached=False
 for i in range(20,len(s)-1):
  if close[i]<.8*ma20[i]:breached=True
  if not (breached and close[i]>=.8*ma20[i] and close[i]>ma5[i] and close[i-1]<=ma5[i-1]):continue
  breached=False
  if s.Date.iloc[i]<pd.Timestamp("2010-01-01"):continue
  entry=i+1;buy=op[entry]
  for target,days in configs:
   if entry+days-1>=len(s):continue
   if s.Date.iloc[entry+days-1]>pd.Timestamp("2026-10-27"):continue
   price=close[entry+days-1];exitidx=entry+days-1;reason="time"
   for k in range(entry,entry+days):
    if op[k]>=buy*(1+target/100):
     price=op[k];exitidx=k;reason="target_gap";break
    if hi[k]>=buy*(1+target/100):
     price=buy*(1+target/100);exitidx=k;reason="target";break
   trades[(target,days)].append({"date":str(s.Date.iloc[entry].date()),"exit_date":str(s.Date.iloc[exitidx].date()),"code":code,"market":s.Market.iloc[i],"ret_pct":100*(price/buy-1),"reason":reason,"hold_days":exitidx-entry+1})
os.makedirs("results/envelope20_nostop",exist_ok=True)
summary=[]
for (target,days),rows in trades.items():
 t=pd.DataFrame(rows);t.to_csv(f"results/envelope20_nostop/trades_{target}_{days}.csv",index=False)
 for period,d in [("2010-2022",t[t.date<"2023-01-01"]),("2023-2026",t[t.date>="2023-01-01"]),("all",t)]:
  if d.empty:continue
  r=d.ret_pct
  summary.append({"period":period,"target_pct":target,"max_days":days,"trades":len(d),"target_hit_pct":round(100*(d.reason!="time").mean(),2),"win_pct":round(100*(r>0).mean(),2),"mean_return_pct":round(r.mean(),3),"median_return_pct":round(r.median(),3),"mean_loser_pct":round(r[r<0].mean(),3),"worst_pct":round(r.min(),2),"mean_hold_days":round(d.hold_days.mean(),2),"mean_return_after_0.165pct_cost":round(r.mean()-.165,3)})
res=pd.DataFrame(summary)
res.to_csv("results/envelope20_nostop/summary.csv",index=False)
print(res.to_string(index=False),flush=True)
