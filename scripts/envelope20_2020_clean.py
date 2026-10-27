import os
import numpy as np
import pandas as pd
START=pd.Timestamp("2020-01-01")
END=pd.Timestamp("2026-10-27")
CAPITAL=10000000.0
data=[]
for year in range(2019,2027):
 d=pd.read_parquet(f"marcap/data/marcap-{year}.parquet")
 if "Date" not in d:d=d.reset_index()
 data.append(d.loc[d.Market.isin(["KOSPI","KOSDAQ"]),["Date","Code","Open","High","Low","Close"]])
df=pd.concat(data,ignore_index=True)
df.Date=pd.to_datetime(df.Date)
df=df[df.Date<=END].copy()
df.Code=df.Code.astype(str).str.zfill(6)
df=df[df.Code.str.fullmatch(r"\d{6}")]
for c in ["Open","High","Low","Close"]:df[c]=pd.to_numeric(df[c],errors="coerce")
df=df.dropna()
df=df[(df[["Open","High","Low","Close"]]>0).all(axis=1)]
df=df.sort_values(["Code","Date"]).drop_duplicates(["Code","Date"])
dates=pd.DatetimeIndex(sorted(df.Date.unique()))
idx={d:i for i,d in enumerate(dates)}
bars={}
signals={}
anomalies=[]
for code,s in df.groupby("Code",sort=False):
 s=s.reset_index(drop=True)
 op=s.Open.to_numpy(float); hi=s.High.to_numpy(float); close=s.Close.to_numpy(float)
 di=np.array([idx[d] for d in s.Date])
 ma20=pd.Series(close).rolling(20).mean().to_numpy()
 ma5=pd.Series(close).rolling(5).mean().to_numpy()
 # Exclude extreme corporate-action-like daily price jumps; cannot reliably model share-count adjustments.
 suspicious=np.zeros(len(s),dtype=bool)
 suspicious[1:]=(op[1:]/close[:-1]>1.4)|(op[1:]/close[:-1]<.6)|(close[1:]/close[:-1]>1.4)|(close[1:]/close[:-1]<.6)
 if suspicious.any():
  anomalies.extend([{"code":code,"date":str(s.Date.iloc[k].date()),"previous_close":close[k-1],"open":op[k],"close":close[k]} for k in np.flatnonzero(suspicious)])
 # reject entire securities with suspicious corporate-action gaps, rather than silently misprice positions
 if suspicious.any():continue
 breached=False
 for i in range(20,len(s)-1):
  if close[i]<.8*ma20[i]:breached=True
  if not (breached and close[i]>=.8*ma20[i] and close[i]>ma5[i] and close[i-1]<=ma5[i-1]):continue
  breached=False
  k=i+1
  if s.Date.iloc[k]<START:continue
  signals.setdefault(int(di[k]),[]).append((code,k))
 bars[code]={"di":di,"op":op,"hi":hi,"close":close,"map":{int(t):int(k) for k,t in enumerate(di)}}
cash=CAPITAL; positions={};ledger=[];curve=[];missing_exits=0
for t,date in enumerate(dates):
 if date<START:continue
 # Exit old positions; no same-day reuse of sale proceeds for purchases at opening.
 exits=[]
 for code,p in list(positions.items()):
  b=bars[code]; k=b["map"].get(t)
  age=t-p["entry_t"]+1
  if k is not None:
   p["last_close"]=b["close"][k];p["last_seen_t"]=t
   price=None;reason=None
   if b["op"][k]>=p["entry_price"]*1.08:price=b["op"][k];reason="gap_target"
   elif b["hi"][k]>=p["entry_price"]*1.08:price=p["entry_price"]*1.08;reason="target"
   elif age>=30:price=b["close"][k];reason="time"
   if price is not None:exits.append((code,price,reason))
  elif age>=30:
   # No price at exit date: retain as suspended for 30 further market days; then write down to zero.
   if t-p["last_seen_t"]>=30:exits.append((code,0.0,"missing_price_writeoff"))
 # At opening use cash before same-day exits and marked equity using previous close for held stocks.
 marked=cash+sum(p["shares"]*p["last_close"] for p in positions.values())
 for code,k in sorted(signals.get(t,[])):
  if code in positions or len(positions)>=10:continue
  b=bars[code]
  price=b["op"][k]
  allocation=min(cash,marked/10)
  shares=int(allocation//price)
  if shares<1:continue
  cost=shares*price;cash-=cost
  positions[code]={"shares":shares,"cost":cost,"entry_price":price,"entry_t":t,"entry_date":date,"last_close":b["close"][k],"last_seen_t":t}
  if b["hi"][k]>=price*1.08:exits.append((code,price*1.08,"target"))
 for code,price,reason in exits:
  p=positions.pop(code,None)
  if p is None:continue
  proceeds=p["shares"]*price*(1-.00165)
  cash+=proceeds
  ledger.append({"code":code,"entry_date":p["entry_date"],"exit_date":date,"reason":reason,"net_profit_krw":round(proceeds-p["cost"]),"return_pct":round(100*(proceeds/p["cost"]-1),3)})
 equity=cash+sum(p["shares"]*p["last_close"] for p in positions.values())
 curve.append({"date":date,"equity":equity,"cash":cash,"positions":len(positions)})
eq=pd.DataFrame(curve);eq["year"]=eq.date.dt.year;eq["dd"]=eq.equity/eq.equity.cummax()-1
annual=[]
for year,g in eq.groupby("year"):
 prior=eq[eq.date<g.date.iloc[0]]
 start=float(prior.equity.iloc[-1]) if len(prior) else CAPITAL
 end=float(g.equity.iloc[-1])
 annual.append({"year":year,"start_krw":round(start),"end_krw":round(end),"net_profit_krw":round(end-start),"return_pct":round(100*(end/start-1),2),"trades":sum(x["exit_date"].year==year for x in ledger)})
end=float(eq.equity.iloc[-1]);elapsed=(eq.date.iloc[-1]-eq.date.iloc[0]).days/365.25
print("ANOMALY_EXCLUDED_CODES",len(set(x["code"] for x in anomalies)),flush=True)
print(pd.DataFrame(annual).to_string(index=False),flush=True)
print("FINAL",{"end":round(end),"net_profit":round(end-CAPITAL),"cagr_pct":round(100*((end/CAPITAL)**(1/elapsed)-1),2),"mdd_pct":round(100*eq.dd.min(),2),"trades":len(ledger),"open_positions":len(positions),"writeoffs":sum(x["reason"]=="missing_price_writeoff" for x in ledger)},flush=True)
os.makedirs("results/envelope20_2020_clean",exist_ok=True)
pd.DataFrame(annual).to_csv("results/envelope20_2020_clean/annual.csv",index=False)
pd.DataFrame(ledger).to_csv("results/envelope20_2020_clean/trades.csv",index=False)
pd.DataFrame(anomalies).to_csv("results/envelope20_2020_clean/excluded_anomalies.csv",index=False)
eq.to_csv("results/envelope20_2020_clean/equity.csv",index=False)
