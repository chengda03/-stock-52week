import os
import numpy as np
import pandas as pd
parts=[]
for y in range(2009,2027):
 d=pd.read_parquet(f"marcap/data/marcap-{y}.parquet")
 if "Date" not in d.columns:d=d.reset_index()
 parts.append(d.loc[d.Market.isin(["KOSPI","KOSDAQ"]),["Date","Code","Open","High","Low","Close"]])
df=pd.concat(parts,ignore_index=True);del parts
df.Date=pd.to_datetime(df.Date)
df=df[df.Date<="2026-10-27"].copy()
df.Code=df.Code.astype(str).str.zfill(6)
df=df[df.Code.str.fullmatch(r"\d{6}")]
for c in ["Open","High","Low","Close"]:df[c]=pd.to_numeric(df[c],errors="coerce")
df=df.dropna(subset=["Open","High","Low","Close"])
df=df[(df[["Open","High","Low","Close"]]>0).all(axis=1)]
df=df.sort_values(["Code","Date"]).drop_duplicates(["Code","Date"])
dates=pd.DatetimeIndex(sorted(df.Date.unique()))
date_idx={d:i for i,d in enumerate(dates)}
bars={}; signals={}
for code,s in df.groupby("Code",sort=False):
 s=s.reset_index(drop=True)
 close=s.Close.to_numpy(float); op=s.Open.to_numpy(float); hi=s.High.to_numpy(float)
 ma20=pd.Series(close).rolling(20).mean().to_numpy()
 ma5=pd.Series(close).rolling(5).mean().to_numpy()
 di=np.array([date_idx[x] for x in s.Date])
 breached=False
 for i in range(20,len(s)-1):
  if close[i]<.8*ma20[i]:breached=True
  if not (breached and close[i]>=.8*ma20[i] and close[i]>ma5[i] and close[i-1]<=ma5[i-1]):continue
  breached=False
  k=i+1
  if s.Date.iloc[k]<pd.Timestamp("2010-01-01"):continue
  signals.setdefault(int(di[k]),[]).append((code,k))
 bars[code]=(di,op,hi,close)
print("signals",sum(map(len,signals.values())),flush=True)
# Capital starts at 10m KRW, 10 slots. Buy on next open, sell at 8% limit or 30th trading bar close.
# No stop. Transaction costs: 0.165% round trip (deduct 0.165% from sale proceeds).
# Signals sorted by code to eliminate arbitrary nondeterminism. Mark-to-market at last available close.
cash=10000000.0
positions={}
last_close={}
ledger=[]
equity_curve=[]
skipped=0
for t,date in enumerate(dates):
 if date<pd.Timestamp("2010-01-01"):continue
 # exits before entries: same-day target exit uses day's high, entries occur at open.
 # existing positions are processed before new buys; proceeds from intraday exits NOT available for same-day opening.
 to_exit=[]
 for code,p in positions.items():
  di,op,hi,cl=bars[code]
  k=p["k"]
  if k>=len(di) or di[k]!=t:continue
  last_close[code]=cl[k]
  age=k-p["entry_k"]+1
  price=None;reason=None
  if op[k]>=p["entry_price"]*1.08:
   price=op[k];reason="target_gap"
  elif hi[k]>=p["entry_price"]*1.08:
   price=p["entry_price"]*1.08;reason="target"
  elif age>=30:
   price=cl[k];reason="time"
  if price is not None:to_exit.append((code,p,price,reason,date))
  else:p["k"]+=1
 # new buys at open with cash available from prior sessions, 1/10 of marked portfolio value per slot.
 candidates=sorted(signals.get(t,[]))
 for code,k in candidates:
  if code in positions or len(positions)>=10:continue
  di,op,hi,cl=bars[code]
  if k>=len(di) or di[k]!=t:continue
  marked=cash+sum(p["shares"]*last_close.get(c,p["entry_price"]) for c,p in positions.items())
  allocation=min(cash,marked/10)
  if allocation<1000:continue
  shares=int(allocation//op[k])
  if shares<=0:continue
  cost=shares*op[k]
  cash-=cost
  positions[code]={"shares":shares,"entry_price":op[k],"entry_date":date,"entry_k":k,"k":k,"cost":cost}
  last_close[code]=cl[k]
  # allow same-day exit at 8% high; conservatively ignore ordering uncertainty of high vs open (high always after open)
  p=positions[code]
  price=None;reason=None
  if hi[k]>=op[k]*1.08:price=op[k]*1.08;reason="target"
  if price is not None:to_exit.append((code,p,price,reason,date))
  else:p["k"]+=1
 for code,p,price,reason,exitdate in to_exit:
  if code not in positions:continue
  proceeds=p["shares"]*price*(1-.00165)
  cash+=proceeds
  ledger.append({"code":code,"entry_date":p["entry_date"],"exit_date":exitdate,"reason":reason,"entry_krw":p["cost"],"net_return_pct":100*(proceeds/p["cost"]-1),"profit_krw":proceeds-p["cost"]})
  del positions[code]
 # Update marks for all held codes when their bar is available; forward-fill on suspensions.
 for code,p in positions.items():
  di,op,hi,cl=bars[code]
  k=p["k"]-1
  if k>=0 and k<len(di) and di[k]==t:last_close[code]=cl[k]
 equity=cash+sum(p["shares"]*last_close.get(code,p["entry_price"]) for code,p in positions.items())
 equity_curve.append({"date":date,"equity":equity,"cash":cash,"positions":len(positions)})
eq=pd.DataFrame(equity_curve)
eq["year"]=eq.date.dt.year
eq["peak"]=eq.equity.cummax()
eq["drawdown_pct"]=100*(eq.equity/eq.peak-1)
years=[]
for year,g in eq.groupby("year"):
 first=eq.loc[eq.date<g.date.iloc[0],"equity"]
 start=float(first.iloc[-1]) if len(first) else 10000000.
 end=float(g.equity.iloc[-1])
 years.append({"year":int(year),"start_krw":round(start),"end_krw":round(end),"profit_krw":round(end-start),"return_pct":round(100*(end/start-1),2),"year_mdd_pct":round(g.drawdown_pct.min(),2),"trades_exited":sum(x["exit_date"].year==year for x in ledger)})
last=float(eq.equity.iloc[-1]);elapsed=(eq.date.iloc[-1]-eq.date.iloc[0]).days/365.25
cagr=(last/10000000.)**(1/elapsed)-1
print(pd.DataFrame(years).to_string(index=False),flush=True)
print("FINAL",{"final_equity_krw":round(last),"net_profit_krw":round(last-10000000),"CAGR_pct":round(cagr*100,2),"MDD_pct":round(eq.drawdown_pct.min(),2),"completed_trades":len(ledger),"open_positions":len(positions),"mean_positions":round(eq.positions.mean(),2),"mean_net_trade_pct":round(pd.DataFrame(ledger).net_return_pct.mean(),3)},flush=True)
os.makedirs("results/envelope20_portfolio",exist_ok=True)
pd.DataFrame(years).to_csv("results/envelope20_portfolio/annual.csv",index=False)
pd.DataFrame(ledger).to_csv("results/envelope20_portfolio/trades.csv",index=False)
eq.to_csv("results/envelope20_portfolio/equity.csv",index=False)
