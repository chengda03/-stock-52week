import pandas as pd, numpy as np
from pathlib import Path
START="2010-01-01"; END="2026-10-09"; INITIAL=100_000_000
frames=[]
for year in range(2008,2027):
 p=Path(f"marcap/data/marcap-{year}.parquet")
 if p.exists():
  x=pd.read_parquet(p).reset_index()
  if "Date" not in x:x=x.rename(columns={x.columns[0]:"Date"})
  frames.append(x)
df=pd.concat(frames,ignore_index=True)
df["Date"]=pd.to_datetime(df.Date)
df=df[(df.Date<=END)&df.Market.isin(["KOSPI","KOSDAQ"])].copy()
df=df[df.Code.astype(str).str.zfill(6)!="011930"]
df=df[~df.Name.astype(str).str.contains("스팩|SPAC|우B?$|우C$",regex=True)]
for c in ["Open","High","Low","Close","Volume","Amount","Rank"]:df[c]=pd.to_numeric(df[c],errors="coerce")
df=df.sort_values(["Code","Date"]).drop_duplicates(["Code","Date"])
g=df.groupby("Code",sort=False)
df["ma10"]=g.Close.transform(lambda s:s.rolling(10,min_periods=10).mean())
df["v20"]=g.Volume.transform(lambda s:s.shift(1).rolling(20,min_periods=20).mean())
df["prevclose"]=g.Close.shift(1)
df["chg"]=df.Close/df.prevclose-1
df["body"]=df.Close/df.Open-1
df["upper"]=(df.High-df.Close)/(df.High-df.Low).replace(0,np.nan)
df["big"]=(df.chg>=.05)&(df.chg<=.12)&(df.body>=.025)&(df.upper<=.20)&(df.Volume>=df.v20*1.5)&(df.Rank<=500)&(df.Close>=2000)&(df.Amount>=3e9)
df["bull"]=df.Close>df.Open
df=df.sort_values(["Date","Code"])
daily={d:x.set_index("Code") for d,x in df.groupby("Date")}
dates=[d for d in sorted(daily) if pd.Timestamp(START)<=d<=pd.Timestamp(END)]
# 3-day rule: no +5% within 3 trading days -> exit next open; +5% -> add 20m, retain until 21st day.
# Conservative intraday order: stop before high target; signals on close, buys/sells at next open except intraday hard stops.
def simulate(tol=.01,holding=21):
 cash=INITIAL; positions={}; pending=[]; last={}; setups={}; equity=[]; trades=[]; closed=[]; n_signals=0;n_add=0
 for d in dates:
  day=daily[d];last.update(day.Close.dropna().to_dict())
  # next-open exits and entries
  for code,p in list(positions.items()):
   if p.get("exit_next") and code in day.index:
    px=float(day.at[code,"Open"]); qty=p["qty"];cash+=qty*px*.9975
    closed.append((code,d,p["first"],(qty*px*.9975-p["spent"])/p["spent"],p["added"],p["reason"]))
    del positions[code]
  for code in pending:
   if code in positions or code not in day.index:continue
   px=float(day.at[code,"Open"])
   if np.isfinite(px) and px>0:
    qty=int(min(10_000_000,cash/1.0002)/px)
    if qty>0:
     cost=qty*px*1.0002;cash-=cost
     positions[code]={"qty":qty,"spent":cost,"first":px,"days":0,"added":False,"exit_next":False,"reason":""}
  pending=[]
  for code,p in list(positions.items()):
   if code not in day.index:continue
   r=day.loc[code];o,h,l,c=[float(r[z]) for z in ["Open","High","Low","Close"]]
   if not all(np.isfinite(z) for z in [o,h,l,c]):continue
   p["days"]+=1
   stop=p["first"]*.96
   # Gap-through stop executes at open; otherwise stop level. Stop applies to entire position.
   if l<=stop:
    px=min(o,stop);cash+=p["qty"]*px*.9975
    closed.append((code,d,p["first"],(p["qty"]*px*.9975-p["spent"])/p["spent"],p["added"],"stop"))
    del positions[code];continue
   if not p["added"] and h>=p["first"]*1.05 and p["days"]<=3:
    px=max(o,p["first"]*1.05)
    qty=int(min(20_000_000,cash/1.0002)/px)
    if qty>0:
     cost=qty*px*1.0002;cash-=cost;p["qty"]+=qty;p["spent"]+=cost;p["added"]=True;n_add+=1
   if not p["added"] and p["days"]>=3:
    p["exit_next"]=True;p["reason"]="no5in3"
   if p["days"]>=holding:
    p["exit_next"]=True;p["reason"]="day21"
  # scan new big candles; then 2-7 sessions later bullish candle with close +/-1% of MA10
  for code,r in day.iterrows():
   if bool(r.big):setups[code]=0
   elif code in setups:setups[code]+=1
  candidates=[]
  for code,age in list(setups.items()):
   if age>7:del setups[code];continue
   if age<2 or code not in day.index or code in positions:continue
   r=day.loc[code]
   if bool(r.bull) and pd.notna(r.ma10) and abs(float(r.Close)/float(r.ma10)-1)<=tol:
    candidates.append((code,float(r.Amount)));del setups[code];n_signals+=1
  candidates.sort(key=lambda t:-t[1])
  pending=[code for code,_ in candidates]
  nav=cash+sum(p["qty"]*float(last.get(code,p["first"])) for code,p in positions.items())
  equity.append((d,nav))
 eq=pd.Series(dict(equity)); yrs=(eq.index[-1]-eq.index[0]).days/365.25
 cagr=(eq.iloc[-1]/INITIAL)**(1/yrs)-1;mdd=(eq/eq.cummax()-1).min()
 win=np.mean([z[3]>0 for z in closed]) if closed else np.nan
 annual=eq.resample("YE").last().pct_change();annual.iloc[0]=eq[eq.index.year==eq.index[0].year].iloc[-1]/INITIAL-1
 return dict(tol=tol,signals=n_signals,closed=len(closed),added=n_add,win=round(win*100,2),final=round(eq.iloc[-1]),cagr=round(cagr*100,2),mdd=round(mdd*100,2),annual={str(i.year):round(v*100,2) for i,v in annual.items()},reasons=pd.Series([x[5] for x in closed]).value_counts().to_dict())
for tol in [.005,.01,.02]:
 print("RESULT",simulate(tol),flush=True)
