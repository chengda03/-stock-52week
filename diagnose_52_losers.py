import pandas as pd,numpy as np,itertools
from pathlib import Path
frames=[]
for y in range(2019,2027):
 p=Path(f"marcap/data/marcap-{y}.parquet")
 if p.exists():
  x=pd.read_parquet(p).reset_index()
  if "Date" not in x:x=x.rename(columns={x.columns[0]:"Date"})
  frames.append(x)
df=pd.concat(frames,ignore_index=True)
df["Date"]=pd.to_datetime(df.Date)
df=df[(df.Date<="2026-10-09") & df.Market.isin(["KOSPI","KOSDAQ"])].copy()
df=df[~df.Name.astype(str).str.contains("스팩|SPAC|우B?$|우C$|우\\(",regex=True)]
for c in ["Open","High","Low","Close","Amount","Rank","Volume"]:df[c]=pd.to_numeric(df[c],errors="coerce")
df=df.sort_values(["Code","Date"]).drop_duplicates(["Code","Date"])
g=df.groupby("Code",sort=False)
for n in [10,15,20,50,90,200]:
 df[f"ma{n}"]=g.Close.transform(lambda x:x.rolling(n,min_periods=n).mean())
for n in [60,252]:
 df[f"b{n}"]=df.Close>g.High.transform(lambda x:x.shift(1).rolling(n,min_periods=n).max())
df["priorhigh252"]=g.High.transform(lambda x:x.shift(1).rolling(252,min_periods=252).max())
df["volume_avg20"]=g.Volume.transform(lambda x:x.shift(1).rolling(20,min_periods=20).mean())
df["momentum60"]=df.Close/g.Close.shift(60)-1
df["range_close"]=(df.Close-df.Low)/(df.High-df.Low).replace(0,np.nan)
df["break_strength"]=df.Close/df.priorhigh252-1
df["volume_ratio"]=df.Volume/df.volume_avg20
df["mom126"]=df.Close/g.Close.shift(126)-1
df["mom252"]=df.Close/g.Close.shift(252)-1
df["eligible"]=(df.Rank<=500)&(df.Amount>=3e9)&(df.Close>=2000)
# Market breadth: fraction of KOSPI stocks above 90d average, based on previous day only
breadth=(df[(df.Market=="KOSPI") & df.ma90.notna()].assign(above=lambda x:x.Close>x.ma90).groupby("Date").above.mean().shift(1))
df=df.sort_values(["Date","Code"])
daily={}
for d,x in df.groupby("Date"):
 daily[d]=x.set_index("Code")
dates=[d for d in sorted(daily) if d>=pd.Timestamp("2020-01-01")]
results=[]
configs=[("baseline",0)]
for i,(feature,cutoff) in enumerate(configs):
 period=252;market="breadth30";stop=20;slots=10;ranking="amount";hardstop=0.10
 cash=1e8;positions={}; sell=set();buy=[];eq=[];wins=0;trades=0;last={};entries={};pending_features={};records=[];signal_date={}
 for d in dates:
  day=daily[d];last.update(day.Close.to_dict())
  for code in tuple(sell):
   if code in positions and code in day.index:
    px=float(day.at[code,"Open"])
    if np.isfinite(px) and px>0:
     qty,cost=positions.pop(code);cash+=qty*px*(1-.0025);trades+=1;wins+=(px*(1-.0025)>cost*1.0002)
     e=entries.pop(code,{});records.append(dict(code=code,entry_date=str(e.get("entry_date","")),exit_date=str(d.date()),pnl_pct=100*(px*.9975/(cost*1.0002)-1),holding_days=(d-e["entry_date"]).days if "entry_date" in e else None,**e.get("features",{})))
  sell.clear()
  for code in buy:
   if len(positions)>=slots or code in positions or code not in day.index:continue
   px=float(day.at[code,"Open"])
   if not np.isfinite(px) or px<=0:continue
   nav=cash+sum(q*last.get(k,c) for k,(q,c) in positions.items())
   budget=min(cash/1.0002,nav/slots)
   qty=int(budget/px)
   if qty>0:
    cash-=qty*px*1.0002;positions[code]=(qty,px)
    entries[code]={"entry_date":d,"features":pending_features.get(code,{})}
  buy=[];pending_features={}
  nav=cash+sum(q*last.get(k,c) for k,(q,c) in positions.items());eq.append(nav)
  for code in positions:
   if code in day.index and pd.notna(day.at[code,f"ma{stop}"]) and (day.at[code,"Close"]<day.at[code,f"ma{stop}"] or day.at[code,"Close"]<positions[code][1]*(1-hardstop)):sell.add(code)
  threshold=int(market.replace("breadth",""))/100
  active=(pd.notna(breadth.get(d,np.nan)) and breadth.get(d,0)>=threshold)
  if not active:sell.update(positions.keys())
  if active:
   candidates=day[day.eligible & day[f"b{period}"] & (day.Close>day.ma200)]
   if feature=="strength":candidates=candidates[candidates.break_strength>=cutoff]
   if feature=="volume":candidates=candidates[candidates.volume_ratio>=cutoff]
   if feature=="close":candidates=candidates[candidates.range_close>=cutoff]
   if feature=="momentum":candidates=candidates[candidates.momentum60<=cutoff]
   if feature=="combo":candidates=candidates[(candidates.volume_ratio>=1.5)&(candidates.range_close>=0.8)&(candidates.momentum60<=0.4)]
   col="Amount" if ranking=="amount" else "mom252"
   candidates=candidates.sort_values(col,ascending=False)
   buy=[k for k in candidates.index if k not in positions][:slots]
   for k in buy:
    row=candidates.loc[k]
    pending_features[k]=dict(momentum60=float(row.momentum60),volume_ratio=float(row.volume_ratio),range_close=float(row.range_close),break_strength=float(row.break_strength),ma20_gap=float(row.Close/row.ma20-1),daily_gain=float(row.Close/row.Open-1),breadth=float(breadth.get(d,np.nan)),amount=float(row.Amount),rank=float(row.Rank))
 e=np.array(eq);yrs=(dates[-1]-dates[0]).days/365.25
 cagr=100*((e[-1]/1e8)**(1/yrs)-1);mdd=100*np.min(e/np.maximum.accumulate(e)-1)
 results.append(dict(feature=feature,cutoff=cutoff,hardstop_pct=round(hardstop*100),breakout=period,market=market,stop=stop,slots=slots,ranking=ranking,cagr=round(cagr,2),mdd=round(mdd,2),final=round(e[-1]),trades=trades,win=round(100*wins/trades,2) if trades else None))
 print(f"{i+1}/{len(configs)} {period} {market} {stop} {slots} {ranking} CAGR={cagr:.2f} MDD={mdd:.2f}",flush=True)
rec=pd.DataFrame(records);rec.to_csv("52_trade_diagnostics.csv",index=False)
rec["outcome"]=np.select([rec.pnl_pct<=-10,rec.pnl_pct<0,rec.pnl_pct>=20],["loss10","loss","winner20"],default="other")
print("TRADE COUNTS");print(rec.groupby("outcome").agg(n=("code","size"),avg_pnl=("pnl_pct","mean")).to_string())
cols=["momentum60","volume_ratio","range_close","break_strength","ma20_gap","daily_gain","breadth","amount","rank","holding_days"]
print("FEATURES BY OUTCOME");print(rec.groupby("outcome")[cols].agg(["median","mean"]).round(3).to_string())
print("LOSS10 VS WINNER20");print(rec[rec.outcome.isin(["loss10","winner20"])].groupby("outcome")[cols].median().round(3).to_string())
res=pd.DataFrame(results);res.to_csv("optimization_trade_diagnostics.csv",index=False)
print("TOP CAGR");print(res.sort_values("cagr",ascending=False).head(15).to_string(index=False))
print("TARGET");print(res[(res.cagr>=30)&(res.mdd>=-15)].to_string(index=False))
