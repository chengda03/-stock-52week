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
df["gap20"]=df.Close/df.ma20-1
df["range_close"]=(df.Close-df.Low)/(df.High-df.Low).replace(0,np.nan)
df["break_strength"]=df.Close/df.priorhigh252-1
df["volume_ratio"]=df.Volume/df.volume_avg20
df["mom126"]=df.Close/g.Close.shift(126)-1
df["mom252"]=df.Close/g.Close.shift(252)-1
df["eligible"]=(df.Rank<=500)&(df.Close>=2000)
# Market breadth: fraction of KOSPI stocks above 90d average, based on previous day only
breadth=(df[(df.Market=="KOSPI") & df.ma90.notna()].assign(above=lambda x:x.Close>x.ma90).groupby("Date").above.mean().shift(1))
df=df.sort_values(["Date","Code"])
daily={}
for d,x in df.groupby("Date"):
 daily[d]=x.set_index("Code")
dates=[d for d in sorted(daily) if d>=pd.Timestamp("2020-01-01")]
results=[]
configs=[(0.0,8)]
for i,(slippage,slots) in enumerate(configs):
 period=252;market="breadth35";stop=20;ranking="amount";hardstop=0.10;min_amount=3e9
 cash=1e8;positions={}; sell=set();buy=[];eq=[];wins=0;trades=0;last={};navdates=[];records=[]
 for d in dates:
  day=daily[d];last.update(day.Close.to_dict())
  for code in tuple(sell):
   if code in positions and code in day.index:
    px=float(day.at[code,"Open"])*(1-slippage)
    if np.isfinite(px) and px>0:
     qty,cost=positions.pop(code);cash+=qty*px*(1-.0025);trades+=1;wins+=(px*(1-.0025)>cost*1.0002);records.append((str(d.date()),code,"SELL",qty,px))
  sell.clear()
  for code in buy:
   if len(positions)>=slots or code in positions or code not in day.index:continue
   px=float(day.at[code,"Open"])*(1+slippage)
   if not np.isfinite(px) or px<=0:continue
   nav=cash+sum(q*last.get(k,c) for k,(q,c) in positions.items())
   budget=min(cash/1.0002,nav/slots)
   qty=int(budget/px)
   if qty>0:cash-=qty*px*1.0002;positions[code]=(qty,px);records.append((str(d.date()),code,"BUY",qty,px))
  buy=[]
  nav=cash+sum(q*last.get(k,c) for k,(q,c) in positions.items());eq.append(nav);navdates.append(d)
  for code in positions:
   if code in day.index and pd.notna(day.at[code,f"ma{stop}"]) and (day.at[code,"Close"]<day.at[code,f"ma{stop}"] or day.at[code,"Close"]<positions[code][1]*(1-hardstop)):sell.add(code)
  threshold=0.35
  active=(pd.notna(breadth.get(d,np.nan)) and breadth.get(d,0)>=threshold)
  if not active:sell.update(positions.keys())
  if active:
   candidates=day[day.eligible & (day.Amount>=min_amount) & day[f"b{period}"] & (day.Close>day.ma200) & (day.gap20<=0.40)]
   col="Amount" if ranking=="amount" else "mom252"
   candidates=candidates.sort_values(col,ascending=False)
   buy=[k for k in candidates.index if k not in positions][:slots]
 e=np.array(eq);yrs=(dates[-1]-dates[0]).days/365.25
 cagr=100*((e[-1]/1e8)**(1/yrs)-1);mdd=100*np.min(e/np.maximum.accumulate(e)-1)
 results.append(dict(slippage_pct=round(slippage*100,2),hardstop_pct=round(hardstop*100),breakout=period,market=market,stop=stop,slots=slots,ranking=ranking,cagr=round(cagr,2),mdd=round(mdd,2),final=round(e[-1]),trades=trades,win=round(100*wins/trades,2) if trades else None))
 navseries=pd.Series(eq,index=pd.DatetimeIndex(navdates));annual=navseries.resample("YE").last().pct_change()*100;annual.iloc[0]=(navseries.loc[navseries.index.year==annual.index[0].year].iloc[-1]/1e8-1)*100
 for start,end in [("2020-01-01","2023-12-31"),("2024-01-01","2026-10-09")]:
  segment=navseries.loc[start:end]
  if len(segment):
   initial=1e8 if start=="2020-01-01" else navseries.loc[navseries.index<segment.index[0]].iloc[-1]
   period_years=(segment.index[-1]-segment.index[0]).days/365.25
   seg_cagr=((segment.iloc[-1]/initial)**(1/period_years)-1)*100
   seg_mdd=((segment/np.maximum.accumulate(np.r_[initial,segment.values])[1:]-1).min())*100
   print("PERIOD",start,end,"START_NAV",round(initial),"END_NAV",round(segment.iloc[-1]),"RETURN_PCT",round((segment.iloc[-1]/initial-1)*100,2),"CAGR",round(seg_cagr,2),"MDD",round(seg_mdd,2),flush=True)
 print("AUDIT slippage",slippage,"ANNUAL",annual.round(2).to_dict(),"MAX_DD_DATE",str((navseries/navseries.cummax()-1).idxmin().date()),"BUY_SELL_RECORDS",len(records),flush=True)
 if slippage==0:
  pd.DataFrame(records,columns=["date","code","side","qty","price"]).to_csv("52_trade_ledger.csv",index=False)
  navseries.rename("equity").to_csv("52_daily_equity.csv")
 print(f"{i+1}/{len(configs)} {period} {market} {stop} {slots} {ranking} CAGR={cagr:.2f} MDD={mdd:.2f}",flush=True)
res=pd.DataFrame(results);res.to_csv("52_period_comparison.csv",index=False)
print("TOP CAGR");print(res.sort_values("cagr",ascending=False).head(15).to_string(index=False))
print("BEST BALANCED");print(res.sort_values(["mdd","cagr"],ascending=[False,False]).head(15).to_string(index=False))
