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
df=df[df.Code.astype(str).str.zfill(6)!="011930"]
df=df[~df.Name.astype(str).str.contains("스팩|SPAC|우B?$|우C$|우\\(",regex=True)]
for c in ["Open","High","Low","Close","Amount","Rank","Volume"]:df[c]=pd.to_numeric(df[c],errors="coerce")
df=df.sort_values(["Code","Date"]).drop_duplicates(["Code","Date"])
g=df.groupby("Code",sort=False)
for n in [10,15,20,25,30,50,90,200]:
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
df["vol20"]=g.Close.transform(lambda x:x.pct_change().rolling(20,min_periods=20).std())
df["eligible"]=(df.Rank<=500)&(df.Close>=2000)
# Market breadth: fraction of KOSPI stocks above 90d average, based on previous day only
breadth=(df[(df.Market=="KOSPI") & df.ma90.notna()].assign(above=lambda x:x.Close>x.ma90).groupby("Date").above.mean().shift(1))
df=df.sort_values(["Date","Code"])
daily={}
for d,x in df.groupby("Date"):
 daily[d]=x.set_index("Code")
dates=[d for d in sorted(daily) if d>=pd.Timestamp("2020-01-01")]
results=[]
configs=list(itertools.product([6,8,10,12],[10,15,20,25,30],[0.30,0.35,0.40]))
for i,(slot_count,stop_length,breadth_threshold) in enumerate(configs):
 ranking="momentum60"
 start_date,end_date="2020-01-01","2026-10-09"
 period=252;market=f"breadth{int(breadth_threshold*100)}";stop=stop_length;hardstop=0.10;min_amount=3e9;slippage=0.0;slots=slot_count
 cash=1e8;positions={}; sell=set();buy=[];eq=[];wins=0;trades=0;last={};navdates=[];records=[];costs={};pnl=[]
 period_dates=[d for d in dates if pd.Timestamp(start_date)<=d<=pd.Timestamp(end_date)]
 for d in period_dates:
  day=daily[d];last.update(day.Close.to_dict())
  for code in tuple(sell):
   if code in positions and code in day.index:
    px=float(day.at[code,"Open"])*(1-slippage)
    if np.isfinite(px) and px>0:
     qty,cost=positions.pop(code);cash+=qty*px*(1-.0025);trades+=1;wins+=(px*(1-.0025)>cost*1.0002);records.append((str(d.date()),code,"SELL",qty,px));pnl.append((code,str(d.date()),round(qty*(px*(1-.0025)-cost*1.0002)),round((px*(1-.0025)/(cost*1.0002)-1)*100,2),qty))
  sell.clear()
  for code in buy:
   if len(positions)>=slots or code in positions or code not in day.index:continue
   px=float(day.at[code,"Open"])*(1+slippage)
   if not np.isfinite(px) or px<=0:continue
   nav=cash+sum(q*(float(day.at[k,"Open"]) if k in day.index and pd.notna(day.at[k,"Open"]) else last.get(k,c)) for k,(q,c) in positions.items())
   budget=min(cash/1.0002,nav/slots)
   qty=int(budget/px)
   if qty>0:cash-=qty*px*1.0002;positions[code]=(qty,px);records.append((str(d.date()),code,"BUY",qty,px))
  buy=[]
  nav=cash+sum(q*last.get(k,c) for k,(q,c) in positions.items());eq.append(nav);navdates.append(d)
  for code in positions:
   if code in day.index and pd.notna(day.at[code,f"ma{stop}"]) and (day.at[code,"Close"]<day.at[code,f"ma{stop}"] or day.at[code,"Close"]<positions[code][1]*(1-hardstop)):sell.add(code)
  threshold=breadth_threshold
  active=(pd.notna(breadth.get(d,np.nan)) and breadth.get(d,0)>=threshold)
  if not active:sell.update(positions.keys())
  if active:
   candidates=day[day.eligible & (day.Amount>=min_amount) & day[f"b{period}"] & (day.Close>day.ma200) & (day.gap20<=0.40)]
   col={"low_break_strength":"break_strength","low_vol20":"vol20"}.get(ranking,ranking)
   candidates=candidates.sort_values(col,ascending=ranking in ["low_break_strength","low_vol20"])
   buy=[k for k in candidates.index if k not in positions][:slots]
 e=np.array(eq);yrs=(period_dates[-1]-period_dates[0]).days/365.25
 cagr=100*((e[-1]/1e8)**(1/yrs)-1);mdd=100*np.min(e/np.maximum.accumulate(e)-1)
 results.append(dict(start=start_date,end=end_date,slippage_pct=round(slippage*100,2),hardstop_pct=round(hardstop*100),breakout=period,market=market,stop=stop,slots=slots,ranking=ranking,cagr=round(cagr,2),mdd=round(mdd,2),final=round(e[-1]),trades=trades,win=round(100*wins/trades,2) if trades else None))
 navseries=pd.Series(eq,index=pd.DatetimeIndex(navdates));annual=navseries.resample("YE").last().pct_change()*100;annual.iloc[0]=(navseries.loc[navseries.index.year==annual.index[0].year].iloc[-1]/1e8-1)*100
 if False: print("TOP_2026_PNL",sorted(pnl,key=lambda x:x[2],reverse=True)[:25],flush=True)
 if False: print("BOTTOM_2026_PNL",sorted(pnl,key=lambda x:x[2])[:10],flush=True)
 if False: print("OPEN_POSITIONS",[(k,v[0],v[1],last.get(k)) for k,v in positions.items()],flush=True)
 print("INDEPENDENT_PERIOD",start_date,end_date,"START_NAV",100000000,"END_NAV",round(e[-1]),"RETURN_PCT",round((e[-1]/1e8-1)*100,2),"CAGR",round(cagr,2),"MDD",round(mdd,2),flush=True)
 print("AUDIT slippage",slippage,"ANNUAL",annual.round(2).to_dict(),"MAX_DD_DATE",str((navseries/navseries.cummax()-1).idxmin().date()),"BUY_SELL_RECORDS",len(records),flush=True)
 if False:
  pd.DataFrame(records,columns=["date","code","side","qty","price"]).to_csv("52_full_ex011930_10slots_ledger.csv",index=False)
  navseries.rename("equity").to_csv("52_full_ex011930_10slots_equity.csv")
 print(f"{i+1}/{len(configs)} {period} {market} {stop} {slots} {ranking} CAGR={cagr:.2f} MDD={mdd:.2f}",flush=True)
res=pd.DataFrame(results);res.to_csv("52_improved_conditions.csv",index=False)
print("TOP CAGR");print(res.sort_values("cagr",ascending=False).head(15).to_string(index=False))
print("BEST BALANCED");print(res.sort_values(["mdd","cagr"],ascending=[False,False]).head(15).to_string(index=False))

print("TARGET CAGR>=50 AND MDD>=-20");print(res[(res.cagr>=50)&(res.mdd>=-20)].to_string(index=False))
