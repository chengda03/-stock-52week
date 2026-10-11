import os
import numpy as np
import pandas as pd
START=pd.Timestamp('2010-01-01'); END=pd.Timestamp('2026-10-09')
cols=['Date','Code','Market','Open','High','Low','Close','Volume']
frames=[]
for year in range(2009,2027):
 d=pd.read_parquet(f'marcap/data/marcap-{year}.parquet')
 if 'Date' not in d.columns:d=d.reset_index()
 frames.append(d[cols])
df=pd.concat(frames,ignore_index=True)
df=df[df.Market.isin(['KOSPI','KOSDAQ'])].copy()
df.Date=pd.to_datetime(df.Date)
df.Code=df.Code.astype(str).str.zfill(6)
df=df[df.Code.str.fullmatch(r'\d{6}')]
for c in ['Open','High','Low','Close','Volume']:df[c]=pd.to_numeric(df[c],errors='coerce')
df=df.dropna(subset=['Open','High','Low','Close','Volume'])
df=df[(df.Open>0)&(df.High>0)&(df.Low>0)&(df.Close>0)]
df=df.sort_values(['Code','Date']).drop_duplicates(['Code','Date'])
g=df.groupby('Code',sort=False)
df['prev']=g.Close.shift()
df['v20']=g.Volume.transform(lambda s:s.shift().rolling(20,min_periods=20).mean())
df['high60']=g.High.transform(lambda s:s.shift().rolling(60,min_periods=60).max())
df['ma5_prev']=g.Close.transform(lambda s:s.shift().rolling(5,min_periods=5).mean())
df['ma10_prev']=g.Close.transform(lambda s:s.shift().rolling(10,min_periods=10).mean())
df['setup']=(df.Close/df.prev-1).between(.09,.12)&(df.Close/df.Open-1>=.025)&((df.High-df.Close)/(df.High-df.Low).replace(0,np.nan)<=.2)&(df.Volume>=1.5*df.v20)
rows=[]
for code,sub in df.groupby('Code',sort=False):
 sub=sub.reset_index(drop=True)
 signals=np.flatnonzero(sub.setup.to_numpy())
 for j in signals:
  if j+25>=len(sub):continue
  sig=sub.iloc[j]
  if not START<=sig.Date<=END:continue
  if sub.iloc[j+25].Date>END:continue
  selected=bool(sig.Close>sig.high60 and 1.5<=sig.Volume/sig.v20<2)
  for strategy in ['next_open','day3_open','day5_open','ma5_touch','ma10_touch']:
   k=None;buy=None
   if strategy.endswith('_open'):
    k={'next_open':1,'day3_open':3,'day5_open':5}[strategy]
    buy=float(sub.iloc[j+k].Open)
   else:
    ma='ma5_prev' if strategy=='ma5_touch' else 'ma10_prev'
    for t in range(1,6):
     bar=sub.iloc[j+t];level=float(bar[ma])
     if np.isfinite(level) and bar.Low<=level:
      k=t;buy=float(min(bar.Open,level));break
   if k is None or buy<=0:continue
   for horizon in [10,20]:
    if j+k+horizon-1>=len(sub):continue
    for stop in [.05,.07,.10]:
     target=buy*1.04;floor=buy*(1-stop);exit_price=None;outcome=None
     # Entry-day intraday movement is unknown after a limit touch; start exit monitoring next day for touch strategies.
     begin=0 if strategy.endswith('_open') else 1
     for t in range(begin,horizon):
      b=sub.iloc[j+k+t]
      if b.Open>=target:exit_price=float(b.Open);outcome='target';break
      if b.Open<=floor:exit_price=float(b.Open);outcome='stop_gap';break
      if b.Low<=floor:exit_price=floor;outcome='stop';break
      if b.High>=target:exit_price=target;outcome='target';break
     if exit_price is None:
      exit_price=float(sub.iloc[j+k+horizon-1].Close);outcome='time'
     rows.append((sig.Date,code,selected,strategy,k,horizon,stop,outcome,(exit_price/buy-1)*100))
out=pd.DataFrame(rows,columns=['date','code','selected','strategy','entry_day','horizon','stop','outcome','return_pct'])
out['period']=np.where(out.date<pd.Timestamp('2023-01-01'),'2010-2022','2023-2026')
out['universe']='all'
out2=out[out.selected].copy();out2['universe']='newhigh60_vol1.5to2'
out=pd.concat([out,out2],ignore_index=True)
summary=out.groupby(['period','universe','strategy','horizon','stop']).agg(
 signals=('return_pct','size'),
 win_pct=('return_pct',lambda s:round(100*(s>0).mean(),2)),
 target_pct=('outcome',lambda s:round(100*(s=='target').mean(),2)),
 mean_return_pct=('return_pct',lambda s:round(s.mean(),3)),
 median_return_pct=('return_pct',lambda s:round(s.median(),3)),
 stop_pct=('outcome',lambda s:round(100*s.isin(['stop','stop_gap']).mean(),2)),
 avg_entry_day=('entry_day',lambda s:round(s.mean(),2))
).reset_index()
os.makedirs('results/prob4',exist_ok=True)
summary.to_csv('results/prob4/entry_timing_comparison.csv',index=False)
# Denominator for entry fill rates: all eligible setups with 25 future bars, using next-open rows
den=out[(out.strategy=='next_open')&(out.horizon==10)&(out.stop==.05)].groupby(['period','universe']).size()
summary['entry_fill_pct']=summary.apply(lambda r:round(100*r.signals/den.loc[(r.period,r.universe)],2),axis=1)
summary.to_csv('results/prob4/entry_timing_comparison.csv',index=False)
print(summary.to_string(index=False),flush=True)
