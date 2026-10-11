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
 a=sub.reset_index(drop=True)
 O=a.Open.to_numpy();H=a.High.to_numpy();L=a.Low.to_numpy();C=a.Close.to_numpy();V=a.Volume.to_numpy()
 M=a.ma5_prev.to_numpy();M10=a.ma10_prev.to_numpy();D=a.Date.to_numpy()
 for j in np.flatnonzero(a.setup.to_numpy()):
  if j+30>=len(a) or not START<=a.Date.iloc[j]<=END or a.Date.iloc[j+30]>END:continue
  t=next((j+k for k in range(1,6) if np.isfinite(M[j+k]) and L[j+k]<=M[j+k]),None)
  if t is None:continue
  buy=min(O[t],M[t]);period='2010-2022' if a.Date.iloc[j]<pd.Timestamp('2023-01-01') else '2023-2026'
  conditions={
   'all':True,
   'touch_day_1to2':t-j<=2,
   'touch_day_3to5':t-j>=3,
   'signal_vol_1p5to2':1.5<=V[j]/a.v20.iloc[j]<2,
   'signal_vol_2to3':2<=V[j]/a.v20.iloc[j]<3,
   'signal_vol_3plus':V[j]/a.v20.iloc[j]>=3,
   'pullback_vol_le50':V[t]<=V[j]*.5,
   'pullback_vol_gt50':V[t]>V[j]*.5,
   'touch_above_ma10':np.isfinite(M10[t]) and buy>=M10[t],
   'touch_below_ma10':np.isfinite(M10[t]) and buy<M10[t],
   'signal_above_ma10':np.isfinite(M10[j]) and C[j]>M10[j],
   'touch_close_above_ma5':C[t]>=M[t],
   'touch_close_below_ma5':C[t]<M[t],
   'signal_close_high60':C[j]>=a.high60.iloc[j] if np.isfinite(a.high60.iloc[j]) else False,
   'signal_not_high60':C[j]<a.high60.iloc[j] if np.isfinite(a.high60.iloc[j]) else False,
   'touch_red':C[t]<O[t],
   'touch_green':C[t]>O[t],
  }
  for horizon in [10]:
   for stop in [.05,.07,.10]:
    floor=buy*(1-stop);target=buy*1.04;price=C[t+horizon-1];outcome='time'
    for z in range(t,t+horizon):
     if z==t:
      if L[z]<=floor:price=floor;outcome='stop';break
      continue
     if O[z]>=target:price=O[z];outcome='target';break
     if O[z]<=floor:price=O[z];outcome='stop_gap';break
     if L[z]<=floor:price=floor;outcome='stop';break
     if H[z]>=target:price=target;outcome='target';break
    ret=(price/buy-1)*100
    for name,yes in conditions.items():
     if yes:rows.append((period,name,stop,ret,outcome))
out=pd.DataFrame(rows,columns=['period','filter','stop','ret','outcome'])
res=out.groupby(['period','filter','stop']).agg(n=('ret','size'),win_pct=('ret',lambda s:round(100*(s>0).mean(),2)),mean_pct=('ret',lambda s:round(s.mean(),3)),stop_pct=('outcome',lambda s:round(100*s.isin(['stop','stop_gap']).mean(),2))).reset_index()
os.makedirs('results/prob4',exist_ok=True)
res.to_csv('results/prob4/ma5_failure_filters.csv',index=False)
print(res.to_string(index=False))
