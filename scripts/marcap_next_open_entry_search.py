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
 O=a.Open.to_numpy();H=a.High.to_numpy();L=a.Low.to_numpy();C=a.Close.to_numpy()
 M=a.ma5_prev.to_numpy()
 for j in np.flatnonzero(a.setup.to_numpy()):
  if j+35>=len(a) or not START<=a.Date.iloc[j]<=END or a.Date.iloc[j+35]>END:continue
  period='2010-2022' if a.Date.iloc[j]<pd.Timestamp('2023-01-01') else '2023-2026'
  entries=[('next_open',j+1)]
  red=next((t for t in range(j+1,j+6) if C[t]<O[t]),None)
  if red is not None:entries.append(('first_red_next_open',red+1))
  green=next((t for t in range(j+2,j+7) if C[t]>O[t] and C[t]>=M[t] and np.any(C[j+1:t]<O[j+1:t])),None)
  if green is not None:entries.append(('red_then_green_ma5_next_open',green+1))
  if C[j+1]<O[j+1]:entries.append(('day1_red_day2_open',j+2))
  for name,entry in entries:
   buy=O[entry]
   if not np.isfinite(buy) or buy<=0:continue
   for stop in [.05,.07,.10]:
    floor=buy*(1-stop);target=buy*1.04;price=C[entry+9];outcome='time'
    for z in range(entry,entry+10):
     if z==entry:
      if L[z]<=floor:price=floor;outcome='stop';break
      continue
     if O[z]>=target:price=O[z];outcome='target';break
     if O[z]<=floor:price=O[z];outcome='stop_gap';break
     if L[z]<=floor:price=floor;outcome='stop';break
     if H[z]>=target:price=target;outcome='target';break
    rows.append((period,name,stop,(price/buy-1)*100,outcome))
out=pd.DataFrame(rows,columns=['period','strategy','stop','ret','outcome'])
res=out.groupby(['period','strategy','stop']).agg(n=('ret','size'),win_pct=('ret',lambda s:round(100*(s>0).mean(),2)),mean_pct=('ret',lambda s:round(s.mean(),3)),stop_pct=('outcome',lambda s:round(100*s.isin(['stop','stop_gap']).mean(),2))).reset_index()
os.makedirs('results/prob4',exist_ok=True)
res.to_csv('results/prob4/next_open_entry_search.csv',index=False)
print(res.to_string(index=False))
