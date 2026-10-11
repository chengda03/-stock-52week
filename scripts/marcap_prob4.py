import os
import numpy as np
import pandas as pd

START='2010-01-01';END='2026-10-09'
COLS=['Date','Code','Market','Open','High','Low','Close','Volume']
frames=[]
for y in range(2009,2027):
 p=f'marcap/data/marcap-{y}.parquet'
 d=pd.read_parquet(p)
 if 'Date' not in d.columns:d=d.reset_index()
 d=d[COLS]
 d=d[d.Market.isin(['KOSPI','KOSDAQ'])].copy()
 frames.append(d)
 print('year',y,'rows',len(d),flush=True)
df=pd.concat(frames,ignore_index=True);del frames
df['Date']=pd.to_datetime(df.Date)
df['Code']=df.Code.astype(str).str.zfill(6)
df=df[df.Code.str.fullmatch(r'\d{6}')].copy()
for c in ['Open','High','Low','Close','Volume']:
 df[c]=pd.to_numeric(df[c],errors='coerce')
df=df.dropna(subset=['Open','High','Low','Close','Volume'])
df=df[(df.Open>0)&(df.High>0)&(df.Low>0)&(df.Close>0)]
df=df.sort_values(['Code','Date']).drop_duplicates(['Code','Date'])
g=df.groupby('Code',sort=False)
df['prev']=g.Close.shift()
df['v20']=g.Volume.transform(lambda x:x.shift(1).rolling(20,min_periods=20).mean())
df['setup']=(df.Close/df.prev-1).between(.05,.12)&((df.Close/df.Open-1)>=.025)&(((df.High-df.Close)/(df.High-df.Low).replace(0,np.nan))<=.2)&(df.Volume>=1.5*df.v20)
results=[]
for code,sub in df.groupby('Code',sort=False):
 sub=sub.reset_index(drop=True)
 ix=np.flatnonzero(sub.setup.to_numpy())
 for j in ix:
  if j+5>=len(sub):continue
  signal=sub.iloc[j]
  if not (pd.Timestamp(START)<=signal.Date<=pd.Timestamp(END)):continue
  entry=sub.iloc[j+1]
  if entry.Date>pd.Timestamp(END):continue
  buy=float(entry.Open)
  if buy<=0:continue
  hit=None;stop=None;first=None;duration=None
  for daynum in range(1,6):
   bar=sub.iloc[j+daynum]
   if bar.Date>pd.Timestamp(END):break
   if hit is None and bar.High>=buy*1.04:hit=daynum
   if stop is None and bar.Low<=buy*.96:stop=daynum
   if first is None:
    if bar.Open>=buy*1.04:first='target';duration=daynum
    elif bar.Open<=buy*.96:first='stop'
    elif bar.Low<=buy*.96:first='stop'
    elif bar.High>=buy*1.04:first='target';duration=daynum
  results.append({'date':signal.Date,'code':code,'market':signal.Market,'signal_gain':float(signal.Close/signal.prev-1),'body':float(signal.Close/signal.Open-1),'volume_ratio':float(signal.Volume/signal.v20),'target_hit':hit is not None,'target_first':first=='target','stop_first':first=='stop','target_day':hit if hit is not None else np.nan,'first_day':duration if duration is not None else np.nan,'ma200':np.nan})
trades=pd.DataFrame(results)
if trades.empty:raise RuntimeError('No eligible signals')
trades['year']=trades.date.dt.year
def stats(d):
 return pd.Series({'signals':len(d),'hit4_pct':round(d.target_hit.mean()*100,2),'first4_pct':round(d.target_first.mean()*100,2),'stopfirst_pct':round(d.stop_first.mean()*100,2),'avg_target_days':round(d.loc[d.target_hit,'target_day'].mean(),2),'avg_first_days':round(d.loc[d.target_first,'first_day'].mean(),2)})
os.makedirs('results/prob4',exist_ok=True)
stats(trades).to_frame('value').to_csv('results/prob4/summary.csv')
trades.groupby('year').apply(stats,include_groups=False).to_csv('results/prob4/annual.csv')
trades.groupby('market').apply(stats,include_groups=False).to_csv('results/prob4/markets.csv')
trades.assign(gain_bin=pd.cut(trades.signal_gain,[.05,.07,.09,.12],include_lowest=True)).groupby('gain_bin',observed=True).apply(stats,include_groups=False).to_csv('results/prob4/gain_bins.csv')
trades.assign(volume_bin=pd.cut(trades.volume_ratio,[1.5,2,3,5,float('inf')],include_lowest=True)).groupby('volume_bin',observed=True).apply(stats,include_groups=False).to_csv('results/prob4/volume_bins.csv')
trades.to_csv('results/prob4/trades.csv',index=False)
print(stats(trades).to_string(),flush=True)
