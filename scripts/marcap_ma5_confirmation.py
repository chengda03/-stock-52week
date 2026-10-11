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

# Entry confirmation: first MA5 touch within five sessions, then bullish confirmation
# during the next three sessions. Buy next day's open; no future information at entry.
records=[]; denominators=[]
for code,sub in df.groupby('Code',sort=False):
 a=sub[['Date','Open','High','Low','Close','Volume','ma5_prev','setup']].reset_index(drop=True)
 dt=a.Date.to_numpy();op=a.Open.to_numpy(float);hi=a.High.to_numpy(float)
 lo=a.Low.to_numpy(float);cl=a.Close.to_numpy(float);vol=a.Volume.to_numpy(float)
 ma=a.ma5_prev.to_numpy(float);sigidx=np.flatnonzero(a.setup.to_numpy())
 for j in sigidx:
  if j+30>=len(a) or not START<=pd.Timestamp(dt[j])<=END or pd.Timestamp(dt[j+30])>END:continue
  period='2010-2022' if pd.Timestamp(dt[j])<pd.Timestamp('2023-01-01') else '2023-2026'
  denominators.append((period,code,int(j)))
  touch=next((k for k in range(1,6) if np.isfinite(ma[j+k]) and lo[j+k]<=ma[j+k]),None)
  if touch is None:continue
  signal_volume=vol[j]
  for confirm in ['bull','bull_above_prevclose','bull_break_prevhigh']:
   for volume_rule in ['any','vol_le_50','vol_le_30','vol_2day_decline']:
    entry=None
    for k in range(touch, min(touch+3,8)):
     t=j+k
     if k==touch:continue # confirm only after touch day
     bull=cl[t]>op[t]
     if confirm=='bull_above_prevclose':bull=bull and cl[t]>cl[t-1]
     if confirm=='bull_break_prevhigh':bull=bull and cl[t]>hi[t-1]
     if not bull:continue
     if volume_rule=='vol_le_50' and not vol[j+touch]<=signal_volume*.5:continue
     if volume_rule=='vol_le_30' and not vol[j+touch]<=signal_volume*.3:continue
     if volume_rule=='vol_2day_decline' and not (touch>=2 and vol[j+touch]<vol[j+touch-1]<vol[j+touch-2]):continue
     entry=t+1;break
    if entry is None:continue
    buy=op[entry]
    if buy<=0:continue
    for horizon in [10,20]:
     for stop in [.05,.07,.10]:
      target=buy*1.04;floor=buy*(1-stop);price=cl[entry+horizon-1];outcome='time'
      for t in range(entry,entry+horizon):
       if op[t]>=target:price=op[t];outcome='target';break
       if op[t]<=floor:price=op[t];outcome='stop_gap';break
       if lo[t]<=floor:price=floor;outcome='stop';break
       if hi[t]>=target:price=target;outcome='target';break
      records.append((period,confirm,volume_rule,horizon,stop,code,(price/buy-1)*100,outcome,entry-j))
out=pd.DataFrame(records,columns=['period','confirmation','volume_rule','horizon','stop','code','return_pct','outcome','entry_day'])
den=pd.DataFrame(denominators,columns=['period','code','j']).groupby('period').size()
summ=out.groupby(['period','confirmation','volume_rule','horizon','stop']).agg(
 signals=('return_pct','size'),
 win_pct=('return_pct',lambda s:round(100*(s>0).mean(),2)),
 target_pct=('outcome',lambda s:round(100*(s=='target').mean(),2)),
 mean_return_pct=('return_pct',lambda s:round(s.mean(),3)),
 stop_pct_rate=('outcome',lambda s:round(100*s.isin(['stop','stop_gap']).mean(),2)),
 avg_entry_day=('entry_day',lambda s:round(s.mean(),2))
).reset_index()
summ['entry_fill_pct']=summ.apply(lambda r:round(100*r.signals/den.loc[r.period],2),axis=1)
os.makedirs('results/prob4',exist_ok=True)
summ.to_csv('results/prob4/ma5_bull_confirmation.csv',index=False)
print(summ.to_string(index=False),flush=True)
