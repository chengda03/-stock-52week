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
df['ma20']=g.Close.transform(lambda x:x.shift(1).rolling(20,min_periods=20).mean())
df['ma60']=g.Close.transform(lambda x:x.shift(1).rolling(60,min_periods=60).mean())
df['ma120']=g.Close.transform(lambda x:x.shift(1).rolling(120,min_periods=120).mean())
df['ma200']=g.Close.transform(lambda x:x.shift(1).rolling(200,min_periods=200).mean())
df['prev20']=g.Close.shift(20)
df['high20']=g.High.transform(lambda x:x.shift(1).rolling(20,min_periods=20).max())
df['high60']=g.High.transform(lambda x:x.shift(1).rolling(60,min_periods=60).max())
df['high252']=g.High.transform(lambda x:x.shift(1).rolling(252,min_periods=252).max())
df['setup']=(df.Close/df.prev-1).between(.05,.12)&((df.Close/df.Open-1)>=.025)&(((df.High-df.Close)/(df.High-df.Low).replace(0,np.nan))<=.2)&(df.Volume>=1.5*df.v20)
results=[]
for code,sub in df.groupby('Code',sort=False):
 sub=sub.reset_index(drop=True)
 ix=np.flatnonzero(sub.setup.to_numpy())
 for j in ix:
  if j+20>=len(sub):continue
  signal=sub.iloc[j]
  if not (pd.Timestamp(START)<=signal.Date<=pd.Timestamp(END)):continue
  entry=sub.iloc[j+1]
  if entry.Date>pd.Timestamp(END):continue
  buy=float(entry.Open)
  if buy<=0:continue
  hit=None;stop=None;first=None;duration=None
  for daynum in range(1,21):
   bar=sub.iloc[j+daynum]
   if bar.Date>pd.Timestamp(END):break
   if hit is None and bar.High>=buy*1.04:hit=daynum
   if stop is None and bar.Low<=buy*.96:stop=daynum
   if first is None:
    if bar.Open>=buy*1.04:first='target';duration=daynum
    elif bar.Open<=buy*.96:first='stop'
    elif bar.Low<=buy*.96:first='stop'
    elif bar.High>=buy*1.04:first='target';duration=daynum
  highs=[float(sub.iloc[j+k].High) for k in range(1,21) if sub.iloc[j+k].Date<=pd.Timestamp(END)]
  results.append({'hit4_5':any(x>=buy*1.04 for x in highs[:5]),'hit4_10':any(x>=buy*1.04 for x in highs[:10]),'hit4_15':any(x>=buy*1.04 for x in highs[:15]),'hit4_20':any(x>=buy*1.04 for x in highs[:20]),'date':signal.Date,'code':code,'market':signal.Market,'signal_gain':float(signal.Close/signal.prev-1),'body':float(signal.Close/signal.Open-1),'volume_ratio':float(signal.Volume/signal.v20),'target_hit':hit is not None,'target_first':first=='target','stop_first':first=='stop','target_day':hit if hit is not None else np.nan,'first_day':duration if duration is not None else np.nan,'above20':bool(signal.Close>signal.ma20),'above60':bool(signal.Close>signal.ma60),'above120':bool(signal.Close>signal.ma120),'above200':bool(signal.Close>signal.ma200),'newhigh20':bool(signal.Close>signal.high20),'newhigh60':bool(signal.Close>signal.high60),'newhigh252':bool(signal.Close>signal.high252),'prev20return':float(signal.prev/signal.prev20-1) if pd.notna(signal.prev20) else np.nan,'close_position':float((signal.Close-signal.Low)/(signal.High-signal.Low)) if signal.High>signal.Low else np.nan})
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

# Target-hit-only condition screening, no stop-loss used.
screen=[]
def add(label,d):
 if len(d)>=100:
  screen.append({'condition':label,'signals':len(d),'hit4_pct':round(d.target_hit.mean()*100,2),'avg_target_days':round(d.loc[d.target_hit,'target_day'].mean(),2)})
add('all',trades)
for k in ['above20','above60','above120','above200','newhigh20','newhigh60','newhigh252']:
 add(k,trades[trades[k]])
for a,b in [(.05,.07),(.07,.09),(.09,.12)]:
 add(f'gain_{a}_{b}',trades[trades.signal_gain.between(a,b)])
for lo,hi in [(-1,-.1),(-.1,0),(0,.1),(.1,.2),(.2,10)]:
 add(f'prior20_{lo}_{hi}',trades[trades.prev20return.between(lo,hi,inclusive='left')])
for key in ['above200','newhigh20','newhigh60','newhigh252']:
 for lo,hi in [(.05,.07),(.07,.09),(.09,.12)]:
  d=trades[trades[key]&trades.signal_gain.between(lo,hi)]
  add(f'{key}_gain_{lo}_{hi}',d)
for k1,k2 in [('above200','newhigh20'),('above200','newhigh60'),('above200','newhigh252'),('newhigh20','newhigh60'),('above120','newhigh252')]:
 d=trades[trades[k1]&trades[k2]]
 add(f'{k1}_and_{k2}',d)
 for lo,hi in [(.07,.09),(.09,.12)]:
  add(f'{k1}_and_{k2}_gain_{lo}_{hi}',d[d.signal_gain.between(lo,hi)])
pd.DataFrame(screen).sort_values(['hit4_pct','signals'],ascending=[False,False]).to_csv('results/prob4/condition_screen.csv',index=False)
print('TOP +4% CONDITIONS (stop ignored)')
print(pd.DataFrame(screen).sort_values('hit4_pct',ascending=False).head(20).to_string(index=False),flush=True)

# Horizon comparison: entry at next-day open, no stop-loss.
rows=[]
for label,d in [
 ('all',trades),
 ('gain_9_12',trades[trades.signal_gain.between(.09,.12)]),
 ('newhigh252_gain_9_12',trades[trades.newhigh252&trades.signal_gain.between(.09,.12)]),
 ('prior20_over20',trades[trades.prev20return>=.2]),
 ('newhigh60_gain_9_12',trades[trades.newhigh60&trades.signal_gain.between(.09,.12)])
]:
 for days in [5,10,15,20]:
  rows.append({'condition':label,'days':days,'signals':len(d),'hit4_pct':round(d[f'hit4_{days}'].mean()*100,2)})
pd.DataFrame(rows).to_csv('results/prob4/horizon_comparison.csv',index=False)
print('5/10/15 DAY COMPARISON')
print(pd.DataFrame(rows).to_string(index=False),flush=True)

# Systematic multi-factor screen: discovery 2010-2022, out-of-sample 2023-2026.
# Every feature is measured on signal day or earlier. No stop-loss.
from itertools import combinations
x=trades.copy()
factors={}
for lo,hi in [(.05,.07),(.07,.09),(.09,.12)]:
 factors[f'gain_{int(lo*100)}_{int(hi*100)}']=x.signal_gain.between(lo,hi)
for name in ['above20','above60','above120','above200','newhigh20','newhigh60','newhigh252']:
 factors[name]=x[name].fillna(False)
for lo,hi in [(1.5,2),(2,3),(3,5),(5,1000000)]:
 factors[f'volume_{lo}_{hi}']=x.volume_ratio.between(lo,hi,inclusive='left')
for lo,hi in [(-10,-.1),(-.1,0),(0,.1),(.1,.2),(.2,10)]:
 factors[f'prior20_{lo}_{hi}']=x.prev20return.between(lo,hi,inclusive='left')
for lo,hi in [(.8,.9),(.9,.95),(.95,1.001)]:
 factors[f'closepos_{lo}_{hi}']=x.close_position.between(lo,hi,inclusive='left')
for lo,hi in [(.025,.04),(.04,.06),(.06,10)]:
 factors[f'body_{lo}_{hi}']=x.body.between(lo,hi,inclusive='left')
factors['KOSPI']=x.market.eq('KOSPI')
factors['KOSDAQ']=x.market.eq('KOSDAQ')
factor_names=list(factors)
train=x.date<pd.Timestamp('2023-01-01')
test=~train
screen_rows=[]
for n in [1,2,3]:
 for names in combinations(factor_names,n):
  # mutually exclusive categories cannot coexist
  if len({v.split('_')[0] for v in names if v.startswith(('gain_','volume_','prior20_','closepos_','body_'))}) < sum(v.startswith(('gain_','volume_','prior20_','closepos_','body_')) for v in names):continue
  if 'KOSPI' in names and 'KOSDAQ' in names:continue
  mask=np.ones(len(x),dtype=bool)
  for name in names:mask &= factors[name].to_numpy()
  tr=x.loc[mask & train]
  te=x.loc[mask & test]
  if len(tr)<200 or len(te)<60:continue
  row={'condition':' + '.join(names),'train_n':len(tr),'test_n':len(te)}
  for days in [5,10,15,20]:
   row[f'train_{days}d_pct']=round(tr[f'hit4_{days}'].mean()*100,2)
   row[f'test_{days}d_pct']=round(te[f'hit4_{days}'].mean()*100,2)
  screen_rows.append(row)
screen_df=pd.DataFrame(screen_rows)
screen_df=screen_df.sort_values(['test_15d_pct','test_n'],ascending=[False,False])
screen_df.to_csv('results/prob4/factor_search_all.csv',index=False)
# Filter for durable out-of-sample candidates rather than ranking only in-sample peaks.
for threshold in [50,60,80]:
 for days in [5,10,15,20]:
  keep=screen_df[(screen_df[f'train_{days}d_pct']>=threshold)&(screen_df[f'test_{days}d_pct']>=threshold)]
  keep.sort_values([f'test_{days}d_pct','test_n'],ascending=[False,False]).head(30).to_csv(f'results/prob4/threshold_{threshold}_{days}d.csv',index=False)
print('TOP VALIDATED 15D FACTOR COMBINATIONS',flush=True)
print(screen_df.head(25).to_string(index=False),flush=True)
