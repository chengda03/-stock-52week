import os,glob,math
import numpy as np,pandas as pd
from collections import defaultdict
paths=sorted(glob.glob('marcap/data/marcap-20*.parquet'))
paths=[p for p in paths if 2009<=int(os.path.basename(p)[7:11])<=2026]
frames=[]
for p in paths:
 d=pd.read_parquet(p, columns=['Date','Code','Market','Open','High','Low','Close','Volume'])
 d['Code']=d['Code'].astype(str).str.zfill(6)
 if 'Date' not in d.columns:
  d=d.reset_index()
 d=d[['Date','Code','Market','Open','High','Low','Close','Volume']]
 d=d[d['Market'].isin(['KOSPI','KOSDAQ'])]
 for col in ['Open','High','Low','Close','Volume']:
  d[col]=pd.to_numeric(d[col],errors='coerce',downcast='float')
 frames.append(d)
if not frames:
 raise RuntimeError('No annual marcap Parquet files found')
df=pd.concat(frames,ignore_index=True,copy=False)
del frames
df['Date']=pd.to_datetime(df['Date'])
df=df[(df.Date>='2009-01-01')&(df.Date<='2026-10-09')&(df.Market.isin(['KOSPI','KOSDAQ']))].copy()
print('Loaded rows:',len(df),'tickers:',df.Code.nunique(),flush=True)
df=df[df.Code.str.fullmatch(r'\d{6}',na=False)]
for c in ['Open','High','Low','Close','Volume']:
 df[c]=pd.to_numeric(df[c],errors='coerce')
df=df.sort_values(['Code','Date'])
df=df.drop_duplicates(['Code','Date'],keep='last')
df['prev']=df.groupby('Code').Close.shift()
df['v20']=df.groupby('Code').Volume.transform(lambda s:s.shift().rolling(20).mean())
df['ma10']=df.groupby('Code').Close.transform(lambda s:s.rolling(10).mean())
df['gain']=df.Close/df.prev-1
df['body']=df.Close/df.Open-1
df['top']=(df.High-df.Close)/(df.High-df.Low).replace(0,np.nan)
df['setup']=df.gain.between(.05,.12)&(df.body>=.025)&(df.top<=.2)&(df.Volume>=1.5*df.v20)
signals=[]
for code,g in df.groupby('Code',sort=False):
 g=g.reset_index(drop=True)
 setup=np.flatnonzero(g.setup.to_numpy())
 last=-100
 for j in setup:
  for k in range(j+2,min(j+8,len(g)-1)):
   r=g.iloc[k]
   if abs(r.Close/r.ma10-1)<=.01 and r.Close>r.Open and k>last:
    signals.append((g.iloc[k+1].Date,code,float(g.iloc[k+1].Open)))
    last=k+1;break
signals.sort()
daily={d:sub.set_index('Code') for d,sub in df[['Date','Code','Open','High','Low','Close']].groupby('Date',sort=True)}
del df
dates=sorted(d for d in daily if d>=pd.Timestamp('2010-01-01'))
sig=defaultdict(list)
for s in signals:sig[s[0]].append(s)
results=[]
for take in [.05,.10,None]:
 cash=100000000.;pos={};trades=[];equity=[];year={}
 print('Starting scenario',take,flush=True)
 for date in dates:
  rows=daily[date]
  for code,p in list(pos.items()):
   if code not in rows.index:continue
   r=rows.loc[code]
   if isinstance(r,pd.DataFrame):r=r.iloc[0]
   p['days']+=1
   if r.Open<=p['stop']:price=float(r.Open);reason='stop_gap'
   elif r.Low<=p['stop']:price=p['stop'];reason='stop'
   elif take is not None and r.High>=p['avg']*(1+take):price=p['avg']*(1+take);reason='take'
   elif p['days']>=3:price=float(r.Close);reason='time'
   else:price=None;reason=''
   if price is not None:
    cash+=p['qty']*price*.9985
    trades.append({'date':str(date.date()),'code':code,'return':(price*p['qty']*.9985-p['cost'])/p['cost'],'reason':reason})
    del pos[code];continue
   if not p['added'] and r.High>=p['first']*1.05:
    price=max(float(r.Open),p['first']*1.05)
    budget=min(cash,p['initial'])
    q=int(budget/(price*1.00015))
    if q>0:
     cash-=q*price*1.00015;p['cost']+=q*price*1.00015;p['qty']+=q
     p['avg']=p['cost']/p['qty'];p['added']=True
  for _,code,price in sig.get(date,[]):
   if code in pos or len(pos)>=8 or price<=0:continue
   budget=min(cash, (cash+sum(v['qty']*float(rows.loc[c].Close) for c,v in pos.items() if c in rows.index))/8)
   qty=int(budget/(price*1.00015))
   if qty<1:continue
   cost=qty*price*1.00015;cash-=cost
   pos[code]={'qty':qty,'cost':cost,'first':price,'avg':cost/qty,'stop':price*.96,'days':0,'added':False,'initial':cost}
  value=cash+sum(p['qty']*float(rows.loc[c].Close) for c,p in pos.items() if c in rows.index)
  equity.append((date,value))
 e=pd.DataFrame(equity,columns=['date','equity']).set_index('date')
 if e.empty:continue
 yr=e.equity.resample('YE').last().pct_change()
 yr.iloc[0]=e.equity.iloc[0]/100000000-1
 years=(e.index[-1]-e.index[0]).days/365.25
 summary={'take':str(take),'final':float(e.equity.iloc[-1]),'cagr':float((e.equity.iloc[-1]/100000000)**(1/years)-1),'mdd':float((e.equity/e.equity.cummax()-1).min()),'win_rate':float(np.mean([t['return']>0 for t in trades])) if trades else None,'trades':len(trades)}
 results.append(summary)
 e.to_csv('equity_'+str(take)+'.csv')
 pd.DataFrame(trades).to_csv('trades_'+str(take)+'.csv',index=False)
 yr.rename('return').to_csv('annual_'+str(take)+'.csv')
pd.DataFrame(results).to_csv('summary.csv',index=False)
print(pd.DataFrame(results).to_string(index=False))
