import glob,os
import numpy as np,pandas as pd
from collections import defaultdict
FILES=[f'marcap/data/marcap-{y}.parquet' for y in range(2009,2027)]
parts=[]
for p in FILES:
 if not os.path.isfile(p):raise FileNotFoundError(p)
 d=pd.read_parquet(p)
 if 'Date' not in d.columns:d=d.reset_index()
 cols=['Date','Code','Market','Open','High','Low','Close','Volume']
 if set(cols)-set(d):raise ValueError(f'{p} columns: {list(d.columns)}')
 d=d[cols]
 d=d[d.Market.isin(['KOSPI','KOSDAQ'])].copy()
 parts.append(d)
 print('loaded',p,len(d),flush=True)
df=pd.concat(parts,ignore_index=True)
del parts
df['Date']=pd.to_datetime(df.Date)
df=df[(df.Date>='2009-01-01')&(df.Date<='2026-10-09')].copy()
df['Code']=df.Code.astype(str).str.zfill(6)
df=df[df.Code.str.fullmatch(r'\d{6}')]
for c in ['Open','High','Low','Close','Volume']:df[c]=pd.to_numeric(df[c],errors='coerce')
df=df.dropna(subset=['Open','High','Low','Close','Volume'])
df=df[(df.Open>0)&(df.Close>0)].sort_values(['Code','Date']).drop_duplicates(['Code','Date'])
print('valid records',len(df),'tickers',df.Code.nunique(),flush=True)
g=df.groupby('Code',sort=False)
df['prev']=g.Close.shift()
df['vol20']=g.Volume.transform(lambda s:s.shift().rolling(20).mean())
df['ma10']=g.Close.transform(lambda s:s.rolling(10).mean())
df['setup']=(df.Close/df.prev-1).between(.05,.12)&(df.Close/df.Open-1>=.025)&((df.High-df.Close)/(df.High-df.Low).replace(0,np.nan)<=.2)&(df.Volume>=1.5*df.vol20)
signals=defaultdict(list)
for code,sub in df.groupby('Code',sort=False):
 sub=sub.reset_index(drop=True)
 for j in np.flatnonzero(sub.setup.to_numpy()):
  for k in range(j+2,min(j+8,len(sub)-1)):
   r=sub.iloc[k]
   if r.ma10>0 and abs(r.Close/r.ma10-1)<=.01 and r.Close>r.Open:
    entry=sub.iloc[k+1]
    if entry.Date>=pd.Timestamp('2010-01-01'):
     signals[entry.Date].append((code,float(entry.Open)))
    break
print('signals',sum(map(len,signals.values())),flush=True)
bars={day:sub.set_index('Code') for day,sub in df[['Date','Code','Open','High','Low','Close']].groupby('Date') if day>=pd.Timestamp('2010-01-01')}
del df
os.makedirs('results',exist_ok=True)
summaries=[]
for target,limit in [(.10,20),(.15,20),(.10,10)]:
 cash=100000000.;positions={};trades=[];curve=[];adds=0
 for day,rows in sorted(bars.items()):
  for code,p in list(positions.items()):
   if code not in rows.index:continue
   bar=rows.loc[code];p['age']+=1
   price=None;reason=''
   if bar.Open<=p['stop']:price=float(bar.Open);reason='gap'
   elif bar.Low<=p['stop']:price=p['stop'];reason='stop'
   else:
    if not p['added'] and bar.High>=p['entry']*1.05:
     ap=max(float(bar.Open),p['entry']*1.05)
     qty=int(min(cash,p['initial'])/(ap*1.00015))
     if qty:
      cost=qty*ap*1.00015;cash-=cost;p['cost']+=cost;p['qty']+=qty
      p['avg']=p['cost']/p['qty'];p['added']=True;adds+=1
    if bar.High>=p['avg']*(1+target):price=max(float(bar.Open),p['avg']*(1+target));reason='take'
    elif p['age']>=limit:price=float(bar.Close);reason='time'
   if price is not None:
    proceeds=p['qty']*price*(1-.00015-.0015)
    cash+=proceeds
    trades.append({'date':str(day.date()),'code':code,'return':proceeds/p['cost']-1,'reason':reason})
    del positions[code]
  mark=sum(p['qty']*float(rows.loc[code,'Close']) for code,p in positions.items() if code in rows.index)
  for code,entry in signals.get(day,[]):
   if code in positions or len(positions)>=8 or entry<=0 or code not in rows.index:continue
   budget=min(cash,(cash+mark)/16)
   qty=int(budget/(entry*1.00015))
   if not qty:continue
   cost=qty*entry*1.00015;cash-=cost
   positions[code]={'qty':qty,'cost':cost,'initial':cost,'avg':cost/qty,'entry':entry,'stop':entry*.96,'age':0,'added':False}
   mark+=qty*float(rows.loc[code,'Close'])
  equity=cash+sum(p['qty']*float(rows.loc[code,'Close']) for code,p in positions.items() if code in rows.index)
  curve.append((day,equity))
 e=pd.DataFrame(curve,columns=['date','equity']).set_index('date')
 annual=e.equity.resample('YE').last().pct_change()
 annual.iloc[0]=e.equity.resample('YE').last().iloc[0]/100000000-1
 years=(e.index[-1]-e.index[0]).days/365.25
 row={'target_pct':target*100,'max_days':limit,'final':e.equity.iloc[-1],'cagr_pct':((e.equity.iloc[-1]/100000000)**(1/years)-1)*100,'mdd_pct':(e.equity/e.equity.cummax()-1).min()*100,'win_pct':100*sum(t['return']>0 for t in trades)/len(trades) if trades else None,'trades':len(trades),'adds':adds}
 summaries.append(row)
 suffix=f'{int(target*100)}_{limit}'
 annual.rename('return').to_csv(f'results/annual_{suffix}.csv')
 e.to_csv(f'results/equity_{suffix}.csv')
 pd.DataFrame(trades).to_csv(f'results/trades_{suffix}.csv',index=False)
 print('RESULT',row,flush=True)
pd.DataFrame(summaries).to_csv('results/summary.csv',index=False)
