"""
gold_validate.py — validate the strategy on a FULL YEAR of REAL Exness gold+silver.
Run on the laptop with MT5 open + logged into Exness:  py gold_validate.py
Tests: full-stack (sticky-band colour, held cross->retest, 3R), flip-on-cross theory,
EMA62 loser-filter, and colour-flip exit. All on your real XAUUSD/XAGUSD data.
"""
import numpy as np, pandas as pd, MetaTrader5 as mt5
EXNESS_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"
BARS = 6000          # ~1 year of 1H
MULT, K, SW = 1.5, 12, 10
WIN = [(0, 21)]      # gold/silver session (UTC)

def jma(src, L=14, ph=100, pw=1):
    b = 0.45*(L-1)/(0.45*(L-1)+2); a = b**pw; pr = 2.5 if ph > 100 else ph/100+1.5
    n = len(src); e0=np.zeros(n); e1=np.zeros(n); e2=np.zeros(n); j=np.zeros(n); e0[0]=src[0]; j[0]=src[0]
    for i in range(1, n):
        e0[i]=(1-a)*src[i]+a*e0[i-1]; e1[i]=(src[i]-e0[i])*(1-b)+b*e1[i-1]
        e2[i]=(e0[i]+pr*e1[i]-j[i-1])*(1-a)**2+a**2*e2[i-1]; j[i]=e2[i]+j[i-1]
    return j

def resolve(names, base):
    for x in names:
        if x.upper() == base: return x
    hits = [x for x in names if base in x.upper()]
    return sorted(set(hits), key=len)[0] if hits else None

def pf(x):
    x = pd.Series(x).dropna()
    return (x[x>0].sum()/-x[x<0].sum()) if (x<0).any() else 99.0

def analyze(sym, df):
    o,h,l,c = (df[x].values.astype(float) for x in ["open","high","low","close"]); n=len(c)
    tr=np.zeros(n); tr[0]=h[0]-l[0]
    for i in range(1,n): tr[i]=max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1]))
    atr=np.zeros(n); atr[:14]=tr[:14].mean()
    for i in range(14,n): atr[i]=(atr[i-1]*13+tr[i])/14
    j=jma((h+l+c)/3); ub=j+MULT*atr; lb=j-MULT*atr; fu=ub.copy(); fl=lb.copy(); bl=np.ones(n,bool)
    for i in range(1,n):
        fu[i]=ub[i] if (ub[i]<fu[i-1] or c[i-1]>fu[i-1]) else fu[i-1]
        fl[i]=lb[i] if (lb[i]>fl[i-1] or c[i-1]<fl[i-1]) else fl[i-1]
        bl[i]=True if c[i]>fu[i-1] else (False if c[i]<fl[i-1] else bl[i-1])
    fg=np.zeros(n,bool); frd=np.zeros(n,bool)
    for i in range(1,n):
        if bl[i] and not bl[i-1]: fg[i]=True
        if not bl[i] and bl[i-1]: frd[i]=True
    e5=pd.Series(c).ewm(span=5,adjust=False).mean().values
    e13=pd.Series(c).ewm(span=13,adjust=False).mean().values
    e62=pd.Series(c).ewm(span=62,adjust=False).mean().values
    hr=((df["time"].values//3600)%24).astype(int)
    insess=lambda hh: any(a<=hh<b for a,b in WIN)
    def sim(E,px,lng,exit_flip=False):
        sw=min(l[E-SW:E+1]) if lng else max(h[E-SW:E+1]); stop=sw-0.1*atr[E] if lng else sw+0.1*atr[E]; R=abs(px-stop)
        if R<=0: return None
        tp=px+(1 if lng else -1)*3*R
        for bb in range(E+1,n):
            if (l[bb]<=stop) if lng else (h[bb]>=stop): return -1.0
            if (h[bb]>=tp) if lng else (l[bb]<=tp): return 3.0
            if exit_flip and ((not bl[bb]) if lng else bl[bb]): return (c[bb]-px)/R if lng else (px-c[bb])/R
        return None
    rows=[]; i=62
    while i<n-1:
        d="BUY" if (o[i]<e13[i] and c[i]>e13[i] and bl[i]) else ("SELL" if (o[i]>e13[i] and c[i]<e13[i] and not bl[i]) else None)
        if not (d and insess(int(hr[i])) and atr[i]>0): i+=1; continue
        lng=d=="BUY"; entry=None
        for e in range(i+1,min(i+1+K,n)):
            ok=bl[e] if lng else not bl[e]
            if not ok or ((c[e]<e13[e]) if lng else (c[e]>e13[e])): break
            if (l[e]<=e5[e]) if lng else (h[e]>=e5[e]): entry=e; break
        fa=fg if lng else frd
        rows.append(dict(flip_on=bool(fa[max(0,i-1):i+1].any()),
            e62=(((c[i]>e62[i]) if lng else (c[i]<e62[i])) and ((e62[i]>e62[i-10]) if lng else (e62[i]<e62[i-10]))),
            r_ret=(sim(entry,e5[entry],lng) if entry is not None else None),
            r_exit=(sim(entry,e5[entry],lng,True) if entry is not None else None)))
        i=(entry+1) if entry is not None else i+1
    D=pd.DataFrame(rows); ret=D.r_ret.dropna()
    print(f"\n===== {sym}  ({n} bars) =====")
    print(f"Full-stack 3R:      n={len(ret):4} PF={pf(ret):.2f} totR={ret.sum():+.0f} WR={100*(ret>0).mean():.0f}%")
    print(f"  flip-on-cross:    n={D[D.flip_on].r_ret.dropna().shape[0]:4} PF={pf(D[D.flip_on].r_ret):.2f}")
    print(f"  no-flip:          n={D[~D.flip_on].r_ret.dropna().shape[0]:4} PF={pf(D[~D.flip_on].r_ret):.2f}")
    allL=(ret<0).sum(); k=D[D.e62].r_ret.dropna(); keptL=(k<0).sum()
    print(f"  EMA62-aligned:    keep {len(k)}/{len(ret)} losers-removed={100*(allL-keptL)/max(1,allL):.0f}% PF->{pf(k):.2f} totR->{k.sum():+.0f}")
    ex=D.r_exit.dropna()
    print(f"  colour-flip exit: n={len(ex):4} PF={pf(ex):.2f} totR={ex.sum():+.0f} WR={100*(ex>0).mean():.0f}%")

if not mt5.initialize(path=EXNESS_PATH):
    print("MT5 init failed:", mt5.last_error()); raise SystemExit
acc=mt5.account_info(); print("Connected:", acc.company if acc else "?", "|", acc.server if acc else "?")
names=[s.name for s in mt5.symbols_get()]
for base in ["XAUUSD","XAGUSD"]:
    sym=resolve(names, base)
    if not sym: print(f"{base}: not found"); continue
    mt5.symbol_select(sym, True)
    rates=mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_H1, 0, BARS)
    if rates is None or len(rates)<200: print(f"{sym}: no data"); continue
    analyze(sym, pd.DataFrame(rates))
mt5.shutdown()
print("\nDone. Paste me the output.")
