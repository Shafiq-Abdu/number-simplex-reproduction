#!/usr/bin/env python3
"""Synthetic number-geometry recovery; independent, checkpointed, resumable.
Requires numpy, scipy, pandas, scikit-learn. Run one model per compute node.
"""
import argparse, json, os, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold

BINS=(60,75,90,100,150,180,225,300,450,900)
GAMMAS=(0.2,0.5,0.8)
COUNTS=np.array([53,67,52,62,51,48,65,49,47])
MODELS=('linear','log','curve','regular_simplex','irregular_simplex','random_highd','null')

def geometry(model, rng):
    k=np.arange(1,10,dtype=float)
    if model=='linear': z=(k-5)[:,None]
    elif model=='log': z=np.log(k)[:,None]
    elif model=='curve':
        t=np.linspace(-1,1,9); z=np.column_stack((t,t*t))
    elif model=='regular_simplex': z=np.eye(9)-np.ones((9,9))/9
    elif model=='irregular_simplex': z=(np.eye(9)-np.ones((9,9))/9)@np.diag(np.linspace(.5,1.5,9))
    elif model=='random_highd': z=rng.normal(size=(9,12))
    elif model=='null': z=np.zeros((9,1))
    else: raise ValueError(model)
    z=z-z.mean(axis=0)
    if np.linalg.norm(z)>0: z=z/np.sqrt(np.mean(np.sum(z*z,axis=1)))
    return z

def load_rates(path, seed):
    if path:
        df=pd.read_csv(path)
        if 'subject' in df: df=df[df.subject.astype(str)=='YFU']
        if 'neuron' in df: df=df.sort_values('neuron')
        r=df['mean_firing_rate_hz'].to_numpy(float)
        if len(r)!=62 or not np.all(np.isfinite(r)) or np.any(r<=0):
            raise ValueError('Expected 62 positive YFU rates in mean_firing_rate_hz')
        return r, 'real_YFU_rates'
    rng=np.random.default_rng(seed)
    r=np.clip(rng.lognormal(np.log(7.5),.85,62),.135,36.471)
    r*=9.0463/r.mean()
    return r, 'synthetic_fallback_rates_NOT_YFU'


def generate(model, rates, seed, strength):
    rng=np.random.default_rng(seed)
    z=geometry(model,rng)
    # Smooth temporal filters: 5 ms bins, 900 ms, calibrated neuron means.
    # Orthogonalization across stimulus classes is NOT performed after Poisson sampling.
    d=z.shape[1]; t=np.arange(180)*.005
    basis=rng.normal(size=(d,62,5))
    knots=np.linspace(0,.9,5)
    w=np.stack([np.interp(t,knots,basis[j,n]) for j in range(d) for n in range(62)])
    w=w.reshape(d,62,180)
    w/=np.maximum(np.max(np.abs(w),axis=2,keepdims=True),1e-12)
    effect=np.einsum('kd,dnt->knt',z,w)
    # Normalize maximum absolute modulation per neuron, preserving rank upper bound.
    effect/=np.maximum(np.max(np.abs(effect),axis=(0,2),keepdims=True),1e-12)
    lam=rates[None,:,None]*.005*(1+strength*effect)
    if np.any(lam<=0): raise ValueError('Nonpositive lambda; lower strength')
    y=np.repeat(np.arange(1,10),COUNTS)
    x=rng.poisson(lam[y-1])
    perm=rng.permutation(len(y))
    x=x[perm]; y=y[perm]
    return x,y,lam,z

def bin_data(x,ms):
    return x.reshape(x.shape[0],x.shape[1],900//ms,ms//5).sum(axis=-1).astype(float)

def cv_best(xbins,y,neuron,folds):
    best=(-1.,None,None)
    for ms in BINS:
        xn=xbins[ms][:,neuron,:]
        for gamma in GAMMAS:
            pred=np.empty(len(y),dtype=int)
            for tr,te in folds:
                mask=xn[tr].var(axis=0)>0
                if not mask.any():
                    pred[te]=1; continue
                lda=LinearDiscriminantAnalysis(solver='lsqr',shrinkage=gamma,priors=np.ones(9)/9)
                lda.fit(xn[tr][:,mask],y[tr]); pred[te]=lda.predict(xn[te][:,mask])
            acc=np.mean(pred==y)
            if acc>best[0]: best=(float(acc),ms,gamma)
    return best

def get_folds(y):
    return list(StratifiedKFold(n_splits=10,shuffle=True,random_state=42).split(np.zeros(len(y)),y))

def spectrum_metrics(c):
    c=c-c.mean(axis=0,keepdims=True)
    s=np.linalg.svd(c,compute_uv=False)
    v=s*s
    if v.sum()==0: return {'effective_rank':0.,'pc1_fraction':0.,'pc2_fraction':0.,'numerical_rank':0,'singular_values':s.tolist()}
    p=v/v.sum(); p=p[p>0]
    return {'effective_rank':float(np.exp(-np.sum(p*np.log(p)))),
            'pc1_fraction':float(v[0]/v.sum()),
            'pc2_fraction':float(v[:2].sum()/v.sum()),
            'numerical_rank':int(np.linalg.matrix_rank(c)),
            'singular_values':s.tolist()}



def centroid(x,y): return np.stack([x[y==k].mean(axis=0) for k in range(1,10)])

def fit_lda_features(xbins,y,selected):
    # Descriptive, in-sample geometry; use held-out geometry separately below.
    cols=[]
    for n,ms,gamma in selected:
        a=xbins[ms][:,n,:]
        mask=a.var(axis=0)>0
        if not mask.any(): continue
        a=a[:,mask]
        if a.shape[1]==1:
            cols.append((a-a.mean(axis=0))/max(a.std(axis=0)[0],1e-10))
            continue
        lda=LinearDiscriminantAnalysis(solver='eigen',shrinkage=gamma,priors=np.ones(9)/9)
        try:
            lda.fit(a,y)
            v=lda.transform(a)[:,:min(3,a.shape[1],8)]
        except (ValueError,np.linalg.LinAlgError): continue
        sd=v.std(axis=0); good=sd>1e-10
        if good.any(): cols.append((v[:,good]-v[:,good].mean(axis=0))/sd[good])
    return np.concatenate(cols,axis=1) if cols else np.empty((len(y),0))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',required=True,choices=MODELS)
    ap.add_argument('--rates-csv',default=None)
    ap.add_argument('--output',default='geometry_results')
    ap.add_argument('--permutations',type=int,default=99)
    ap.add_argument('--seed',type=int,default=2026)
    ap.add_argument('--strength',type=float,default=.6)
    ap.add_argument('--replicates',type=int,default=1)
    args=ap.parse_args()
    if not 0<=args.strength<1: ap.error('strength must be in [0,1)')
    if args.permutations<1: ap.error('permutations must be >=1')
    base=Path(args.output)/args.model; base.mkdir(parents=True,exist_ok=True)
    rates,rate_source=load_rates(args.rates_csv,args.seed)
    for rep in range(args.replicates):
        out=base/f'rep_{rep:03d}'; out.mkdir(exist_ok=True)
        rngseed=args.seed+100003*rep
        x,y,lam,z=generate(args.model,rates,rngseed,args.strength)
        xbins={ms:bin_data(x,ms) for ms in BINS}
        folds=get_folds(y)
        meta={'model':args.model,'replicate':rep,'seed':rngseed,'strength':args.strength,
              'rate_source':rate_source,'permutations':args.permutations,
              'ground_truth_affine_rank':int(np.linalg.matrix_rank(z,tol=1e-10)),
              'true_mean_centroid_metrics':spectrum_metrics(lam.reshape(9,-1))}
        (out/'metadata.json').write_text(json.dumps(meta,indent=2))
        path=out/'neuron_selection.csv'
        if path.exists():
            rows=pd.read_csv(path).to_dict('records')
            rows=[r for r in rows if int(r['permutations'])==args.permutations]
        else: rows=[]
        done={int(r['neuron']) for r in rows}
        for n in range(62):
            if n in done: continue
            start=time.time()
            obs,ms,gamma=cv_best(xbins,y,n,folds)
            # Distinct reproducible shuffle stream per neuron; re-optimize for every shuffle.
            rngp=np.random.default_rng(rngseed+1000000+n)
            null=[]
            for j in range(args.permutations):
                yp=rngp.permutation(y)
                a,_,_=cv_best(xbins,yp,n,get_folds(yp))
                null.append(a)
            p=(1+np.sum(np.asarray(null)>=obs))/(args.permutations+1)
            rows.append({'neuron':n,'best_bin_ms':ms,'best_gamma':gamma,
                         'observed_accuracy':obs,'null_mean':float(np.mean(null)),
                         'p_value':float(p),'selected':bool(p<=.125),
                         'permutations':args.permutations})
            pd.DataFrame(rows).sort_values('neuron').to_csv(path,index=False)
            print(f'{args.model} rep={rep} neuron={n+1}/62 p={p:.3f} selected={p<=.125} elapsed={time.time()-start:.1f}s',flush=True)
        df=pd.DataFrame(rows).sort_values('neuron')
        chosen=[(int(r.neuron),int(r.best_bin_ms),float(r.best_gamma)) for r in df.itertuples() if r.selected]
        # Full temporal raw centroid geometry, and selected supervised LDA geometry.
        raw=centroid(x.reshape(len(y),-1),y)
        lda_x=fit_lda_features(xbins,y,chosen)
        report={'metadata':meta,'n_selected':len(chosen),'selected_neurons':[q[0] for q in chosen],
                'raw_temporal_centroids':spectrum_metrics(raw),
                'lda_centroids_in_sample':spectrum_metrics(centroid(lda_x,y)) if lda_x.shape[1] else None,
                'lda_feature_count':int(lda_x.shape[1]),
                'warning':'LDA centroids are in-sample and may be inflated by supervised fitting and selection; not independent validation.'}
        (out/'summary.json').write_text(json.dumps(report,indent=2))
        print('FINISHED',args.model,'rep',rep,'selected',len(chosen),'effective rank',report['lda_centroids_in_sample'],flush=True)
if __name__=='__main__': main()
