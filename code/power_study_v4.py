"""Frozen discrepancy compression and power-based design. All observations synthetic."""
from extended_study_v3 import *
from scipy.stats import qmc
from scipy.linalg import cholesky, solve_triangular
import sys
V4=OUT/'v4';V4.mkdir(exist_ok=True)
CACHE=np.load(V3/'design_means.npz'); X=CACHE['candidate']; TRAIN=CACHE['train'];KS=np.array([1,2,3,4,6]); LEVEL=.05

def transform(x,U,p=3,C=None,A=None):
    if C is None:C=np.eye(len(x))
    L=cholesky(C,lower=True);W=solve_triangular(L,np.eye(len(x)),lower=True)
    if A is None:A=basis(x,p)
    Q=qr(W@A,mode='full')[0][:,A.shape[1]:]
    T=Q.T@W
    Z=np.einsum('pm,smj->spj',T,U).reshape(len(U),-1)
    return T,Z

def learn(x,U,p=3,C=None,A=None):
    T,Z=transform(x,U,p,C,A)
    unit=Z/np.maximum(np.linalg.norm(Z,axis=1,keepdims=True),1e-15)
    _,s,v=np.linalg.svd(unit,full_matrices=False)
    ks=KS[KS<=len(s)];ens=np.cumsum((Z@v.T)**2,axis=1)[:,ks-1].min(axis=0)
    pp=ncf.sf(f.ppf(.95,ks,len(x)*5),ks,len(x)*5,2*ens/(.02*SIGBASE)**2)
    k=int(ks[np.argmax(pp)])
    return T,v,s,k,float(max(pp))

def power(E,k,d,noise=.01):return ncf.sf(f.ppf(.95,k,d),k,d,np.maximum(2*np.asarray(E)/(noise*SIGBASE)**2,1e-14))

def design():
    old=json.loads((V3/'sensor_designs.json').read_text());ds={};tr=[]
    for m in [7,9,13,17]:
      uniform=np.array(old[f'{m}:uniform']['indices']);rob=np.array(old[f'{m}:robust-discrimination']['indices'])
      cache={}
      def score(sel):
        key=tuple(sel)
        if key not in cache:cache[key]=learn(X[sel],TRAIN[:,sel])[-1]
        return cache[key]
      starts=[uniform,rob,np.sort(np.random.default_rng(2026110504+m).choice(41,m,replace=False))]
      found=[exchange(score,start,41,tr,f'power_m{m}_start{i}',maxiter=12) for i,start in enumerate(starts)]
      best,val=max(found,key=lambda a:a[1])
      for name,sel in [('uniform',uniform),('energy',rob),('power',best)]:
        T,v,s,k,ss=learn(X[sel],TRAIN[:,sel]);ds[f'{m}:{name}']={'indices':sel.tolist(),'positions':X[sel].tolist(),'rank':k,'train_power_at_2pct':ss,'singular_values':s.tolist()}
      print('design',m,'rank',ds[f'{m}:power']['rank'],'score',val,flush=True)
    (V4/'frozen_designs.json').write_text(json.dumps(ds,indent=2));pd.DataFrame(tr).to_csv(V4/'design_trace.csv',index=False)

def holdout():
    ds=json.loads((V4/'frozen_designs.json').read_text());pars=[];means=[];ref=[]
    for domain,lo,hi in [('interpolation',[1.15,.35,-.35,0],[1.75,.65,.35,.3]),('stress',[1.75,.25,-.5,-.25],[1.97,.8,.5,.45])]:
      pts=qmc.scale(qmc.Sobol(4,scramble=True,seed=4431 if domain=='interpolation' else 4432).random_base2(8),lo,hi)
      for a,k,c2,c3 in pts:
        u=predict(a,k,X,TIMES,128,coeff=np.array([1,c2,c3]));means.append(u);pars.append(dict(domain=domain,alpha=a,kappa=k,c2=c2,c3=c3))
        if len(pars)%32==0:
          u96=predict(a,k,X,TIMES,96,coeff=np.array([1,c2,c3]));ref.append(dict(case=len(pars)-1,max_abs=float(np.max(abs(u-u96)))))
    # Original 12 cases are an explicitly separate legacy comparison, never used to train.
    for i,(a,k,c) in enumerate(itertools.product([1.25,1.6,1.85],[.45,.6],[[1,.22,.12],[1,-.25,.28]])):
      means.append(CACHE['holdout'][i]);pars.append(dict(domain='legacy',alpha=a,kappa=k,c2=c[1],c3=c[2]))
    U=np.array(means);pd.DataFrame(pars).to_csv(V4/'test_scenarios.csv',index=False);np.savez_compressed(V4/'test_means.npz',means=U)
    pd.DataFrame(ref).to_csv(V4/'reference_refinement.csv',index=False)
    rows=[];ab=[]
    for key,design in ds.items():
      m=int(key.split(':')[0]);name=key.split(':')[1];sel=design['indices'];x=X[sel];T,v,s,k,_=learn(x,TRAIN[:,sel]);_,Z=transform(x,U[:,sel]);energy_all=np.sum(Z*Z,axis=1)
      _,nominal=transform(x,predict(1.5,.5,x,TIMES,128)[None]);local=nominal[0]/np.linalg.norm(nominal[0]);k99=int(np.searchsorted(np.cumsum(s*s)/sum(s*s),.99)+1)
      options={'omnibus':(Z.shape[1],energy_all),'compressed':(k,np.sum((Z@v[:k].T)**2,axis=1)),'local-rank1':(1,(Z@local)**2),'POD99':(k99,np.sum((Z@v[:k99].T)**2,axis=1))}
      for meth,(rank,ee) in options.items():
       for noise in [.005,.01,.02,.03]:
        pp=power(ee,rank,m*5,noise)
        for i,val in enumerate(pp):rows.append(dict(design=name,m=m,method=meth,rank=rank,case=i,noise=noise,power=val,retained=ee[i]/max(energy_all[i],1e-30),**pars[i]))
      if m==9:
       for rank in [1,2,3,4,6,10,20]:
        ee=np.sum((Z@v[:rank].T)**2,axis=1)
        for i,val in enumerate(power(ee,rank,m*5)):ab.append(dict(design=name,rank=rank,case=i,power=val,retained=ee[i]/max(energy_all[i],1e-30),**pars[i]))
    pd.DataFrame(rows).to_csv(V4/'holdout.csv',index=False);pd.DataFrame(ab).to_csv(V4/'rank_ablation.csv',index=False)
    print(pd.DataFrame(rows).query('m==9 and noise==.01').groupby(['domain','design','method']).power.agg(['mean','min']).round(4),flush=True)

def calibration():
    ds=json.loads((V4/'frozen_designs.json').read_text());rows=[];runs=[]
    for name in ['uniform','energy','power']:
      sel=ds['9:'+name]['indices'];x=X[sel];c=np.array([1,.22,.12]);T,v,sv,k,_=learn(x,TRAIN[:,sel]);rankfull=30
      means={'spectral':spectral(1.85,.45,x,TIMES,c),'restricted':predict(1.85,.45,x,TIMES,128,coeff=c)}
      rng=np.random.default_rng(5112026);e1=rng.normal(size=(5000,9,5))*SIG;e2=rng.normal(size=(5000,9,5))*SIG;var=np.sum((e1-e2)**2,axis=(1,2))/90
      for truth,u in means.items():
       avg=u[None]+(e1+e2)/2;z=np.einsum('pm,rmj->rpj',T,avg).reshape(5000,-1)
       for method,rank,energy0 in [('omnibus',30,np.sum(z*z,axis=1)),('compressed',k,np.sum((z@v[:k].T)**2,axis=1))]:
        stat=2*energy0/(rank*var);rejected=stat>f.ppf(.95,rank,45);lo,hi=wilson(rejected.sum(),len(rejected));rows.append(dict(design=name,truth=truth,method=method,n=5000,rate=rejected.mean(),low=lo,high=hi))
       # Nonlinear residual on the first 300 of exactly the same datasets.
       for i in range(300):
        pp,cc,rss,ok=varpro(x,avg[i]);runs.append(dict(design=name,truth=truth,rep=i,success=ok,alpha_hat=pp[0],kappa_hat=pp[1],residual_reject=2*rss/(40*var[i])>f.ppf(.95,40,45),compressed_reject=2*np.sum((z[i]@v[:k].T)**2)/(k*var[i])>f.ppf(.95,k,45),omnibus_reject=2*np.sum(z[i]**2)/(30*var[i])>f.ppf(.95,30,45)))
    pd.DataFrame(rows).to_csv(V4/'calibration.csv',index=False);pd.DataFrame(runs).to_csv(V4/'paired_residual.csv',index=False)

def robustness():
    ds=json.loads((V4/'frozen_designs.json').read_text());rows=[];support=[]
    sel=ds['9:power']['indices'];x=X[sel];rng=np.random.default_rng(6412026)
    for distribution in ['Gaussian','Student-t4']:
     for ell in [0,.05,.15,.35]:
      C=np.eye(9) if ell==0 else .3*np.eye(9)+.7*np.exp(-abs(x[:,None]-x[None,:])/ell);L=cholesky(C,lower=True)
      T,v,s,k,_=learn(x,TRAIN[:,sel],C=C)
      e=rng.normal(size=(2,3000,9,5)) if distribution=='Gaussian' else rng.standard_t(4,size=(2,3000,9,5))/np.sqrt(2)
      e=np.einsum('ij,brjt->brit',L,e)*SIG;W=np.linalg.inv(L);diff=np.einsum('ij,rjt->rit',W,e[0]-e[1]);var=np.sum(diff*diff,axis=(1,2))/90
      for truth,u in [('spectral',spectral(1.85,.45,x,TIMES,np.array([1,.22,.12]))),('restricted',predict(1.85,.45,x,TIMES,128,coeff=np.array([1,.22,.12])))]:
       z=np.einsum('pm,rmj->rpj',T,u[None]+(e[0]+e[1])/2).reshape(3000,-1)
       for meth,rank,energy0 in [('omnibus',30,np.sum(z*z,axis=1)),('compressed',k,np.sum((z@v[:k].T)**2,axis=1))]:
        rej=2*energy0/(rank*var)>f.ppf(.95,rank,45);lo,hi=wilson(rej.sum(),len(rej));rows.append(dict(noise=distribution,length=ell,truth=truth,method=meth,n=3000,rate=rej.mean(),low=lo,high=hi))
    pd.DataFrame(rows).to_csv(V4/'noise_stress.csv',index=False)
    sel=ds['17:power']['indices'];x=X[sel];rng=np.random.default_rng(7412026);e=rng.normal(size=(2,1000,17,5))*SIG;var=np.sum((e[0]-e[1])**2,axis=(1,2))/170
    for p in [3,5,7]:
      T,v,s,k,_=learn(x,TRAIN[:,sel],p=p)
      for delta in [0,.04,.08,.16]:
       c=np.array([1,.22,.12,0,delta]);freq=np.arange(1,6)*np.pi/2;spec=(np.sin(np.outer(x+1,freq))*c)@np.exp(-.45*freq[:,None]**1.85*TIMES)
       for truth,u in [('spectral',spec),('restricted',predict(1.85,.45,x,TIMES,128,coeff=c))]:
        z=np.einsum('pm,rmj->rpj',T,u[None]+(e[0]+e[1])/2).reshape(1000,-1)
        for meth,rank,energy0 in [('omnibus',z.shape[1],np.sum(z*z,axis=1)),('compressed',k,np.sum((z@v[:k].T)**2,axis=1))]:
         rej=2*energy0/(rank*var)>f.ppf(.95,rank,85);lo,hi=wilson(rej.sum(),len(rej));support.append(dict(support=p,delta=delta,truth=truth,method=meth,rank=rank,n=1000,rate=rej.mean(),low=lo,high=hi))
    pd.DataFrame(support).to_csv(V4/'support_stress.csv',index=False)

def twod4():
    old=json.loads((V3/'sensor_designs_2d.json').read_text());rows=[];spectra=[]
    pts=qmc.scale(qmc.Sobol(2,scramble=True,seed=8412026).random_base2(7),[1.1,1.1],[1.97,1.97]);modes=np.array([[1,1],[2,1],[1,2],[3,3]])
    for name in ['uniform','robust']:
      xy=np.array(old[name]);A=np.prod(np.sin((xy[:,None,:]+1)*modes[None,:,:]*np.pi/2),axis=2)
      train=np.array([product2d(xy,TIMES,ax=a,ay=b,n=96) for a,b in itertools.product([1.2,1.5,1.75],repeat=2)])
      T,v,s,k,score=learn(xy[:,0],train,A=A);spectra.append(dict(design=name,rank=k,singular_values=s.tolist()))
      for i,(a,b) in enumerate(pts):
       U=product2d(xy,TIMES,ax=a,ay=b,n=128);Z=(T@U).ravel()
       for method,rank,e in [('omnibus',25,sum(Z*Z)),('compressed',k,sum((Z@v[:k].T)**2))]:
        for noise in [.005,.01,.02,.03]:rows.append(dict(design=name,method=method,rank=rank,case=i,ax=a,ay=b,noise=noise,power=float(power(e,rank,45,noise))))
    pd.DataFrame(rows).to_csv(V4/'twod_holdout.csv',index=False);(V4/'twod_ranks.json').write_text(json.dumps(spectra,indent=2))

if __name__=='__main__':
    part=sys.argv[1];start=time.perf_counter();globals()[part]()
    (V4/(part+'_runtime.json')).write_text(json.dumps(dict(seconds=time.perf_counter()-start,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),seed_rules='Explicit constants in the source; no observed test outcomes select designs.'),indent=2))
