"""Mechanism, finite-budget design, nuisance fitting and 2D directional extension.
All design choices are deterministic; held-out scenarios/noise never select designs.
"""
from independent_galerkin import *
from scipy.linalg import qr
from scipy.stats import f, ncf
from scipy.optimize import least_squares
import itertools
V3=OUT/'v3'; V3.mkdir(exist_ok=True)
TIMES=np.array([.02,.05,.1,.2,.4]); SIG=.011615502616

def basis(x,p=3): return np.sin(np.outer(np.asarray(x)+1,np.arange(1,p+1)*np.pi/2))
def energy(U,A):
    q=qr(A,mode='full')[0][:,A.shape[1]:]
    return np.sum(np.einsum('mp,smj->spj',q,U)**2,axis=(1,2))
def powfun(e,m,j=5,p=3,sigma=SIG):
    nc=2*np.asarray(e)/sigma**2;df=(m-p)*j;d=m*j
    return ncf.sf(f.ppf(.95,df,d),df,d,np.maximum(nc,1e-12))
def wilson(k,n):
    p=k/n;z=1.95996398454;den=1+z*z/n;c=(p+z*z/(2*n))/den;v=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return c-v,c+v

def exchange(score,start,count,trace,name,maxiter=20):
    sel=np.sort(start);value=score(sel);calls=1
    for it in range(maxiter):
        best=value;chosen=sel
        rest=np.setdiff1d(np.arange(count),sel)
        for j in range(len(sel)):
            for q in rest:
                cand=sel.copy();cand[j]=q;cand.sort();v=score(cand);calls+=1
                if v>best+1e-12:best=v;chosen=cand
        trace.append(dict(method=name,iteration=it,score=value,evaluations=calls))
        if best<=value+1e-12:break
        sel,value=chosen,best
    return sel,value

def aperture():
    rows=[];z=np.linspace(-1,1,31)
    for a in [1.2,1.5,1.8]:
      for parity,c in [('mixed',[1,.3,.2]),('even',[1,0,.2])]:
       for radius in np.geomspace(.025,.95,65):
        x=radius*z;u=predict(a,.5,x,TIMES,128,coeff=np.array(c))
        e=energy(u[None],basis(x))[0]
        rows.append(dict(alpha=a,parity=parity,radius=radius,energy=e,noncentrality=2*e/SIG**2,power=powfun(e,31)))
    df=pd.DataFrame(rows);df.to_csv(V3/'aperture.csv',index=False)
    slopes=[]
    for (a,p),g in df.groupby(['alpha','parity']):
        small=g[g.radius<=.12];slope=np.polyfit(np.log(small.radius),np.log(small.energy),1)[0]
        slopes.append(dict(alpha=a,parity=p,measured_power=slope,predicted_power=6 if p=='mixed' else 8))
    pd.DataFrame(slopes).to_csv(V3/'aperture_slopes.csv',index=False)
    x=np.linspace(-.98,.98,241);times=np.linspace(.01,.8,101)
    r=predict(1.5,.5,x,times,128);s=spectral(1.5,.5,x,times)
    np.savez_compressed(V3/'fields_1d.npz',x=x,t=times,restricted=r,spectral=s,difference=r-s)
    print('Aperture',slopes,flush=True)

def sensor_design():
    x=np.linspace(-.95,.95,41);A=basis(x);train=[];meta=[]
    for a,k,c2,c3 in itertools.product([1.15,1.45,1.75],[.35,.65],[-.35,.35],[0,.3]):
        c=np.array([1,c2,c3]);train.append(predict(a,k,x,TIMES,96,coeff=c));meta.append(dict(alpha=a,kappa=k,c2=c2,c3=c3))
    U=np.array(train);norm=energy(U,A);assert np.min(norm)>1e-12
    pd.DataFrame(meta).to_csv(V3/'design_training_scenarios.csv',index=False)
    local=predict(1.5,.5,x,TIMES,96)[None];lnorm=energy(local,A)[0]
    freq=np.arange(1,4)*np.pi/2;z=freq[:,None]**1.5*TIMES;ex=np.exp(-.5*z);B=A*COEF
    ja=B@(-.5*z*np.log(freq[:,None])*ex);jk=B@(-z*ex)
    J=np.stack([ja,jk],axis=-1)
    def score(s):return float(np.min(energy(U[:,s],A[s])/norm))
    def loc(s):return float(energy(local[:,s],A[s])[0]/lnorm)
    def dop(s):return float(np.linalg.slogdet(np.einsum('mti,mtj->ij',J[s],J[s])+1e-14*np.eye(2))[1])
    trace=[];designs={};rng=np.random.default_rng(2026110501)
    for m in [7,9,13,17]:
        uniform=np.round(np.linspace(0,40,m)).astype(int)
        central=np.round(np.linspace(13,27,m)).astype(int) if m<=15 else np.arange(12,29)
        if len(np.unique(central))<m: central=np.arange((41-m)//2,(41+m)//2)
        cheb=np.unique(np.round((np.cos(np.linspace(np.pi,0,m))+1)*20).astype(int))
        while len(cheb)<m:cheb=np.sort(np.r_[cheb,np.setdiff1d(uniform,cheb)[0]])
        starts=[uniform]+[np.sort(rng.choice(41,m,replace=False)) for _ in range(3)]
        found=[]
        for si,start in enumerate(starts):found.append(exchange(score,start,41,trace,f'robust_m{m}_start{si}'))
        rob,rv=max(found,key=lambda a:a[1]);lo,_=exchange(loc,uniform,41,trace,f'local_m{m}');dd,_=exchange(dop,uniform,41,trace,f'Dopt_m{m}')
        for name,sel in [('uniform',uniform),('central',central),('Chebyshev',cheb),('D-optimal',dd),('local-discrimination',lo),('robust-discrimination',rob)]:designs[(m,name)]=sel.tolist()
        print('design',m,'robust',rv,'uniform',score(uniform),'x',x[rob],flush=True)
    pd.DataFrame(trace).to_csv(V3/'exchange_trace.csv',index=False)
    (V3/'sensor_designs.json').write_text(json.dumps({str(m)+':'+n:dict(indices=s,positions=x[s].tolist()) for (m,n),s in designs.items()},indent=2))
    # Save frozen design definitions before evaluating held-out cases.
    hold=[];hmeta=[]
    for a,k,c in itertools.product([1.25,1.6,1.85],[.45,.6],[[1,.22,.12],[1,-.25,.28]]):
        hold.append(predict(a,k,x,TIMES,128,coeff=np.array(c)));hmeta.append(dict(alpha=a,kappa=k,c2=c[1],c3=c[2]))
    H=np.array(hold);hnorm=energy(H,A);out=[];rand=[]
    for (m,name),sel in designs.items():
      ens=energy(H[:,sel],A[sel])
      for noise in [.005,.01,.02,.03]:
       for i,ee in enumerate(ens):out.append(dict(m=m,method=name,case=i,noise=noise,energy=ee,efficiency=ee/hnorm[i],power=powfun(ee,m,sigma=noise*SIGBASE),**hmeta[i]))
    for m in [7,9,13,17]:
      for seed in range(200):
       sel=np.sort(np.random.default_rng(3000000+m*1000+seed).choice(41,m,replace=False));ens=energy(H[:,sel],A[sel])
       for i,ee in enumerate(ens):rand.append(dict(m=m,random_design=seed,case=i,power=powfun(ee,m),efficiency=ee/hnorm[i]))
    pd.DataFrame(out).to_csv(V3/'design_holdout.csv',index=False);pd.DataFrame(rand).to_csv(V3/'random_design_holdout.csv',index=False)
    np.savez_compressed(V3/'design_means.npz',candidate=x,train=U,holdout=H)
    # Independently evaluate reference-basis change on all held-out candidate blocks.
    refinement=[]
    for i,h in enumerate(hmeta):
      v=predict(h['alpha'],h['kappa'],x,TIMES,96,coeff=np.array([1,h['c2'],h['c3']]))
      refinement.append(dict(case=i,max_abs=float(abs(v-H[i]).max())))
    pd.DataFrame(refinement).to_csv(V3/'design_reference_refinement.csv',index=False)

def varpro(x,y,p=3):
    freq=np.arange(1,p+1)*np.pi/2;A=basis(x,p)
    def solve(pv):
      D=(A[:,:,None]*np.exp(-pv[1]*freq[None,:,None]**pv[0]*TIMES)).transpose(0,2,1).reshape(-1,p)
      c=np.linalg.lstsq(D,y.ravel(),rcond=None)[0];return D@c-y.ravel(),c
    fits=[least_squares(lambda z:solve(z)[0],start,bounds=([.8,.1],[1.95,1]),max_nfev=70,ftol=1e-8,xtol=1e-8,gtol=1e-8) for start in [[1.1,.4],[1.7,.65]]]
    res=min(fits,key=lambda r:sum(r.fun**2));return res.x,solve(res.x)[1],float(sum(res.fun**2)),bool(res.success)

def nuisance():
    designs=json.loads((V3/'sensor_designs.json').read_text());rows=[]
    # Train-only designs, test noise independent from selection. Unknown initial amplitudes.
    for name in ['uniform','central','D-optimal','local-discrimination','robust-discrimination']:
      x=np.array(designs['9:'+name]['positions']);A=basis(x);Q=qr(A,mode='full')[0][:,3:];d=45;q=30
      for a in [1.6,1.85]:
       c=np.array([1,.22,.12]);meanR=predict(a,.45,x,TIMES,128,coeff=c);meanS=spectral(a,.45,x,TIMES,c)
       for rep in range(200):
        seed=41000000+int(a*100)*1000+rep;rng=np.random.default_rng(seed);e1=rng.normal(size=(9,5))*SIG;e2=rng.normal(size=(9,5))*SIG
        var=np.sum(((e1-e2)/np.sqrt(2))**2)/d
        for truth,u in [('spectral',meanS),('restricted',meanR)]:
         avg=u+(e1+e2)/2;p,cc,rss,ok=varpro(x,avg)
         ps=2*np.sum((Q.T@avg)**2)/(q*var);rs=2*rss/((d-5)*var)
         rows.append(dict(design=name,alpha=a,rep=rep,seed=seed,truth=truth,projection_F=ps,residual_F=rs,projection_reject=ps>f.ppf(.95,q,d),residual_reject=rs>f.ppf(.95,d-5,d),alpha_hat=p[0],kappa_hat=p[1],c1_hat=cc[0],c2_hat=cc[1],c3_hat=cc[2],success=ok))
       print('unknown-amplitude',name,a,flush=True)
    pd.DataFrame(rows).to_csv(V3/'nuisance_runs.csv',index=False)
    summ=[]
    for keys,g in pd.DataFrame(rows).groupby(['design','alpha','truth']):
     for method in ['projection','residual']:
      rr=g[method+'_reject'];lo,hi=wilson(rr.sum(),len(rr));summ.append(dict(design=keys[0],alpha=keys[1],truth=keys[2],method=method,n=len(rr),rate=rr.mean(),low=lo,high=hi))
    pd.DataFrame(summ).to_csv(V3/'nuisance_summary.csv',index=False)

def product2d(coords,t,ax=1.3,ay=1.7,kx=.5,ky=.35,n=96,model='restricted',fdn=None):
    modes=[(1,1),(2,1),(1,2),(3,3)];coef=[1,.2,.15,.1];out=np.zeros((len(coords),len(t)))
    def one(a,k,x,m):
      if model=='spectral':return np.sin(m*np.pi*(x[:,None]+1)/2)*np.exp(-k*(m*np.pi/2)**a*np.asarray(t))
      if fdn:
       xx,lam,v,_=eig(fdn,a);init=np.sin(m*np.pi*(xx+1)/2);proj=project(xx,v,v.T@init,x);return proj@np.exp(-k*lam[:,None]*np.asarray(t))
      c=np.zeros(m);c[-1]=1;return predict(a,k,x,t,n,coeff=c)
    for c,(mx,my) in zip(coef,modes):out+=c*one(ax,kx,coords[:,0],mx)*one(ay,ky,coords[:,1],my)
    return out

def twod():
    grid=np.linspace(-.96,.96,81);xx,yy=np.meshgrid(grid,grid);pts=np.c_[xx.ravel(),yy.ravel()];tt=np.array([.05,.2,.6])
    R=product2d(pts,tt,n=128);S=product2d(pts,tt,model='spectral')
    np.savez_compressed(V3/'fields_2d.npz',x=grid,t=tt,restricted=R.reshape(81,81,3),spectral=S.reshape(81,81,3))
    coords=np.array(list(itertools.product(np.linspace(-.9,.9,9),repeat=2)));modes=np.array([[1,1],[2,1],[1,2],[3,3]])
    A=np.prod(np.sin((coords[:,None,:]+1)*modes[None,:,:]*np.pi/2),axis=2)
    pars=[(1.2,1.5),(1.5,1.2),(1.7,1.7)];U=np.array([product2d(coords,TIMES,ax=a,ay=b) for a,b in pars]);norm=energy(U,A)
    uniform=np.array([i for i,(a,b) in enumerate(coords) if np.isclose(a,[-.9,0,.9]).any() and np.isclose(b,[-.9,0,.9]).any()])
    # Avoid the mode-3 aliasing of exact equispaced triples through candidate choices.
    uniform=np.array([i*9+j for i in [1,4,7] for j in [1,4,7]])
    central=np.array([i*9+j for i in [3,4,5] for j in [3,4,5]])
    def score(s):
     if np.linalg.matrix_rank(A[s])<4:return -1.
     return float(np.min(energy(U[:,s],A[s])/norm))
    tr=[];rob,_=exchange(score,uniform,81,tr,'robust_2d',maxiter=15);pd.DataFrame(tr).to_csv(V3/'exchange_2d.csv',index=False)
    (V3/'sensor_designs_2d.json').write_text(json.dumps({name:coords[sel].tolist() for name,sel in [('uniform',uniform),('central',central),('robust',rob)]},indent=2))
    rows=[];checks=[]
    for a,b in [(1.3,1.7),(1.6,1.4),(1.85,1.85)]:
      ref=product2d(coords,TIMES,ax=a,ay=b,n=128);g96=product2d(coords,TIMES,ax=a,ay=b,n=96)
      checks.append(dict(ax=a,ay=b,method='Galerkin96',max_abs=np.max(abs(ref-g96)),rmse=np.sqrt(np.mean((ref-g96)**2))))
      for n in [127,255,511]:
       fd=product2d(coords,TIMES,ax=a,ay=b,fdn=n);checks.append(dict(ax=a,ay=b,method='FD'+str(n),max_abs=np.max(abs(ref-fd)),rmse=np.sqrt(np.mean((ref-fd)**2))))
      for name,sel in [('uniform',uniform),('central',central),('robust',rob)]:
       ee=energy(ref[None,sel],A[sel])[0]
       for noise in [.005,.01,.02,.03]:rows.append(dict(ax=a,ay=b,design=name,noise=noise,power=powfun(ee,9,p=4,sigma=noise*1.2),energy=ee))
    pd.DataFrame(rows).to_csv(V3/'twod_holdout.csv',index=False);pd.DataFrame(checks).to_csv(V3/'twod_refinement.csv',index=False)
    print('2D complete',flush=True)

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('part',choices=['aperture','design','nuisance','twod']);args=p.parse_args();ts=time.perf_counter()
 {'aperture':aperture,'design':sensor_design,'nuisance':nuisance,'twod':twod}[args.part]()
 (V3/(args.part+'_runtime.json')).write_text(json.dumps({'seconds':time.perf_counter()-ts,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},indent=2))
