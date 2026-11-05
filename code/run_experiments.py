"""Reproducible synthetic study; no measured data, no new discretization claim."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
import json, time, platform, hashlib, sys
import numpy as np
import pandas as pd
import scipy
from scipy.linalg import toeplitz, eigh, solve
from scipy.special import gammaln, gamma
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits
threadpool_limits(limits=1)
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'code/outputs'; OUT.mkdir(exist_ok=True)
T=np.array([.02,.05,.1,.2,.4]); TT=np.array([.55,.75,1.])
XW=np.linspace(-.875,.875,15); XC=np.linspace(-.35,.35,15); XT=np.linspace(-.9375,.9375,31)
BLOCKS=[(XW,T),(XC,T),(XT,TT)]; CUT=[0,75,150,243]
COEF=np.array([1.,.3,.2]); MODES=np.arange(1,4); FREQ=MODES*np.pi/2
AG=np.linspace(.8,1.95,231); KG=np.linspace(.1,1.,181)
SIGBASE=float(np.max(np.sin(np.outer(np.linspace(-1,1,10001)+1,FREQ))@COEF))

def spectral(a,k,x,t,c=COEF):
    return (np.sin(np.outer(x+1,FREQ))*c)@np.exp(-k*FREQ[:,None]**a*t)
def initial(x): return np.sin(np.outer(x+1,FREQ))@COEF

def matrix(n,a):
    h=2/(n+1); g=np.empty(n); g[0]=np.exp(gammaln(a+1)-2*gammaln(a/2+1))
    for j in range(1,n): g[j]=g[j-1]*(j-1-a/2)/(j+a/2)
    return toeplitz(g)*h**(-a)
def eig(n,a):
    x=np.linspace(-1,1,n+2)[1:-1]; w,v=eigh(matrix(n,a),check_finite=False)
    return x,w,v,v.T@initial(x)
def project(x,v,c,obs):
    # Piecewise-linear interpolation in physical x; exact zero exterior endpoints.
    fullx=np.r_[-1,x,1]; fullv=np.vstack([np.zeros(v.shape[1]),v,np.zeros(v.shape[1])])
    idx=np.clip(np.searchsorted(fullx,obs)-1,0,len(fullx)-2)
    q=(obs-fullx[idx])/(fullx[idx+1]-fullx[idx])
    return ((1-q[:,None])*fullv[idx]+q[:,None]*fullv[idx+1])*c

def direct(n,a,k,blocks=BLOCKS):
    x,w,v,c=eig(n,a)
    return np.concatenate([(project(x,v,c,xx)@np.exp(-k*w[:,None]*tt)).ravel() for xx,tt in blocks])
def savecsv(name,rows):
    df=pd.DataFrame(rows); df.to_csv(OUT/name,index=False); return df

def forward_validation():
    rows=[]; checks=[]; grids=[]; profiles=[]; truths={}
    for a in [1.2,1.5,1.8]:
        rhs=2**a*gamma(1+a/2)*gamma((1+a)/2)/np.sqrt(np.pi)
        for n in [31,63,127,255,511,1023]:
            h=2/(n+1); x=np.linspace(-1,1,n+2)[1:-1]
            exact=(1-x*x)**(a/2); sol=solve(matrix(n,a),np.full(n,rhs),assume_a='pos')
            rows.append(dict(alpha=a,n=n,h=h,l2=np.sqrt(h*np.sum((sol-exact)**2)),linf=np.max(abs(sol-exact))))
        x,w,v,c=eig(1023,a); times=np.r_[0,T,TT]
        u=v@(np.exp(-.5*w[:,None]*times)*c[:,None]); h=2/1024
        for j,t in enumerate(times):
            checks.append(dict(alpha=a,t=t,min_u=u[:,j].min(),mass=h*u[:,j].sum(),energy=h*np.sum(u[:,j]**2),min_eigenvalue=w[0]))
        truth=np.concatenate([(project(x,v,c,xx)@np.exp(-.5*w[:,None]*tt)).ravel() for xx,tt in BLOCKS]); truths[a]=truth
        for n in [127,255,511]:
            pred=direct(n,a,.5)
            for b,name in enumerate(['wide','central','test']):
                z=slice(CUT[b],CUT[b+1]); d=pred[z]-truth[z]
                grids.append(dict(alpha=a,n=n,block=name,rmse=np.sqrt(np.mean(d*d)),max_abs=np.max(abs(d))))
        for j,xx in enumerate(XT):
            for it,t in enumerate(TT):
                profiles.append(dict(alpha=a,x=xx,t=t,restricted=truth[150:].reshape(31,3)[j,it],spectral=spectral(a,.5,XT,TT)[j,it]))
    savecsv('convergence.csv',rows);savecsv('physics_checks.csv',checks);savecsv('grid_checks.csv',grids);savecsv('profiles.csv',profiles)
    np.savez_compressed(OUT/'truth.npz',**{str(a):v for a,v in truths.items()})
    return truths

def build_surrogate():
    file=OUT/'restricted_surrogate.npz'
    if file.exists():
        z=np.load(file); values=z['values']; assert np.array_equal(z['alpha'],AG) and np.array_equal(z['kappa'],KG)
    else:
        values=np.empty((len(AG),len(KG),243))
        for i,a in enumerate(AG):
            x,w,v,c=eig(255,a); arr=[]
            for xx,tt in BLOCKS:
                p=project(x,v,c,xx)
                expo=np.exp(-KG[:,None,None]*w[None,:,None]*tt[None,None,:])
                arr.append(np.einsum('xm,kmt->kxt',p,expo,optimize=True).reshape(len(KG),-1))
            values[i]=np.concatenate(arr,axis=1)
            if i%40==0: print('surrogate',i,len(AG),flush=True)
        np.savez_compressed(file,alpha=AG,kappa=KG,values=values)
    return RegularGridInterpolator((AG,KG),values,bounds_error=True)

def inverse(truths,sur):
    rows=[]; obs=[]; noiseless=[]
    for a in [1.2,1.5,1.8]:
        for layout,b in [('wide',0),('central',1)]:
            x,t=BLOCKS[b]; z=slice(CUT[b],CUT[b+1]); true=truths[a][z]
            funcs={'restricted':lambda p,z=z:sur(np.asarray(p)[None,:])[0,z],
                   'spectral':lambda p,x=x,t=t:spectral(*p,x,t).ravel()}
            for model,fun in funcs.items():
                fit=fit_model(fun,true)
                noiseless.append(dict(alpha=a,layout=layout,model=model,alpha_hat=fit.x[0],kappa_hat=fit.x[1],rmse=np.sqrt(np.mean(fit.fun**2))))
            for noise in [.002,.01,.03]:
                for rep in range(50):
                    seed=20261105+int(a*10)*100000+int(noise*1000)*100+rep
                    eps=np.random.default_rng(seed).standard_normal(75)*SIGBASE*noise
                    y=true+eps
                    obs.append(dict(alpha=a,layout=layout,noise=noise,rep=rep,seed=seed,y=y))
                    for model,fun in funcs.items():
                        fit=fit_model(fun,y); ah,kh=fit.x
                        pred=sur(fit.x[None,:])[0,150:] if model=='restricted' else spectral(ah,kh,XT,TT).ravel()
                        j=fit.jac; cov=(SIGBASE*noise)**2*np.linalg.pinv(j.T@j)
                        se=np.sqrt(np.maximum(np.diag(cov),0))
                        rows.append(dict(truth='restricted',alpha=a,kappa=.5,layout=layout,noise=noise,rep=rep,seed=seed,model=model,alpha_hat=ah,kappa_hat=kh,alpha_se=se[0],alpha_cover=abs(ah-a)<=1.96*se[0],kappa_cover=abs(kh-.5)<=1.96*se[1],train_rmse=np.sqrt(np.mean(fit.fun**2)),test_rel_l2=np.linalg.norm(pred-truths[a][150:])/np.linalg.norm(truths[a][150:]),success=fit.success,nfev=fit.nfev,boundary=(min(ah-.8,1.95-ah,kh-.1,1.-kh)<1e-5)))
                print('inverse',a,layout,noise,flush=True)
        # exact spectral data / spectral fit control, same wide observations
        for noise in [.002,.01,.03]:
            for rep in range(50):
                seed=40261105+int(a*10)*100000+int(noise*1000)*100+rep
                y=spectral(a,.5,XW,T).ravel()+np.random.default_rng(seed).standard_normal(75)*SIGBASE*noise
                fun=lambda p:spectral(*p,XW,T).ravel(); fit=fit_model(fun,y); ah,kh=fit.x
                se=np.sqrt(np.diag((SIGBASE*noise)**2*np.linalg.pinv(fit.jac.T@fit.jac)))
                pred=spectral(ah,kh,XT,TT).ravel(); target=spectral(a,.5,XT,TT).ravel()
                rows.append(dict(truth='spectral',alpha=a,kappa=.5,layout='wide',noise=noise,rep=rep,seed=seed,model='spectral',alpha_hat=ah,kappa_hat=kh,alpha_se=se[0],alpha_cover=abs(ah-a)<=1.96*se[0],kappa_cover=abs(kh-.5)<=1.96*se[1],train_rmse=np.sqrt(np.mean(fit.fun**2)),test_rel_l2=np.linalg.norm(pred-target)/np.linalg.norm(target),success=fit.success,nfev=fit.nfev,boundary=(min(ah-.8,1.95-ah,kh-.1,1.-kh)<1e-5)))
    df=savecsv('inverse_runs.csv',rows); savecsv('noiseless_fits.csv',noiseless)
    np.savez_compressed(OUT/'observations.npz',y=np.stack([o['y'] for o in obs]))
    savecsv('observation_index.csv',[{k:v for k,v in o.items() if k!='y'} for o in obs])
    group=['truth','alpha','layout','noise','model']
    df['alpha_error']=df.alpha_hat-df.alpha;df['kappa_error']=df.kappa_hat-.5
    summary=df.groupby(group).agg(n=('rep','count'),alpha_mean=('alpha_hat','mean'),alpha_sd=('alpha_hat','std'),alpha_bias=('alpha_error','mean'),kappa_mean=('kappa_hat','mean'),alpha_coverage=('alpha_cover','mean'),kappa_coverage=('kappa_cover','mean'),train_rmse=('train_rmse','mean'),test_rel_l2=('test_rel_l2','mean'),boundary_count=('boundary','sum'),failure_count=('success',lambda x:int((~x).sum()))).reset_index()
    summary.to_csv(OUT/'inverse_summary.csv',index=False)
    return df

def fit_model(fun,y):
    fits=[least_squares(lambda p:fun(p)-y,start,bounds=([.8,.1],[1.95,1.]),xtol=1e-10,ftol=1e-10,gtol=1e-10,max_nfev=150) for start in [[1.,.35],[1.5,.5],[1.85,.7]]]
    return min(fits,key=lambda f:np.sum(f.fun**2))

def extra_checks(sur,truths):
    rng=np.random.default_rng(9081); rows=[]
    for i in range(30):
        a=rng.uniform(.82,1.93); k=rng.uniform(.12,.98)
        exact=direct(255,a,k); pred=sur([[a,k]])[0]; d=pred-exact
        rows.append(dict(alpha=a,kappa=k,rmse=np.sqrt(np.mean(d*d)),max_abs=np.max(abs(d)),relative_to_min_sigma=np.max(abs(d))/(.002*SIGBASE)))
    savecsv('surrogate_validation.csv',rows)
    refine=[]
    for a in [1.2,1.5,1.8]:
        y=truths[a][:75]
        for n in [127,255,511]:
            # Direct eigensolve fitting: no parameter interpolation.
            fit=least_squares(lambda p:direct(n,*p,blocks=[BLOCKS[0]])-y,[a,.5],bounds=([.8,.1],[1.95,1.]),diff_step=1e-4,gtol=1e-10,ftol=1e-10,xtol=1e-10,max_nfev=40)
            refine.append(dict(alpha=a,n=n,alpha_hat=fit.x[0],kappa_hat=fit.x[1],rmse=np.sqrt(np.mean(fit.fun**2)),success=fit.success))
    savecsv('inverse_grid_checks.csv',refine)
    # Classical endpoint matrix and eigenbasis consistency.
    n=63; h=2/(n+1); expected=toeplitz(np.r_[2.,-1.,np.zeros(n-2)])/h**2
    endpoint=float(np.max(abs(matrix(n,2)-expected)))
    modes=[]; profiles=[]
    for name,c in [('single',np.array([1.,0,0])),('multiple',COEF)]:
        u=spectral(1.5,.5,XW,T,c).ravel()
        def f(p): return spectral(*p,XW,T,c).ravel()
        da=(f([1.5+1e-5,.5])-f([1.5-1e-5,.5]))/2e-5
        dk=(f([1.5,.5+1e-5])-f([1.5,.5-1e-5]))/2e-5
        s=np.linalg.svd(np.column_stack([da,dk]),compute_uv=False)
        modes.append(dict(initial=name,smax=s[0],smin=s[1],condition=s[0]/s[1]))
        for a in np.linspace(.8,1.95,200):
            k=.5*(np.pi/2)**(1.5-a)
            profiles.append(dict(initial=name,alpha=a,kappa=k,rmse=np.sqrt(np.mean((f([a,k])-u)**2))))
    savecsv('identifiability.csv',modes);savecsv('identifiability_profile.csv',profiles)
    # Independently evaluate the constant integral identity at x=0 by quadrature.
    from scipy.integrate import quad
    quadrature=[]
    for a in [1.2,1.5,1.8]:
        ca=2**a*gamma((1+a)/2)/(np.sqrt(np.pi)*abs(gamma(-a/2)))
        # Stable numerator; avoid subtractive cancellation near zero.
        val=2*ca*(quad(lambda y:-np.expm1((a/2)*np.log1p(-y*y))/y**(1+a),0,1,epsabs=1e-9)[0]+1/a)
        exact=2**a*gamma(1+a/2)*gamma((1+a)/2)/np.sqrt(np.pi)
        quadrature.append(dict(alpha=a,quadrature=val,analytic=exact,abs_error=abs(val-exact)))
    savecsv('quadrature_checks.csv',quadrature)
    return endpoint

def main():
    start=time.time(); truths=forward_validation();print('forward complete',flush=True)
    sur=build_surrogate(); endpoint=extra_checks(sur,truths);print('checks complete',flush=True)
    df=inverse(truths,sur)
    meta=dict(date='2026-11-05',kind='synthetic numerical experiments',python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),processor=platform.processor(),threads=1,runtime_seconds=time.time()-start,noise_scale=SIGBASE,alpha_truth=[1.2,1.5,1.8],kappa_truth=.5,n_truth=1023,n_fit=255,replicates=50,alpha_bounds=[.8,1.95],kappa_bounds=[.1,1.],parameter_grid_spacing=.005,alpha2_matrix_error=endpoint,fit_count=len(df),failures=int((~df.success).sum()),boundary_hits=int(df.boundary.sum()),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'environment.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2),flush=True)
if __name__=='__main__': main()
