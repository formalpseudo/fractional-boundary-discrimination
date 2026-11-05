"""Independent restricted-Laplacian Galerkin solver using a published weighted-Jacobi identity.
No centered-difference matrix or data from the first solver are used in assembly.
Acosta et al., arXiv:1608.08443, weighted polynomial operator eigenvalues.
"""
from run_experiments import *
from scipy.special import eval_jacobi, roots_jacobi
from scipy.linalg import eigh
from functools import lru_cache
V2=OUT/'v2';V2.mkdir(exist_ok=True)

def normalized_jacobi(a,order,x):
    s=a/2; n=np.arange(order)
    lognorm=(2*s+1)*np.log(2)-np.log(2*n+2*s+1)+2*gammaln(n+s+1)-gammaln(n+1)-gammaln(n+2*s+1)
    return eval_jacobi(n[None,:],s,s,np.asarray(x)[:,None])*np.exp(-lognorm/2)

@lru_cache(maxsize=1000)
def assemble(a,order):
    s=a/2;n=np.arange(order)
    # q>=order integrates the mass polynomial exactly against (1-x^2)^(2s).
    xm,wm=roots_jacobi(max(order+8,64),2*s,2*s)
    pm=normalized_jacobi(a,order,xm);mass=pm.T@(wm[:,None]*pm)
    stiffness=np.diag(np.exp(gammaln(n+a+1)-gammaln(n+1)))
    lam,v=eigh(stiffness,mass,check_finite=False)
    # Initial projection uses weight (1-x^2)^s, not its square.
    xq,wq=roots_jacobi(max(2*order,128),s,s)
    pq=normalized_jacobi(a,order,xq)
    return lam,v,pq,xq,wq,mass,stiffness

def predict(a,k,x,t,order=64,coeff=COEF):
    lam,v,pq,xq,wq,_,_=assemble(float(a),order)
    freq=np.arange(1,len(coeff)+1)*np.pi/2
    b=pq.T@(wq*(np.sin(np.outer(xq+1,freq))@coeff))
    btime=v@(np.exp(-k*lam[:,None]*np.asarray(t))*(v.T@b)[:,None])
    basis=(1-np.asarray(x)**2)[:,None]**(a/2)*normalized_jacobi(a,order,x)
    return basis@btime

def main():
    convergence=[];cross=[];poisson=[];checks=[];inverse=[]
    for a in [1.2,1.5,1.8]:
        ref=np.concatenate([predict(a,.5,x,t,128).ravel() for x,t in BLOCKS])
        for order in [8,16,32,64,96]:
            pred=np.concatenate([predict(a,.5,x,t,order).ravel() for x,t in BLOCKS])
            convergence.append(dict(alpha=a,order=order,rmse=np.sqrt(np.mean((pred-ref)**2)),max_abs=np.max(abs(pred-ref))))
        for n in [127,255,511,1023]:
            fd=direct(n,a,.5);d=fd-ref
            for j,name in enumerate(['wide','central','test']):
                z=slice(CUT[j],CUT[j+1]);cross.append(dict(alpha=a,n=n,block=name,rmse=np.sqrt(np.mean(d[z]**2)),max_abs=np.max(abs(d[z]))))
        lam,v,pq,xq,wq,mass,stiff=assemble(a,64)
        # Analytic Poisson problem tested in this basis independently.
        f=gamma(a+1);b=pq.T@(wq*np.full(len(xq),f));cs=np.linalg.solve(stiff,b)
        xt=np.linspace(-.999,.999,501);uh=(1-xt*xt)**(a/2)*(normalized_jacobi(a,64,xt)@cs)
        poisson.append(dict(alpha=a,max_abs=np.max(abs(uh-(1-xt*xt)**(a/2)))))
        checks.append(dict(alpha=a,min_mass_eigenvalue=np.linalg.eigvalsh(mass)[0],min_generator_eigenvalue=lam[0],generalized_eigen_residual=np.linalg.norm(stiff@v-mass@(v*lam))/np.linalg.norm(stiff@v)))
        # Bias of centered differences on a source from a distinct solver.
        y=ref[:75]
        for model in ['restricted_fd255','spectral']:
            fun=(lambda p:direct(255,*p,blocks=[BLOCKS[0]])) if model=='restricted_fd255' else (lambda p:spectral(*p,XW,T).ravel())
            fit=fit_model(fun,y)
            inverse.append(dict(alpha=a,model=model,alpha_hat=fit.x[0],kappa_hat=fit.x[1],rmse=np.sqrt(np.mean(fit.fun**2)),success=fit.success))
        np.savez_compressed(V2/f'galerkin_truth_{a:.1f}.npz',values=ref)
    for name,rows in [('galerkin_convergence',convergence),('cross_solver',cross),('galerkin_poisson',poisson),('galerkin_checks',checks),('independent_inverse',inverse)]:pd.DataFrame(rows).to_csv(V2/(name+'.csv'),index=False)
    print(pd.DataFrame(convergence).to_string(index=False));print(pd.DataFrame(inverse).to_string(index=False))
if __name__=='__main__':main()
