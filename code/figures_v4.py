from power_study_v4 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
F=ROOT/'figures/v4';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':7,'xtick.labelsize':8,'ytick.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'lines.linewidth':1.5,'savefig.bbox':'tight'})
blue='#0072B2';red='#D55E00';green='#009E73';purple='#8B6AA7'
def save(f,n):f.savefig(F/(n+'.pdf'),dpi=600);f.savefig(F/(n+'.png'),dpi=600);plt.close(f)
def title(ax,t):ax.set_title(t,loc='left',fontweight='bold')
ds=json.loads((V4/'frozen_designs.json').read_text());sel=ds['9:power']['indices'];x=X[sel];T,v,s,k,score=learn(x,TRAIN[:,sel]);physical=T.T@v[0].reshape(6,5)
fig,ax=plt.subplots(1,3,figsize=(9,2.65),layout='constrained');ax[0].semilogy(np.arange(1,len(s)+1),s/s[0],'o-',color=blue,ms=3);ax[0].axvline(k,color=red,ls='--');ax[0].set(xlabel='Residual direction',ylabel='Normalized singular value');title(ax[0],'(a) Training discrepancy spectrum')
im=ax[1].imshow(physical,aspect='auto',cmap='RdBu_r',vmin=-abs(physical).max(),vmax=abs(physical).max());ax[1].set_xticks(range(5),TIMES);ax[1].set_yticks(range(9),[f'{z:.2f}' for z in x]);ax[1].set(xlabel='Time',ylabel='Sensor position');fig.colorbar(im,ax=ax[1],pad=.02);title(ax[1],'(b) Frozen leading direction')
a=pd.read_csv(V4/'rank_ablation.csv').query('design=="power" and domain=="stress"');g=a.groupby('rank').agg(retained=('retained','mean'),power=('power','mean'));ax[2].plot(g.index,g.retained,'o-',color=blue,label='Retained discrepancy');ax[2].plot(g.index,g.power,'s-',color=red,label='Detection power');ax[2].set(xlabel='Number of retained directions',ylabel='Mean over 256 stress cases',ylim=(0,1.03));ax[2].legend(frameon=False);title(ax[2],'(c) Signal versus added dimensions');save(fig,'01_compression_mechanism')
h=pd.read_csv(V4/'holdout.csv');fig,axs=plt.subplots(2,2,figsize=(8.5,5.7),layout='constrained')
for j,domain in enumerate(['interpolation','stress']):
 for name,meth,col,ls,label in [('uniform','omnibus',blue,'--','Uniform + omnibus'),('power','omnibus',green,'--','Designed + omnibus'),('power','compressed',red,'-','Designed + compressed'),('power','local-rank1',purple,':','Designed + local rank one')]:
  a=h.query('m==9 and noise==.01 and domain==@domain and design==@name and method==@meth');y=np.sort(a.power);axs[0,j].plot(y,np.arange(1,len(y)+1)/len(y),color=col,ls=ls,label=label)
 axs[0,j].set(xlabel='Detection probability',ylabel='Empirical fraction of scenarios',xlim=(0,1),ylim=(0,1));title(axs[0,j],['(a) 256 interpolation scenarios','(b) 256 stress scenarios'][j]);axs[0,j].legend(frameon=False,loc='upper left',fontsize=6)
 a=h.query('m==9 and noise==.01 and domain==@domain and design=="power"').pivot(index='case',columns='method',values='power');meta=h.query('m==9 and noise==.01 and domain==@domain and design=="power" and method=="compressed"').set_index('case');im=axs[1,j].scatter(meta.alpha,meta.kappa,c=(a.compressed-a.omnibus).loc[meta.index],s=18,cmap='viridis',vmin=0,vmax=.8);axs[1,j].set(xlabel=r'Fractional order $\alpha$',ylabel=r'Diffusivity $\kappa$');title(axs[1,j],['(c) Power gain within training ranges','(d) Power gain beyond training ranges'][j]);fig.colorbar(im,ax=axs[1,j],label='Compressed − omnibus',pad=.02)
save(fig,'02_generalization')
fig,axs=plt.subplots(1,3,figsize=(9.2,2.85),layout='constrained')
for method,col,label in [('omnibus',blue,'Omnibus'),('compressed',red,'Compressed'),('local-rank1',purple,'Local rank one'),('POD99',green,'99% POD')]:
 a=h.query('design=="power" and domain=="stress" and noise==.01 and method==@method');g=a.groupby('m').power.mean();axs[0].plot(g.index,g,'o-',color=col,label=label,ms=3)
 a=h.query('design=="power" and domain=="stress" and m==9 and method==@method');g=a.groupby('noise').power.mean();axs[1].plot(g.index*100,g,'o-',color=col,ms=3)
 a=h.query('design=="power" and domain=="legacy" and m==9 and method==@method');g=a.groupby('noise').power.min();axs[2].plot(g.index*100,g,'o-',color=col,ms=3)
for ax in axs:ax.set_ylim(0,1.03)
axs[0].set(xlabel='Sensor budget',ylabel='Mean stress-case power',xticks=[7,9,13,17]);axs[0].legend(frameon=False);axs[1].set(xlabel='Noise (%)',ylabel='Mean stress-case power');axs[2].set(xlabel='Noise (%)',ylabel='Worst legacy-case power');
for ax,t in zip(axs,['(a) Budget at 1% noise','(b) Expanded stress distribution','(c) Original twelve-case benchmark']):title(ax,t)
save(fig,'03_budget_noise_baselines')
c=pd.read_csv(V4/'calibration.csv');pa=pd.read_csv(V4/'paired_residual.csv');fig,ax=plt.subplots(1,3,figsize=(9,3),layout='constrained')
for j,truth in enumerate(['spectral','restricted']):
 for offset,method,col in [(-.1,'omnibus',blue),(.1,'compressed',red)]:
  z=c.query('truth==@truth and method==@method').set_index('design').loc[['uniform','power']];ax[j].errorbar(np.arange(2)+offset,z.rate,yerr=np.array([z.rate-z.low,z.high-z.rate]),fmt='o',color=col,capsize=3,label=method.capitalize())
 ax[j].set_xticks([0,1],['Uniform','Designed']);ax[j].set_ylabel('Rejection probability');ax[j].set_ylim((0,.09) if j==0 else (0,1.03));ax[j].axhline(.05,color='.6',ls=':');title(ax[j],['(a) Null: 5,000 paired readouts','(b) Alternative: 5,000 readouts'][j])
ax[0].legend(frameon=False)
for i,(column,col,label) in enumerate([('omnibus_reject',blue,'Omnibus'),('residual_reject',green,'Nonlinear residual'),('compressed_reject',red,'Compressed')]):
 z=pa.query('design=="power" and truth=="restricted"')[column];lo,hi=wilson(z.sum(),len(z));ax[2].errorbar(i,z.mean(),yerr=[[z.mean()-lo],[hi-z.mean()]],fmt='o',capsize=3,color=col)
ax[2].set(xticks=[0,1,2],xticklabels=['Omnibus','Residual','Compressed'],ylim=(.5,1.03),ylabel='Alternative rejection');title(ax[2],'(c) Same 300 observed datasets');save(fig,'04_calibration_residual')
c=pd.read_csv(V4/'support_stress.csv');fig,axs=plt.subplots(2,2,figsize=(8,5.3),layout='constrained')
for j,p in enumerate([5,7]):
 for row,truth in enumerate(['spectral','restricted']):
  for meth,col in [('omnibus',blue),('compressed',red)]:
   z=c.query('support==@p and truth==@truth and method==@meth');axs[row,j].errorbar(z.delta,z.rate,yerr=np.array([z.rate-z.low,z.high-z.rate]),fmt='o-',color=col,capsize=2,label=meth.capitalize())
  axs[row,j].set(xlabel='Additional fifth-mode amplitude',ylabel='Rejection probability',ylim=(0,.13) if row==0 else (0,1.03));axs[row,j].axhline(.05,color='.5',ls=':');title(axs[row,j],f'({chr(97+row*2+j)}) '+('Null' if row==0 else 'Alternative')+f', declared support p = {p}')
axs[0,0].legend(frameon=False);save(fig,'05_support_uncertainty')
c=pd.read_csv(V4/'noise_stress.csv');fig,axs=plt.subplots(1,2,figsize=(8,2.9),layout='constrained')
for distribution,ls in [('Gaussian','-'),('Student-t4','--')]:
 for meth,col in [('omnibus',blue),('compressed',red)]:
  for ax,truth in zip(axs,['spectral','restricted']):
   z=c.query('noise==@distribution and method==@meth and truth==@truth');ax.errorbar(z.length,z.rate,yerr=np.array([z.rate-z.low,z.high-z.rate]),fmt='o',ls=ls,color=col,capsize=2,label=f'{meth}, {distribution}',ms=3)
for ax in axs:ax.set(xlabel='Spatial correlation length',ylabel='Rejection probability')
axs[0].set_ylim(0,.1);axs[1].set_ylim(0,1.05);axs[0].axhline(.05,color='.6',ls=':');axs[1].legend(frameon=False,fontsize=6);title(axs[0],'(a) Null calibration');title(axs[1],'(b) Boundary-model detection');save(fig,'06_noise_covariance')
c=pd.read_csv(V4/'twod_holdout.csv');fig,axs=plt.subplots(1,3,figsize=(9.1,2.8),layout='constrained')
for name,col in [('uniform',blue),('robust',red)]:
 for method,ls in [('omnibus','--'),('compressed','-')]:
  z=c.query('design==@name and method==@method and noise==.01');yy=np.sort(z.power);axs[0].plot(yy,np.arange(1,len(yy)+1)/len(yy),color=col,ls=ls,label=f'{name}, {method}')
for j,method in enumerate(['omnibus','compressed']):
 z=c.query('design=="robust" and method==@method and noise==.01');im=axs[j+1].scatter(z.ax,z.ay,c=z.power,vmin=0,vmax=1,cmap='viridis',s=25);axs[j+1].set(xlabel=r'$\alpha_x$',ylabel=r'$\alpha_y$');fig.colorbar(im,ax=axs[j+1],pad=.02,label='Power');title(axs[j+1],['(b) Omnibus','(c) Frozen compression'][j])
axs[0].set(xlabel='Detection probability',ylabel='Fraction of 128 unseen order pairs',xlim=(0,1),ylim=(0,1));axs[0].legend(frameon=False,fontsize=6);title(axs[0],'(a) Two-dimensional generalization');save(fig,'07_twod_generalization')
print('Seven data figures exported.')
