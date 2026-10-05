import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from PIL import Image
import sys, os
# Usage: python make_study_area_figure.py <pipeline_results_dir>
# The results dir is the notebook output folder containing fig_02_rgb_proxy.png,
# cluster_map_consensus.npy and anomaly_score.npy.
RES = sys.argv[1] if len(sys.argv) > 1 else '.'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.5,'axes.linewidth':0.6})
x0,y0,px=636945.0,2752905.0,30.0
H,W=1165,1194
ext=[x0/1000,(x0+W*px)/1000,(y0-H*px)/1000,y0/1000]
rgb=np.array(Image.open(os.path.join(RES,'fig_02_rgb_proxy.png')).convert('RGB'))[79:1957,25:1950]
units=np.load(os.path.join(RES,'cluster_map_consensus.npy')).astype(float); units[units<0]=np.nan
an=np.load(os.path.join(RES,'anomaly_score.npy'))
cols=['#E69F00','#56B4E9','#009E73','#CC79A7']
fig,axs=plt.subplots(1,3,figsize=(7.2,2.75),constrained_layout=True)
axs[0].imshow(rgb,extent=ext); axs[0].set_title('(a) EnMAP L2A true-colour proxy',fontsize=8)
axs[1].set_facecolor('white')
axs[1].imshow(units,cmap=ListedColormap(cols),vmin=-0.5,vmax=3.5,extent=ext,interpolation='nearest')
axs[1].set_title('(b) Spectral units (3×3 consensus)',fontsize=8)
axs[1].legend(handles=[Patch(color=c,label=f'Unit {i}') for i,c in enumerate(cols)],loc='lower left',fontsize=6,frameon=True,framealpha=0.9,handlelength=1,borderpad=0.3,labelspacing=0.2)
v1,v2=np.nanpercentile(an,[2,99.5])
axs[2].set_facecolor('black'); im=axs[2].imshow(an,cmap='magma',vmin=v1,vmax=v2,extent=ext,interpolation='nearest')
axs[2].set_title('(c) Ensemble anomaly score',fontsize=8)
cb=fig.colorbar(im,ax=axs[2],shrink=0.8,pad=0.02); cb.set_label('Robust score',fontsize=7); cb.ax.tick_params(labelsize=6)
for i,ax in enumerate(axs):
    ax.set_xlabel('Easting (km)',fontsize=7)
    if i==0: ax.set_ylabel('Northing (km), UTM 36N',fontsize=7)
    else: ax.set_yticklabels([])
    ax.tick_params(labelsize=6)
    # scale bar 10 km
    sx,sy=ext[1]-13,ext[2]+1.8
    ax.plot([sx,sx+10],[sy,sy],color='k' if i==1 else 'w',lw=2)
    ax.text(sx+5,sy+1.3,'10 km',ha='center',fontsize=6,color='k' if i==1 else 'w')
    ax.annotate('N',xy=(ext[1]-3,ext[3]-2),xytext=(ext[1]-3,ext[3]-7),ha='center',fontsize=7,color='k' if i==1 else 'w',
                arrowprops=dict(arrowstyle='-|>',color='k' if i==1 else 'w',lw=1))
fig.savefig(os.path.join(RES,'fig_study_area.png'),dpi=400)
print('ok')
