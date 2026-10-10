#!/usr/bin/env python3
"""Reproduce sheet illustrations from the actual native lab model.

Requires numpy and matplotlib; run `cargo build --release --manifest-path
core/Cargo.toml --example lab_dump` first. No simulation arrays are guessed.
"""
import json
from pathlib import Path
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'chapters/figures'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'precrossedsheet-v1'})
BLUE='#0072B2';ORANGE='#D55E00';GREEN='#009E73'
def run(dim):
 req={'dim':dim,'n':16,'fraction':.55 if dim==1 else .45,'order':4,'steps':64,'mesh':32,'forceMesh':32,'smoothing':0.0,'ic':'waves'}
 res=subprocess.run([str(ROOT/'core/target/release/examples/lab_dump')],input=json.dumps(req),capture_output=True,text=True,check=True)
 d=json.loads(res.stdout)
 if not d.get('guard',{}).get('ok'):raise RuntimeError(d)
 return d,np.array(d['q']).reshape(-1,dim),np.array(d['x']).reshape(-1,dim),np.array(d['v']).reshape(-1,dim)
def save(fig,name):
 fig.savefig(OUT/(name+'.svg'),bbox_inches='tight',metadata={'Date':None})
 fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight')
 plt.close(fig)
# Exact plane-symmetric growing flow, plus its conservative cell average.
d,q,x,v=run(1);fig,axs=plt.subplots(1,3,figsize=(12,3.4))
axs[0].plot(q[:,0],x[:,0],'-o',color=BLUE,ms=3);axs[0].plot([0,1],[0,1],':',color='#777',label='undeformed');axs[0].set(xlabel='initial label q',ylabel='position x',title='Compression of the position map');axs[0].legend(frameon=False)
axs[1].plot(x[:,0],v[:,0],'-o',color=ORANGE,ms=3);axs[1].set(xlabel='position x',ylabel='dx / dD',title='The cold phase-space curve')
axs[2].stairs(d['rho'],np.linspace(0,1,d['mesh']+1),color=GREEN);axs[2].set(xlabel='position x',ylabel='cell-average ρ / ρ̄',title='Density from interval overlaps')
fig.suptitle(f"1D plane symmetry · D = {d['a']:.3f} · min J = {d['guard']['minimumJacobian']:.3f}",fontsize=12,y=1.02);fig.tight_layout();save(fig,'sheet-1d')
# Keep the same triangulation before and after deformation.
d,q,x,v=run(2);n=d['n'];triangles=[]
for i in range(n-1):
 for j in range(n-1):
  a=i*n+j;b=(i+1)*n+j;c=i*n+j+1;e=(i+1)*n+j+1
  triangles.extend([(a,b,e),(a,e,c)])
fig,axs=plt.subplots(1,2,figsize=(10,4.5))
for ax,pts,title in zip(axs,[q,x],['Uniform labels: equal initial triangle masses','Deformed positions: fixed masses, changing areas']):
 segments=[pts[[t[0],t[1],t[2],t[0]]] for t in triangles]
 ax.add_collection(LineCollection(segments,colors=BLUE,linewidths=.55,alpha=.55));ax.scatter(pts[:,0],pts[:,1],c=q[:,0],cmap='viridis',s=7,zorder=3)
 ax.autoscale();ax.set_aspect('equal');ax.set(xlabel='first coordinate',ylabel='second coordinate',title=title)
fig.suptitle(f"2D position projection · D = {d['a']:.3f} · min simplex J = {d['guard']['minimumSimplexJacobian']:.3f}",fontsize=12,y=1.01);fig.tight_layout();save(fig,'sheet-2d')
# A small patch with six highlighted Kuhn tetrahedra in one cell.
d,q,x,v=run(3);n=d['n'];lo,hi=3,8
idx=lambda i,j,k:(i*n+j)*n+k
edges=[]
for i in range(lo,hi):
 for j in range(lo,hi):
  for k in range(lo,hi):
   a=idx(i,j,k)
   for di,dj,dk in [(1,0,0),(0,1,0),(0,0,1)]:
    if i+di<hi and j+dj<hi and k+dk<hi:edges.append([a,idx(i+di,j+dj,k+dk)])
TETS=[[0,1,3,7],[0,1,5,7],[0,2,3,7],[0,2,6,7],[0,4,5,7],[0,4,6,7]]
corner=[idx(5+(b&1),5+((b>>1)&1),5+((b>>2)&1)) for b in range(8)]
fig=plt.figure(figsize=(10,4.8))
for panel,pts,title in [(1,q,'Initial volume patch'),(2,x,'Deformed patch: tetrahedral connectivity')]:
 ax=fig.add_subplot(1,2,panel,projection='3d');ax.add_collection3d(Line3DCollection([pts[e] for e in edges],colors=BLUE,linewidths=.6,alpha=.35))
 for tid,tet in enumerate(TETS):
  vv=pts[[corner[t] for t in tet]];faces=[vv[list(f)] for f in [(0,1,2),(0,1,3),(0,2,3),(1,2,3)]]
  ax.add_collection3d(Poly3DCollection(faces,facecolors=plt.cm.Set2(tid/6),edgecolors='#444',linewidths=.45,alpha=.12))
 patch=np.array([pts[idx(i,j,k)] for i in range(lo,hi) for j in range(lo,hi) for k in range(lo,hi)])
 ax.scatter(*patch.T,s=5,c=BLUE,alpha=.65)
 mins=patch.min(axis=0);maxs=patch.max(axis=0)
 ax.set_xlim(mins[0]-.02,maxs[0]+.02);ax.set_ylim(mins[1]-.02,maxs[1]+.02);ax.set_zlim(mins[2]-.02,maxs[2]+.02)
 ax.set(xlabel='coordinate 1',ylabel='coordinate 2',zlabel='coordinate 3',title=title);ax.view_init(elev=23,azim=-53);ax.set_box_aspect([1,1,1])
fig.suptitle(f"3D position projection · six tetrahedra per cube · D = {d['a']:.3f}",fontsize=12,y=1.01);fig.tight_layout();save(fig,'sheet-3d')
print('Reproduced 1D, 2D, 3D figures from guarded native Rust outputs.')
