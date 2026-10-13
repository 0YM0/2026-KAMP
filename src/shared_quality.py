import math
from pathlib import Path
import numpy as np
import cv2
import torch
from torch import nn
from PIL import Image

def variants(p,size):
    w,h=size;p=np.asarray(p,float);center=(p[:2]+p[2:4])/2;wh=p[2:4]-p[:2]
    result=[p.copy()]
    if min(wh)<=0 or max(wh)>32:return np.asarray(result)
    for dx in [0,-.125,.125]:
        for dy in [0,-.125,.125]:
            for sw in [1,.75,4/3]:
                for sh in [1,.75,4/3]:
                    if (dx,dy,sw,sh)==(0,0,1,1):continue
                    xy=center+wh*np.array([dx,dy]);dims=wh*np.array([sw,sh]);q=p.copy()
                    q[:2]=np.maximum(xy-dims/2,0);q[2:4]=np.minimum(xy+dims/2,[w,h])
                    if np.all(q[2:4]>q[:2]):result.append(q)
    z=np.asarray(result);assert np.array_equal(z[0],p) and np.all(z[:,4]==p[4])
    return z


def component_features(gray,source,candidates):
    """Native grayscale context, no labels/marker mask/camera/absolute coordinates/score."""
    p=np.asarray(source);center=(p[:2]+p[2:4])/2;wh=p[2:4]-p[:2]
    side=int(math.ceil(np.clip(4*max(wh),48,128)));left,top=np.floor(center-side/2).astype(int)
    patch=np.asarray(gray.crop((left,top,left+side,top+side)),np.float32)
    bg=cv2.morphologyEx(patch,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
    residual=np.maximum(bg-patch,0);yy,xx=np.mgrid[:side,:side];xx=xx+left+.5;yy=yy+top+.5
    local=(np.abs(xx-center[0])<=wh[0])&(np.abs(yy-center[1])<=wh[1])
    weights=residual*local;peak=float(weights.max());weights*=weights>=max(peak*.5,1.)
    count,labels=cv2.connectedComponents((weights>0).astype(np.uint8),connectivity=8)
    if count>1:
        chosen=max(range(1,count),key=lambda i:(float(weights[labels==i].max()),float(weights[labels==i].sum())))
        weights*=labels==chosen
    mass=max(float(weights.sum()),1e-6);cx=float((weights*xx).sum()/mass) if weights.sum()>0 else center[0];cy=float((weights*yy).sum()/mass) if weights.sum()>0 else center[1]
    sx=float(np.sqrt((weights*(xx-cx)**2).sum()/mass));sy=float(np.sqrt((weights*(yy-cy)**2).sum()/mass))
    pm=float(np.median(patch));ps=max(float(patch.std()),2.);normal=(patch-pm)/ps;rows=[]
    for q in candidates:
        qc=(q[:2]+q[2:4])/2;qw=np.maximum(q[2:4]-q[:2],1e-3)
        inside=(xx>=q[0])&(xx<=q[2])&(yy>=q[1])&(yy<=q[3]);pixels=normal[inside];r=residual[inside]/max(peak,1.)
        if not len(pixels):pixels=np.array([0.]);r=np.array([0.])
        f=[*np.log(qw/16),*np.log(qw/wh),*((qc-center)/wh),*((qc-[cx,cy])/qw),sx/qw[0],sy/qw[1],
           peak/64,pm/255,ps/16,np.log(max(wh.prod(),1)/256),float(pixels.mean()),float(pixels.std()),float(pixels.min()),
           *np.quantile(pixels,[.05,.25,.5,.75,.95]),float(r.mean()),float(r.max()),float((r>.5).mean()),
           float(weights[inside].sum()/mass),float(inside.sum()/side**2),float((cx>=q[0])&(cx<=q[2])&(cy>=q[1])&(cy<=q[3]))]
        rows.append(f)
    a=np.asarray(rows,np.float32);base=np.repeat(a[0:1],len(a),axis=0)
    result=np.c_[a,base,a-base];assert np.isfinite(result).all()
    return result


class Quality(nn.Module):
    def __init__(self,n):
        super().__init__();self.layers=nn.Sequential(nn.Linear(n,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,x):return self.layers(x).sigmoid().squeeze(1)


def choose(a,q,size_only=False):
    gain=q-q[0];valid=(q>=.55)&(gain>=.10)
    if size_only:
        centers=(a[:,:2]+a[:,2:4])/2
        valid &= np.all(np.isclose(centers,centers[0],rtol=0,atol=1e-6),axis=1)
    valid[0]=False;ids=np.flatnonzero(valid)
    return int(max(ids,key=lambda i:(float(q[i]),-int(i)))) if len(ids) else 0


class Refiner:
    def __init__(self,checkpoint):
        ck=torch.load(checkpoint,weights_only=True);self.model=Quality(84);self.model.load_state_dict(ck['state']);self.model.eval()
        self.mean=np.asarray(ck['mean'],np.float32);self.std=np.asarray(ck['std'],np.float32)
    def predict(self,image_path,predictions):
        gray=Image.open(image_path).convert('L');out=np.asarray(predictions,float).reshape(-1,5).copy();ledger=[]
        with torch.no_grad():
            for pi,p in enumerate(out.copy()):
                a=variants(p,gray.size);x=component_features(gray,p,a);q=self.model(torch.from_numpy(np.clip((x-self.mean)/self.std,-8,8))).numpy();idx=choose(a,q)
                out[pi]=a[idx];ledger.append(dict(prediction_index=pi,selected=idx,source_box=p.tolist(),selected_box=a[idx].tolist(),predicted_source_quality=float(q[0]),predicted_selected_quality=float(q[idx]),predicted_gain=float(q[idx]-q[0])))
        assert np.array_equal(out[:,4],np.asarray(predictions,float).reshape(-1,5)[:,4]);return out.tolist(),ledger
