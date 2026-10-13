"""Local texture prototypes; no labels, filenames or learned detector inputs.

Conditional prototype: detrended residual, robust scale, separable AR(1)
covariance, Gaussian conditioning on observed neighbors. This is not a full
implementation of Galerne/Leclaire's stationary Gaussian texture algorithm.
"""
import hashlib
import cv2
import numpy as np
from xray_marker_processing import restore, edit_mask


def robust_sigma(a):
    if not len(a): return 0.
    return float(1.4826*np.median(np.abs(a-np.median(a))))


def covariance(a, b, sigma, rho_x, rho_y):
    distance = np.abs(a[:,None,:]-b[None,:,:])
    return sigma*sigma * rho_y**distance[:,:,0] * rho_x**distance[:,:,1]


def posterior(missing, observed, values, sigma, rho_x, rho_y):
    kmm = covariance(missing,missing,sigma,rho_x,rho_y)
    if len(observed):
        koo = covariance(observed,observed,sigma,rho_x,rho_y)
        koo += np.eye(len(observed))*sigma*sigma*1e-4
        kmo = covariance(missing,observed,sigma,rho_x,rho_y)
        solution = np.linalg.solve(koo,np.column_stack([values,kmo.T]))
        mean = kmo@solution[:,0]
        cov = kmm-kmo@solution[:,1:]
    else:
        mean = np.zeros(len(missing)); cov = kmm
    cov = (cov+cov.T)/2
    try:
        factor = np.linalg.cholesky(cov+np.eye(len(missing))*sigma*sigma*1e-6)
        fallback = False
    except np.linalg.LinAlgError:
        eigenvalues,vectors=np.linalg.eigh(cov)
        factor=vectors@np.diag(np.sqrt(np.maximum(eigenvalues,sigma*sigma*1e-6)))
        fallback=True
    return mean,factor,fallback


def texture_restore(gray, mask, method, gain=1., run_seed=20261006):
    if not mask.any(): return gray.copy(), []
    if method not in ('conditional','patch','iid','adaptive'): raise ValueError(method)
    base = restore(gray,mask,'harmonic',0)
    trend = cv2.GaussianBlur(base.astype(float),(0,0),1.2)
    residual = gray.astype(float)-trend
    n,labels,stats,_ = cv2.connectedComponentsWithStats(mask.astype(np.uint8),connectivity=8)
    result = base.copy(); logs=[]
    for ci in range(1,n):
        x,y,w,h,_ = map(int,stats[ci]);comp=labels==ci
        x0,x1=max(0,x-8),min(gray.shape[1],x+w+8)
        y0,y1=max(0,y-8),min(gray.shape[0],y+h+8)
        local=comp[y0:y1,x0:x1];all_mask=mask[y0:y1,x0:x1]
        r=residual[y0:y1,x0:x1];t=trend[y0:y1,x0:x1]
        ring=edit_mask(local,6)&~edit_mask(all_mask,1)
        values=r[ring]
        sigma=robust_sigma(values)
        log=dict(component=ci,sigma=sigma,method=method,gain=gain,masked_pixels=int(local.sum()),
                 status='sampled',conditioning_pixels=0,rho_x=0.,rho_y=0.,factorization_fallback=False)
        if len(values)<12 or sigma<.25:
            log['status']='baseline_fallback_insufficient_texture';logs.append(log);continue
        centered=r-float(np.median(values))
        reliable=(~all_mask)&(np.abs(centered)<=3.5*sigma)
        correlations=[]
        for axis in [1,0]:
            if axis==1:
                valid=ring[:,:-1]&ring[:,1:]&reliable[:,:-1]&reliable[:,1:]
                a,b=centered[:,:-1][valid],centered[:,1:][valid]
            else:
                valid=ring[:-1,:]&ring[1:,:]&reliable[:-1,:]&reliable[1:,:]
                a,b=centered[:-1,:][valid],centered[1:,:][valid]
            rho=float(np.mean(a*b)/(sigma*sigma)) if len(a)>=8 else 0.
            correlations.append(float(np.clip(rho,-.35,.6)))
        rho_x,rho_y=correlations;log.update(rho_x=rho_x,rho_y=rho_y)
        payload=gray[y0:y1,x0:x1][~all_mask].tobytes()+local.tobytes()+str(run_seed).encode()
        seed=int.from_bytes(hashlib.sha256(payload).digest()[:8],'little')
        rng=np.random.default_rng(seed)
        coordinates=np.argwhere(local)
        if method in ('conditional','iid','adaptive'):
            observed_mask=edit_mask(local,2)&~all_mask
            observed=np.argwhere(observed_mask)
            if len(observed)>192:
                observed=observed[np.linspace(0,len(observed)-1,192).astype(int)]
            obs_values=np.clip(centered[observed[:,0],observed[:,1]],-3.5*sigma,3.5*sigma)
            if method in ('conditional','adaptive'):
                mean,factor,fallback=posterior(coordinates,observed,obs_values,sigma,rho_x,rho_y)
                sampled=mean+factor@rng.standard_normal(len(coordinates))
                log.update(conditioning_pixels=len(observed),factorization_fallback=fallback)
            else:
                sampled=sigma*rng.standard_normal(len(coordinates))
            yy,xx=coordinates[:,0],coordinates[:,1]
            if method=='adaptive':
                # Preserve the harmonic structural mean. Only a bounded,
                # zero-mean conditional innovation adds local microtexture.
                # Strong observed edges and nearby dark structure suppress it.
                innovation=np.clip(sampled-mean,-2*sigma,2*sigma)
                gx=cv2.Sobel(t,cv2.CV_64F,1,0)/8
                gy=cv2.Sobel(t,cv2.CV_64F,0,1)/8
                edge=np.hypot(gx,gy)
                # Gate only edges clearly stronger than the residual scale.
                edge_gate=1/(1+(edge/max(3*sigma,.75))**4)
                # Require a compact local minimum: at least 3 of 4 neighbors
                # two pixels away must be brighter. A broad product edge
                # typically has only one or two brighter directions.
                structural=base[y0:y1,x0:x1].astype(float)
                padded=np.pad(structural,2,mode='reflect')
                differences=np.stack([padded[:-4,2:-2]-structural,
                    padded[4:,2:-2]-structural,padded[2:-2,:-4]-structural,
                    padded[2:-2,4:]-structural])
                compact=np.sort(differences,axis=0)[1]
                # A four-sigma contrast avoids guarding ordinary noise minima.
                dark=(~all_mask)&(compact>max(8.,4*sigma))
                dn,dl,ds,_=cv2.connectedComponentsWithStats(dark.astype(np.uint8))
                compact_dark=np.zeros_like(dark)
                for di in range(1,dn):
                    if ds[di,cv2.CC_STAT_AREA]<=9:compact_dark|=dl==di
                dark=compact_dark
                protect=edit_mask(dark,3)
                weights=gain*edge_gate[yy,xx]*(~protect[yy,xx])
                blended=base[y0:y1,x0:x1][yy,xx]+weights*innovation
                log.update(mean_gain=float(weights.mean()),protected_fraction=float(protect[yy,xx].mean()),
                           innovation_clip_sigma=2.,dark_guard_radius=3)
            else:
                target=t[yy,xx]+float(np.median(values))+sampled
                blended=(1-gain)*base[y0:y1,x0:x1][yy,xx]+gain*target
            result[y0:y1,x0:x1][yy,xx]=np.clip(np.rint(blended),0,255).astype(np.uint8)
        else:
            # Exemplar residual prototype: safe observed 3x3 donor blocks,
            # context match using visible residuals and local mean brightness.
            safe=cv2.erode(reliable.astype(np.uint8),np.ones((3,3),np.uint8))>0
            safe[0,:]=safe[-1,:]=False;safe[:,0]=safe[:,-1]=False
            donors=np.argwhere(safe)
            if not len(donors):
                log['status']='baseline_fallback_no_safe_donor';logs.append(log);continue
            if len(donors)>256:donors=donors[rng.choice(len(donors),256,replace=False)]
            donor_patches=np.stack([centered[dy-1:dy+2,dx-1:dx+2] for dy,dx in donors])
            donor_level=t[donors[:,0],donors[:,1]]
            pending=local.copy();generated=np.zeros_like(centered)
            ordering=coordinates[rng.permutation(len(coordinates))]
            for py,px in ordering:
                if not pending[py,px]:continue
                # Patches clipped at image borders are handled by their actual
                # offsets; no outside-mask pixels are written.
                ry0,ry1=max(0,py-1),min(local.shape[0],py+2)
                rx0,rx1=max(0,px-1),min(local.shape[1],px+2)
                off_y0,off_y1=ry0-py+1,ry1-py+1;off_x0,off_x1=rx0-px+1,rx1-px+1
                candidates=donor_patches[:,off_y0:off_y1,off_x0:off_x1]
                observed_context=~all_mask[ry0:ry1,rx0:rx1]
                context=centered[ry0:ry1,rx0:rx1]
                cost=np.mean((candidates[:,observed_context]-context[observed_context])**2,axis=1) if observed_context.any() else np.zeros(len(donors))
                cost+=.1*(donor_level-t[py,px])**2
                order=np.argsort(cost)[:min(4,len(donors))];chosen=int(rng.choice(order))
                write=pending[ry0:ry1,rx0:rx1]
                generated[ry0:ry1,rx0:rx1][write]=candidates[chosen][write]
                pending[ry0:ry1,rx0:rx1][write]=False
            yy,xx=coordinates[:,0],coordinates[:,1]
            target=t[yy,xx]+float(np.median(values))+generated[yy,xx]
            blended=(1-gain)*base[y0:y1,x0:x1][yy,xx]+gain*target
            result[y0:y1,x0:x1][yy,xx]=np.clip(np.rint(blended),0,255).astype(np.uint8)
            log['donor_patches']=len(donors)
        logs.append(log)
    result[~mask]=gray[~mask]
    return result,logs
