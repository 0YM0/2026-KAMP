import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils import ops
from ultralytics.models.yolo.detect.predict import DetectionPredictor
CAL_GROUPS=['capture_000','capture_009']
def iou(box,targets):
    targets=np.asarray(targets,dtype=float).reshape(-1,4)
    low=np.maximum(box[:2],targets[:,:2]);high=np.minimum(box[2:4],targets[:,2:4])
    area=np.prod(np.maximum(high-low,0),axis=1)
    ba=np.prod(np.maximum(np.asarray(box[2:4])-box[:2],0))
    ta=np.prod(np.maximum(targets[:,2:4]-targets[:,:2],0),axis=1)
    return area/np.maximum(ba+ta-area,1e-12)

def match(predictions,targets,threshold=.5):
    predictions=np.asarray(predictions,dtype=float).reshape(-1,5);targets=np.asarray(targets,dtype=float).reshape(-1,4)
    used=set();matches=[];false=[]
    for pi in np.argsort(-predictions[:,4],kind='stable'):
        overlap=iou(predictions[pi,:4],targets)
        for j in used:overlap[j]=-1
        gi=int(overlap.argmax()) if len(overlap) else -1
        if gi>=0 and overlap[gi]>=threshold:used.add(gi);matches.append((int(pi),gi,float(overlap[gi])))
        else:false.append(int(pi))
    return matches,false,[j for j in range(len(targets)) if j not in used]

def summarize(cases,threshold,match_iou=.5):
    tp=fp=fn=images_missed=0;details=[]
    for c in cases:
        pred=np.asarray(c['predictions'],dtype=float).reshape(-1,5);pred=pred[pred[:,4]>=threshold]
        hits,false,missing=match(pred,c['targets'],match_iou)
        tp+=len(hits);fp+=len(false);fn+=len(missing);images_missed+=bool(missing)
        details.append(dict(image_id=c['image_id'],camera=c['camera'],date=c['date'],tp=len(hits),fp=len(false),fn=len(missing),
            predictions=len(pred),gt_boxes=len(c['targets']),min_side_px=c['min_side_px']))
    return dict(confidence=threshold,match_iou=match_iou,images=len(cases),tp=tp,fp=fp,fn=fn,
                precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),fp_per_image=fp/max(len(cases),1),
                images_with_any_miss=images_missed),details

def targets(row):
    w,h=int(row['width']),int(row['height']);out=[]
    for line in Path(row['label_path']).read_text().splitlines():
        cls,x,y,bw,bh=map(float,line.split());assert cls==0
        out.append([(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h])
    return out

def roi_tensor(gray,proposal):
    p=np.asarray(proposal,float)[:4];center=(p[:2]+p[2:])/2
    side=int(math.ceil(np.clip(4*max(p[2:]-p[:2]),48,128)))
    left,top=np.floor(center-side/2).astype(int)
    crop=gray.crop((left,top,left+side,top+side)).resize((64,64),Image.Resampling.BILINEAR)
    pixels=np.asarray(crop,np.float32)/255
    pixels=np.clip((pixels-pixels.mean())/max(pixels.std(),.02),-4,4)/4
    # Relative proposal position is available at inference, independent of labels.
    yy,xx=np.mgrid[:64,:64];xx=left+(xx+.5)*side/64;yy=top+(yy+.5)*side/64
    mask=((xx>=p[0])&(xx<=p[2])&(yy>=p[1])&(yy<=p[3])).astype(np.float32)
    return np.stack([pixels,mask])

def input_boxes(raw):
    p=torch.tensor(raw['xywh_score'],dtype=torch.float32).reshape(-1,5)
    return torch.cat([ops.xywh2xyxy(p[:,:4]),p[:,4,None]],1).numpy()

def scaled_boxes(raw):
    p=input_boxes(raw).copy();p[:,:4]=ops.scale_boxes(tuple(raw['input_shape']),torch.from_numpy(p[:,:4].copy()),tuple(raw['original_shape'])).numpy()
    return p

def geometry_features(p):
    size=np.maximum(np.asarray(p[2:4])-p[:2],1e-3)
    return np.asarray([np.log(size[0]/16),np.log(size[1]/16),np.log(max(float(p[4]),.001))],np.float32)

def quality_nms(boxes,quality,threshold=.5,transfer_score=True,max_det=100):
    p=np.asarray(boxes,float).reshape(-1,5);q=np.asarray(quality,float).reshape(-1)
    assert len(p)==len(q) and np.isfinite(q).all()
    remaining=list(range(len(p)));out=[];chosen=[]
    while remaining and len(out)<max_det:
        index=max(remaining,key=lambda i:(q[i],p[i,4],-i))
        overlaps=iou(p[index,:4],p[remaining,:4]);cluster=[i for i,ov in zip(remaining,overlaps) if ov>threshold or i==index]
        row=p[index].copy()
        if transfer_score:row[4]=max(p[i,4] for i in cluster)
        out.append(row);chosen.append(index);remove=set(cluster);remaining=[i for i in remaining if i not in remove]
    order=sorted(range(len(out)),key=lambda i:(-out[i][4],i))
    return np.asarray([out[i] for i in order],float).reshape(-1,5),[chosen[i] for i in order]

class CapturePredictor(DetectionPredictor):
    captured=[]
    def postprocess(self,preds,img,orig_imgs,**kwargs):
        raw=preds[0] if isinstance(preds,(tuple,list)) else preds
        assert raw.shape[1]==5 and tuple(img.shape[-2:])==(640,640)
        assert isinstance(orig_imgs,list)
        for matrix,original,path in zip(raw,orig_imgs,self.batch[0]):
            p=matrix.detach().cpu().T.numpy().copy();p=p[p[:,4]>.001]
            self.captured.append(dict(image_id=Path(path).stem,input_shape=list(img.shape[-2:]),original_shape=list(original.shape[:2]),xywh_score=p.tolist()))
        return super().postprocess(preds,img,orig_imgs,**kwargs)

class Verifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(nn.Conv2d(2,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,24,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(24,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Flatten())
        self.head=nn.Sequential(nn.Linear(2051,128),nn.ReLU(),nn.Linear(128,2))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
    def forward(self,x,g):return self.head(torch.cat([self.features(x),g],1))

def sample_candidates(boxes,gt):
    p=np.asarray(boxes,float).reshape(-1,5)
    if not len(p):return np.zeros(0,dtype=int),np.zeros(0),np.zeros(0,dtype=int)
    overlap=np.stack([iou(b[:4],gt) for b in p]);best=overlap.max(1);near=overlap.argmax(1)
    selected=set()
    for gi in range(len(gt)):
        ids=[i for i in range(len(p)) if near[i]==gi and best[i]>.1]
        selected.update(sorted(ids,key=lambda i:(-best[i],-p[i,4],i))[:8])
    ambiguous=np.flatnonzero((best>.1)&(best<.5));selected.update(sorted(ambiguous,key=lambda i:(-p[i,4],i))[:24])
    negatives=sorted(np.flatnonzero(best<=.1),key=lambda i:(-p[i,4],i));selected.update(negatives[:24])
    rest=[i for i in negatives if i not in selected]
    if rest:selected.update(rest[i] for i in np.linspace(0,len(rest)-1,min(8,len(rest)),dtype=int))
    return np.asarray(sorted(selected),int),best,near

def pack(raw,train):
    xs=[];gs=[];ys=[];binary=[];groups=[];ledger=[]
    assert [r['image_id'] for r in raw]==[r['image_id'] for r in train]
    for image_index,(r,row) in enumerate(zip(raw,train)):
        p=scaled_boxes(r);gt=targets(row);indices,quality,nearest=sample_candidates(p,gt)
        gray=Image.open(row['image_path']).convert('L')
        for index in indices:
            b=p[index];q=float(quality[index]);label=1 if q>=.5 else 0 if q<=.1 else -1
            xs.append(roi_tensor(gray,b));gs.append(geometry_features(b));ys.append(q);binary.append(label)
            groups.append(image_index*4+int(nearest[index]) if q>.1 else -1)
            ledger.append(dict(image_id=row['image_id'],capture_group=row['capture_group'],role='calibration' if row['capture_group'] in CAL_GROUPS else 'fit',
                raw_index=int(index),original_score=float(b[4]),target_iou=q,binary_target=label))
    assert len(xs)>0
    return dict(x=torch.from_numpy(np.stack(xs)),g=torch.from_numpy(np.stack(gs)),q=torch.tensor(ys),binary=torch.tensor(binary),
                group=torch.tensor(groups),cal=torch.tensor([r['role']=='calibration' for r in ledger]),ledger=ledger)

def loss_parts(logits,quality,binary,group,rank=True):
    pos=binary==1;neg=binary==0;terms=[]
    if pos.any():terms.append(nn.functional.binary_cross_entropy_with_logits(logits[pos,0],torch.ones_like(logits[pos,0])))
    if neg.any():terms.append(nn.functional.binary_cross_entropy_with_logits(logits[neg,0],torch.zeros_like(logits[neg,0])))
    cls=sum(terms)/len(terms) if terms else logits[:,0].sum()*0
    q=logits[:,1].sigmoid();ql=nn.functional.smooth_l1_loss(q,quality,beta=.1)
    pairs=(group[:,None]==group[None,:])&(group[:,None]>=0)&((quality[:,None]-quality[None,:])>=.1)
    ranking=torch.relu(.05-(q[:,None]-q[None,:]))[pairs].mean() if rank and pairs.any() else q.sum()*0
    return cls+ql+.1*ranking,dict(binary=float(cls.detach()),quality=float(ql.detach()),rank=float(ranking.detach()))

def estimates(model,raw,image_path):
    p=scaled_boxes(raw)
    if not len(p):return np.empty((0,2))
    gray=Image.open(image_path).convert('L');result=[]
    with torch.no_grad():
        for start in range(0,len(p),128):
            chunk=p[start:start+128];x=torch.from_numpy(np.stack([roi_tensor(gray,b) for b in chunk]));g=torch.from_numpy(np.stack([geometry_features(b) for b in chunk]))
            result.append(model(x,g).sigmoid().numpy())
    return np.concatenate(result)

def apply_method(raw,est,method):
    boxes=input_boxes(raw);est=np.asarray(est);assert est.shape==(len(boxes),2) and np.isfinite(est).all()
    if method=='quality_keep':out,ids=quality_nms(boxes,est[:,1],threshold=.5,transfer_score=False)
    elif method=='verifier07':
        out=boxes.copy();out[:,4]*=est[:,0];out,_=quality_nms(out,np.zeros(len(out)),threshold=.7,transfer_score=False)
    elif method=='joint_cluster':
        out,ids=quality_nms(boxes,est[:,0]*est[:,1],threshold=.5,transfer_score=True)
        out[:,4]*=est[ids,0]
        out=out[np.argsort(-out[:,4],kind='stable')]
    else:raise ValueError(method)
    out=out[out[:,4]>=.001]
    if len(out):out[:,:4]=ops.scale_boxes(tuple(raw['input_shape']),torch.tensor(out[:,:4],dtype=torch.float32),tuple(raw['original_shape'])).numpy()
    return out.tolist()

def ap_metrics(cases):
    """Official Ultralytics IoU matching + AP integration on cached NMS outputs."""
    import torch
    from ultralytics.engine.validator import BaseValidator
    from ultralytics.utils.metrics import ap_per_class
    obj = SimpleNamespace(iouv=torch.linspace(.5, .95, 10))
    correct, confidence, nt = [], [], 0
    for c in cases:
        pred = np.asarray(c['predictions'], float).reshape(-1, 5)
        targets = np.asarray(c['targets'], float).reshape(-1, 4)
        overlap = np.array([iou(t, pred[:, :4]) for t in targets]).reshape(len(targets), len(pred))
        tp = BaseValidator.match_predictions(obj, torch.zeros(len(pred)), torch.zeros(len(targets)), torch.tensor(overlap))
        correct.append(tp.numpy()); confidence.extend(pred[:, 4]); nt += len(targets)
    ap = ap_per_class(np.concatenate(correct), np.array(confidence), np.zeros(len(confidence)), np.zeros(nt))[5]
    return dict(map50=float(ap[:, 0].mean()), map50_95=float(ap.mean()), ap_by_iou=ap[0].tolist(),
                note='Cached conf≥0.001 / NMS IoU0.7 / square640; official Ultralytics AP matching. Not the training log evaluation geometry.')

