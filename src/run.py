"""Portable preprocessing, fixed-split inference and evaluation."""
import argparse,csv,json,hashlib,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'runtime');os.environ['MPLCONFIGDIR']=str(ROOT/'runtime/matplotlib');os.environ['YOLO_OFFLINE']='true'
import cv2,numpy as np,torch
from PIL import Image,ImageDraw
import model as m
from xray_marker_processing import color_mask
from xray_texture_processing import texture_restore

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2))
def rows(stage=None):
    rr=list(csv.DictReader((ROOT/'data/manifest.csv').open()))
    rr=[r for r in rr if stage is None or r['split']==stage]
    return [dict(r,image_path=str(ROOT/r['image_path']),label_path=str(ROOT/r['label_path'])) for r in rr]
def check_weights():
    for n,digest in read(ROOT/'configs/model.json')['weights_sha256'].items():assert sha(ROOT/'models'/n)==digest,n
def preprocess(source):
    rgb=np.asarray(Image.open(source).convert('RGB'));gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);mask=color_mask(rgb,10)
    restored,_=texture_restore(gray,mask,'adaptive',1.,20261006)
    assert np.array_equal(restored[~mask],gray[~mask])
    return restored
def case(r):
    return dict(image_id=r['image_id'],image_path=str(Path(r['image_path']).relative_to(ROOT)),targets=m.targets(r),camera=r['camera'],date=r['date_group'],min_side_px=float(r['min_box_side_px']))
def inference(paths,output):
    check_weights();torch.set_num_threads(4)
    detector=m.YOLO(str(ROOT/'models/detector.pt'));verifier=m.Verifier();verifier.load_state_dict(torch.load(ROOT/'models/verifier.pt',weights_only=True));verifier.eval()
    m.CapturePredictor.captured=[]
    for start in range(0,len(paths),8):detector.predict([str(p) for p in paths[start:start+8]],imgsz=640,rect=False,device='cpu',batch=8,conf=.001,iou=.7,max_det=100,verbose=False,predictor=m.CapturePredictor)
    from shared_quality import Refiner
    quality=Refiner(ROOT/'models/shared_quality.pt')
    raw=m.CapturePredictor.captured;assert len(raw)==len(paths);results=[]
    for p,r in zip(paths,raw):
        pred=m.apply_method(r,m.estimates(verifier,r,str(p)),'joint_cluster')
        pred,box_changes=quality.predict(str(p),pred)
        high=any(b[4]>=.2 for b in pred);weak=any(b[4]>=.02 for b in pred)
        results.append(dict(image_id=Path(p).stem,predictions=pred,refinement=box_changes,route='alarm_hold' if high else 'weak_review' if weak else 'unconfirmed_no_candidate'))
    save(output,results);return results
def evaluate(stage):
    assert stage in ['validation','test'];rr=rows(stage)
    pred=inference([r['image_path'] for r in rr],ROOT/'results'/stage/'predictions.json');cs=[];flat=[]
    for r,p in zip(rr,pred):
        assert r['image_id']==p['image_id'];cs.append(dict(case(r),**p))
        for box in p['predictions']:flat.append(dict(image_id=p['image_id'],x1=box[0],y1=box[1],x2=box[2],y2=box[3],score=box[4],class_id=0,route=p['route']))
    save(ROOT/'results'/stage/'cases.json',cs)
    metrics=[m.summarize(cs,t,ov)[0] for t in [.2,.02] for ov in [.5,.75]]
    ap=m.ap_metrics(cs);ap['note']='YOLO11s + paired-flip ROI + joint_cluster + shared-quality coordinate refinement; square640, floor.001; official Ultralytics AP matching.'
    save(ROOT/'results'/stage/'metrics.json',dict(metrics=metrics,ap=ap,frozen_weights=read(ROOT/'configs/model.json')['weights_sha256'],previous_test_exposure=True if stage=='test' else None))
    with (ROOT/'results'/stage/'predictions.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['image_id','x1','y1','x2','y2','score','class_id','route']);w.writeheader();w.writerows(flat)
    with (ROOT/'results'/stage/'image_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['image_id','prediction_count','count_at_020','count_at_002','route']);w.writeheader()
        for c in cs:w.writerow(dict(image_id=c['image_id'],prediction_count=len(c['predictions']),count_at_020=sum(b[4]>=.2 for b in c['predictions']),count_at_002=sum(b[4]>=.02 for b in c['predictions']),route=c['route']))
    for c in cs:
        im=Image.open(ROOT/c['image_path']).convert('RGB');draw=ImageDraw.Draw(im)
        for box in c['predictions']:
            if box[4]>=.02:draw.rectangle(box[:4],outline=(240,70,0) if box[4]>=.2 else (240,150,0),width=1)
        path=ROOT/'results'/stage/'overlays'/f'{c["image_id"]}.png';path.parent.mkdir(exist_ok=True);im.save(path)
    print(stage,metrics,ap,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['preprocess','validation','test','predict']);p.add_argument('--input',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    if a.stage=='preprocess':
        for i,r in enumerate(rows()):
            out=ROOT/'data/processed/images'/f'{r["image_id"]}.png';old=sha(out);Image.fromarray(preprocess(ROOT/'data/raw/images'/f'{r["image_id"]}.bmp')).save(out);assert sha(out)==old,r['image_id']
            if (i+1)%50==0:print('preprocessed',i+1,flush=True)
    elif a.stage=='predict':
        assert a.input and a.output
        paths=sorted(a.input.glob('*')) if a.input.is_dir() else [a.input];images=[]
        for source in paths:
            if source.suffix.lower() not in ['.bmp','.png','.jpg','.jpeg']:continue
            dest=ROOT/'runtime/predict_inputs'/f'{source.stem}.png';dest.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(preprocess(source)).save(dest);images.append(dest)
        assert images,'No input images';inference(images,a.output)
    else:evaluate(a.stage)
