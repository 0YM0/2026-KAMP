"""Rebuild the frozen detector, group-excluded proposals and paired-flip verifier."""
import argparse,math,time,os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')
import yaml,torch
from run import ROOT,read,save,rows,m,check_weights

def reflected_roi(x):
    assert x.ndim==4 and x.shape[1]==2
    return x.flip(-1)

def paired_loss(model,x,g,q,binary,group):
    a=m.loss_parts(model(x,g),q,binary,group)[0]
    h=m.loss_parts(model(reflected_roi(x),g),q,binary,group)[0]
    return (a+h)/2

def dataset(rr,directory):
    for r in rr:
        for sub,key,ext in [('images','image_path','png'),('labels','label_path','txt')]:
            p=directory/sub/'fit'/f'{r["image_id"]}.{ext}';p.parent.mkdir(parents=True,exist_ok=True)
            if not p.exists():p.symlink_to(r[key])
    return 'images/fit'

def train_detector(device):
    args=yaml.safe_load((ROOT/'configs/detector_train.yaml').read_text());d=ROOT/'retraining/detector_data'
    dataset(rows('train'),d)
    for r in rows('validation'):
        for sub,key,ext in [('images','image_path','png'),('labels','label_path','txt')]:
            p=d/sub/'validation'/f'{r["image_id"]}.{ext}';p.parent.mkdir(parents=True,exist_ok=True)
            if not p.exists():p.symlink_to(r[key])
    config=d/'dataset.yaml';config.write_text(yaml.safe_dump(dict(path=str(d),train='images/fit',val='images/validation',names={0:'defect'})))
    args.update(data=str(config),device=device,project=str(ROOT/'retraining'),name='detector',exist_ok=False)
    model=m.YOLO(str(ROOT/'models/detector_coco_init.pt'));model.overrides.pop('save_dir',None);model.train(**args)

def train_oof(device):
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils.torch_utils import strip_optimizer
    class Trainer(DetectionTrainer):
        def final_eval(self):
            for p in [self.last,self.best]:
                if p.exists():strip_optimizer(p)
    allrows=rows('train');allraw={}
    for group in sorted({r['capture_group'] for r in allrows}):
        fit=[r for r in allrows if r['capture_group']!=group];held=[r for r in allrows if r['capture_group']==group]
        assert not ({r['date_group'] for r in fit}&{r['date_group'] for r in held})
        d=ROOT/'retraining/oof'/group;config=d/'dataset.yaml';dataset(fit,d/'dataset')
        config.write_text(yaml.safe_dump(dict(path=str(d/'dataset'),train='images/fit',val='images/fit',names={0:'defect'})))
        batches=math.ceil(len(fit)/8);epochs=math.ceil(495/batches)
        args=yaml.safe_load((ROOT/'configs/detector_train.yaml').read_text())
        args.update(data=str(config),epochs=math.ceil(1800/batches),warmup_epochs=135/batches,patience=math.ceil(1800/batches),
            val=False,save_period=-1,close_mosaic=0,augmentations=[],device=device,project=str(d/'runs'),name='detector',exist_ok=False)
        model=m.YOLO(str(ROOT/'models/detector_coco_init.pt'))
        model.overrides.pop('save_dir',None)
        fitids={r['image_id'] for r in fit}
        def progress(t):
            assert {m.Path(p).stem for p in t.train_loader.dataset.im_files}==fitids
            if t.epoch+1>=epochs:t.stop=True
        model.add_callback('on_fit_epoch_end',progress);model.train(trainer=Trainer,**args)
        teacher=m.YOLO(str(d/'runs/detector/weights/last.pt'));m.CapturePredictor.captured=[]
        for start in range(0,len(held),8):teacher.predict([r['image_path'] for r in held[start:start+8]],imgsz=640,rect=False,device='cpu',batch=8,conf=.001,iou=.7,max_det=100,verbose=False,predictor=m.CapturePredictor)
        raw=m.CapturePredictor.captured;assert [r['image_id'] for r in raw]==[r['image_id'] for r in held]
        allraw.update({r['image_id']:r for r in raw});save(d/'raw.json',raw)
        print(group,len(fit),len(held),epochs,flush=True)
    save(ROOT/'retraining/oof_train_raw.json',[allraw[r['image_id']] for r in allrows])

def train_verifier(raw_path):
    torch.use_deterministic_algorithms(True);data=m.pack(read(raw_path),rows('train'));data.pop('ledger')
    fit=torch.where(~data['cal'])[0];cal=torch.where(data['cal'])[0];torch.manual_seed(20261007)
    model=m.Verifier();opt=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.001)
    d=ROOT/'retraining/verifier';assert not (d/'history.json').exists(),'Use a new retraining directory';d.mkdir(parents=True,exist_ok=True)
    def calibration():
        model.eval();pieces=[]
        with torch.no_grad():
            for start in range(0,len(cal),256):
                ids=cal[start:start+256];pieces.append(model(data['x'][ids],data['g'][ids]))
            return m.loss_parts(torch.cat(pieces),data['q'][cal],data['binary'][cal],data['group'][cal],rank=False)[0].item()
    best=calibration();bestepoch=0;history=[dict(epoch=0,calibration_loss=best)];torch.save(model.state_dict(),d/'best.pt')
    for epoch in range(1,25):
        model.train();order=fit[torch.randperm(len(fit))];total=0
        for start in range(0,len(order),128):
            ids=order[start:start+128];x=data['x'][ids];g=data['g'][ids];q=data['q'][ids];binary=data['binary'][ids];group=data['group'][ids]
            opt.zero_grad();loss=paired_loss(model,x,g,q,binary,group)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();total+=loss.item()*len(ids)
        current=calibration();history.append(dict(epoch=epoch,train_loss=total/len(fit),calibration_loss=current))
        if current<best:best=current;bestepoch=epoch;torch.save(model.state_dict(),d/'best.pt')
        save(d/'history.json',history);print(history[-1],flush=True)
    torch.save(model.state_dict(),d/'last.pt');save(d/'status.json',dict(best_epoch=bestepoch,source=str(raw_path.relative_to(ROOT))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['detector','oof','verifier']);p.add_argument('--device',default='cpu');p.add_argument('--raw',default='data/oof_train_raw.json');a=p.parse_args()
    check_weights();torch.set_num_threads(4)
    if a.stage=='detector':train_detector(a.device)
    elif a.stage=='oof':train_oof(a.device)
    else:train_verifier(ROOT/a.raw)
