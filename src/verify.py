"""Verify packaged hashes, official labels, output schemas and reported metrics."""
import csv,json
from pathlib import Path
import numpy as np
from PIL import Image
from run import ROOT,read,sha,check_weights,rows,m

def main():
    check_weights();manifest=read(ROOT/'CHECKSUMS.json')
    for path,digest in manifest.items():assert sha(ROOT/path)==digest,path
    rr=rows();assert len(rr)==500 and len({r['image_id'] for r in rr})==500
    for r in rr:
        cid=r['image_id'];assert (ROOT/'data/raw/labels'/f'{cid}.txt').read_bytes()==Path(r['label_path']).read_bytes()
        assert Image.open(r['image_path']).size==(int(r['width']),int(r['height']))
    for stage in ['validation','test']:
        cases=read(ROOT/'results'/stage/'cases.json');metrics=read(ROOT/'results'/stage/'metrics.json')
        assert [c['image_id'] for c in cases]==[r['image_id'] for r in rows(stage)]
        for reported in metrics['metrics']:
            actual=m.summarize(cases,reported['confidence'],reported['match_iou'])[0]
            assert actual==reported
        for c,r in zip(cases,rows(stage)):
            p=np.asarray(c['predictions'],float).reshape(-1,5)
            assert np.isfinite(p).all() and len(p)<=100
            assert ((p[:,0]>=0)&(p[:,1]>=0)&(p[:,2]<=int(r['width']))&(p[:,3]<=int(r['height']))).all()
    raw=read(ROOT/'data/oof_train_raw.json');assert [r['image_id'] for r in raw]==[r['image_id'] for r in rows('train')]
    assert not ({r['image_id'] for r in raw}&{r['image_id'] for r in rows('test')})
    fit=read(ROOT/'data/shared_quality_oof301.json');assert len(fit)==301
    assert {c['image_id'] for c in fit}<={r['image_id'] for r in rows('train')}
    features=np.load(ROOT/'data/shared_quality_features.npz');assert features['peak'].shape==(86913,84)
    for stage in ['validation','test']:
        actual=read(ROOT/'results'/stage/'cases.json');reference=read(ROOT/'results'/stage/'reference_cases.json')
        assert [c['image_id'] for c in actual]==[c['image_id'] for c in reference]
        for a,b in zip(actual,reference):np.testing.assert_allclose(a['predictions'],b['predictions'],rtol=0,atol=1e-6)
    x=__import__('torch').arange(2*4*6).reshape(1,2,4,6);assert __import__('torch').equal(x.flip(-1).flip(-1),x)
    print('PASS',len(manifest),'file hashes;500 official pairs;360 OOF cases;68 validation/72 test schemas and metrics')
if __name__=='__main__':main()
