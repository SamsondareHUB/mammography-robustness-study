from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict
import csv, hashlib, tempfile
import numpy as np
import torch
from PIL import Image
from mammography_robustness.clean_inference import create_model
from mammography_robustness.robustness_inference import infer_matrix, check_coverage, RUNS
from mammography_robustness.degradations import CONDITIONS
from mammography_robustness.predictions import export_prediction_tables, write_image_metadata_csv, write_breast_metadata_csv

torch.set_num_threads(1)
with tempfile.TemporaryDirectory(prefix='synthetic_matrix_') as work:
    p=Path(work)
    records=[]; paths={}; classes={}
    for i,(patient,side,view,label,cls) in enumerate([('synthetic_1','L','CC',0,'normal'),('synthetic_1','L','MLO',1,'malignant'),('synthetic_2','R','CC',0,'benign')]):
        case=f'fixture_{i}'
        r=SimpleNamespace(case_id=case,patient_id=patient,breast_side=side,view=view,label=label)
        records.append(r); classes[case]=cls
        y,x=np.indices((83+i*11,139-i*13))
        paths[case]=p/(case+'.png')
        Image.fromarray(np.uint8((x*7+y*11+i*31)%256)).save(paths[case])
    # Synthetic checkpoints contain random initialization only, never trained weights.
    manifest_hash=hashlib.sha256(b'synthetic manifest').hexdigest()
    split_hash=hashlib.sha256(b'synthetic splits').hexdigest()
    checkpoints=[]
    for name,seed in RUNS:
        torch.manual_seed(seed)
        model=create_model(name)
        path=p/f'{name}_{seed}.pt'
        torch.save(dict(model=name,seed=seed,config=dict(model=name,seed=seed,image_size=224,batch_size=16,normalization='imagenet',pretrained=False),manifest_sha256=manifest_hash,splits_sha256=split_hash,model_state_dict=model.state_dict()),path)
        checkpoints.append(path)
    rows,hashes=infer_matrix(records,paths,classes,checkpoints,manifest_hash,split_hash)
    assert len(rows)==234 and len(hashes)==13
    print('PASS: 78 model/seed/condition combinations, 234 synthetic image predictions; shared input hashes and clean branch equality')
    for broken in (rows[:-1],rows+[rows[0]], [r for r in rows if (r.model,r.seed)!=RUNS[0]]):
        try: check_coverage(broken,records,classes)
        except ValueError: pass
        else: raise AssertionError('Coverage checker accepted incomplete or duplicated output')
    print('PASS: missing row, duplicate row, missing checkpoint rejected')
    export_prediction_tables(rows,p/'image.csv',p/'breast.csv',expected_conditions=CONDITIONS)
    write_image_metadata_csv(rows,p/'image_meta.csv',expected_conditions=CONDITIONS)
    write_breast_metadata_csv(rows,p/'breast_meta.csv',expected_conditions=CONDITIONS)
    def read(name):
        with (p/name).open(newline='') as f: return list(csv.DictReader(f))
    def keyed(rs,fields):
        out={tuple(r[k] for k in fields):r for r in rs}
        assert len(out)==len(rs)
        return out
    tail=('model','seed','degradation','severity')
    image=keyed(read('image.csv'),('case_id',)+tail)
    imeta=keyed(read('image_meta.csv'),('case_id',)+tail)
    breast=keyed(read('breast.csv'),('patient_id','breast_side')+tail)
    bmeta=keyed(read('breast_meta.csv'),('patient_id','breast_side')+tail)
    assert len(image)==len(imeta)==234 and len(breast)==len(bmeta)==156
    assert image.keys()==imeta.keys() and breast.keys()==bmeta.keys()
    groups=defaultdict(list)
    for key,row in image.items():
        assert all(imeta[key][k]==v for k,v in row.items())
        assert imeta[key]['image_id']==row['case_id'] and imeta[key]['original_class']==classes[row['case_id']]
        groups[tuple(row[k] for k in ('patient_id','breast_side')+tail)].append(row)
    for k,row in breast.items():
        views=groups[k]; meta=bmeta[k]
        assert float(row['score'])==max(float(r['score']) for r in views)
        assert int(row['label'])==max(int(r['label']) for r in views)==int(meta['label'])
        assert meta['view_status']==('paired' if len(views)==2 else 'single')
        assert meta['view_label_conflict']==str(int(len({r['label'] for r in views})>1))
    print('PASS: 234 image and 156 breast rows; exact metadata joins, maximum-view score, labels and view flags')
print('SYNTHETIC INTEGRATION: PASS. No private data or actual trained checkpoints used.')
