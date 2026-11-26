"""CPU inference integration using the frozen degradation module."""
from collections import Counter
from PIL import Image
import torch
from .degradations import CONDITIONS, model_input, input_hash
from .clean_inference import predict_images, _load_checkpoint, create_model
from .predictions import ImagePrediction

RUNS = tuple((name, seed) for name in ('mobilenet_v3_small', 'resnet18') for seed in (42,43,44))

def shared_batch(records, image_paths, condition):
    tensors, hashes = [], {}
    for row in records:
        with Image.open(image_paths[row.case_id]) as im:
            tensor, digest = model_input(im, row.case_id, *condition)
        tensors.append(tensor)
        hashes[row.case_id] = (digest, input_hash(tensor))
    return torch.stack(tensors), hashes

def score_batch(model, tensors, batch_size=16):
    model.eval()
    scores=[]
    with torch.inference_mode():
        for batch in tensors.split(batch_size):
            logits=model(batch)
            if logits.shape != (len(batch),2):
                raise ValueError('Invalid logits shape')
            scores.extend(torch.softmax(logits,dim=1)[:,1].tolist())
    if len(scores)!=len(tensors) or any(not 0<=v<=1 for v in scores):
        raise ValueError('Invalid scores')
    return scores

def check_coverage(predictions, records, classes):
    expected={(r.case_id, m, s, d, v) for r in records for m,s in RUNS for d,v in CONDITIONS}
    counts=Counter((p.case_id,p.model,p.seed,p.degradation,p.severity) for p in predictions)
    if set(counts)!=expected or any(n!=1 for n in counts.values()):
        raise ValueError('Incomplete, duplicate or extra case/run/condition keys')
    source={r.case_id:r for r in records}
    for p in predictions:
        r=source[p.case_id]
        if (p.patient_id,p.breast_side,p.view,p.label,p.image_id,p.original_class)!=(r.patient_id,r.breast_side,r.view,r.label,r.case_id,classes[r.case_id]) or not 0<=p.score<=1:
            raise ValueError('Invalid prediction metadata or score')

def infer_matrix(records, image_paths, classes, checkpoint_paths, manifest_hash, split_hash):
    models={}
    for path in checkpoint_paths:
        state,name,seed,size,batch=_load_checkpoint(path,manifest_hash,split_hash)
        if (name,seed) in models or size!=224 or batch!=16 or state['config'].get('normalization')!='imagenet':
            raise ValueError('Checkpoint configuration mismatch')
        model=create_model(name)
        model.load_state_dict(state['model_state_dict'],strict=True)
        model.eval()
        models[name,seed]=model
    if set(models)!=set(RUNS): raise ValueError('Incorrect checkpoint matrix')
    predictions=[]
    hashes={}
    for condition in CONDITIONS:
        tensors,digests=shared_batch(records,image_paths,condition)
        hashes[condition]=digests
        before=input_hash(tensors[0])
        for (name,seed),model in models.items():
            if condition==('clean','none'):
                scores=predict_images(model,records,image_paths,224,16,torch.device('cpu'),normalization='imagenet')
                if scores!=score_batch(model,tensors): raise ValueError('Integrated clean branch differs from original path in this runtime')
            else:
                scores=score_batch(model,tensors)
            if any(input_hash(t)!=digests[r.case_id][1] for r,t in zip(records,tensors,strict=True)):
                raise ValueError('Shared input changed during inference')
            predictions.extend(ImagePrediction(case_id=r.case_id,patient_id=r.patient_id,breast_side=r.breast_side,image_id=r.case_id,view=r.view,label=r.label,score=score,model=name,degradation=condition[0],severity=condition[1],seed=seed,original_class=classes[r.case_id]) for r,score in zip(records,scores,strict=True))
    check_coverage(predictions,records,classes)
    return predictions,hashes
