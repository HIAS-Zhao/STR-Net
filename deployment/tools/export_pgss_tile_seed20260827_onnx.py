#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import numpy as np, torch, onnx, onnxruntime as ort
ROOT=Path('deployment/pgss/seed20260827');SRC=Path('V1_SUPINIT_NOKD_765_20260908');CK=SRC/'SEED_20260827/supinit_nokd_40E/model/model_best.pth';OUT=ROOT/'models/pgss_w32_765_seed20260827_tile48.onnx'
sys.path.insert(0,str(SRC/'code'));from pgss_model import PGSSTinyV1,forward_boundary_aligned_tile
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
class Tile(torch.nn.Module):
 def __init__(self,m):super().__init__();self.m=m
 def forward(self,image,valid): return forward_boundary_aligned_tile(self.m,image,valid)[:,:,8:40,8:40]
d=torch.load(CK,map_location='cpu',weights_only=False);m=PGSSTinyV1(3,int(d['width']),int(d['head_width']));m.load_state_dict(d['model_state_dict']);w=Tile(m.eval()).eval();x=torch.randn(1,3,48,48);v=torch.ones(1,1,48,48,dtype=torch.bool);OUT.parent.mkdir(parents=True,exist_ok=True)
torch.onnx.export(w,(x,v),OUT,input_names=['image','valid'],output_names=['pgss_core_logit'],dynamic_axes={'image':{0:'batch'},'valid':{0:'batch'},'pgss_core_logit':{0:'batch'}},opset_version=17,do_constant_folding=True)
g=onnx.load(OUT);onnx.checker.check_model(g);s=ort.InferenceSession(str(OUT),providers=['CPUExecutionProvider']);y=s.run(None,{'image':x.numpy(),'valid':v.numpy()})[0]; rec={'name':'PGSS-W32-765 tile verifier','checkpoint':str(CK.resolve()),'checkpoint_sha256':sha(CK),'onnx':str(OUT.resolve()),'onnx_sha256':sha(OUT),'inputs':{'image':{'shape':['batch',3,48,48],'dtype':'float32'},'valid':{'shape':['batch',1,48,48],'dtype':'bool'}},'output':{'name':'pgss_core_logit','shape':['batch',1,32,32],'dtype':'float32','raw_logit':True},'onnx_checker':'PASS','onnxruntime':'PASS','finite_onnx':bool(np.isfinite(y).all()),'precision_target':'TensorRT FP16','engine':'NOT_BUILT'}
(root:=ROOT/'audit').mkdir(exist_ok=True); (root/'TILE_EXPORT_AUDIT.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2)+'\n');print(json.dumps(rec,ensure_ascii=False,indent=2))
