import json, torch, numpy as np, os
os.environ['HF_HUB_OFFLINE']='1'
from transformers import AutoTokenizer, AutoModel
SP='${SCISLOP_ROOT}/scratch'
P=json.load(open(f'{SP}/pairs143_text.json'))
tok=AutoTokenizer.from_pretrained('allenai/specter2_base'); mod=AutoModel.from_pretrained('allenai/specter2_base').eval()
torch.set_num_threads(32)
def emb(texts):
    out=[]
    for i in range(0,len(texts),16):
        b=tok(texts[i:i+16],padding=True,truncation=True,max_length=512,return_tensors='pt')
        with torch.no_grad(): h=mod(**b).last_hidden_state[:,0,:]
        out.append(torch.nn.functional.normalize(h,dim=-1))
    return torch.cat(out).numpy()
sep=tok.sep_token
A=emb([p['fars_title']+sep+p['fars_abs'] for p in P]); H=emb([p['hu_title']+sep+p['hu_abs'] for p in P])
S=A@H.T  # S[i,j]= cos(FARS i, anchor j)
n=len(P); diag=np.diag(S)
off=S[~np.eye(n,dtype=bool)]
rank_a=[int(1+np.sum(S[i,:]>S[i,i])) for i in range(n)]   # rank of assigned anchor among anchors for FARS i
rank_f=[int(1+np.sum(S[:,j]>S[j,j])) for j in range(n)]   # rank of FARS among FARS for anchor j
res=dict(n=n, mean_assigned=float(diag.mean()), median_assigned=float(np.median(diag)), mean_random=float(off.mean()), median_random=float(np.median(off)),
 frac_assigned_above_random_median=float((diag>np.median(off)).mean()),
 top1_a=int(sum(r==1 for r in rank_a)), top3_a=int(sum(r<=3 for r in rank_a)), top10_a=int(sum(r<=10 for r in rank_a)), median_rank_a=float(np.median(rank_a)),
 top1_f=int(sum(r==1 for r in rank_f)), top3_f=int(sum(r<=3 for r in rank_f)), top10_f=int(sum(r<=10 for r in rank_f)), median_rank_f=float(np.median(rank_f)))
# by tier and pool
import collections
for key in ['tier','pool']:
    g=collections.defaultdict(list)
    for i,p in enumerate(P): g[p[key]].append(i)
    res['by_'+key]={str(k):dict(n=len(v), mean_cos=float(diag[v].mean()), top1=int(sum(rank_a[i]==1 for i in v)), top3=int(sum(rank_a[i]<=3 for i in v)), median_rank=float(np.median([rank_a[i] for i in v]))) for k,v in g.items()}
# percentile of assigned cos within each FARS row's distribution
res['worst_pairs']=[dict(code=P[i]['code'],cos=float(diag[i]),rank=rank_a[i],tier=P[i]['tier']) for i in np.argsort(diag)[:10]]
json.dump(dict(res, diag=diag.tolist(), rank_a=rank_a, rank_f=rank_f, codes=[p['code'] for p in P]), open(f'{SP}/specter_align.json','w'), indent=1)
print(json.dumps(res,indent=1))
