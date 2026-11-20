import json, math, sys
B="/fs04/scratch2/ub62/ssim0070/benchmark_runs/dx/"
def load(slug,t):
    return {r["_id"]:r for r in map(json.loads,open(f"{B}{slug}/{t}/scores.jsonl"))}
def mc(b,c):
    m=b+c
    return 1.0 if not m else min(1.0,2*sum(math.comb(m,k) for k in range(min(b,c)+1))/2**m)
for t in ("T1.1","T1.2"):
    A=load("dermagent-qwen3.8-flash-next-clean",t); Bs=load("qwen3.8-flash-next",t)
    ids=sorted(set(A)&set(Bs)); n=len(ids)
    up=sum(A[i]["status"]=="unparsed" for i in ids); ub=sum(Bs[i]["status"]=="unparsed" for i in ids)
    print(f"{t} n={n} unparsed agent {up} base {ub}")
    for lab,k,thr in (("top-5 main","top5",1),("top-1 main","top1",1),("top-5 exact","top5",2),("top-1 exact","top1",2)):
        a=[(A[i].get(k) or 0)>=thr for i in ids]; b=[(Bs[i].get(k) or 0)>=thr for i in ids]
        x=sum(p and not q for p,q in zip(a,b)); y=sum(q and not p for p,q in zip(a,b))
        d=(x-y)/n; se=math.sqrt(max(x+y-(x-y)**2/n,0))/n
        print(f"  {lab}: agent {100*sum(a)/n:.1f} base {100*sum(b)/n:.1f} diff {100*d:+.1f} [{100*(d-1.96*se):+.1f},{100*(d+1.96*se):+.1f}] p={mc(x,y):.3g}")
