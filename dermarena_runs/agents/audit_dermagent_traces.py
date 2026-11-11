import json,glob,ast,re,sys,collections as C
root=sys.argv[1]
def lit(s):
    if not isinstance(s,str): return s
    try: return ast.literal_eval(s)
    except Exception: return None
agg=C.Counter(); sizes=C.Counter(); errs=C.Counter(); per=[]
for t in ["T1.1","T1.2"]:
    preds={json.loads(l)["_id"]:json.loads(l) for l in open(f"{root}/{t}/predictions.jsonl")}
    for f in sorted(glob.glob(f"{root}/{t}/traces/*.json")):
        tr=json.load(open(f)); m=tr["messages"]; cid=tr["_id"]
        first=m[0]["content"] if isinstance(m[0]["content"],str) else json.dumps(m[0]["content"]); idmod={i:mo for mo,i in re.findall(r"image \d+ \(([^)]*)\): (IMG_[0-9a-f]{8})",first)}
        aftercrit=False
        crit=foc=fin=reinj=0
        for i,x in enumerate(m):
            if not isinstance(x.get("content"),str): x["content"]=json.dumps(x.get("content"))
            c=x.get("content") or ""
            if x["type"] in("HumanMessage","SystemMessage") and i>0:
                if "Critic Review" in c: crit+=1; aftercrit=True
                if "Focused Re-diagnosis" in c: foc+=1
                if "Final Attempt" in c: fin+=1
                if "image_url" in c: reinj+=1
            if x["type"]=="SystemMessage" and i>0: agg["midconv_system"]+=1
            if x["type"]=="AIMessage":
                for tc in lit(x.get("tool_calls")) or []:
                    a=tc.get("args",{}); n=tc["name"]; agg["call:"+n]+=1
                    ip=a.get("image_path")
                    if ip is not None:
                        if ip not in idmod: errs[f"{n} bad id {ip!r}"]+=1
                        else: agg["img_call_mod:"+idmod[ip]]+=1
                    if n in("panderm_classifier",) or (n=="dermogpt_vqa" and a.get("candidate_diseases")):
                        cd=a.get("candidate_diseases"); k=len([s for s in re.split(r",",cd) if s.strip()]) if isinstance(cd,str) else len(cd or [])
                        sizes[(n,"after_critic" if aftercrit else "round1",k)]+=1
            if x["type"]=="AIMessage" and "<tool_call>" in c: agg["toolcall_text_in_content"]+=1
            if x["type"]=="ToolMessage" and c.startswith("Error"):
                errs[x.get("name")+": "+c[:90].replace("\n"," ")]+=1
        p=preds[cid]
        per.append((t,cid,p["parsed"],p["finish_reason"],p["n_tool_calls"],p["n_llm_calls"],crit,foc,fin,reinj,p["n_images_case"],p["n_images_tool_usable"],p["backbone_reissues"]))
        agg["cases"]+=1; agg["critic_cases"]+=crit>0; agg["focused"]+=foc>0; agg["final_attempt"]+=fin>0; agg["reinject_cases"]+=reinj>0
print(dict(agg)); print("list sizes:",sorted(sizes.items())); print("errors:"); [print("  ",v,k) for k,v in errs.most_common()]
print("t id parsed finish tools llm crit foc fin reinj nimg usable reissue")
for r in per: print(*r)
