# A1 hand-back acceptance checks (HANDOFF_ABLATION_A1_20261116.md §6). Read-only.
import json, glob, os, random, collections
B='/fs04/scratch2/ub62/ssim0070/benchmark_runs'; WI=B+'/labels/findings_kept_img'
R='/fs04/scratch2/ub62/ssim0070/DermArena_release/DermArena_v1'
TASKS={'T1.1':(1980,2806),'T1.2':(1616,2030),'T1.3':(1498,2128)}
BASE={'qwen3.5-9b':'qwen3.5-9b','huatuogpt-3-27b':'huatuogpt-3-27b','glm-4.6v':'glm-4.6v'}
ok_all=True
def chk(c,msg):
    global ok_all
    if not c: ok_all=False
    return ('PASS ' if c else 'FAIL ')+msg
for m in BASE:
    s=f'{m}-findingskept-img'; print(f'\n=== {s}')
    for t,(n,full) in TASKS.items():
        d=f'{B}/dx/{s}/{t}'; out=[]
        rm=json.load(open(f'{d}/RUN_META.json'))
        out.append(chk(rm['code']['dx_benchmark/dx_infer.py']=='87886ed2c5a0','dx_infer hash '+rm['code']['dx_benchmark/dx_infer.py']))
        out.append(chk(rm['input']['path']==f'{WI}/{t}_findings_kept_img.jsonl','input '+os.path.basename(rm['input']['path'])))
        out.append(chk(rm['image_root']==R,'image_root'))
        bm=f'{B}/dx/{BASE[m]}/T1.1/RUN_META.json'
        if os.path.exists(bm): b=json.load(open(bm)); out.append(chk(b['sampling']==rm['sampling'],f'sampling == dx/{BASE[m]} ({rm["sampling"]})'))
        else: out.append(chk(False,f'no base RUN_META {bm}'))
        P=[json.loads(l) for l in open(f'{d}/predictions.jsonl') if l.strip()]
        ids=[p['_id'] for p in P]; inp=[json.loads(l)['_id'] for l in open(f'{WI}/{t}_findings_kept_img.jsonl') if l.strip()]
        out.append(chk(len(P)==n and len(set(ids))==len(ids) and set(ids)==set(inp),f'rows {len(P)}/{n}, dup {len(ids)-len(set(ids))}, ids==input {set(ids)==set(inp)}'))
        sigs=collections.Counter(p.get('run_sig') for p in P); out.append(chk(len(sigs)==1,f'run_sig {dict(sigs)}'))
        # images: count image parts in traces is heavy; use the prompt's image count recorded per row if present, else the POSTCHECK line
        pc=open(f'{d}/POSTCHECK.txt').read() if os.path.exists(f'{d}/POSTCHECK.txt') else ''
        im=[l for l in pc.splitlines() if 'images sent' in l]
        out.append(chk(bool(im) and 'missing 0' in im[-1], 'images: '+(im[-1].split('images sent')[-1].strip()[:60] if im else 'no POSTCHECK line')))
        unp=sum(1 for p in P if p.get('parse_ok') is False or p.get('status')=='unparsed' or not (p.get('content') or p.get('answer') or p.get('prediction')))
        mx=sum(1 for p in P if p.get('finish_reason')=='length')
        S=[json.loads(l) for l in open(f'{d}/scores.jsonl') if l.strip()]
        st=collections.Counter(x['status'] for x in S); unp2=st.get('unparsed',0)
        bad=(unp2+mx)/n
        out.append(chk(bad<=0.01,f'unparsed {unp2} + max-token {mx} = {100*bad:.2f}% (<=1%)'))
        sid={x['_id'] for x in S if x['status']!='no_prediction'}
        out.append(chk(len(S)==full and sid==set(ids),f'scores rows {len(S)}/{full}, non-no_prediction ids == prediction ids {sid==set(ids)}'))
        jf=0; unj=0
        for i in ids:
            f=f'{d}/judge/{i}.json'
            if not os.path.exists(f): unj+=1; continue
            try:
                v=json.load(open(f))
                if v.get('judge_failed') is True: jf+=1
            except Exception: unj+=1
        out.append(chk((unj+jf)/n<=0.02,f'unjudged {unj}, judge_failed(value True) {jf}'))
        jm=json.load(open(f'{d}/judge/JUDGE_META.json')); ref=json.load(open(f'{B}/dx/glm-4.6v-findingskept/{t}/judge/JUDGE_META.json'))
        same=jm['judge']==ref['judge'] and jm['sampling']==ref['sampling'] and jm['code']==ref['code'] and jm['max_attempts']==ref['max_attempts']
        out.append(chk(same,f'judge meta == dx/glm-4.6v-findingskept ({jm["judge"]}, sig {jm["sig"]} vs {ref["sig"]})'))
        print(f'  {t}: '+'\n       '.join(out))
print('\nALL PASS' if ok_all else '\nSOME CHECKS FAILED')
