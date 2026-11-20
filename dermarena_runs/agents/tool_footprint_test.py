"""DermAgent tool GPU-footprint test: can ONE tool process (all 6 tools) run under ~46 GB (a 48 GB L40S)?
Usage (on the L40S node, one GPU): CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /fs04/scratch2/ub62/ssim0070/dermagent/bin/python3 tool_footprint_test.py [threads=12] [n_images=60]
Loads the 6 tools like run_dermarena_v1.py, then calls every image tool on n clinical images with <threads> threads
(the per-class lock serialises each tool like in the run) and the text tools on 30 queries. Reports torch peak
allocated/reserved and nvidia-smi used MiB, sampled every second."""
import os, sys, json, time, threading, subprocess, random, glob
sys.path.insert(0, "/fs04/scratch2/ub62/ssim0070/DermAgent")
sys.path.insert(0, "/fs04/scratch2/ub62/ssim0070/DermAgent/scripts")
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
HC = "/fs04/scratch2/ub62/ssim0070/sglang/hf_cache"
os.environ.update(HF_HOME=HC, HUGGINGFACE_HUB_CACHE=HC + "/hub", TRANSFORMERS_CACHE=HC + "/hub")
import torch
from concurrent.futures import ThreadPoolExecutor
from skin_agent.benchmark_agent import create_tools
import run_dermarena_v1 as R

threads = int(sys.argv[1]) if len(sys.argv) > 1 else 12
n = int(sys.argv[2]) if len(sys.argv) > 2 else 60
Q = "/tmp/dermagent_qdrant_ft"
AG = "/fs04/scratch2/ub62/ssim0070/DermAgent"
if not os.path.isdir(Q + "/img/collection"):
    os.makedirs(Q, exist_ok=True)
    subprocess.run(f"cp -r {AG}/qdrant_storage {Q}/img && cp -r {AG}/qdrant_storage_text {Q}/text", shell=True, check=True)
for f in (Q + "/img/.lock", Q + "/text/.lock"):
    if os.path.exists(f): os.remove(f)
os.environ["QDRANT_PATH"], os.environ["QDRANT_TEXT_PATH"] = Q + "/img", Q + "/text"
peak = {"smi": 0}
stop = False
def sample():
    while not stop:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.split()
            peak["smi"] = max(peak["smi"], int(out[0]))
        except Exception: pass
        time.sleep(1)
threading.Thread(target=sample, daemon=True).start()
t0 = time.time()
tools = create_tools(device="cuda", enabled_tools=["panderm", "make", "dermogpt_vqa", "rag", "text_rag", "ontology"], preload=True, qdrant_path=Q + "/img")
bad = [t.name for t in tools if getattr(t, "_load_failed", False) or (hasattr(t, "is_initialized") and not t.is_initialized)]
print(f"loaded in {time.time()-t0:.0f}s; failed: {bad}; after load: torch alloc {torch.cuda.memory_allocated()/2**30:.1f} GiB, reserved {torch.cuda.memory_reserved()/2**30:.1f} GiB, nvidia-smi {peak['smi']} MiB", flush=True)
R.lock_tool_classes(tools)
T = {t.name: t for t in tools}
imgs = []
rows = [json.loads(l) for l in open("/fs04/scratch2/ub62/ssim0070/DermArena_release/DermArena_v1/tasks/T1.1_disease_screening/test.jsonl")][:400]
for r in rows:
    for im in r.get("images") or []:
        if im.get("modality") == "clinical_photo":
            p = R.RELEASE / im["image_path"]
            if p.exists(): imgs.append(R.rgb_path(p))
random.seed(0); random.shuffle(imgs); imgs = imgs[:n]
cands = "psoriasis, eczema, tinea corporis, lichen planus, basal cell carcinoma, melanoma, nevus, urticaria, cellulitis, scabies"
from skin_agent.utils.image_utils import generate_image_id, register_image_path
def reg(p):  # register an anonymous id exactly like run_open_diagnosis
    iid = generate_image_id(p); register_image_path(iid, p); return iid
ids = [reg(p) for p in imgs]
def job(i):
    iid = ids[i % len(ids)]
    T["panderm_classifier"]._run(image_path=iid, candidate_diseases=cands)
    T["make_concept_annotator"]._run(image_path=iid)
    T["dermogpt_vqa"]._run(image_path=iid, candidate_diseases=cands)
    T["rag_retrieval"]._run(image_path=iid, top_k=5)
    T["text_rag_retrieval"]._run(query="psoriasis versus lichen planus differentiating features", top_k=5)
    T["ontology_query"]._run(disease_name="psoriasis", query_type="hierarchy")
t1 = time.time()
with ThreadPoolExecutor(threads) as ex:
    list(ex.map(job, range(len(ids))))
stop = True
print(f"ran {len(ids)} cases x 6 tools in {time.time()-t1:.0f}s with {threads} threads")
print(f"PEAK torch allocated {torch.cuda.max_memory_allocated()/2**30:.1f} GiB, reserved {torch.cuda.max_memory_reserved()/2**30:.1f} GiB, nvidia-smi {peak['smi']/1024:.1f} GiB")
print("VERDICT:", "FITS 48GB L40S (<= 45 GiB)" if peak["smi"]/1024 <= 45 else "DOES NOT FIT 48GB")
