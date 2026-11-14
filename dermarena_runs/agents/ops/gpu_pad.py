# 11-11: pad every visible GPU up to the most-used one, so SGLang tensor-parallel sees balanced free memory ("memory capacity is
# unbalanced" otherwise). Holds the padding until killed. Writes argv[1] when ready.
import sys, time, torch
n = torch.cuda.device_count(); used = []
for i in range(n):
    f, t = torch.cuda.mem_get_info(i); used.append(t - f)
target = max(used); hold = []
for i in range(n):
    pad = target - used[i] - int(0.3 * 2**30)
    if pad > 2**30:
        hold.append(torch.empty(pad, dtype=torch.uint8, device=f"cuda:{i}"))
        print(f"pad GPU {i}: {pad/2**30:.1f} GB", flush=True)
open(sys.argv[1], "w").write("ok"); print("padding ready", flush=True)
time.sleep(10**7)
