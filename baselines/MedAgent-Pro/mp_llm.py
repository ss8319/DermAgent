"""One LLM-call helper for every MedAgent-Pro module (Planner, CodingAgent, Summary_Module, Deciders).

DermArena v1 paper run (2026-11-05). With no env set this is exactly the original call
(`client.chat.completions.create(model=<module default, gpt-4o>, messages=messages)`).
Env:
  MP_MODEL            backbone served name (e.g. Qwen/Qwen3.5-27B on the local SGLang server; OPENAI_BASE_URL)
  MP_SAMPLING_JSON    creator sampling, identical to the base-model arm (eval/sampling_profiles.py)
  MP_MAX_TOKENS       per-call output budget (32768 = base arm)
  MP_EOS_RETRIES      re-issue an immediate-EOS blank (finish=stop, <=2 tokens, empty) up to N times (= base arm)
  MP_SANITIZE_SPECIAL_TOKENS=1  rewrite literal multimodal placeholders in message TEXT (<|image_pad|> -> [image_pad]):
                      the server counts them as image tokens and answers HTTP 500 (= DermAgent's 11-06 fix). Model output
                      is fed into later prompts here, so one such string would otherwise drop the whole case.
Every call is appended, in full (images replaced by a size+hash placeholder), to the per-case trace list set
with start_case_trace() — the runner writes it to traces/<case>.json (save-every-step rule).
"""
import contextvars
import hashlib
import json
import os
import re
import time

_TRACE = contextvars.ContextVar("mp_trace", default=None)


def start_case_trace():
    lst = []
    _TRACE.set(lst)
    return lst


def _strip(messages):
    out = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            c2 = []
            for part in c:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    u = (part.get("image_url") or {}).get("url", "")
                    c2.append({"type": "image_url", "image_url": f"<base64 image, {len(u)} chars, "
                                                               f"md5 {hashlib.md5(u.encode()).hexdigest()[:10]}>"})
                else:
                    c2.append(part)
            c = c2
        out.append({"role": m.get("role"), "content": c})
    return out


_MM_TOKEN = re.compile(r"<\|(vision_start|vision_end|image_pad|video_pad|vision_pad|object_ref_start|object_ref_end|box_start|box_end)\|>")


def _sanitize(messages):
    """Text only; image parts are untouched. Same token list as skin_agent/benchmark_agent.py."""
    out = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            c = _MM_TOKEN.sub(r"[\1]", c)
        elif isinstance(c, list):
            c = [dict(p, text=_MM_TOKEN.sub(r"[\1]", p["text"])) if isinstance(p, dict) and p.get("type") == "text" else p
                 for p in c]
        out.append(dict(m, content=c))
    return out


def llm_kwargs(model_default="gpt-4o"):
    kw = {"model": os.environ.get("MP_MODEL") or model_default}
    if os.environ.get("MP_SAMPLING_JSON"):
        kw.update(json.loads(os.environ["MP_SAMPLING_JSON"]))
    if os.environ.get("MP_MAX_TOKENS"):
        kw["max_tokens"] = int(os.environ["MP_MAX_TOKENS"])
    return kw


def mp_create(client, messages, model_default="gpt-4o", module=""):
    kw = llm_kwargs(model_default)
    if os.environ.get("MP_SANITIZE_SPECIAL_TOKENS") == "1":
        messages = _sanitize(messages)
    tries = int(os.environ.get("MP_EOS_RETRIES", "0"))
    reissues = 0
    t0 = time.time()
    comp = client.chat.completions.create(messages=messages, **kw)
    while reissues < tries:
        ch = comp.choices[0]
        out = getattr(comp.usage, "completion_tokens", None) if comp.usage else None
        if (ch.message.content or "").strip() or ch.finish_reason != "stop" or (out is not None and out > 2):
            break
        reissues += 1
        comp = client.chat.completions.create(messages=messages, **kw)
    # MP_REASONING_FALLBACK=1 (DermArena runner, 2026-11-06) = base-arm rule llm_client._answer_from: blank content +
    # finish=stop + reasoning > 20 chars -> the reasoning is the answer (copied into message.content so every module,
    # which reads .content only, sees it). Off = original behaviour.
    answer_source = "content"
    ch0 = comp.choices[0]
    if not (ch0.message.content or "").strip():
        answer_source = None
        rs = getattr(ch0.message, "reasoning_content", None) or \
            (getattr(ch0.message, "model_extra", None) or {}).get("reasoning_content") or ""
        if os.environ.get("MP_REASONING_FALLBACK") == "1" and ch0.finish_reason == "stop" and len(rs.strip()) > 20:
            ch0.message.content = rs
            answer_source = "reasoning"
    tr = _TRACE.get()
    if tr is not None:
        ch = comp.choices[0]
        tr.append({
            "module": module, "model": kw["model"], "params": {k: v for k, v in kw.items() if k != "model"},
            "messages": _strip(messages), "content": ch.message.content,
            "reasoning_content": getattr(ch.message, "reasoning_content", None)
                                 or (getattr(ch.message, "model_extra", None) or {}).get("reasoning_content"),
            "finish_reason": ch.finish_reason, "usage": comp.usage.model_dump() if comp.usage else None,
            "eos_reissues": reissues, "answer_source": answer_source, "latency_s": round(time.time() - t0, 2),
        })
    return comp
