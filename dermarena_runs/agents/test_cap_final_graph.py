"""Offline wiring test for DERMAGENT_CAP_FINAL: fake LLM that always requests 3 tool calls; cap 4 → cap_final node
must run once, answer, and end. No GPU, no server."""
import os, sys
os.environ["DERMAGENT_CAP_FINAL"] = "1"
sys.path.insert(0, "/fs04/scratch2/ub62/ssim0070/DermAgent")
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
import skin_agent.benchmark_agent as BA

@tool
def text_rag_retrieval(query: str) -> str:
    """fake text rag"""
    return f"result for {query}"

calls = {"n": 0}
class Fake:
    def bind_tools(self, tools, **k): return self
    def invoke(self, messages, **k):
        calls["n"] += 1
        last = messages[-1]
        if isinstance(last, HumanMessage) and "tool-call budget" in str(last.content):
            return AIMessage(content="FINAL_ANSWER:\n1. psoriasis\n2. eczema", response_metadata={"finish_reason": "stop"})
        return AIMessage(content="", tool_calls=[{"name": "text_rag_retrieval", "args": {"query": f"q{calls['n']}_{i}"}, "id": f"c{calls['n']}_{i}"} for i in range(3)],
                         response_metadata={"finish_reason": "tool_calls"})
BA._ReasoningChatOpenAI = lambda **kw: Fake()
BA.ChatOpenAI = lambda **kw: Fake()
agent = BA.create_benchmark_agent(tools=[text_rag_retrieval], model_name="fake", base_url="http://x", enabled_tools=["text_rag"],
                                  max_tool_calls=4, recursion_limit=30, enable_critic=True, critic_image_reinject=True,
                                  critic_llm_summary=False, max_critic_retries=2)
st = {"messages": [HumanMessage(content="case")], "current_image": None, "task_type": "open_diagnosis", "dataset": None,
      "options": None, "concepts": None, "tool_call_count": 0, "critic_retry_count": 0, "tools_called": [],
      "last_critic_feedback": None, "evidence_summaries": [], "image_reinject_count": 0, "tool_call_cache": {}, "text_rag_queries": []}
res = agent.invoke(st, config={"configurable": {"thread_id": "t"}, "recursion_limit": 30})
for m in res["messages"]:
    print(type(m).__name__, repr(str(m.content))[:90], getattr(m, "tool_calls", "") or "", (getattr(m, "response_metadata", {}) or {}).get("cap_final", ""))
last = res["messages"][-1]
assert last.response_metadata.get("cap_final") and "FINAL_ANSWER" in last.content, "cap_final did not produce the answer"
print("TEST OK: llm calls", calls["n"])
