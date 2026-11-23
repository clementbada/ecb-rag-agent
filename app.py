"""Demo interface (Streamlit).

    streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from ecbrag import config  # noqa: E402
from ecbrag.agent.graph import build_agent  # noqa: E402
from ecbrag.tracing import flush, traced  # noqa: E402

st.set_page_config(page_title="ECB Speeches Analyst", layout="wide")

st.markdown("""
<style>
  html, body, [class*="css"] { font-family: "Inter", "Helvetica Neue", Arial, sans-serif; }
  .block-container { padding-top: 2.5rem; max-width: 1180px; }
  h1 { font-weight: 300; letter-spacing: -0.02em; }
  .kicker { text-transform: uppercase; letter-spacing: 0.18em; font-size: 0.72rem; color: #8a8a8a; }
  .answer { border-left: 2px solid #1f2a44; padding: 0.4rem 0 0.4rem 1.2rem; font-size: 1.05rem; line-height: 1.65; }
  .metric-label { text-transform: uppercase; letter-spacing: 0.12em; font-size: 0.68rem; color: #8a8a8a; }
  .metric-value { font-size: 1.5rem; font-weight: 300; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading index")
def get_agent():
    return build_agent()


agent = get_agent()
cfg = config.best_config()

st.markdown('<div class="kicker">Knowledge base · 500 ECB speeches · 2020 – 2026</div>', unsafe_allow_html=True)
st.title("ECB Speeches Analyst")
st.caption(f"Retrieval: {cfg['mode']} on index {cfg['variant']} · Model: {config.CHAT_MODEL}")

examples = [
    "What share of day-to-day transactions in the euro area was made in cash in 2025?",
    "How did euro area headline inflation change between January and May 2026?",
    "Which ECB speaker gave the most speeches in 2025?",
    "What has Frank Elderson said about nature-related risks for banks?",
]
choice = st.selectbox("Example questions", [""] + examples)
question = st.text_input("Your question", value=choice)

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Searching the speeches"):
        run = traced(agent.run, "agent-run")(question.strip())
        flush()

    st.markdown(f'<div class="answer">{run["answer"]}</div>', unsafe_allow_html=True)
    st.write("")

    cols = st.columns(5)
    stats = [("Iterations", run["iterations"]), ("Tool calls", run["tool_calls"]),
             ("Tokens", f"{run['tokens_in'] + run['tokens_out']:,}"),
             ("Cost", f"${run['cost_usd']:.4f}"), ("Latency", f"{run['latency_ms'] / 1000:.1f} s")]
    for col, (label, value) in zip(cols, stats):
        col.markdown(f'<div class="metric-label">{label}</div><div class="metric-value">{value}</div>',
                     unsafe_allow_html=True)

    st.subheader("Sources")
    chunks = {c["chunk_id"]: c for c in agent.tools.retriever.index.chunks}
    docs = {d["doc_id"]: d for d in agent.tools.docs}
    rows = []
    for s in run["sources"]:
        meta = chunks.get(s) or docs.get(s) or {}
        rows.append({"id": s, "speaker": meta.get("speakers", ""), "date": meta.get("date", ""),
                     "title": meta.get("title", "")})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        for s in run["sources"]:
            if s in chunks:
                with st.expander(f"Passage {s}"):
                    st.write(chunks[s]["text"])
    else:
        st.caption("No source cited.")

    st.subheader("Agent trace")
    trace = [{"step": t.get("iteration"), "type": t["type"], "name": t.get("name", "llm"),
              "status": t.get("status", ""), "tokens": t.get("tokens_in", 0) + t.get("tokens_out", 0),
              "latency_ms": t["latency_ms"], "detail": t.get("arguments", "")} for t in run["trace"]]
    st.dataframe(pd.DataFrame(trace), hide_index=True, use_container_width=True)
    st.caption(f"Terminated by: {run['terminated_by']} · hallucinated tool calls: {run['hallucinated_calls']} "
               f"· tool errors: {run['tool_errors']}")
