# ui/streamlit_app.py
import streamlit as st
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from local_reasoner import LocalMemoryAgent

# Page config
st.set_page_config(
    page_title="🧠 AI Memory Agent",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize agent once
if "agent" not in st.session_state:
    st.session_state.agent = LocalMemoryAgent(device="cuda")
    st.session_state.messages = []

agent = st.session_state.agent

# Sidebar
with st.sidebar:
    st.title("🛠️ Agent Controls")
    if st.button("🧹 Clear All Memory"):
        agent.memory = []
        agent.facts = {}
        agent._save_memory()
        agent._save_facts()
        agent._rebuild_chroma_index()
        st.session_state.messages = []
        st.success("Memory cleared!")
    
    topic = st.text_input("🗑️ Forget Topic", placeholder="e.g., salary")
    if st.button("Forget") and topic:
        if agent.forget_memory(topic):
            st.success(f"Forgotten memories about '{topic}'")
        else:
            st.warning(f"No memories found about '{topic}'")
    
    st.subheader("📌 Known Facts")
    st.json(agent.facts)

    st.subheader("🧠 Memory Entries")
    st.write(f"Total: {len(agent.memory)}")
    with st.expander("View Memories"):
        for i, m in enumerate(agent.memory):
            st.caption(f"[{i+1}] {m}")

# Main chat
st.title("🧠 AI Memory Agent (Local, Private, GPU-Accelerated)")
st.caption("Speak naturally. Agent remembers facts, avoids hallucination, and runs 100% on your machine.")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input
if prompt := st.chat_input("Ask anything..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = agent.answer_with_memory(prompt)
            answer = result["answer"]
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})