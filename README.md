# 🧠 Memora — Your Private, Memory-Aware AI Agent

> **Runs 100% on your machine. Remembers what matters. Never lies.**  
> Built for ethical hackers, AI researchers, and privacy-first users.

[![Streamlit](https://static.streamlit.io/badge/streamlit.svg)](https://streamlit.io)
[![Ollama](https://img.shields.io/badge/Ollama-Supported-orange)](https://ollama.com)
[![Privacy](https://img.shields.io/badge/Privacy-Local_Only-green)](https://github.com/NarlaRohan050/AI-Agent)

Memora is a **hybrid GPU/CPU AI agent** that combines **Mistral + Mistral-Instruct** for truthful, memory-augmented reasoning — with **zero data sent to the cloud**.

## ✨ Features

- ✅ **Persistent memory** using ChromaDB + sentence embeddings  
- ✅ **Fact extraction**: name, interests, goals — never invented  
- ✅ **Hallucination prevention**: refuses to answer unknowns  
- ✅ **Dynamic GPU/CPU load balancing** — avoids VRAM crashes  
- ✅ **Secure memory deletion**: `/forget salary` removes all traces  
- ✅ **Name conflict resolution**: “I’m Jordan” → confirmation prompt  
- ✅ **Redundancy blocking**: avoids duplicate memories  
- ✅ **NSA-ready**: no telemetry, no cloud, full data sovereignty  

## 🚀 How to Run (Local Only)

> ⚠️ **Critical**: Do **NOT** run inside **OneDrive**, **Dropbox**, or any cloud-synced folder.  
> Move your project to a local path like `C:\AI-Agent-main` to avoid `PermissionError`.

### 1. Install Ollama
- Download and install: [https://ollama.com/download/OllamaSetup.exe](https://ollama.com/download/OllamaSetup.exe)
- Restart your terminal after installation

### 2. Pull the quantized Mistral model
```bash
ollama pull mistral:7b-instruct-v0.2-q5_K_M

git clone https://github.com/NarlaRohan050/AI-Agent.git
cd AI-Agent
pip install -r requirements.txt
streamlit run ui/streamlit_app.py
python test_agent_max_capacity.py
