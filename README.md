# 🧠 Memora — Your Private, Memory-Aware AI Agent

> **Runs 100% on your machine. Remembers what matters. Never lies.**  
> Built for ethical hackers, AI researchers, and privacy-first users.

[![Streamlit](https://static.streamlit.io/badge/streamlit.svg)](https://streamlit.io)  
[![Ollama](https://img.shields.io/badge/Ollama-Supported-orange)](https://ollama.com)  
[![Privacy](https://img.shields.io/badge/Privacy-Local_Only-green)](https://github.com/NarlaRohan050/AI-Agent)

---

## 🚀 Overview

**Memora** is a **hybrid GPU/CPU AI agent** that combines **Mistral** and **Mistral-Instruct** to deliver truthful, memory-aware reasoning—**completely offline** with **zero data sent to the cloud**.

It is designed for developers and researchers who want full **privacy, control, and transparency** in AI systems.

---

## ✨ Features

* ✅ **Persistent memory** using ChromaDB + sentence embeddings  
* ✅ **Fact extraction** (name, interests, goals) — never invented  
* ✅ **Hallucination prevention** — refuses to answer unknowns  
* ✅ **Dynamic GPU/CPU load balancing** — prevents VRAM crashes  
* ✅ **Secure memory deletion** — `/forget salary` removes all traces  
* ✅ **Name conflict resolution** — prompts for confirmation  
* ✅ **Redundancy blocking** — avoids duplicate memories  
* ✅ **Privacy-first** — no telemetry, no cloud

---

## ⚠️ Important Notice

Do **NOT** run this project inside cloud-synced folders such as:

* OneDrive  
* Dropbox  
* Google Drive  

✅ Move the project to a local directory like: `C:\AI-Agent`

---

## 🛠️ Requirements

* Python **3.10 or higher**  
* **Ollama** installed  
* GPU recommended (CPU fallback supported)  
* Windows / Linux / macOS  

---

## 📦 Installation & Setup (Local Only)

### 1️⃣ Install Ollama

Download and install Ollama:

```bash
[https://ollama.com/download/OllamaSetup.exe](https://ollama.com/download/OllamaSetup.exe)
````

> Restart your terminal after installation.

-----

### 2️⃣ Pull the Quantized Mistral Model

```bash
ollama pull mistral:7b-instruct-v0.2-q5_K_M
```

-----

### 3️⃣ Clone the Repository

```bash
git clone [https://github.com/NarlaRohan050/AI-Agent.git](https://github.com/NarlaRohan050/AI-Agent.git)
cd AI-Agent
```

-----

### 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

-----

### 5️⃣ Run the Application

```bash
streamlit run ui/streamlit_app.py
```

> Open the local URL shown in the terminal.

-----

### 6️⃣ Optional: Run Capacity Test

```bash
python test_agent_max_capacity.py
```

-----

## 🔐 Privacy Guarantee

  * ❌ No cloud APIs  
  * ❌ No telemetry  
  * ❌ No background uploads  
  * ✅ All data stays **on your machine**

-----

## 📁 Project Structure

```yaml
AI-Agent/
│── src/ # Core agent logic
│── ui/ # Streamlit UI
│── models/ # Local models
│── data/ # Persistent memory
│── requirements.txt
│── test_agent_max_capacity.py
│── README.md
```

-----

## 📄 License

This project is intended for **educational and research purposes**.  
Refer to the repository for detailed license information.

-----

✅ Built with **privacy, control, and transparency**.

```

Now that your `README.md` is complete, do you need help with any other Git commands or steps for your project?
```
