# src/local_reasoner.py
import os
import json
import re
import subprocess
import time
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

# Memory file (relative to src/)
MEMORY_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "memory", "memory.json"))
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "all-MiniLM-L6-v2"))
REASONING_MODEL_NAME = "mistral"  # set to the name shown by `ollama list` (you have mistral:latest)

# ensure directories exist for memory
os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)


class LocalMemoryAgent:
    def __init__(self, device: str | None = None):
        # device auto-detection
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Loading local embedding model on [{self.device}] ...")
        if not os.path.exists(MODEL_DIR):
            raise FileNotFoundError(f"Embedding model folder not found at {MODEL_DIR}")
        self.model = SentenceTransformer(MODEL_DIR, device=self.device)
        print(f"✅ Embedding model loaded successfully on {self.device.upper()}!")

        # FAISS index and memory arrays
        self.embed_dim = 384  # all-MiniLM-L6-v2 dimension
        self.index = None
        self.memory = []               # list[str] of cleaned summary entries
        self.memory_embeddings = None  # torch tensor or None

        # initialize memory
        self._init_memory()

        # check ollama availability (non-fatal)
        self.ollama_available = self._check_ollama_and_model()
        if not self.ollama_available:
            print("⚠️ Ollama or the reasoning model is not available. Reasoning will fallback or return warnings.")
        else:
            print(f"✅ Ollama and model '{REASONING_MODEL_NAME}' are available locally.")

    # -------------------------
    # Memory persistence
    # -------------------------
    def _init_memory(self):
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                entries = data.get("entries", [])
                self.memory = entries
                if len(self.memory) > 0:
                    # create embeddings tensor (on device)
                    self.memory_embeddings = self.model.encode(self.memory, convert_to_tensor=True, show_progress_bar=False)
                # create FAISS index and populate from embeddings (if any)
                self.index = faiss.IndexFlatL2(self.embed_dim)
                if self.memory_embeddings is not None:
                    # convert to numpy on CPU for FAISS
                    emb_np = self.memory_embeddings.cpu().numpy().astype(np.float32)
                    self.index.add(emb_np)
                print(f"✅ Loaded memory with {len(self.memory)} items.")
                return
            except Exception as e:
                print(f"⚠️ Failed to load memory file (starting fresh): {e}")

        # if no file or load failed, initialize empty
        self.memory = []
        self.memory_embeddings = None
        self.index = faiss.IndexFlatL2(self.embed_dim)
        self._save_memory()
        print("🆕 Created new empty memory store.")

    def _save_memory(self):
        # save JSON of entries
        try:
            with open(MEMORY_FILE, "w", encoding="utf-8") as f:
                json.dump({"entries": self.memory}, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Failed to save memory file: {e}")

    # -------------------------
    # Ollama utilities (robust)
    # -------------------------
    def _check_ollama_and_model(self) -> bool:
        """Check if `ollama` is installed and the target model is present (non-fatal)."""
        try:
            subprocess.run(["ollama", "--version"], capture_output=True, check=True)
        except Exception:
            return False
        try:
            res = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
            out = res.stdout.lower()
            # user pulled 'mistral:latest' — check substring
            return REASONING_MODEL_NAME.lower() in out
        except Exception:
            return False

    def _extract_content_from_str(self, txt: str) -> str:
        """
        When ollama.chat binding returns a stringified object like:
        "model='mistral' ... message=Message(role='assistant', content=' ... ')"
        we try to extract the content='...'.
        """
        if not txt:
            return ""
        # try regex for content='...'
        m = re.search(r"content=(?:\"|')(.+?)(?:\"|')(?:,|\))", txt, flags=re.DOTALL)
        if m:
            return m.group(1).strip()
        # fallback: remove "Message(...)" wrappers if any
        m2 = re.search(r"Message\([^)]*content=(?:\"|')(.+?)(?:\"|')", txt, flags=re.DOTALL)
        if m2:
            return m2.group(1).strip()
        return txt.strip()

    def query_llm(self, prompt: str, timeout: int = 60) -> str:
        """
        Query the local reasoning model. Try Python ollama bindings first (if installed);
        if that fails, fallback to subprocess `ollama run <model>` and read stdout.
        Always return a cleaned string.
        """
        # Try Python ollama package if available
        try:
            import ollama as _ollama  # local binding (may or may not exist)
            try:
                resp = _ollama.chat(model=REASONING_MODEL_NAME, messages=[{"role": "user", "content": prompt}])
                # resp may be dict, object, or string
                if isinstance(resp, dict) and "message" in resp and isinstance(resp["message"], dict):
                    content = resp["message"].get("content", "")
                    if isinstance(content, str):
                        return content.strip()
                # If object-like
                if hasattr(resp, "message") and hasattr(resp.message, "content"):
                    return str(resp.message.content).strip()
                if isinstance(resp, str):
                    return self._extract_content_from_str(resp)
                # fallback to string conversion
                return self._extract_content_from_str(str(resp))
            except Exception:
                # fall through to subprocess fallback
                pass
        except Exception:
            # ollama binding not installed; fall back to subprocess
            pass

        # Subprocess fallback: run `ollama run <model>` and send prompt via stdin
        try:
            proc = subprocess.run(["ollama", "run", REASONING_MODEL_NAME],
                                  input=prompt.encode("utf-8"),
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  timeout=timeout)
            if proc.returncode != 0:
                serr = proc.stderr.decode("utf-8", errors="ignore").strip()
                # sometimes stderr contains useful info; include it minimally
                return f"⚠️ Reasoning model error: {serr[:300]}"
            out = proc.stdout.decode("utf-8", errors="ignore").strip()
            return self._extract_content_from_str(out)
        except subprocess.TimeoutExpired:
            return "⚠️ Reasoning timed out."
        except Exception as e:
            return f"⚠️ Error invoking reasoning model: {e}"

    # -------------------------
    # Semantic summarization & memory management
    # -------------------------
    def summarize_context(self, text: str) -> str:
        """
        Produce a short semantic summary for storing in memory.
        This asks the reasoning model to convert raw user input or an exchange into a concise memory entry.
        """
        # short prompt that asks for a one-line conceptual summary
        short_prompt = (
            "Convert the following user statement or short exchange into a single short memory summary "
            "that captures intent, topics, and entities (one sentence):\n\n"
            f"{text}\n\n"
            "Short summary:"
        )
        raw = self.query_llm(short_prompt, timeout=40)
        # clean result if it contains message wrappers
        summary = self._extract_content_from_str(raw)
        # final safety trim
        summary = summary.replace("\n", " ").strip()
        if len(summary) == 0:
            # fallback to a truncated raw form
            summary = text.strip()
            if len(summary) > 200:
                summary = summary[:200].rstrip() + "..."
        return summary

    def add_memory(self, summary: str):
        """Add a cleaned summary to memory (embedding + FAISS + save)."""
        try:
            # append to memory list
            self.memory.append(summary)
            # update embeddings (torch tensor on device)
            self.memory_embeddings = self.model.encode(self.memory, convert_to_tensor=True, show_progress_bar=False)
            # update FAISS index: rebuild (safe for small memory). For larger memories, use incremental insertion with numpy.
            emb_np = self.memory_embeddings.cpu().numpy().astype(np.float32)
            self.index.reset()
            if emb_np.size > 0:
                self.index.add(emb_np)
            self._save_memory()
            print(f"🧩 Added to memory: {summary}")
        except Exception as e:
            print(f"⚠️ Failed to add memory: {e}")

    # -------------------------
    # Retrieval & reasoning pipeline
    # -------------------------
    def search_memory(self, query: str, top_k: int = 3) -> list:
        """Return top_k memory summaries most relevant to the query (semantic search)."""
        if not self.memory or self.memory_embeddings is None:
            return []
        q_emb = self.model.encode(query, convert_to_tensor=True, show_progress_bar=False)
        cos_scores = util.pytorch_cos_sim(q_emb, self.memory_embeddings)[0]
        topk = min(top_k, len(self.memory))
        vals, idx = torch.topk(cos_scores, k=topk)
        results = [self.memory[i] for i in idx.cpu().numpy().tolist()]
        return results

    def query_with_context(self, query: str, top_k: int = 3) -> dict:
        """
        Retrieve context, ask the reasoning model to answer using that context,
        then store a short semantic summary of the interaction (concept-level).
        """
        contexts = self.search_memory(query, top_k=top_k)
        context_block = ("\n".join(contexts)) if contexts else "No prior memory available."

        reasoning_prompt = (
            "You are a helpful reasoning assistant that uses stored short memory summaries.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question:\n{query}\n\n"
            "Answer concisely and clearly, and if the context is relevant use it."
        )
        answer = self.query_llm(reasoning_prompt, timeout=60)

        # Build compact interaction string and summarize conceptually before storing
        interaction_text = f"User: {query}\nAI: {answer}"
        summary = self.summarize_context(interaction_text)
        # add to memory
        self.add_memory(summary)

        return {"answer": answer, "context": contexts}
