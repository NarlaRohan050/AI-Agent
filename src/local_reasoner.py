# src/local_reasoner.py
import os
import json
import shutil
import subprocess
import threading
import time
import datetime
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

# -------- CONFIG ----------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MEMORY_DIR = os.path.join(BASE_DIR, "memory")
os.makedirs(MEMORY_DIR, exist_ok=True)

MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "models", "all-MiniLM-L6-v2"))
MEMORY_FILE = os.path.join(MEMORY_DIR, "memory.json")
FACTS_FILE = os.path.join(MEMORY_DIR, "facts.json")
BACKUP_DIR = os.path.join(MEMORY_DIR, "backups")
REASONING_BASE = "mistral"
REASONING_INSTRUCT = "mistral:instruct"
EMBED_DIM = 384
COMPACT_THRESHOLD = 200
MAX_COMPACT_CLUSTER = 30
# --------------------------

os.makedirs(BACKUP_DIR, exist_ok=True)


class LocalMemoryAgent:
    def __init__(self, device: str | None = None):
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Loading embedding model on [{self.device}] ...")
        if not os.path.exists(MODEL_DIR):
            raise FileNotFoundError(f"Embedding model not found at {MODEL_DIR}")
        self.model = SentenceTransformer(MODEL_DIR, device=self.device)
        print(f"✅ Embedding model loaded on {self.device.upper()}")

        self.embed_dim = EMBED_DIM
        self.index = faiss.IndexFlatL2(self.embed_dim)
        self.memory = []
        self.memory_embeddings = None
        self._load_memory()

        self.facts = {}
        self._load_facts()

        self.parallel_mode = True
        self.last_oom_timestamp = None
        self._lock = threading.Lock()
        self._pending_name_conflict = False
        self.ollama_ok = self._check_ollama_models()
        if not self.ollama_ok:
            print("⚠️ Ollama or one of the models not found - reasoning will fallback to subprocess or return warnings.")
        else:
            print(f"✅ Ollama sees models: {REASONING_BASE} and {REASONING_INSTRUCT} (or similar)")

    # -------------------------
    # Persistence / Checkpoint
    # -------------------------
    def _load_memory(self):
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.memory = data.get("entries", [])
                if self.memory:
                    self.memory_embeddings = self.model.encode(self.memory, convert_to_tensor=True, show_progress_bar=False)
                    emb_np = self.memory_embeddings.cpu().numpy().astype(np.float32)
                    self.index.reset()
                    if emb_np.size > 0:
                        self.index.add(emb_np)
                print(f"✅ Loaded memory ({len(self.memory)} entries).")
                return
            except Exception as e:
                print(f"⚠️ Failed to load memory file: {e} — starting fresh.")
        self.memory = []
        self.memory_embeddings = None
        self.index.reset()
        self._save_memory()

    def _save_memory(self):
        tmp = MEMORY_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"entries": self.memory}, f, indent=2, ensure_ascii=False)
        os.replace(tmp, MEMORY_FILE)

    def _load_facts(self):
        if os.path.exists(FACTS_FILE):
            try:
                with open(FACTS_FILE, "r", encoding="utf-8") as f:
                    self.facts = json.load(f)
            except Exception as e:
                print(f"⚠️ Failed to load facts: {e}")
                self.facts = {}
        else:
            self.facts = {}

    def _save_facts(self):
        tmp = FACTS_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.facts, f, indent=2, ensure_ascii=False)
        os.replace(tmp, FACTS_FILE)

    def backup_memory(self):
        ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        dst = os.path.join(BACKUP_DIR, f"memory_{ts}.json")
        try:
            shutil.copy2(MEMORY_FILE, dst)
            print(f"🗄️ Memory backed up to {dst}")
        except Exception as e:
            print(f"⚠️ Backup failed: {e}")

    # -------------------------
    # Ollama utils
    # -------------------------
    def _check_ollama_models(self) -> bool:
        try:
            subprocess.run(["ollama", "--version"], capture_output=True, check=True)
        except Exception:
            return False
        try:
            res = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
            out = res.stdout.lower()
            base = REASONING_BASE.split(":")[0]
            return base in out
        except Exception:
            return False

    def _call_ollama_run(self, model_name: str, prompt: str, timeout: int = 800):
        try:
            proc = subprocess.run(["ollama", "run", model_name],
                                  input=prompt.encode("utf-8"),
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  timeout=timeout)
            stderr = proc.stderr.decode("utf-8", errors="ignore")
            stdout = proc.stdout.decode("utf-8", errors="ignore").strip()
            if proc.returncode != 0:
                if "out of memory" in stderr.lower() or "oom" in stderr.lower():
                    raise RuntimeError("CUDA_OOM")
                return f"⚠️ Reasoning model error: {stderr.strip()[:400]}"
            return self._extract_content_from_str(stdout)
        except subprocess.TimeoutExpired:
            return "⚠️ Reasoning timed out."
        except RuntimeError as e:
            if "CUDA_OOM" in str(e):
                self.last_oom_timestamp = time.time()
                raise
            raise
        except Exception as e:
            return f"⚠️ Error running model {model_name}: {e}"

    def _extract_content_from_str(self, txt: str) -> str:
        if not txt:
            return ""
        m = re.search(r"content=(?:\"|')(.+?)(?:\"|')", txt, flags=re.DOTALL)
        if m:
            return m.group(1).strip()
        return txt.strip()

    def query_model(self, model_name: str, prompt: str, timeout: int = 60):
        try:
            import ollama as _ollama
            resp = _ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
            if isinstance(resp, dict) and "message" in resp:
                return resp["message"].get("content", "").strip()
            if hasattr(resp, "message") and hasattr(resp.message, "content"):
                return str(resp.message.content).strip()
            if isinstance(resp, str):
                return self._extract_content_from_str(resp)
            return str(resp)
        except Exception:
            pass
        return self._call_ollama_run(model_name, prompt, timeout=timeout)

    # -------------------------
    # Structured Fact Extraction (FIXED - INDUSTRIAL ROBUST)
    # -------------------------
    def extract_structured_facts(self, text: str) -> dict:
        prompt = (
            "Extract structured facts from the conversation below. "
            "Output ONLY a valid JSON object with keys: user_name, interests, goals, preferences. "
            "Rules:\n"
            "- user_name must be a SINGLE string (e.g., 'Alex'), NEVER a list.\n"
            "- interests and goals must be lists of strings.\n"
            "- If no facts, output {}.\n\n"
            f"Conversation:\n{text}\n\n"
        )
        try:
            raw = self.query_model(REASONING_INSTRUCT, prompt, timeout=30)
            # Extract first valid JSON object
            json_match = re.search(r"\{[^{}]*\}", raw)
            if not json_match:
                json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                clean = {}
                for key, value in parsed.items():
                    if key == "user_name":
                        # Enforce string
                        if isinstance(value, list) and value:
                            clean[key] = str(value[0]).strip()
                        elif isinstance(value, str):
                            clean[key] = value.strip()
                        # else: ignore invalid
                    elif key in ("interests", "goals", "preferences"):
                        # Enforce list of non-empty strings
                        if isinstance(value, str) and value.strip():
                            clean[key] = [value.strip()]
                        elif isinstance(value, list):
                            clean[key] = [str(item).strip() for item in value if str(item).strip()]
                        else:
                            clean[key] = []
                    else:
                        # Other keys: store if valid
                        if value not in (None, "", [], {}):
                            clean[key] = value
                return clean
        except Exception as e:
            print(f"⚠️ Fact extraction failed: {e}")
        return {}

    # -------------------------
    # Memory operations
    # -------------------------
    def add_memory(self, summary: str):
        with self._lock:
            try:
                self.memory.append(summary)
                self.memory_embeddings = self.model.encode(self.memory, convert_to_tensor=True, show_progress_bar=False)
                emb_np = self.memory_embeddings.cpu().numpy().astype(np.float32)
                self.index.reset()
                if emb_np.size > 0:
                    self.index.add(emb_np)
                self._save_memory()
            except Exception as e:
                print(f"⚠️ Failed to append memory: {e}")

            if len(self.memory) > COMPACT_THRESHOLD:
                print("🔔 Memory exceeded threshold — scheduling compaction in background.")
                threading.Thread(target=self.compact_memory, daemon=True).start()

    def search_memory(self, query: str, top_k: int = 5) -> list:
        if not self.memory or self.memory_embeddings is None or self.index.ntotal == 0:
            return []
        q_emb = self.model.encode([query], convert_to_tensor=True, show_progress_bar=False)
        cos_scores = util.pytorch_cos_sim(q_emb, self.memory_embeddings)[0]
        topk = min(top_k, len(self.memory))
        vals, idx = torch.topk(cos_scores, k=topk)
        results = [self.memory[i] for i in idx.cpu().numpy().tolist()]
        return results

    # -------------------------
    # Summarization
    # -------------------------
    def summarize_text_via_model(self, text: str, short: bool = True) -> str:
        prefix = "Summarize the following user/AI exchange into a single concise memory sentence capturing intent and topics:"
        prompt = f"{prefix}\n\n{text}\n\nSummary:"
        try:
            out = self.query_model(REASONING_INSTRUCT, prompt, timeout=45)
            if isinstance(out, str) and out.startswith("⚠️"):
                return (text[:200] + "...").replace("\n", " ")
            return self._extract_content_from_str(out).replace("\n", " ").strip()
        except RuntimeError as e:
            print("⚠️ Summarization OOM detected; falling back to simple truncation.")
            return (text[:200] + "...").replace("\n", " ")

    # -------------------------
    # Compaction
    # -------------------------
    def compact_memory(self, n_clusters: int | None = None):
        with self._lock:
            n = len(self.memory)
            if n <= 1:
                print("ℹ️ Not enough memory items to compact.")
                return
            if n_clusters is None:
                n_clusters = max(1, min(MAX_COMPACT_CLUSTER, n // 8))
            print(f"🧹 Compacting {n} memories into {n_clusters} clusters ...")
            try:
                emb_np = self.memory_embeddings.cpu().numpy().astype(np.float32)
                kmeans = faiss.Kmeans(d=self.embed_dim, k=n_clusters, niter=20, verbose=False)
                kmeans.train(emb_np)
                _, assignments = kmeans.index.search(emb_np, 1)
                assignments = assignments.reshape(-1)
                clusters = {}
                for idx, c in enumerate(assignments):
                    clusters.setdefault(int(c), []).append(self.memory[idx])

                new_memory = []
                for cid, texts in clusters.items():
                    sample = "\n".join(texts[:10])
                    summary = self.summarize_text_via_model(sample)
                    new_memory.append(summary)

                self.backup_memory()
                self.memory = new_memory
                self.memory_embeddings = self.model.encode(self.memory, convert_to_tensor=True, show_progress_bar=False)
                emb_np2 = self.memory_embeddings.cpu().numpy().astype(np.float32)
                self.index.reset()
                if emb_np2.size > 0:
                    self.index.add(emb_np2)
                self._save_memory()
                print(f"✅ Compaction done. New memory size: {len(self.memory)}")
            except Exception as e:
                print(f"⚠️ Compaction failed: {e}")

    # -------------------------
    # Hybrid inference
    # -------------------------
    def _parallel_infer_and_merge(self, prompt: str, timeout: int = 60) -> str:
        prompt_base = f"Deep reasoning draft (do not polish). User: {prompt}"
        prompt_instruct = f"Answer concisely and clearly. User: {prompt}"

        responses = {}
        errors = []
        try:
            with ThreadPoolExecutor(max_workers=2) as ex:
                futs = {
                    ex.submit(self.query_model, REASONING_BASE, prompt_base, timeout): REASONING_BASE,
                    ex.submit(self.query_model, REASONING_INSTRUCT, prompt_instruct, timeout): REASONING_INSTRUCT,
                }
                for fut in as_completed(futs):
                    model_name = futs[fut]
                    try:
                        res = fut.result()
                        responses[model_name] = res
                    except RuntimeError as re:
                        errors.append((model_name, str(re)))
                    except Exception as e:
                        errors.append((model_name, str(e)))
        except Exception as e:
            errors.append(("executor", str(e)))

        if any("CUDA_OOM" in str(err) or "out of memory" in str(err).lower() for _, err in errors):
            self.parallel_mode = False
            self.last_oom_timestamp = time.time()
            print("⚠️ GPU OOM detected while running in parallel. Switching to sequential mode for now.")

        if REASONING_BASE in responses and REASONING_INSTRUCT in responses:
            try:
                combined = f"Base model response:\n{responses[REASONING_BASE]}\n\nInstruct model response:\n{responses[REASONING_INSTRUCT]}"
                merge_prompt = (
                    "You are a meta-reasoner. Combine the two responses below: keep correctness from the base draft "
                    "and clarity from the instruct response. Produce a single final answer.\n\n" + combined + "\n\nFinal answer:"
                )
                merged = self.query_model(REASONING_INSTRUCT, merge_prompt, timeout=timeout)
                return merged
            except Exception as e:
                print(f"⚠️ Merge failed, returning instruct response: {e}")
                return responses.get(REASONING_INSTRUCT, responses.get(REASONING_BASE, "⚠️ No response."))
        if responses:
            return responses.get(REASONING_INSTRUCT) or next(iter(responses.values()))
        return "⚠️ All reasoning calls failed."

    def _sequential_infer(self, prompt: str, timeout: int = 60) -> str:
        try:
            draft = self.query_model(REASONING_BASE, f"Draft reasoning: {prompt}", timeout=timeout)
            refine_prompt = f"Refine the draft into a concise final answer. Draft:\n{draft}\n\nFinal:"
            final = self.query_model(REASONING_INSTRUCT, refine_prompt, timeout=timeout)
            return final
        except RuntimeError as re:
            print("⚠️ OOM during sequential pipeline; falling back to single instruct.")
            try:
                return self.query_model(REASONING_INSTRUCT, f"Answer: {prompt}", timeout=timeout)
            except Exception as e:
                return f"⚠️ Reasoning failed: {e}"
        except Exception as e:
            return f"⚠️ Reasoning failed: {e}"

    # -------------------------
    # Main Answer Pipeline (FULLY FIXED)
    # -------------------------
    def answer_with_memory(self, query: str, top_k: int = 5) -> dict:
        contexts = self.search_memory(query, top_k=top_k)

        # === HANDLE CONFIRMATION REPLIES ===
        if self._pending_name_conflict:
            confirm_match = re.search(
                r"\b(?:yes|yeah|yep|sure|okay|ok)\b.*?\b(?:i\s*am|i'm)\s+([A-Za-z][a-z]*(?:\s+[A-Za-z][a-z]*)?)",
                query, re.IGNORECASE
            )
            if confirm_match:
                confirmed_name = " ".join(w.capitalize() for w in confirm_match.group(1).split())
                self.facts["user_name"] = confirmed_name
                self._save_facts()
                self._pending_name_conflict = False
                return {
                    "answer": f"✅ Got it! I'll call you {confirmed_name} from now on.",
                    "context": contexts
                }
            elif re.search(r"\b(?:no|nope|nah|nevermind)\b", query, re.IGNORECASE):
                self._pending_name_conflict = False
                current = self.facts.get("user_name", "you")
                return {
                    "answer": f"✅ Alright, I'll keep calling you {current}.",
                    "context": contexts
                }

        # Normalize query for name detection
        normalized_query = re.sub(r"\bi\s*am\b", "i am", query, flags=re.IGNORECASE)
        normalized_query = re.sub(r"\bmy\s+name\s+is\b", "my name is", normalized_query, flags=re.IGNORECASE)

        # Extract names
        name_matches = re.findall(
            r"(?:my name is|i am|call me|this is|actually[,\s]*i['’]?m)\s+([A-Za-z][a-z]*(?:\s+[A-Za-z][a-z]*)?)",
            normalized_query, re.IGNORECASE
        )
        declared_names = [
            " ".join(w.capitalize() for w in name.strip().split())
            for name in name_matches
        ]

        existing_name = self.facts.get("user_name")
        should_add = True

        # Handle name declaration
        if declared_names:
            new_name = declared_names[0]
            if existing_name:
                if existing_name.lower() != new_name.lower():
                    self._pending_name_conflict = True
                    return {
                        "answer": f"I currently know you as {existing_name}, but you just said you're {new_name}. "
                                  f"Should I switch to calling you {new_name}? (Say: 'Yes, I am {new_name}' or 'No')",
                        "context": contexts
                    }
                else:
                    should_add = False
            else:
                self.facts["user_name"] = new_name
                self._save_facts()
                should_add = False

        # Fallback: ensure name is captured
        if "user_name" not in self.facts and declared_names:
            self.facts["user_name"] = declared_names[0]
            self._save_facts()

        # Build structured context
        structured_context = ""
        if self.facts:
            parts = []
            if "user_name" in self.facts:
                parts.append(f"User name: {self.facts['user_name']}")
            if "interests" in self.facts:
                parts.append(f"Interests: {', '.join(self.facts['interests'])}")
            if "goals" in self.facts:
                parts.append(f"Goals: {', '.join(self.facts['goals'])}")
            if parts:
                structured_context = "Known facts:\n- " + "\n- ".join(parts) + "\n"

        # Skip memory for identity queries
        if re.search(r"\b(what is my name|who am i|am i|my name)\b", query, re.IGNORECASE):
            if self.facts.get("user_name"):
                should_add = False

        semantic_context = "\n".join(contexts) if contexts else "No prior conversation."
        full_context = (
            "You are a helpful AI assistant. Use the following verified facts and conversation history to answer.\n"
            "NEVER invent personal details. If something isn't known, say so.\n\n"
            f"{structured_context}\n"
            f"Conversation history:\n{semantic_context}\n\n"
            f"User question: {query}"
        )

        # Reasoning
        now = time.time()
        if self.parallel_mode:
            try:
                answer = self._parallel_infer_and_merge(full_context, timeout=75)
            except RuntimeError:
                answer = self._sequential_infer(full_context, timeout=75)
        else:
            if self.last_oom_timestamp and now - self.last_oom_timestamp < 600:
                try:
                    answer = self._sequential_infer(full_context, timeout=75)
                except Exception:
                    answer = self.query_model(REASONING_INSTRUCT, f"Answer truthfully: {full_context}", timeout=75)
            else:
                self.parallel_mode = True
                answer = self._parallel_infer_and_merge(full_context, timeout=75)

        # Update facts with MERGING for lists
        interaction = f"User: {query}\nAI: {answer}"
        new_facts = self.extract_structured_facts(interaction)
        if new_facts:
            for key, value in new_facts.items():
                if key in ("interests", "goals") and key in self.facts:
                    if isinstance(value, list) and isinstance(self.facts[key], list):
                        merged = list(set(self.facts[key] + value))
                        self.facts[key] = merged
                    else:
                        self.facts[key] = value if isinstance(value, list) else [value]
                else:
                    self.facts[key] = value
            self._save_facts()

        # Fallback: ensure name is captured
        if "user_name" not in self.facts and declared_names:
            self.facts["user_name"] = declared_names[0]
            self._save_facts()

        # Add to memory (with deduplication)
        if should_add:
            summary = self.summarize_text_via_model(interaction)
            if self.memory:
                emb = self.model.encode([summary], convert_to_tensor=True)
                sims = util.pytorch_cos_sim(emb, self.memory_embeddings)[0]
                if torch.max(sims) < 0.92:
                    self.add_memory(summary)
            else:
                self.add_memory(summary)

        return {"answer": answer, "context": contexts}