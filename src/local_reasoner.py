# src/local_reasoner.py
import os
import json
import shutil
import subprocess
import threading
import time
import datetime
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import chromadb

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
        self.model = SentenceTransformer(MODEL_DIR, device=self.device, trust_remote_code=True)
        print(f"✅ Embedding model loaded on {self.device.upper()}")

        self.embed_dim = EMBED_DIM
        self.memory = []
        self._load_memory()

        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=MEMORY_DIR)
        self.collection = self.chroma_client.get_or_create_collection(
            name="memory",
            embedding_function=None,
            metadata={"hnsw:space": "cosine"}
        )
        self._rebuild_chroma_index()

        self.facts = {}
        self._load_facts()

        self.parallel_mode = True
        self.last_oom_timestamp = None
        self._lock = threading.Lock()
        self._pending_name_conflict = False
        self.interaction_count = 0
        self.ollama_ok = self._check_ollama_models()
        if not self.ollama_ok:
            print("⚠️ Ollama or one of the models not found - reasoning will fallback to subprocess or return warnings.")
        else:
            print(f"✅ Ollama sees models: {REASONING_BASE} and {REASONING_INSTRUCT} (or similar)")

    # -------------------------
    # Dynamic GPU/CPU Load Balancing
    # -------------------------
    def _get_device_for_embeddings(self):
        """Return device based on VRAM usage"""
        if self.device != "cuda" or not torch.cuda.is_available():
            return "cpu"
        
        try:
            torch.cuda.synchronize()
            total = torch.cuda.get_device_properties(0).total_memory
            reserved = torch.cuda.memory_reserved(0)
            allocated = torch.cuda.memory_allocated(0)
            used = reserved  # Use reserved as it includes cached memory
            usage_percent = (used / total)
            
            if usage_percent > 0.80:  # 80% threshold
                print(f"MemoryWarning VRAM usage {usage_percent:.2%} > 80% - using CPU for embeddings")
                return "cpu"
            return self.device
        except:
            return self.device

    # -------------------------
    # Memory & ChromaDB (GPU/CPU-AWARE)
    # -------------------------
    def _load_memory(self):
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.memory = data.get("entries", [])
                print(f"✅ Loaded memory ({len(self.memory)} entries).")
                return
            except Exception as e:
                print(f"⚠️ Failed to load memory file: {e} — starting fresh.")
        self.memory = []
        self._save_memory()

    def _save_memory(self):
        tmp = MEMORY_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"entries": self.memory}, f, indent=2, ensure_ascii=False)
        os.replace(tmp, MEMORY_FILE)

    def _rebuild_chroma_index(self):
        try:
            ids = [str(i) for i in range(len(self.memory))]
            if ids:
                # Use dynamic device selection
                emb_device = self._get_device_for_embeddings()
                emb_tensor = self.model.encode(self.memory, device=emb_device, convert_to_tensor=True, show_progress_bar=False)
                embeddings = emb_tensor.cpu().numpy().astype(np.float32).tolist()
                self.collection.add(ids=ids, documents=self.memory, embeddings=embeddings)
            print(f"✅ ChromaDB index rebuilt with {len(self.memory)} entries.")
        except Exception as e:
            print(f"⚠️ Failed to rebuild ChromaDB index: {e}")

    def add_memory(self, summary: str):
        with self._lock:
            try:
                self.memory.append(summary)
                # Use dynamic device selection
                emb_device = self._get_device_for_embeddings()
                emb_tensor = self.model.encode(summary, device=emb_device, convert_to_tensor=True)
                embeddings = [emb_tensor.cpu().numpy().astype(np.float32).tolist()]
                self.collection.add(ids=[str(len(self.memory) - 1)], documents=[summary], embeddings=embeddings)
                self._save_memory()
                self.interaction_count += 1
                if self.interaction_count % 5 == 0:
                    self._save_memory()
            except Exception as e:
                print(f"⚠️ Failed to append memory: {e}")
                self._rebuild_chroma_index()

    def search_memory(self, query: str, top_k: int = 3):  # Reduced for speed
        if not self.memory:
            return []
        try:
            # Use dynamic device selection
            emb_device = self._get_device_for_embeddings()
            query_emb_tensor = self.model.encode(query, device=emb_device, convert_to_tensor=True, show_progress_bar=False)
            query_emb = query_emb_tensor.cpu().numpy().astype(np.float32).tolist()
            results = self.collection.query(query_embeddings=[query_emb], n_results=top_k, include=["documents"])
            return [doc for doc in results["documents"][0]] if results["documents"] else []
        except Exception as e:
            print(f"⚠️ ChromaDB search failed: {e}")
            return []

    def forget_memory(self, topic: str):
        to_remove = [i for i, mem in enumerate(self.memory) if topic.lower() in mem.lower()]
        if not to_remove:
            return False
        for i in sorted(to_remove, reverse=True):
            del self.memory[i]
        self._rebuild_chroma_index()
        self._save_memory()
        return True

    # -------------------------
    # Persistence & Ollama (Optimized for Speed)
    # -------------------------
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

    def _call_ollama_run(self, model_name: str, prompt: str, timeout: int = 30):  # Reduced timeout
        try:
            env = os.environ.copy()
            env["OLLAMA_NUM_CTX"] = "1024"
            env["OLLAMA_NUM_THREADS"] = "6"
            env["OLLAMA_TEMPERATURE"] = "0.3"
            
            proc = subprocess.run(
                ["ollama", "run", model_name],
                input=prompt.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                env=env
            )
            stderr = proc.stderr.decode("utf-8", errors="ignore")
            stdout = proc.stdout.decode("utf-8", errors="ignore").strip()
            if proc.returncode != 0:
                if "out of memory" in stderr.lower() or "oom" in stderr.lower():
                    raise RuntimeError("SYSTEM_OOM")
                return f"⚠️ Reasoning model error: {stderr.strip()[:400]}"
            return self._extract_content_from_str(stdout)
        except subprocess.TimeoutExpired:
            return "⚠️ Reasoning timed out."
        except RuntimeError as e:
            if "SYSTEM_OOM" in str(e):
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

    def query_model(self, model_name: str, prompt: str, timeout: int = 25):  # Reduced timeout
        try:
            import ollama as _ollama
            resp = _ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "num_ctx": 1024,
                    "num_threads": 6,
                    "temperature": 0.3,
                    "num_predict": 200
                }
            )
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
    # Structured Fact Extraction (Optimized)
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
            raw = self.query_model(REASONING_INSTRUCT, prompt, timeout=20)
            json_match = re.search(r"\{[^{}]*\}", raw)
            if not json_match:
                json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                clean = {}
                for key, value in parsed.items():
                    if key == "user_name":
                        if isinstance(value, list) and value:
                            clean[key] = str(value[0]).strip()
                        elif isinstance(value, str):
                            clean[key] = value.strip()
                    elif key in ("interests", "goals", "preferences"):
                        if isinstance(value, str) and value.strip():
                            clean[key] = [value.strip()]
                        elif isinstance(value, list):
                            clean[key] = [str(item).strip() for item in value if str(item).strip()]
                        else:
                            clean[key] = []
                    else:
                        if value not in (None, "", [], {}):
                            clean[key] = value
                return clean
        except Exception as e:
            print(f"⚠️ Fact extraction failed: {e}")
        return {}

    # -------------------------
    # Summarization & Inference (Speed-Optimized)
    # -------------------------
    def summarize_text_via_model(self, text: str, short: bool = True) -> str:
        prefix = "Summarize to 1 sentence:"
        prompt = f"{prefix}\n\n{text}\n\nSummary:"
        try:
            out = self.query_model(REASONING_INSTRUCT, prompt, timeout=15)
            if isinstance(out, str) and out.startswith("⚠️"):
                return (text[:150] + "...").replace("\n", " ")
            return self._extract_content_from_str(out).replace("\n", " ").strip()
        except RuntimeError as e:
            print("⚠️ Summarization OOM detected; truncating.")
            return (text[:150] + "...").replace("\n", " ")

    def compact_memory(self, n_clusters: int | None = None):
        print("ℹ️ ChromaDB manages memory efficiently — compaction not needed.")

    def _parallel_infer_and_merge(self, prompt: str, timeout: int = 25):  # Reduced timeout
        prompt_base = f"Draft: {prompt}"
        prompt_instruct = f"Answer: {prompt}"

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
                    except Exception as e:
                        responses[model_name] = f"⚠️ Error: {str(e)[:100]}"
        except Exception as e:
            responses = {REASONING_BASE: f"⚠️ Parallel error: {e}"}

        if REASONING_BASE in responses and REASONING_INSTRUCT in responses:
            try:
                merge_prompt = f"Combine: A={responses[REASONING_BASE]}, B={responses[REASONING_INSTRUCT]}. Final:"
                return self.query_model(REASONING_INSTRUCT, merge_prompt, timeout=15)
            except:
                return responses.get(REASONING_INSTRUCT, responses.get(REASONING_BASE, "⚠️ No response."))
        return responses.get(REASONING_INSTRUCT) or next(iter(responses.values()))

    def _sequential_infer(self, prompt: str, timeout: int = 25):  # Reduced timeout
        try:
            draft = self.query_model(REASONING_BASE, f"Draft: {prompt}", timeout=timeout)
            refine_prompt = f"Refine: {draft}. Final:"
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
    # Main Answer Pipeline (Perfect Integrity Maintained)
    # -------------------------
    def answer_with_memory(self, query: str, top_k: int = 3) -> dict:
        contexts = self.search_memory(query, top_k=top_k)

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

        if re.search(r"\b(clear|delete|forget)\s+my\s+memory\b", query, re.IGNORECASE):
            self.memory = []
            self._rebuild_chroma_index()
            self._save_memory()
            return {"answer": "🧹 All memories cleared!", "context": []}

        normalized_query = re.sub(r"\bi\s*am\b", "i am", query, flags=re.IGNORECASE)
        normalized_query = re.sub(r"\bmy\s+name\s+is\b", "my name is", normalized_query, flags=re.IGNORECASE)

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

        if "user_name" not in self.facts and declared_names:
            self.facts["user_name"] = declared_names[0]
            self._save_facts()

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

        now = time.time()
        if self.parallel_mode:
            try:
                answer = self._parallel_infer_and_merge(full_context, timeout=25)
            except RuntimeError:
                answer = self._sequential_infer(full_context, timeout=25)
        else:
            if self.last_oom_timestamp and now - self.last_oom_timestamp < 600:
                try:
                    answer = self._sequential_infer(full_context, timeout=25)
                except Exception:
                    answer = self.query_model(REASONING_INSTRUCT, f"Answer truthfully: {full_context}", timeout=25)
            else:
                self.parallel_mode = True
                answer = self._parallel_infer_and_merge(full_context, timeout=25)

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

        if "user_name" not in self.facts and declared_names:
            self.facts["user_name"] = declared_names[0]
            self._save_facts()

        if should_add:
            summary = self.summarize_text_via_model(interaction)
            self.add_memory(summary)

        return {"answer": answer, "context": contexts}