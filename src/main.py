# src/main.py
import os
import sys
from local_reasoner import LocalMemoryAgent

def main():
    agent = LocalMemoryAgent()

    print("\n" + "="*60)
    print("🧠 Local Hybrid Context Memory Agent")
    print(f"Embedding device: {agent.device}")
    print(f"Parallel fusion mode: {agent.parallel_mode}")
    print(f"Memory entries: {len(agent.memory)}")
    print("="*60)

    print("\nCommands:")
    print("  /exit            - quit")
    print("  /backup          - create memory backup now")
    print("  /compact         - run compaction (cluster+summarize)")
    print("  /status          - show status")
    print("  /reindex         - rebuild FAISS index from embeddings")
    print("  /clear           - CLEAR ALL memory and facts (confirm required)")
    print("  any other text   - treated as user query")

    while True:
        try:
            user_input = input("\n🔍 Ask (or command): ").strip()
            if not user_input:
                continue

            if user_input == "/exit":
                print("👋 Goodbye!")
                break

            elif user_input == "/backup":
                agent.backup_memory()

            elif user_input == "/compact":
                agent.compact_memory()

            elif user_input == "/status":
                print(f"\n📊 STATUS:")
                print(f"Memory entries: {len(agent.memory)}")
                print(f"Structured facts: {agent.facts}")
                print(f"FAISS index size: {agent.index.ntotal}")

            elif user_input == "/reindex":
                if agent.memory:
                    emb_np = agent.memory_embeddings.cpu().numpy().astype(np.float32)
                    agent.index.reset()
                    agent.index.add(emb_np)
                    print("✅ FAISS index rebuilt.")
                else:
                    print("⚠️ No memory to reindex.")

            elif user_input == "/clear":
                print("⚠️ Warning: This will erase all memory and facts permanently.")
                confirm = input("Type 'CONFIRM' to proceed: ").strip()
                if confirm == "CONFIRM":
                    agent.facts = {}
                    agent.memory = []
                    agent._save_facts()
                    agent._save_memory()
                    agent.index.reset()
                    agent.memory_embeddings = None
                    agent._pending_name_conflict = False
                    print("✅ All memory and facts cleared.")
                else:
                    print("❌ Clear aborted.")

            else:
                result = agent.answer_with_memory(user_input)
                print(f"\n💬 AI Agent Answer:\n{result['answer']}")
                if result["context"]:
                    print(f"\n📚 Retrieved Contexts:")
                    for i, ctx in enumerate(result["context"], 1):
                        print(f"  {i}. {ctx}")
                else:
                    print("\n📚 No related memory found yet.")
                print("-" * 80)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n💥 Unexpected error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()