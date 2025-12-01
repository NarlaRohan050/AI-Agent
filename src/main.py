# src/main.py
import os
import sys
from local_reasoner import LocalMemoryAgent

def main():
    agent = LocalMemoryAgent()

    print("\n" + "="*60)
    print("🧠 Local Hybrid Context Memory Agent")
    print(f"Embedding device: {agent.device}")
    print(f"Vector DB: ChromaDB (persistent, efficient)")
    print(f"Memory entries: {len(agent.memory)}")
    print("="*60)

    print("\nCommands:")
    print("  /exit            - quit")
    print("  /backup          - create memory backup now")
    print("  /status          - show status")
    print("  /clear           - CLEAR ALL memory and facts (confirm required)")
    print("  /forget <topic>  - delete memories containing <topic>")
    print("  /export          - save full conversation history")
    print("  Or say: 'Clear my memory', 'Forget about X', etc.")
    print("  Any other text   - treated as user query")

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

            elif user_input == "/status":
                print(f"\n📊 STATUS:")
                print(f"Memory entries: {len(agent.memory)}")
                print(f"Structured facts: {agent.facts}")
                if torch.cuda.is_available():
                    print(f"GPU VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB used")

            elif user_input == "/clear":
                print("⚠️ Warning: This will erase all memory and facts permanently.")
                confirm = input("Type 'CONFIRM' to proceed: ").strip()
                if confirm == "CONFIRM":
                    agent.facts = {}
                    agent.memory = []
                    agent._save_facts()
                    agent._save_memory()
                    agent._rebuild_chroma_index()
                    agent._pending_name_conflict = False
                    print("✅ All memory and facts cleared.")
                else:
                    print("❌ Clear aborted.")

            elif user_input.startswith("/forget "):
                topic = user_input[8:].strip()
                if agent.forget_memory(topic):
                    print(f"✅ Memories about '{topic}' forgotten.")
                else:
                    print(f"❌ No memories found about '{topic}'.")

            elif user_input == "/export":
                export_path = os.path.join(agent.memory_dir, "conversation_export.txt")
                with open(export_path, "w", encoding="utf-8") as f:
                    for i, mem in enumerate(agent.memory):
                        f.write(f"[{i+1}] {mem}\n")
                print(f"📤 Conversation exported to: {export_path}")

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