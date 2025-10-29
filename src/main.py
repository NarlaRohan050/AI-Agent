# src/main.py
from local_reasoner import LocalMemoryAgent
import torch

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = LocalMemoryAgent(device=device)
    print("\n🧠 Local Reasoning Agent with Contextual Memory Ready!\n")

    while True:
        query = input("🔍 Ask me something (or type 'exit'): ").strip()
        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            print("👋 Goodbye! Memory saved.")
            break

        result = agent.query_with_context(query)

        print("\n💬 AI Agent Answer:")
        print(result.get("answer", "⚠️ No answer generated."))

        if result.get("context"):
            print("\n📚 Retrieved Contexts:")
            for c in result["context"]:
                print(f"- {c}")
        else:
            print("\n📚 No related memory found yet.")
        print("\n" + "-" * 80 + "\n")
