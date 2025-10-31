# test_agent_max_capacity.py
import os
import sys
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from local_reasoner import LocalMemoryAgent

def print_test(title):
    print("\n" + "="*80)
    print(f"🧪 TEST: {title}")
    print("="*80)

def run_query(agent, query, expected_contains=None, should_not_contain=None):
    print(f"\n👤 User: {query}")
    start = time.time()
    result = agent.answer_with_memory(query, top_k=5)
    latency = time.time() - start
    answer = result["answer"]
    context = result["context"]
    print(f"🤖 AI ({latency:.2f}s): {answer}")
    if context:
        print(f"📚 Context snippets: {len(context)} retrieved")
        for i, c in enumerate(context[:2], 1):
            print(f"   [{i}] {c[:100]}...")
    else:
        print("📚 No context retrieved.")

    if expected_contains:
        for phrase in expected_contains:
            assert phrase.lower() in answer.lower(), f"Expected '{phrase}' in answer, but got: {answer}"
    if should_not_contain:
        for phrase in should_not_contain:
            assert phrase.lower() not in answer.lower(), f"Should NOT contain '{phrase}', but got: {answer}"
    return answer, context

def main():
    print("🚀 Starting MAX CAPACITY & CORRECTNESS TEST SUITE")
    agent = LocalMemoryAgent(device="cuda")  # GPU

    agent.memory = []
    agent.facts = {}
    agent._save_memory()
    agent._save_facts()
    agent.index.reset()
    agent.memory_embeddings = None

    # TEST 1: Identity & Redundancy
    print_test("Identity Declaration & Redundancy Prevention")
    run_query(agent, "My name is Alex.")
    run_query(agent, "What is my name?", expected_contains=["Alex"])
    run_query(agent, "I am Alex.")
    run_query(agent, "Who am I?", expected_contains=["Alex"])
    print(f"✅ Memory entries: {len(agent.memory)}")

    # TEST 2: Conflict Resolution
    print_test("Identity Conflict Handling")
    run_query(agent, "Actually, I'm Jordan.")
    run_query(agent, "Yes, I am Jordan.")
    run_query(agent, "What is my name now?", expected_contains=["Jordan"])

    # TEST 3: Semantic Understanding & Merging
    print_test("Semantic Understanding & Interest Merging")
    run_query(agent, "I enjoy ethical hacking and AI research.")
    run_query(agent, "I like penetration testing and machine learning.")
    run_query(agent, "List my interests.", expected_contains=["ethical hacking", "AI", "penetration testing", "machine learning"])

    print("\n" + "="*80)
    print("🎉 ALL TESTS PASSED!")
    print("="*80)

if __name__ == "__main__":
    main()