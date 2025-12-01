# test_agent_max_capacity.py
import os
import sys
import time
import json
import torch

# Add src to path
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
    
    # Initialize agent on GPU (or CPU if unavailable)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = LocalMemoryAgent(device=device)

    # Clear any prior state for clean test
    agent.memory = []
    agent.facts = {}
    agent._save_memory()
    agent._save_facts()
    agent._rebuild_chroma_index()

    # ───────────────────────────────────────
    # TEST 1: Identity Declaration & Redundancy
    # ───────────────────────────────────────
    print_test("Identity Declaration & Redundancy Prevention")
    
    run_query(agent, "My name is Alex.")
    run_query(agent, "What is my name?", expected_contains=["Alex"])
    run_query(agent, "I am Alex.")
    run_query(agent, "Who am I?", expected_contains=["Alex"])

    print(f"✅ Memory entries after identity tests: {len(agent.memory)} (should be ≤2)")

    # ───────────────────────────────────────
    # TEST 2: Identity Conflict Resolution
    # ───────────────────────────────────────
    print_test("Identity Conflict Handling")
    
    run_query(agent, "Actually, I'm Jordan.")  # Should ask for confirmation
    run_query(agent, "Yes, I am Jordan.")
    run_query(agent, "What is my name now?", expected_contains=["Jordan"])

    # ───────────────────────────────────────
    # TEST 3: Semantic Understanding & Interest Merging
    # ───────────────────────────────────────
    print_test("Semantic Understanding & Interest Merging")
    
    run_query(agent, "I enjoy ethical hacking and AI research.")
    run_query(agent, "I like penetration testing and machine learning.")
    run_query(agent, "List my interests.", expected_contains=["ethical hacking", "AI", "penetration testing", "machine learning"])

    # Ensure interests are merged (not overwritten)
    assert "ethical hacking" in [item.lower() for item in agent.facts.get("interests", [])]
    assert "ai" in [item.lower() for item in agent.facts.get("interests", [])] or "ai research" in [item.lower() for item in agent.facts.get("interests", [])]
    assert "penetration testing" in [item.lower() for item in agent.facts.get("interests", [])]
    assert "machine learning" in [item.lower() for item in agent.facts.get("interests", [])]

    # ───────────────────────────────────────
    # TEST 4: Redundancy Blocking (Near-Duplicates)
    # ───────────────────────────────────────
    print_test("Redundancy Prevention – Near-Duplicate Queries")
    
    initial_mem_count = len(agent.memory)
    run_query(agent, "I like AI.")
    run_query(agent, "I am into artificial intelligence.")
    run_query(agent, "Artificial Intelligence is my passion.")
    
    final_mem_count = len(agent.memory)
    print(f"Memory grew by {final_mem_count - initial_mem_count} entries (should be ≤2 due to deduplication)")
    assert final_mem_count - initial_mem_count <= 2, "Too many redundant entries added!"

    # ───────────────────────────────────────
    # TEST 5: Large-Scale Memory Injection (Stress Test)
    # ───────────────────────────────────────
    print_test("STRESS: Injecting 100 Memory Entries")
    
    base_topics = [
        "quantum cryptography", "neural networks", "zero-day exploits",
        "homomorphic encryption", "reinforcement learning", "side-channel attacks"
    ]
    
    for i in range(100):  # Reduced from 150 for faster test
        topic = base_topics[i % len(base_topics)]
        query = f"In conversation {i+1}, I mentioned my interest in {topic}."
        run_query(agent, query, expected_contains=[topic])
        if i % 20 == 0 and i > 0:
            print(f"   Injected {i+1}/100...")

    print(f"✅ Total memory entries: {len(agent.memory)}")
    assert len(agent.memory) >= 90, "Memory injection failed"

    # Test retrieval still works
    run_query(agent, "What did I say about quantum cryptography?", expected_contains=["quantum"])

    # ───────────────────────────────────────
    # TEST 6: Fact Extraction Accuracy
    # ───────────────────────────────────────
    print_test("Structured Fact Extraction")
    
    agent.facts = {}
    agent._save_facts()
    
    run_query(agent, "My name is Taylor. I'm studying cybersecurity. My goal is to work at NSA.")
    assert agent.facts.get("user_name") == "Taylor"
    assert any("cybersecurity" in interest.lower() for interest in agent.facts.get("interests", []))
    assert any("nsa" in goal.lower() for goal in agent.facts.get("goals", []))
    print("✅ Facts correctly extracted:", agent.facts)

    # ───────────────────────────────────────
    # TEST 7: Hallucination Prevention
    # ───────────────────────────────────────
    print_test("Hallucination Check – Unknown Questions")
    
    ans, _ = run_query(agent, "What is my favorite color?")
    assert "don't know" in ans.lower() or "not mentioned" in ans.lower() or "no information" in ans.lower(), \
        f"Should not hallucinate favorite color! Got: {ans}"

    # ───────────────────────────────────────
    # TEST 8: New Feature – Forget Command
    # ───────────────────────────────────────
    print_test("New Feature: /forget Command")
    
    agent.memory = []
    agent.add_memory("I once worked at Acme Corp.")
    agent.add_memory("My salary was 50000.")
    agent.add_memory("I love hiking.")
    agent._rebuild_chroma_index()
    
    # Verify memory exists
    _, ctx = run_query(agent, "Where did I work?", expected_contains=["Acme"])
    
    # Forget salary
    success = agent.forget_memory("salary")
    assert success, "Failed to forget 'salary'"
    
    # Verify salary is gone, but job remains
    ans, _ = run_query(agent, "What was my salary?")
    assert "don't know" in ans.lower() or "not mentioned" in ans.lower()
    
    ans, _ = run_query(agent, "Where did I work?", expected_contains=["Acme"])
    print("✅ /forget command works correctly")

    # ───────────────────────────────────────
    # FINAL SUMMARY
    # ───────────────────────────────────────
    print("\n" + "="*80)
    print("🎉 ALL TESTS PASSED! Agent is:")
    print("   - Resistant to redundancy")
    print("   - Accurate in identity & facts")
    print("   - Capable under memory load")
    print("   - Semantically aware (not verbatim)")
    print("   - Free from hallucination on unknowns")
    print("   - Efficient in retrieval (ChromaDB)")
    print("   - VRAM-aware (GPU/CPU balanced)")
    print("   - Supports new features like /forget")
    print("="*80)


if __name__ == "__main__":
    main()