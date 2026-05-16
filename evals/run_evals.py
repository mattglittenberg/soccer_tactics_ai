import json
from pathlib import Path
from src.assistant import chat 

def run_evals(questions_path: str = "evals/questions.json") -> dict:
    questions = json.loads(Path(questions_path).read_text())
    results = []

    for q in questions:
        print(f"Running: {q['id']}...")

        # Reset history for each question
        answer = chat(q["question"])
        answer_lower = answer.lower()

        # Score: did the answer mention required terms?
        hits = [term for term in q["must_mention"]
                if term.lower() in answer_lower]
        misses = [term for term in q["must_mention"]
                  if term.lower() not in answer_lower]
        violations = [term for term in q["must_not_mention"]
                      if term.lower() in answer_lower]

        score = len(hits) / len(q["must_mention"]) if q["must_mention"] else 1.0

        results.append({
            "id": q["id"],
            "category": q["category"],
            "score": round(score, 2),
            "hits": hits,
            "misses": misses,
            "violations": violations,
            "answer_preview": answer[:200] + "..."
        })

    # Summary
    avg_score = sum(r["score"] for r in results) / len(results)
    print(f"\n{'='*50}")
    print(f"EVAL RESULTS — {len(results)} questions")
    print(f"Average score: {avg_score:.0%}")
    print(f"{'='*50}")

    for r in results:
        status = "✅" if r["score"] == 1.0 else "⚠️" if r["score"] >= 0.5 else "❌"
        print(f"{status} {r['id']} ({r['category']}): {r['score']:.0%}")
        if r["misses"]:
            print(f"   Missing: {r['misses']}")

    return {"average_score": avg_score, "results": results}

if __name__ == "__main__":
    run_evals()