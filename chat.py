from rag_feat.main import search_index

query = str(input("Ask Something:  "))

results = search_index(query)
print(f"THese are results {results}")

answer = results["answer"]
confidence_score = results["score"]

print(f"Answer: {answer}\n\nConfidence Score: {confidence_score}")
