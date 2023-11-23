"""
This is the place to connect the pot request of the API
All you need is to feed the user request in to the search_index() function the return the `result` back
you can comment query = str(input("Ask Something:  ")) after testing

"""

import os
from rag_feat.main import search_index
from langchain.embeddings.openai import OpenAIEmbeddings

open_ai= os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(openai_api_key=open_ai)

query = str(input("Ask Something:  "))

results = search_index(query, embeddings)

answer = results["answer"]
confidence_score = results["score"]

print(f"Answer: {answer}\n\nConfidence Score: {confidence_score}")
