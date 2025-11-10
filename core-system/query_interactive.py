import os
from dotenv import load_dotenv
from openai import OpenAI
import faiss
import numpy as np
import json

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

print("Loading saved embeddings...\n")

# Load saved data
index = faiss.read_index("faiss_index.bin")
chunks = np.load("chunks.npy", allow_pickle=True).tolist()
with open("metadata.json", "r") as f:
    metadata = json.load(f)

companies = sorted(set(m['company'] for m in metadata))
print(f"Loaded {index.ntotal} vectors from {len(companies)} companies: {', '.join(companies)}")
print(f"Each question costs ~$0.005 (half a cent)\n")

def ask_question(question, company_filter=None):
    q_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    )
    q_embedding = np.array([q_response.data[0].embedding]).astype('float32')
    
    k = 50 if company_filter else 6
    distances, indices = index.search(q_embedding, k)
    
    if company_filter:
        filtered_results = []
        for idx in indices[0]:
            if metadata[idx]["company"].lower() == company_filter.lower():
                filtered_results.append(idx)
                if len(filtered_results) >= 6:
                    break
        result_indices = filtered_results
    else:
        result_indices = indices[0][:6]
    
    context_chunks = []
    sources = []
    for idx in result_indices:
        context_chunks.append(chunks[idx])
        sources.append(f"{metadata[idx]['company']} - {metadata[idx]['filename']}")
    
    context = "\n\n".join(context_chunks)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are analyzing earnings call transcripts from multiple companies. Answer based only on the provided context. When relevant, mention which company you're referring to."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        temperature=0
    )
    
    return response.choices[0].message.content, list(set(sources))[:3]

print("=" * 60)
print(" INTERACTIVE MULTI-COMPANY Q&A")
print("=" * 60)
print("\nCommands:")
print("  - Just type your question for cross-company analysis")
print("  - Type 'filter:salesforce' before question to filter by company")
print("  - Type 'quit' to exit\n")
print("Example questions:")
print("  - Compare AI strategies across companies")
print("  - filter:microsoft What is Azure's revenue growth?")
print("  - What are the biggest challenges mentioned?")
print("  - Which company is most bullish on AI?\n")

question_count = 0
while True:
    user_input = input("Your question: ")
    
    if user_input.lower() in ['quit', 'exit', 'q']:
        print(f"\nAsked {question_count} questions. Total cost: ~${question_count * 0.005:.3f}")
        break
    
    # Check for company filter
    company_filter = None
    if user_input.lower().startswith("filter:"):
        parts = user_input.split(" ", 1)
        if len(parts) == 2:
            company_filter = parts[0].replace("filter:", "").strip()
            question = parts[1]
        else:
            print("Invalid filter format. Use: filter:company Your question here")
            continue
    else:
        question = user_input
    
    print("-" * 60)
    answer, sources = ask_question(question, company_filter)
    print(f"{answer}")
    print(f"Sources: {', '.join(sources)}\n")
    question_count += 1