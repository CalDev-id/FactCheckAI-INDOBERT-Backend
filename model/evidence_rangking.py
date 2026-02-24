import json
import torch
from sentence_transformers import SentenceTransformer

# Load data from data.json
with open('/kaggle/input/mendaley-combined/mendaley_combined.json', 'r') as f:
    data = json.load(f)

# Initialize empty lists for corpus and queries
corpus = []
queries = []
evidence_metadata = []
labels = []

# Process JSON data to fill corpus, queries, and labels
for item in data:
    queries.append({"id": item['id'], "query": item['query']})
    labels.append(item.get('label', ''))  # Assuming the label is stored under the key 'label'
    for evidence in item['evidence']:
        corpus.append(evidence['article'])
        evidence_metadata.append({
            "url": evidence.get('url', ''),
            "date": evidence.get('date', ''),
            "author": evidence.get('author', '')
        })

# Load the pre-trained SentenceTransformer model
embedder = SentenceTransformer("multi-qa-mpnet-base-dot-v1")

# Encode the corpus to get embeddings
corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

# Set the number of top results to retrieve
top_k = min(10, len(corpus))

# Initialize a list to store results
results = []

# Find and store the closest sentences for each query based on cosine similarity
for query_item, label in zip(queries, labels):
    query_id = query_item['id']
    query_text = query_item['query']
    query_embedding = embedder.encode(query_text, convert_to_tensor=True)

    # Compute cosine similarity scores between the query embedding and corpus embeddings
    similarity_scores = torch.matmul(query_embedding, corpus_embeddings.T).squeeze()
    scores, indices = torch.topk(similarity_scores, k=top_k)

    query_result = {
        "id": query_id,
        "query": query_text,
        "label": label,  # Add the label here
        "results": []
    }

    for score, idx in zip(scores, indices):
        idx_item = idx.item()
        query_result["results"].append({
            "corpus": corpus[idx_item],
            "score": score.item(),
            "url": evidence_metadata[idx_item]['url'],
            "date": evidence_metadata[idx_item]['date'],
            "author": evidence_metadata[idx_item]['author']
        })

    results.append(query_result)

# Write results to output JSON file
with open('/kaggle/working/mendaley_sentenced.json', 'w') as f:
    json.dump(results, f, indent=4)

print("Results saved to output.json")
