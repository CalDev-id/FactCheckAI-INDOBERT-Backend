import json

json_path = "updated_json_file.json"
with open(json_path, "r") as f:
    data = json.load(f)

# Count the number of articles
num_articles = len(data)
print(f"Jumlah artikel yang berhasil di-scrape: {num_articles}")

# Count the total number of evidence entries
num_evidence = 0
for item in data:
    evidences = item.get("results", [])
    if isinstance(evidences, list):
        num_evidence += len(evidences)

print(f"Jumlah evidence total: {num_evidence}")
