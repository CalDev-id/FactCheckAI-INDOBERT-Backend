import csv
import json
import random

input_files = [
    "../claim_only/train.csv",
    "../claim_only/test.csv",
    "../claim_only/validation.csv",
]

rows = []
for csv_path in input_files:
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "query": row.get("Headline", ""),
                    "label": row.get("label", ""),
                    "corpus": row.get("text", ""),
                }
            )

random.seed(42)
random.shuffle(rows)

num_total = len(rows)
num_train = int(num_total * 0.8)
train_rows = rows[:num_train]
test_rows = rows[num_train:]

num_val = int(len(train_rows) * 0.2)
val_rows = train_rows[:num_val]

with open("train.json", "w") as f:
    json.dump(train_rows, f, ensure_ascii=False, indent=2)

with open("validation.json", "w") as f:
    json.dump(val_rows, f, ensure_ascii=False, indent=2)

with open("test.json", "w") as f:
    json.dump(test_rows, f, ensure_ascii=False, indent=2)

print(f"Total data: {num_total}")
print(f"Train: {len(train_rows)}")
print(f"Validation: {len(val_rows)}")
print(f"Test: {len(test_rows)}")
