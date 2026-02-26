import json
import random
from pathlib import Path
from typing import Any


def load_json_list(file_path: Path) -> list[dict[str, Any]]:
    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"Format JSON bukan list di {file_path.name}")

    return data


def main() -> None:
    current_dir = Path(__file__).resolve().parent
    input_path = current_dir / "scored_combined_flat.json"

    if not input_path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {input_path.name}")

    data = load_json_list(input_path)

    for item in data:
        if "label" in item and item["label"] is not None:
            item["label"] = int(item["label"])

    rng = random.Random(42)
    rng.shuffle(data)

    total = len(data)
    train_size = int(total * 0.8)

    train_data = data[:train_size]
    test_data = data[train_size:]

    val_size = int(len(train_data) * 0.2)
    val_data = train_data[:val_size]

    output_paths = {
        "train.json": train_data,
        "test.json": test_data,
        "val.json": val_data,
    }

    for name, payload in output_paths.items():
        output_path = current_dir / name
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

    print(
        "Split selesai: "
        f"train={len(train_data)}, test={len(test_data)}, val={len(val_data)}"
    )


if __name__ == "__main__":
    main()
