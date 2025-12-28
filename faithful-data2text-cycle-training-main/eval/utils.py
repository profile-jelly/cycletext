import json
import os

def save_results(results: dict, filename: str):
    os.makedirs(os.path.dirname(filename), exist_ok=True) if os.path.dirname(filename) else None

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Results saved to {filename}")
