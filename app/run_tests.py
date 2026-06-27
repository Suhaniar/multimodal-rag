import requests
import json
import time
import pandas as pd
import os

BASE_URL = "http://127.0.0.1:8000"

# ---------- helpers ----------
def upload(file_path):
    with open(file_path, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/ingest",
            files={"files": (os.path.basename(file_path), f)}
        )
    return r.json()

def query(question):
    payload = {
        "question": question,
        "top_k": 5,
        "alpha": 0.7,
        "include_sources": True
    }
    start = time.time()
    r = requests.post(f"{BASE_URL}/query", json=payload)
    latency = round(time.time() - start, 3)
    return r.json(), latency

def get_health():
    return requests.get(f"{BASE_URL}/health").json()

def delete_file(name):
    return requests.delete(f"{BASE_URL}/delete", json={"filenames":[name]}).json()

# ---------- runner ----------
def run():
    with open("test_cases.json") as f:
        tests = json.load(f)

    results = []

    print("\n🚀 Starting RAG Test Suite...\n")

    for t in tests:
        print(f"Running Test {t['id']} - {t['name']}")

        result = {
            "id": t["id"],
            "name": t["name"],
            "type": t["type"]
        }

        try:
            # --- SYSTEM TEST ---
            if t.get("endpoint") == "/health":
                res = get_health()
                result["status"] = "PASS" if res.get("status") == "ok" else "FAIL"
                result["response"] = res

            # --- INGEST ---
            elif t.get("endpoint") == "/ingest":
                if "file" in t:
                    res = upload(t["file"])
                else:
                    res = upload(t["files"][0])
                result["status"] = "PASS"
                result["response"] = res

            # --- DELETE ---
            elif t.get("endpoint") == "/delete":
                res = delete_file(t["file"])
                result["status"] = "PASS"
                result["response"] = res

            # --- QUERY ---
            else:
                response, latency = query(t["question"])
                answer = response.get("llama_answer", "")
                grounded = response.get("grounded", False)

                result["answer"] = answer
                result["grounded"] = grounded
                result["latency"] = latency

                # validation logic
                status = "PASS"

                if "expected_contains" in t:
                    status = "PASS" if t["expected_contains"].lower() in answer.lower() else "FAIL"

                if t.get("expect_grounded") is False:
                    status = "PASS" if grounded is False else "FAIL"

                result["status"] = status
                result["response"] = response

        except Exception as e:
            result["status"] = "FAIL"
            result["error"] = str(e)

        results.append(result)

    # save outputs
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    pd.DataFrame(results).to_excel("test_results.xlsx", index=False)

    print("\n✅ Completed! Results saved to JSON + Excel")

if __name__ == "__main__":
    run()