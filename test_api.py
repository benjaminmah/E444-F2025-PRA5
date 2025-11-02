import time, csv, pathlib
import requests
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict

BASE_URL = "http://ece444-pra5-env.eba-uxzhdqtz.us-east-2.elasticbeanstalk.com/predict"
N_CALLS = 100
TIMEOUT = 15
OUT_DIR = pathlib.Path("results"); OUT_DIR.mkdir(exist_ok=True)

TESTS = {
    "fake_1": "Breaking! Aliens invade New York City!",
    "fake_2": "Celebrity admits immortality serum works after hidden trials.",
    "real_1": "Bank of Canada holds key interest rate steady this month.",
    "real_2": "Toronto Raptors clinch a playoff berth after road win."
}

FIELDS = ["case","iter","status_code","latency_ms","prediction","timestamp_start_iso"]

def run_case(name, text):
    rows = []
    for i in range(1, N_CALLS + 1):
        t0 = time.time()
        try:
            r = requests.post(BASE_URL, json={"text": text}, timeout=TIMEOUT)
            latency = (time.time() - t0) * 1000
            pred = None
            try:
                pred = r.json().get("prediction")
            except Exception:
                pred = None
            rows.append({
                "case": name,
                "iter": i,
                "status_code": r.status_code,
                "latency_ms": round(latency, 2),
                "prediction": pred,
                "timestamp_start_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)),
            })
        except requests.exceptions.RequestException:
            latency = (time.time() - t0) * 1000
            rows.append({
                "case": name,
                "iter": i,
                "status_code": -1,
                "latency_ms": round(latency, 2),
                "prediction": None,
                "timestamp_start_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)),
            })

    # per-case CSV (exactly 100 rows)
    with open(OUT_DIR / f"{name}_latency.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader(); w.writerows(rows)
    return rows

def main():
    all_rows = []
    for name, text in TESTS.items():
        print(f"Running {name}…")
        all_rows.extend(run_case(name, text))

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / "latency_all_cases.csv", index=False)

    # Only successful calls for latency stats/plot
    ok = df[df["status_code"] == 200]
    plt.figure(figsize=(9,6))
    ax = ok.boxplot(column="latency_ms", by="case", grid=False, return_type=None)
    plt.title("API Latency per Test Case (ms)"); plt.suptitle("")
    plt.xlabel("Test Case"); plt.ylabel("Latency (ms)")

    # Compute and annotate averages on the plot
    avgs = ok.groupby("case")["latency_ms"].mean().round(2)
    max_by_case: Dict[str, float] = ok.groupby("case")["latency_ms"].max().to_dict()
    # Pandas boxplot orders categories alphabetically by default
    categories = sorted(avgs.index.tolist())
    for idx, case in enumerate(categories, start=1):
        mean_val = avgs[case]
        y = max(max_by_case.get(case, mean_val), mean_val) * 1.02
        plt.text(idx, y, f"{mean_val:.2f} ms", ha="center", va="bottom", fontsize=9, color="#333")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "latency_boxplot.png", dpi=300)

    with open(OUT_DIR / "latency_averages.txt", "w") as f:
        f.write("Average latency (ms) by case\n")
        for k, v in avgs.items():
            f.write(f"{k}: {v}\n")

    print("\nSaved:")
    print(" - results/fake_1_latency.csv (100 rows)")
    print(" - results/fake_2_latency.csv (100 rows)")
    print(" - results/real_1_latency.csv (100 rows)")
    print(" - results/real_2_latency.csv (100 rows)")
    print(" - results/latency_all_cases.csv")
    print(" - results/latency_boxplot.png")
    print(" - results/latency_averages.txt")
    print("\nAverages (ms):\n", avgs.to_string())

if __name__ == "__main__":
    main()
