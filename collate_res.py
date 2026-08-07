import pickle
import pandas as pd
import os


DATA_DIR = "data"
RESULTS_DIR = os.path.join(DATA_DIR, "results")
OUTPUT_FILE = os.path.join(DATA_DIR, "results.csv")


def main():
    all_results = []
    for file in os.listdir(RESULTS_DIR):
        if file.endswith(".pkl"):
            with open(os.path.join(RESULTS_DIR, file), "rb") as f:
                result = pickle.load(f)
                all_results.extend(result)

    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Collated {len(all_results)} rows into {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
