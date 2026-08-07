import os
import pickle
import pandas as pd


def main():
    DATA_DIR = "data"

    for dir_name in os.listdir(DATA_DIR):
        dir_path = os.path.join(DATA_DIR, dir_name)

        # Only process directories starting with "results"
        if not (os.path.isdir(dir_path) and dir_name.startswith("results")):
            continue

        all_results = []

        for file in os.listdir(dir_path):
            if file.endswith(".pkl"):
                file_path = os.path.join(dir_path, file)

                with open(file_path, "rb") as f:
                    result = pickle.load(f)

                all_results.extend(result)

        if not all_results:
            print(f"No results found in {dir_name}, skipping.")
            continue

        df = pd.DataFrame(all_results)

        output_file = os.path.join(DATA_DIR, f"{dir_name}.csv")
        df.to_csv(output_file, index=False)

        print(f"Collated {len(all_results)} rows from {dir_name} into {output_file}")


if __name__ == "__main__":
    main()
