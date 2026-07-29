import numpy as np
import zarr
import pickle
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor


DATA_DIR = "data"
LATENTS_PATH = os.path.join(DATA_DIR, "latents.zarr")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.pkl")
RESULTS_PATH = os.path.join(DATA_DIR, "results.csv")
SEED = 42


def run_probe(latents, model_name, tgt_name, tgt, probe_name):
    if probe_name == "ridge":
        clf = make_pipeline(
            StandardScaler(), RidgeCV(alphas=np.logspace(-4, 4, 30))
        )
    elif probe_name == "xgboost":
        clf = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            verbosity=0,
        )
    elif probe_name == "mlp":
        clf = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(64,),
                activation="relu",
                solver="adam",
                batch_size=256,
                learning_rate_init=1e-3,
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                random_state=SEED,
            ),
        )
    else:
        raise ValueError(f"Unknown probe: {probe_name}")

    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(clf, latents, tgt, cv=cv, scoring="r2")

    return [
        {
            "model": model_name,
            "target": tgt_name,
            "probe": probe_name,
            "score": score,
            "split": i,
        }
        for i, score in enumerate(scores)
    ]


def main():
    root = zarr.open(LATENTS_PATH, mode="r")

    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)

    preds = {
        "lat": np.array([r["center_lat"] for r in metadata]),
        "lon": np.array([r["center_lon"] for r in metadata]),
    }

    model_names = [
        "terramind_v1_tiny",
        "terramind_v1_small",
        "terramind_v1_base",
        "terramind_v1_large",
    ]
    probe_names = ["ridge", "xgboost", "mlp"]

    records = []

    with ThreadPoolExecutor(max_workers=6) as executor:
        for model_name in model_names:
            latents = np.asarray(root[model_name][:])

            futures = [
                executor.submit(
                    run_probe,
                    latents,
                    model_name,
                    tgt_name,
                    tgt,
                    probe_name,
                )
                for tgt_name, tgt in preds.items()
                for probe_name in probe_names
            ]

            for future in as_completed(futures):
                records.extend(future.result())

    df = pd.DataFrame.from_records(records)
    df.to_csv(RESULTS_PATH, index=False)


if __name__ == "__main__":
    main()
