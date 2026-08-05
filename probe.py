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
from sklearn.decomposition import PCA
from xgboost import XGBRegressor


DATA_DIR = "data"
LATENTS_PATH = os.path.join(DATA_DIR, "latents.zarr")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.pkl")
RESULTS_PATH = os.path.join(DATA_DIR, "results.csv")
SEED = 42


def apply_pca(latents, n_components):
    scaler = StandardScaler()
    latents = scaler.fit_transform(latents)
    pca = PCA(n_components=n_components, random_state=SEED)
    latents_pca = pca.fit_transform(latents)
    return latents_pca


def run_probe(latents, model_name, tgt_name, tgt, probe_name, n_components, control):
    if n_components is not None:
        latents = apply_pca(latents, n_components)

    if control:
        np.random.shuffle(latents)

    if probe_name == "ridge":
        clf = make_pipeline(
            StandardScaler(),
            RidgeCV()
        )
    elif probe_name == "xgboost":
        clf = XGBRegressor(random_state=SEED)
    elif probe_name == "mlp":
        clf = make_pipeline(
            StandardScaler(),
            MLPRegressor(random_state=SEED),
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
            "n_components": n_components,
            "control": control,
        }
        for i, score in enumerate(scores)
    ]


def main():
    root = zarr.open(LATENTS_PATH, mode="r")

    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)

    preds_base = {
        "lat": np.array([r["center_lat"] for r in metadata]),
        "lon": np.array([r["center_lon"] for r in metadata]),
    }

    preds_additional = {
        "lon_sin": np.array([np.sin(np.radians(r["center_lon"])) for r in metadata]),
        "lon_cos": np.array([np.cos(np.radians(r["center_lon"])) for r in metadata]),
    }

    model_names = [
        "terramind_v1_tiny",
        "terramind_v1_small",
        "terramind_v1_base",
        "terramind_v1_large",
    ]
    num_layers_dict = {
        "terramind_v1_tiny": 12,
        "terramind_v1_small": 12,
        "terramind_v1_base": 12,
        "terramind_v1_large": 24,
    }
    probe_names = ["ridge", "xgboost", "mlp"]

    records = []

    with ThreadPoolExecutor(max_workers=256) as executor:
        for model_name in model_names:
            num_layers = num_layers_dict[model_name]
            for layer in range(num_layers):
                latents = np.asarray(root[model_name][f"layer_{layer}"][:])

                futures = [
                    executor.submit(
                        run_probe,
                        latents,
                        model_name,
                        tgt_name,
                        tgt,
                        probe_name,
                        None,
                        control
                    )
                    for tgt_name, tgt in preds_base.items()
                    for probe_name in probe_names
                    for control in [False, True]
                ]

                if layer == num_layers - 1:
                    futures += [
                        executor.submit(
                            run_probe,
                            latents,
                            model_name,
                            tgt_name,
                            tgt,
                            probe_name,
                            None,
                            control
                        )
                        for tgt_name, tgt in preds_additional.items()
                        for probe_name in probe_names
                        for control in [False, True]
                    ]

                    if model_name != "terramind_v1_tiny":
                        futures += [
                            executor.submit(
                                run_probe,
                                latents,
                                model_name,
                                tgt_name,
                                tgt,
                                probe_name,
                                192,
                                control
                            )
                            for tgt_name, tgt in preds_base.items()
                            for probe_name in probe_names
                            for control in [False, True]
                        ]

                for future in as_completed(futures):
                    results = future.result()
                    results["layer"] = layer
                    records.extend(results)

    df = pd.DataFrame.from_records(records)
    df.to_csv(RESULTS_PATH, index=False)


if __name__ == "__main__":
    main()
