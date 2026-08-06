import numpy as np
import zarr
import pickle
import os
import sys
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import PCA
from xgboost import XGBRegressor


DATA_DIR = "data"
LATENTS_PATH = os.path.join(DATA_DIR, "latents.zarr")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.pkl")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
SEED = 42


def apply_pca(latents, n_components):
    scaler = StandardScaler()
    latents = scaler.fit_transform(latents)
    pca = PCA(n_components=n_components, random_state=SEED)
    latents_pca = pca.fit_transform(latents)
    return latents_pca


def apply_random_projection(latents, n_components):
    scaler = StandardScaler()
    latents = scaler.fit_transform(latents)
    rp = GaussianRandomProjection(n_components=n_components, random_state=SEED)
    latents_rp = rp.fit_transform(latents)
    return latents_rp


def run_probe(
        latents_root,
        model_name,
        tgt_name,
        tgt,
        probe_name,
        layer,
        n_components,
        dim_reduction,
        control
):
    latents = np.asarray(latents_root[:])

    assert dim_reduction in ["pca", "random", "none"], \
        "dim_reduction must be one of 'pca', 'random', or 'none'"
    assert not (dim_reduction != "none" and n_components is None), \
        "n_components must be specified if dim_reduction is not 'none'"
    assert probe_name in ["ridge", "xgboost", "mlp"], \
        "probe_name must be one of 'ridge', 'xgboost', or 'mlp'"

    if dim_reduction == "pca":
        latents = apply_pca(latents, n_components)
    elif dim_reduction == "random":
        latents = apply_random_projection(latents, n_components)

    n_components = latents.shape[1]

    if control:
        rng = np.random.default_rng(SEED)
        rng.shuffle(latents)

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

    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(clf, latents, tgt, cv=cv, scoring="r2")

    return [
        {
            "model": model_name,
            "target": tgt_name,
            "probe": probe_name,
            "score": score,
            "fold": i,
            "layer": layer,
            "n_components": n_components,
            "dim_reduction": dim_reduction,
            "control": control,
        }
        for i, score in enumerate(scores)
    ]


def main():
    if len(sys.argv) < 2:
        print("Usage: python probe.py <run_id>")
        sys.exit(1)

    id = int(sys.argv[1])

    os.makedirs(RESULTS_DIR, exist_ok=True)

    results_path = os.path.join(RESULTS_DIR, f"{id}.pkl")
    if os.path.exists(results_path):
        sys.exit(0)

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
    num_layers_dict = {
        "terramind_v1_tiny": 12,
        "terramind_v1_small": 12,
        "terramind_v1_base": 12,
        "terramind_v1_large": 24,
    }
    probe_names = ["ridge", "xgboost", "mlp"]

    run_probe_args = []
    for model_name in model_names:
        num_layers = num_layers_dict[model_name]
        for layer in range(num_layers):
            latents_root = root[model_name][f"layer_{layer}"]

            run_probe_args.extend(
                [
                    (latents_root, model_name, tgt_name,
                     tgt, probe_name, layer, None, "none", control)
                    for tgt_name, tgt in preds.items()
                    for probe_name in probe_names
                    for control in [False, True]
                ]
            )

            if layer == num_layers - 1 and model_name != "terramind_v1_tiny":
                run_probe_args.extend(
                    [
                        (latents_root, model_name, tgt_name, tgt,
                         probe_name, layer, 192, proj, control)
                        for tgt_name, tgt in preds.items()
                        for probe_name in probe_names
                        for proj in ["pca", "random"]
                        for control in [False, True]
                    ]
                )

    res = run_probe(*run_probe_args[id])

    with open(results_path, "wb") as f:
        pickle.dump(res, f)


if __name__ == "__main__":
    main()
