import numpy as np
import zarr
import pickle
import os
import sys
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
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


def get_probe(probe_name):
    if probe_name == "ridge":
        return make_pipeline(
            StandardScaler(),
            RidgeCV()
        )
    elif probe_name == "xgboost":
        return XGBRegressor(random_state=SEED)
    elif probe_name == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(random_state=SEED),
        )
    else:
        raise ValueError(f"Invalid probe name: {probe_name}")


def compute_geodesic_distance(lat1, lon1, lat2, lon2):
    R = 6371.0  # Radius of the Earth in kilometers

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * \
        np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def run_probe(
        latents_root,
        model_name,
        lats,
        lons,
        probe_name,
        layer,
        n_components,
        dim_reduction,
        circular_encoding
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

    reg_lat = get_probe(probe_name)
    reg_lon = get_probe(probe_name)

    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

    entries = []
    for i, (train_idx, test_idx) in enumerate(cv.split(latents)):
        latents_train, latents_test = latents[train_idx], latents[test_idx]
        lats_train, lats_test = lats[train_idx], lats[test_idx]
        lons_train, lons_test = lons[train_idx], lons[test_idx]

        if circular_encoding:
            lons_train = np.column_stack([
                np.sin(np.radians(lons_train)),
                np.cos(np.radians(lons_train))
            ])

        reg_lat.fit(latents_train, lats_train)
        reg_lon.fit(latents_train, lons_train)

        lat_preds = reg_lat.predict(latents_test)
        lon_preds = reg_lon.predict(latents_test)

        r2_lat = r2_score(lats_test, lat_preds)

        if circular_encoding:
            sin_lon_true = np.sin(np.radians(lons_test))
            cos_lon_true = np.cos(np.radians(lons_test))

            r2_sin_lon = r2_score(sin_lon_true, lon_preds[:, 0])
            r2_cos_lon = r2_score(cos_lon_true, lon_preds[:, 1])

            lon_preds = np.degrees(np.arctan2(lon_preds[:, 0], lon_preds[:, 1]))

        else:
            r2_lon = r2_score(lons_test, lon_preds)

        distances = compute_geodesic_distance(
            lats_test, lons_test, lat_preds, lon_preds
        )

        entry = {
            "model": model_name,
            "probe": probe_name,
            "layer": layer,
            "n_components": n_components,
            "dim_reduction": dim_reduction,
            "circular_encoding": circular_encoding,
            "fold": i,
            "pred_error_km": np.mean(distances),
            "r2_lat": r2_lat,
        }

        if circular_encoding:
            entry.update({
                "r2_sin_lon": r2_sin_lon,
                "r2_cos_lon": r2_cos_lon,
            })
        else:
            entry.update({
                "r2_lon": r2_lon,
            })

        entries.append(entry)

    return entries


def main():
    if len(sys.argv) < 2:
        print("Usage: python probe.py <run_id>")
        sys.exit(1)

    id = int(sys.argv[1])

    os.makedirs(RESULTS_DIR, exist_ok=True)

    res_path = os.path.join(RESULTS_DIR, f"{id}.pkl")
    if os.path.exists(res_path):
        sys.exit(0)

    root = zarr.open(LATENTS_PATH, mode="r")

    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)

    lats = np.array([r["center_lat"] for r in metadata])
    lons = np.array([r["center_lon"] for r in metadata])
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
                    (latents_root, model_name, lats, lons,
                     probe_name, layer, None, "none", True)
                    for probe_name in probe_names
                ]
            )

            if layer == num_layers - 1:
                # Add no circular encoding runs
                run_probe_args.extend(
                    [
                        (latents_root, model_name, lats, lons,
                         probe_name, layer, None, "none", False)
                        for probe_name in probe_names
                    ]
                )

                if model_name != "terramind_v1_tiny":
                    # Add PCA and random projection runs
                    TINY_MODEL_DIM = 192
                    run_probe_args.extend(
                        [
                            (latents_root, model_name, lats, lons, probe_name,
                             layer, TINY_MODEL_DIM, proj, True)
                            for probe_name in probe_names
                            for proj in ["pca", "random"]
                        ]
                    )

    res = run_probe(*run_probe_args[id])

    with open(res_path, "wb") as f:
        pickle.dump(res, f)


if __name__ == "__main__":
    main()
