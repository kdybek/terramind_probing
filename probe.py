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
MAIN_RES_DIR = os.path.join(DATA_DIR, "results_main")
CIRC_ENC_RES_DIR = os.path.join(DATA_DIR, "results_circ_enc")
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

    clf = get_probe(probe_name)

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


def test_earth_pred(
        latents_root,
        model_name,
        lats,
        lons,
        probe_name,
        circular_encoding
):
    latents = np.asarray(latents_root[:])

    clf_lat = get_probe(probe_name)
    clf_lon = get_probe(probe_name)

    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

    mean_distances = []
    for train_idx, test_idx in cv.split(latents):
        latents_train, latents_test = latents[train_idx], latents[test_idx]
        lats_train, lats_test = lats[train_idx], lats[test_idx]
        lons_train, lons_test = lons[train_idx], lons[test_idx]

        if circular_encoding:
            lons_train = np.concatenate(
                [np.sin(np.radians(lons_train))[:, np.newaxis],
                 np.cos(np.radians(lons_train))[:, np.newaxis]], axis=1
            )

        clf_lat.fit(latents_train, lats_train)
        clf_lon.fit(latents_train, lons_train)

        lat_preds = clf_lat.predict(latents_test)
        lon_preds = clf_lon.predict(latents_test)

        if circular_encoding:
            lon_preds = np.degrees(np.arctan2(lon_preds[:, 0], lon_preds[:, 1]))

        distances = compute_geodesic_distance(
            lat_preds, lon_preds, lats_test, lons_test
        )

        mean_distances.append(np.mean(distances))

    return [
        {
            "model": model_name,
            "probe": probe_name,
            "mean_distance": mean_distance,
            "fold": i,
            "circular_encoding": circular_encoding,
        }
        for i, mean_distance in enumerate(mean_distances)
    ]


def main():
    if len(sys.argv) < 2:
        print("Usage: python probe.py <run_id>")
        sys.exit(1)

    id = int(sys.argv[1])

    os.makedirs(MAIN_RES_DIR, exist_ok=True)
    os.makedirs(CIRC_ENC_RES_DIR, exist_ok=True)

    main_res_path = os.path.join(MAIN_RES_DIR, f"{id}.pkl")
    circ_enc_res_path = os.path.join(CIRC_ENC_RES_DIR, f"{id}.pkl")
    if os.path.exists(main_res_path) or os.path.exists(circ_enc_res_path):
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

    test_earth_pred_args = [
        (root[model_name][f"layer_{num_layers_dict[model_name] - 1}"],
         model_name, preds["lat"], preds["lon"], probe_name, circular_encoding)
        for model_name in model_names
        for probe_name in probe_names
        for circular_encoding in [False, True]
    ]

    if id < len(run_probe_args):
        res = run_probe(*run_probe_args[id])

        with open(main_res_path, "wb") as f:
            pickle.dump(res, f)
    else:
        res = test_earth_pred(*test_earth_pred_args[id - len(run_probe_args)])

        with open(circ_enc_res_path, "wb") as f:
            pickle.dump(res, f)


if __name__ == "__main__":
    main()
