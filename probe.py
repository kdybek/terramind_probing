import numpy as np
import zarr
import pickle
import os
import sys
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler


DATA_DIR = "data"
LATENTS_PATH = os.path.join(DATA_DIR, "latents.zarr")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.pkl")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
SEED = 42
N_INTERACTION_SAMPLES = 30


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


def encode_target(lats, lons, coordinate_encoding):
    if coordinate_encoding == "none":
        return np.column_stack((lats, lons))
    elif coordinate_encoding == "sincos":
        lat_rad = np.radians(lats)
        lon_rad = np.radians(lons)

        return np.column_stack([
            np.sin(lat_rad),
            np.cos(lat_rad),
            np.sin(lon_rad),
            np.cos(lon_rad),
        ])
    elif coordinate_encoding == "spherical":
        lat_rad = np.radians(lats)
        lon_rad = np.radians(lons)
        x = np.cos(lat_rad) * np.cos(lon_rad)
        y = np.cos(lat_rad) * np.sin(lon_rad)
        z = np.sin(lat_rad)
        return np.column_stack((x, y, z))
    else:
        raise ValueError(f"Unknown coordinate encoding: {coordinate_encoding}")


def decode_target(encoded, coordinate_encoding):
    if coordinate_encoding == "none":
        assert encoded.shape[1] == 2, "Expected 2D coordinates for 'none' encoding"
        return encoded[:, 0], encoded[:, 1]
    elif coordinate_encoding == "sincos":
        assert encoded.shape[1] == 4, "Expected 4D coordinates for 'sincos' encoding"
        lat_rad = np.arctan2(encoded[:,0], encoded[:,1])
        lon_rad = np.arctan2(encoded[:,2], encoded[:,3])

        return np.degrees(lat_rad), np.degrees(lon_rad)
    elif coordinate_encoding == "spherical":
        assert encoded.shape[1] == 3, "Expected 3D coordinates for 'spherical' encoding"
        x, y, z = encoded[:, 0], encoded[:, 1], encoded[:, 2]

        # Normalize the vector to ensure it's on the unit sphere
        norm = np.sqrt(x**2 + y**2 + z**2)
        x /= norm
        y /= norm
        z /= norm

        lat_rad = np.arcsin(np.clip(z, -1.0, 1.0))  # Clip to avoid numerical issues
        lon_rad = np.arctan2(y, x)
        return np.degrees(lat_rad), np.degrees(lon_rad)
    else:
        raise ValueError(f"Unknown coordinate encoding: {coordinate_encoding}")


def add_interaction_features(latents, n_interactions, seed):
    if n_interactions == 0:
        return latents

    rng = np.random.default_rng(seed)
    _, n_features = latents.shape

    assert n_interactions <= (n_features * (n_features + 1)) // 2, \
        "n_interactions exceeds the number of possible unique interactions"

    # All (i,j) with i <= j
    rows, cols = np.triu_indices(n_features)

    idx = rng.choice(rows.size, size=n_interactions, replace=False)

    interaction_features = (
        latents[:, rows[idx]] *
        latents[:, cols[idx]]
    )

    return np.hstack((latents, interaction_features))


def run_probe(
        latents_root,
        model_name,
        lats,
        lons,
        coordinate_encoding,
        n_interactions,
):
    latents = np.asarray(latents_root[:])

    outer_cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    model = make_pipeline(
        StandardScaler(),
        TransformedTargetRegressor(
            regressor=RidgeCV(
                alphas=np.logspace(-2, 8, 11)
            ),
            transformer=StandardScaler()
        )
    )

    results = []

    y = encode_target(lats, lons, coordinate_encoding=coordinate_encoding)

    for interaction_seed in range(N_INTERACTION_SAMPLES):

        X = add_interaction_features(
            latents,
            n_interactions=n_interactions,
            seed=interaction_seed
        )

        fold_scores = []

        for train_idx, val_idx in outer_cv.split(X):

            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y[train_idx]
            lats_val = lats[val_idx]
            lons_val = lons[val_idx]

            model.fit(X_train, y_train)

            pred_encoded = model.predict(X_val)

            lats_pred, lons_pred = decode_target(
                pred_encoded, coordinate_encoding=coordinate_encoding
            )

            distances = [
                compute_geodesic_distance(lat1, lon1, lat2, lon2)
                for lat1, lon1, lat2, lon2 in zip(lats_val, lons_val, lats_pred, lons_pred)
            ]

            fold_scores.append(np.mean(distances))

        results.append({
            "interaction_seed": interaction_seed,
            "n_interactions": n_interactions,
            "model_name": model_name,
            "coordinate_encoding": coordinate_encoding,
            "mean_pred_err_km": np.mean(fold_scores),
            "std_pred_err_km": np.std(fold_scores),
        })

    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python probe.py <run_id>")
        sys.exit(1)

    run_id = int(sys.argv[1])

    os.makedirs(RESULTS_DIR, exist_ok=True)

    res_path = os.path.join(RESULTS_DIR, f"{run_id}.pkl")
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

    run_probe_args = []
    for model_name in model_names:
        num_layers = num_layers_dict[model_name]
        latents_root = root[model_name][f"layer_{num_layers - 1}"]

        for coordinate_encoding in ["none", "sincos", "spherical"]:
            for n_interactions in [0, 10, 50, 100, 500, 1000]:
                run_probe_args.append((
                    latents_root,
                    model_name,
                    lats,
                    lons,
                    coordinate_encoding,
                    n_interactions,
                ))

    res = run_probe(*run_probe_args[run_id])

    with open(res_path, "wb") as f:
        pickle.dump(res, f)


if __name__ == "__main__":
    main()
