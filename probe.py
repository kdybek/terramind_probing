import numpy as np
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
import zarr
import pickle
import os
import sys
import h3


DATA_DIR = "data"
LATENTS_PATH = os.path.join(DATA_DIR, "latents.zarr")
METADATA_PATH = os.path.join(DATA_DIR, "metadata_3.pkl")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
SEED = 42


def assign_h3_blocks(lat, lon, resolution):
    """
    Assign points to H3 spatial blocks.

    Parameters
    ----------
    lat : array-like
        Latitude in degrees
    lon : array-like
        Longitude in degrees
    resolution : int
        H3 resolution (0-15)

    Returns
    -------
    ndarray
        H3 cell IDs
    """

    return [
        h3.latlng_to_cell(
            float(la),
            float(lo),
            resolution
        )
        for la, lo in zip(lat, lon)
    ]


def run_probe(
        latents_root,
        metadata,
        lats,
        lons,
        resolution,
        model_name,
        target,
        regressor_type
    ):
    latents = np.asarray(latents_root[:])

    cv = GroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    results = []

    groups = assign_h3_blocks(lats, lons, resolution)

    y = np.array([r[target] for r in metadata]).reshape(-1, 1)

    if regressor_type == "linear":
        regressor = Pipeline([
            ("x_scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0, random_state=SEED))
        ])
    elif regressor_type == "mlp":
        regressor = Pipeline([
            ("x_scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                learning_rate_init=1e-3,
                batch_size=512,
                max_iter=300,
                early_stopping=False,
                random_state=SEED
            ))
        ])

    for i, (train_idx, val_idx) in enumerate(cv.split(latents, y, groups=groups)):
        X_train, X_val = latents[train_idx], latents[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        y_train_unscaled = y_train.copy()
        y_val_unscaled = y_val.copy()

        y_scaler = StandardScaler()
        y_train = y_scaler.fit_transform(y_train)
        y_val = y_scaler.transform(y_val)

        regressor.fit(X_train, y_train)

        y_pred_train = regressor.predict(X_train).reshape(-1, 1)
        y_pred_val = regressor.predict(X_val).reshape(-1, 1)

        y_pred_train_unscaled = y_scaler.inverse_transform(y_pred_train)
        y_pred_val_unscaled = y_scaler.inverse_transform(y_pred_val)

        train_rmse = root_mean_squared_error(
            y_train,
            y_pred_train
        )

        val_rmse = root_mean_squared_error(
            y_val,
            y_pred_val
        )

        train_rmse_unscaled = root_mean_squared_error(
            y_train_unscaled,
            y_pred_train_unscaled
        )

        val_rmse_unscaled = root_mean_squared_error(
            y_val_unscaled,
            y_pred_val_unscaled
        )

        train_r2 = r2_score(y_train, y_pred_train)
        val_r2 = r2_score(y_val, y_pred_val)

        results.append({
            "fold": i,
            "model_name": model_name,
            "target": target,
            "resolution": resolution,
            "regressor": regressor_type,
            "train_rmse": train_rmse,
            "val_rmse": val_rmse,
            "train_rmse_unscaled": train_rmse_unscaled,
            "val_rmse_unscaled": val_rmse_unscaled,
            "train_r2": train_r2,
            "val_r2": val_r2
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
    targets = ["bio01", "bio04", "bio12", "bio15"]
    resolutions = [2, 3, 4, 5, 6, 7]

    run_probe_args = []
    for model_name in model_names:
        num_layers = num_layers_dict[model_name]
        latents_root = root[model_name]["all"][f"layer_{num_layers - 1}"]

        for target in targets:
            for regressor_type in ["linear", "mlp"]:
                for resolution in resolutions:
                    run_probe_args.append((
                        latents_root,
                        metadata,
                        lats,
                        lons,
                        resolution,
                        model_name,
                        target,
                        regressor_type
                    ))

    res = run_probe(*run_probe_args[run_id])

    with open(res_path, "wb") as f:
        pickle.dump(res, f)


if __name__ == "__main__":
    main()
