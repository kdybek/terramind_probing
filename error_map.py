import numpy as np
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GroupKFold
import zarr
import pickle
import os
import sys
import pandas as pd


DATA_DIR = "data"
LATENTS_PATH = os.path.join(DATA_DIR, "latents.zarr")
METADATA_PATH = os.path.join(DATA_DIR, "metadata_3.pkl")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
SEED = 42
RFF_COMPONENT_COUNT = 16384


def create_spatial_groups(
    lats,
    lons,
    lat_bin_size=10,
    lon_bin_size=10
):

    lat_bins = np.floor(
        (lats + 90) / lat_bin_size
    )

    lon_bins = np.floor(
        (lons + 180) / lon_bin_size
    )

    groups = lat_bins.astype(int) * 1000 + lon_bins.astype(int)

    return groups


def run_probe_error_map(
        latents_root,
        metadata,
        groups,
        model_name,
        target,
        layer,
        modality_group,
        regressor_type,
        lats,
        lons
    ):
    latents = np.asarray(latents_root[:])

    cv = GroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    results = []

    y = np.array([r[target] for r in metadata]).reshape(-1, 1)

    for capacity in [RFF_COMPONENT_COUNT]:
        if regressor_type == "rff":
            regressor = Pipeline([
                ("x_scaler", StandardScaler()),
                ("rff", RBFSampler(
                    gamma="scale",
                    n_components=capacity,
                    random_state=SEED
                )),
                ("regressor", Ridge(alpha=1.0, random_state=SEED))
            ])
        elif regressor_type == "rf":
            regressor = RandomForestRegressor(
                max_depth=capacity,
                n_estimators=300,
                max_features="sqrt",
                random_state=SEED,
                n_jobs=-1
            )
        elif regressor_type == "baseline":
            regressor = DummyRegressor(strategy="mean")

        for i, (train_idx, val_idx) in enumerate(cv.split(latents, y, groups=groups)):
            X_train, X_val = latents[train_idx], latents[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            lats_val = lats[val_idx]
            lons_val = lons[val_idx]

            y_scaler = StandardScaler()
            y_train = y_scaler.fit_transform(y_train)
            y_val = y_scaler.transform(y_val)

            regressor.fit(X_train, y_train)

            y_pred_val = regressor.predict(X_val).reshape(-1, 1)

            y_pred_unscaled = y_scaler.inverse_transform(y_pred_val)
            y_true_unscaled = y_scaler.inverse_transform(y_val)

            for y_pred, y_true, lat, lon, y_pred_uns, y_true_uns in zip(y_pred_val, y_val, lats_val, lons_val, y_pred_unscaled, y_true_unscaled, strict=True):
                error = y_pred - y_true
                error_unscaled = y_pred_uns - y_true_uns

                results.append({
                    "fold": i,
                    "model_name": model_name,
                    "target": target,
                    "layer": layer,
                    "modality_group": modality_group,
                    "regressor": regressor_type,
                    "capacity": capacity,
                    "lat": lat,
                    "lon": lon,
                    "error": error.item(),
                    "error_unscaled": error_unscaled.item()
                })

    return results


def main():
    res_path = os.path.join(RESULTS_DIR, f"error_map.csv")

    root = zarr.open(LATENTS_PATH, mode="r")

    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)

    lats = np.array([r["center_lat"] for r in metadata])
    lons = np.array([r["center_lon"] for r in metadata])

    groups = create_spatial_groups(lats, lons)

    all_results = []

    for target in ["bio01", "bio04", "bio12", "bio15"]:
        res = run_probe_error_map(
            latents_root=root["terramind_v1_base"]["all"]["layer_11"],
            metadata=metadata,
            groups=groups,
            model_name="terramind_v1_base",
            target=target,
            layer=11,
            modality_group="all",
            regressor_type="rff",
            lats=lats,
            lons=lons
        )

        all_results.extend(res)

    all_results_df = pd.DataFrame(all_results)

    with open(res_path, "w") as f:
        all_results_df.to_csv(f, index=False)


if __name__ == "__main__":
    main()
