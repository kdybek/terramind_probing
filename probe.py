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


DATA_DIR = "data"
LATENTS_PATH = os.path.join(DATA_DIR, "latents.zarr")
METADATA_PATH = os.path.join(DATA_DIR, "metadata_3.pkl")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
SEED = 42
RFF_COMPONENT_COUNT_LIST = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
RF_TREE_DEPTH_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


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

'''
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


class MyMultiOutputRegressor:
    def __init__(self, estimator_factory):
        self.estimator_factory = estimator_factory
        self.estimators_ = []

    def fit(self, X, y):
        self.estimators_ = []

        for i in range(y.shape[1]):
            estimator = self.estimator_factory()
            estimator.fit(X, y[:, i])
            self.estimators_.append(estimator)

        return self

    def predict(self, X):
        return np.stack(
            [est.predict(X) for est in self.estimators_],
            axis=1
        )


def run_probe_coords(
        latents_root,
        model_name,
        lats,
        lons,
        coordinate_encoding,
        fm_rank
):
    latents = np.asarray(latents_root[:])

    groups = create_spatial_groups(lats, lons)

    outer_cv = GroupShuffleSplit(
        n_splits=5,
        test_size=0.2,
        random_state=SEED
    )

    results = []

    y = encode_target(lats, lons, coordinate_encoding=coordinate_encoding)

    for i, (train_idx, val_idx) in enumerate(outer_cv.split(latents, y, groups=groups)):
        X_train, X_val = latents[train_idx], latents[val_idx]
        y_train = y[train_idx]
        lats_train, lats_val = lats[train_idx], lats[val_idx]
        lons_train, lons_val = lons[train_idx], lons[val_idx]

        fm = MyMultiOutputRegressor(
            lambda: myfm.MyFMRegressor(
                rank=fm_rank,
                random_seed=SEED
            )
        )

        fm.fit(X_train, y_train)

        pred_encoded_val = fm.predict(X_val)
        pred_encoded_train = fm.predict(X_train)

        r2_train = r2_score(y_train, pred_encoded_train, multioutput='variance_weighted')
        r2_val = r2_score(y[val_idx], pred_encoded_val, multioutput='variance_weighted')

        lats_pred_train, lons_pred_train = decode_target(
            pred_encoded_train, coordinate_encoding=coordinate_encoding
        )
        lats_pred_val, lons_pred_val = decode_target(
            pred_encoded_val, coordinate_encoding=coordinate_encoding
        )

        distances_train = [
            compute_geodesic_distance(lat1, lon1, lat2, lon2)
            for lat1, lon1, lat2, lon2 in zip(lats_train, lons_train, lats_pred_train, lons_pred_train)
        ]
        distances_val = [
            compute_geodesic_distance(lat1, lon1, lat2, lon2)
            for lat1, lon1, lat2, lon2 in zip(lats_val, lons_val, lats_pred_val, lons_pred_val)
        ]

        results.append({
            "fold": i,
            "fm_rank": fm_rank,
            "model_name": model_name,
            "coordinate_encoding": coordinate_encoding,
            "mean_train_err_km": np.mean(distances_train),
            "median_train_err_km": np.median(distances_train),
            "p90_train_err_km": np.percentile(distances_train,90),
            "mean_val_err_km": np.mean(distances_val),
            "median_val_err_km": np.median(distances_val),
            "p90_val_err_km": np.percentile(distances_val,90),
            "r2_train": r2_train,
            "r2_val": r2_val
        })

    return results
'''

def run_probe(
        latents_root,
        metadata,
        groups,
        model_name,
        target,
        layer,
        modality_group,
        regressor_type
    ):
    latents = np.asarray(latents_root[:])

    cv = GroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    results = []

    y = np.array([r[target] for r in metadata]).reshape(-1, 1)

    if regressor_type == "rff":
        capacity_list = RFF_COMPONENT_COUNT_LIST
    elif regressor_type == "rf":
        capacity_list = RF_TREE_DEPTH_LIST
    elif regressor_type == "baseline":
        capacity_list = [None]
    else:
        raise ValueError(f"Unknown regressor type: {regressor_type}")

    for capacity in capacity_list:
        print(f"capacity: {capacity}, model: {model_name}, target: {target}, layer: {layer}, modality_group: {modality_group}, regressor_type: {regressor_type}")
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
                "layer": layer,
                "modality_group": modality_group,
                "regressor": regressor_type,
                "capacity": capacity,
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

    groups = create_spatial_groups(lats, lons)
    
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
    modality_groups = ["optical", "radar", "vegetation", "terrain", "landcover"]

    run_probe_args = []
    for model_name in model_names:
        num_layers = num_layers_dict[model_name]
        latents_root = root[model_name]["all"][f"layer_{num_layers - 1}"]

        for target in targets:
            run_probe_args.append((
                latents_root,
                metadata,
                groups,
                model_name,
                target,
                num_layers - 1,
                "all",
                "rff"
            ))

    for model_name in ["terramind_v1_base"]:
        num_layers = num_layers_dict[model_name]

        for layer in range(num_layers - 1):
            latents_root = root[model_name]["all"][f"layer_{layer}"]

            for target in targets:
                run_probe_args.append((
                    latents_root,
                    metadata,
                    groups,
                    model_name,
                    target,
                    layer,
                    "all",
                    "rff"
                ))

    for model_name in ["terramind_v1_base"]:
        num_layers = num_layers_dict[model_name]
    
        for modality_group in modality_groups:
            latents_root = root[model_name][modality_group][f"layer_{num_layers - 1}"]
    
            for target in targets:
                run_probe_args.append((
                    latents_root,
                    metadata,
                    groups,
                    model_name,
                    target,
                    num_layers - 1,
                    modality_group,
                    "rff"
                ))

    for model_name in model_names:
        num_layers = num_layers_dict[model_name]
        latents_root = root[model_name]["all"][f"layer_{num_layers - 1}"]
    
        for target in targets:
            run_probe_args.append((
                latents_root,
                metadata,
                groups,
                model_name,
                target,
                num_layers - 1,
                "all",
                "rf"
            ))

    for model_name in ["terramind_v1_base"]:
        num_layers = num_layers_dict[model_name]
        latents_root = root[model_name]["all"][f"layer_{num_layers - 1}"]

        for target in targets:
            run_probe_args.append((
                latents_root,
                metadata,
                groups,
                model_name,
                target,
                num_layers - 1,
                "all",
                "baseline"
            ))

    res = run_probe(*run_probe_args[run_id])

    with open(res_path, "wb") as f:
        pickle.dump(res, f)


if __name__ == "__main__":
    main()
