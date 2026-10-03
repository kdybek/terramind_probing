import numpy as np
import zarr
import pickle
import os
import sys
from sklearn.linear_model import Ridge
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV, GroupKFold


DATA_DIR = "data"
LATENTS_PATH = os.path.join(DATA_DIR, "latents.zarr")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.pkl")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
SEED = 42
N_INTERACTION_SAMPLES = 30


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

    groups = (
        lat_bins.astype(int) * 1000
        +
        lon_bins.astype(int)
    )

    return groups


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



def add_interaction_features(latents, hessian_frac, seed):
    if hessian_frac == 0.0:
        return latents

    rng = np.random.default_rng(seed)
    _, n_features = latents.shape

    n_interactions = round(hessian_frac * (n_features * (n_features + 1)) // 2)

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


def svd_RRR(X, Y, rnk, lambda_=0):
    """
        Perform Ridge Regularized Reduced Rank Regression (RRR) using SVD.
        Parameters:
            X : np.ndarray
                Input data matrix (n_samples, n_input_neurons).
            Y : np.ndarray
                Output data matrix (n_samples, n_output_neurons).
            rnk : int
                Dimensionaility of communication
            lambda_ : float
                Regularization parameter (default is 0 for no regularization).
        Returns:
            w0 : np.ndarray
                Estimate of the communication strength (n_input_neurons, n_output_neurons).
            urrr : np.ndarray
                Input axes (n_input_neurons, rnk).
            vrrr : np.ndarray
                Output axes, orthonormal (n_output_neurons, rnk).
    """
    # Check if X and Y are 2D arrays
    # Ridge regularization
    XX = X.T @ X + lambda_ * np.eye(X.shape[1])

    # Least squares estimate with ridge
    if np.linalg.cond(XX) < 1e10:
        wridge = np.linalg.solve(XX, X.T @ Y)
    else:
        wridge = np.linalg.pinv(XX) @ (X.T @ Y)

    # SVD of relevant matrix
    _, _, vrrr = np.linalg.svd(Y.T @ X @ wridge)

    # Get the top 'rnk' components
    vrrr = vrrr[:rnk, :].T   # shape: (features, rnk)
    urrr = wridge @ vrrr    # shape: (features, rnk)

    # Construct full RRR estimate
    w0 = urrr @ vrrr.T
    vrrr = vrrr.T  # for compatibility with original code's return

    return w0, urrr, vrrr


class ReducedRankRidge(
    BaseEstimator,
    RegressorMixin
):
    """
    Ridge reduced-rank regression estimator
    using the svd_RRR solver.
    """

    def __init__(
        self,
        rank=2,
        alpha=1.0,
        fit_intercept=True
    ):
        self.rank = rank
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        if self.fit_intercept:
            self.X_mean_ = X.mean(axis=0)
            self.y_mean_ = y.mean(axis=0)

            Xc = X - self.X_mean_
            yc = y - self.y_mean_
        else:
            Xc = X
            yc = y

        W, U, V = svd_RRR(
            Xc,
            yc,
            rnk=self.rank,
            lambda_=self.alpha
        )

        self.coef_ = W.T

        self.input_axes_ = U
        self.output_axes_ = V

        self.n_features_in_ = X.shape[1]

        # Recover intercept
        if self.fit_intercept:
            self.intercept_ = self.y_mean_ - self.X_mean_ @ W
        else:
            self.intercept_ = np.zeros(y.shape[1])

        return self

    def predict(self, X):
        X = np.asarray(X)
        return X @ self.coef_.T + self.intercept_


class MyRidge(
    BaseEstimator,
    RegressorMixin
):
    """
    A custom Ridge regression estimator.
    """
    def __init__(self, alphas, groups):
        self.alphas = alphas
        self.groups = groups

    def fit(self, X, y):
        self.x_scaler_ = StandardScaler()
        self.y_scaler_ = StandardScaler()

        X_scaled = self.x_scaler_.fit_transform(X)
        y_scaled = self.y_scaler_.fit_transform(y)

        inner_cv = GroupKFold(
            n_splits=3,
            shuffle=True,
            random_state=SEED + 1
        )

        self.cv_ = GridSearchCV(
            estimator=Ridge(),
            param_grid={"alpha": self.alphas},
            cv=inner_cv,
            scoring="neg_mean_squared_error",
            n_jobs=-1
        )
        self.cv_.fit(X_scaled, y_scaled, groups=self.groups.copy())

        self.model_ = self.cv_.best_estimator_
        self.coef_ = self.model_.coef_

        return self

    def predict(self, X):
        X_scaled = self.x_scaler_.transform(X)

        y_scaled = self.model_.predict(X_scaled)

        return self.y_scaler_.inverse_transform(y_scaled)


def run_probe(
        latents_root,
        model_name,
        lats,
        lons,
        coordinate_encoding,
        hessian_frac,
        reduced_dim
):
    latents = np.asarray(latents_root[:])

    groups = create_spatial_groups(lats, lons)

    outer_cv = GroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    results = []

    y = encode_target(lats, lons, coordinate_encoding=coordinate_encoding)

    for interaction_seed in range(N_INTERACTION_SAMPLES):
        fold_scores = []
        for train_idx, val_idx in outer_cv.split(latents, y, groups=groups):

            X_train, X_val = latents[train_idx], latents[val_idx]
            y_train = y[train_idx]
            groups_train = groups[train_idx]
            lats_val = lats[val_idx]
            lons_val = lons[val_idx]

            # This is a workaround to pass the groups to the model during fitting
            model = MyRidge(alphas=np.logspace(-2, 8, 11), groups=groups_train)

            if reduced_dim is not None and reduced_dim < X_train.shape[1]:
                selector = RFE(estimator=model, n_features_to_select=reduced_dim, step=0.35)
                X_train = selector.fit_transform(X_train, y_train)
                X_val = selector.transform(X_val)

            len_X_train = len(X_train)

            X_all = np.vstack((X_train, X_val))
            X_all = add_interaction_features(
                X_all,
                hessian_frac=hessian_frac,
                seed=interaction_seed
            )
            X_train = X_all[:len_X_train]
            X_val = X_all[len_X_train:]

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
            "hessian_frac": hessian_frac,
            "reduced_dim": reduced_dim,
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

        for coordinate_encoding in ["none"]:
            for hessian_frac in np.linspace(0.0, 0.1, 6):
                run_probe_args.append((
                    latents_root,
                    model_name,
                    lats,
                    lons,
                    coordinate_encoding,
                    hessian_frac,
                    192
                ))

    res = run_probe(*run_probe_args[run_id])

    with open(res_path, "wb") as f:
        pickle.dump(res, f)


if __name__ == "__main__":
    main()
