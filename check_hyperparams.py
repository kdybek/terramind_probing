from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor


def print_default_params(model, name):
    print("=" * 80)
    print(f"Default hyperparameters for {name}")
    print("=" * 80)

    params = model.get_params()

    for key, value in params.items():
        print(f"{key}: {value}")

    print()


if __name__ == "__main__":

    ridge_cv = RidgeCV()
    mlp = MLPRegressor()
    xgb = XGBRegressor()

    print_default_params(ridge_cv, "sklearn RidgeCV")
    print_default_params(mlp, "sklearn MLPRegressor")
    print_default_params(xgb, "xgboost XGBRegressor")
