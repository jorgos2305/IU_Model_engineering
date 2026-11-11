import pandas as pd

def calculate_deviations(df:pd.DataFrame) -> pd.DataFrame:

    features = ["area", "compactness", "smoothness", "symmetry", "texture"]
    features_to_drop = ["perimeter_mean", "perimeter_worst", "perimeter_se", "radius_mean", "radius_worst", "radius_se"]

    X = df.drop(features_to_drop, axis=1)

    transformed = {}
    for feature in features:
        transformed[f"{feature}_mean"] = X.loc[:, f"{feature}_mean"]
        transformed[f"{feature}_deviation_relative"] = (X.loc[:, f"{feature}_worst"] - X.loc[:, f"{feature}_mean"]) / X.loc[:, f"{feature}_mean"]

    for feature in features[:2]:
        transformed.update(
            {
                f"{feature}_se" : X.loc[:, f"{feature}_se"]
            }
        )

    return pd.DataFrame(transformed, index=X.index)