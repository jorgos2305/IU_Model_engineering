import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA

def _calculate_PCA(df:pd.DataFrame) -> pd.DataFrame:
    X = df.drop(["id", "diagnosis"], axis=1)
    y = df["diagnosis"]

    pca = PCA(n_components=2)
    X_transformed = pca.fit_transform(X)
    return pd.DataFrame(np.c_[X_transformed, y.values], columns=["feature_1", "feature_2", "Diagnosis"])

def plot_PCA(X:pd.DataFrame) -> plt.Figure:

    # calculate PCA
    df_proj = _calculate_PCA(X)
    # create plot
    fig, ax = plt.subplots(figsize=(10,4))
    palette = {
        "B": "#2ca02c",
        "M": "#d62728"
    }
    sns.scatterplot(df_proj, x="feature_1", y="feature_2", hue="Diagnosis", palette=palette, ax=ax)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, linewidth=0.5, linestyle=":")
    ax.set_title("PCA")
    fig.tight_layout()

    return fig