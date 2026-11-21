import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA

class PCAprojector:

    def __init__(self, n_components:int) -> None:
        self.pca = PCA(n_components=n_components)
        self.features = None

    def fit(self, df:pd.DataFrame) -> None:
        self.features = df.columns.to_list()
        self.pca.fit(df)

    def transform(self, df:pd.DataFrame) -> pd.DataFrame:
        if self.features is None:
            raise RuntimeError("PCA has not been fitted yet")

        transformed = self.pca.transform(df)

        return pd.DataFrame(
            transformed,
            columns=("Projection axis 1", "Projection axis 2")
            )

def plot_PCA(df_proj:pd.DataFrame, new_data_point:pd.DataFrame=None) -> plt.Figure:

    # create plot
    fig, ax = plt.subplots(figsize=(8,5))
    # define colors for the diagnosis
    palette = {
        "B": "#2ca02c",
        "M": "#d62728"
    }
    # this plots the complete dataset
    sns.scatterplot(
        df_proj,
        x="Projection axis 1", 
        y="Projection axis 2", 
        hue="Diagnosis", 
        palette=palette, 
        alpha=0.9, 
        ax=ax)
    # plot the decision of the new observation
    if new_data_point is not None:
        ax.scatter(
            new_data_point["Projection axis 1"],
            new_data_point["Projection axis 2"],
            color=palette[new_data_point.loc[0,"Diagnosis"]],
            marker="+",
            label="New data point"
        )
    # format the axis
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, linewidth=0.5, linestyle=":")
    ax.set_title("Tumor diagnosis")
    ax.legend()
    fig.tight_layout()
        
    return fig