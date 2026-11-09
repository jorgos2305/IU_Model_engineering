import streamlit as st
import pandas as pd
from components.plots import plot_PCA
from pathlib import Path

class CancerDoc:

    def __init__(self):

        self.tumor_features = [
            'area_deviation_relative', 'area_mean', 'area_se',
            'compactness_deviation_relative', 'compactness_mean', 'compactness_se',
            'smoothness_deviation_relative', 'smoothness_mean',
            'symmetry_deviation_relative', 'symmetry_mean',
            'texture_deviation_relative', 'texture_mean'
            ]
        self.pca_plot = None

        # predictions
        self.prediction = None
        self.probability_malignant = None

        self.diagnosis_colors = {
            "M" : "#d62728",
            "B" : "#2ca02c"
        }

        # general settings
        st.set_page_config(layout="wide",menu_items={})
        st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)
        
        st.title("IU Model Engineeing - Tumor diagnosis assistant")

        # layout
        self.side_container, self.main_container = st.columns([.2,.5], border=False)

        self.load_pca_plot()
        self.build_sidebar()
        self.build_main_app()

    def load_pca_plot(self):

        data_location = Path(__file__).parents[0] / "artifacts" / "data" / "baseline_data.csv"
        df = pd.read_csv(data_location)
        self.pca_plot = plot_PCA(df)
    
    def build_main_app(self):
        with self.main_container:
            self.diagnosis_container = st.container(border=False)
            with self.diagnosis_container:
                st.header("Diagnosis")

                self.diagnosis_display = st.empty()
                self.probability_display = st.empty()

                self.display_prediction(
                    self.prediction,
                    self.probability_malignant
                )

            self.plot_container = st.container(border=False)
            with self.plot_container:
                self.plot = st.pyplot(self.pca_plot)

    def display_prediction(self, diagnosis, probability):
        if diagnosis is None:
            self.diagnosis_display.info("Awaiting prediction")
            self.probability_display.markdown("**Probability of malignancy:** -")
            return
        
        color = self.diagnosis_colors[diagnosis.lower()]
        self.diagnosis_display.markdown(
            f'<h3 style="color: {color};">{diagnosis.capitalize()}</h3>',
            unsafe_allow_html=True
        )

        if probability is not None:
            self.probability_display.markdown(
                f'<p style="color: {color}; font-size: 20px;">'
                f'Probability of malignancy: {probability:.2%}'
                f'</p>',
                unsafe_allow_html=True
            )
    
    def build_sidebar(self):
        with self.side_container:
            st.header("Tumor features")
            self.form = st.form("feature_input", border=False)
            with self.form:
                for i in range(0, len(self.tumor_features), 2):
                    cols = st.columns(2)

                    for col, feature in zip(cols, self.tumor_features[i:i+2]):
                        with col:
                            st.number_input(feature.replace("_", " ").capitalize(), key=feature, min_value=0)

                submitted = st.form_submit_button("Predict")
    

if __name__ == "__main__":

    CancerDoc()
