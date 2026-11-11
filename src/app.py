import streamlit as st
import requests
import pandas as pd
from components.plots import plot_PCA
from pathlib import Path

class CancerDoc:

    def __init__(self):

        # original data uses 30 features
        # selection happens inside the pipeline but the pipeline expects 30 features
        self.tumor_features = [
            "patient_id",
            "date_diagnosis",                                 
            "radius_mean",
            "texture_mean",
            "perimeter_mean",
            "area_mean",
            "smoothness_mean",
            "compactness_mean",
            "concavity_mean",
            "concave_points_mean",
            "symmetry_mean",
            "fractal_dimension_mean",
            "radius_se",
            "texture_se",
            "perimeter_se",
            "area_se",
            "smoothness_se",
            "compactness_se",
            "concavity_se",
            "concave_points_se",
            "symmetry_se",
            "fractal_dimension_se",
            "radius_worst",
            "texture_worst",
            "perimeter_worst",
            "area_worst",
            "smoothness_worst",
            "compactness_worst",
            "concavity_worst",
            "concave_points_worst",
            "symmetry_worst",
            "fractal_dimension_worst"
            ]
        self.pca_plot = None

        # predictions
        self.prediction = None
        self.probability_malignant = None

        # depending of diagnosis this is the color shown
        self.diagnosis_colors = {
            "M" : "#d62728",
            "B" : "#2ca02c",
            "W" : "#ffbf00",
            None : "#2563a6"
        }

        # general settings
        st.set_page_config(layout="wide",menu_items={})
        
        with st.container(border=True):
            st.header("IU Model Engineeing - Tumor diagnosis assistant")

        # layout
        self.side_container, self.main_container = st.columns([0.5,0.5], border=False)

        self.build_sidebar()
        self.load_pca_plot()
        self.build_main_app()

    def load_pca_plot(self) -> None:
        data_location = Path(__file__).parents[0] / "artifacts" / "data" / "baseline_data.csv"
        df = pd.read_csv(data_location)
        self.pca_plot = plot_PCA(df)

    def build_sidebar(self):    
        
        n_columns = 5
        with self.side_container:
            with st.container(border=True):
                st.header("Tumor features")
            self.form = st.form("feature_input", border=True)
            with self.form:
                for i in range(0, len(self.tumor_features), n_columns):
                    cols = st.columns(n_columns)
                    for col, feature in zip(cols, self.tumor_features[i:i+n_columns]):
                        display_label = feature.replace("_", " ").capitalize()
                        with col:
                            if feature == "patient_id":
                                st.text_input(display_label, key=feature, value="")
                                continue
                            elif feature == "date_diagnosis":
                                st.date_input(display_label, key=feature)
                                continue
                            st.number_input(display_label, key=feature, min_value=0.0, format="%.6f")

                submitted = st.form_submit_button("Predict", key="predict_button")
                if submitted:
                    self.diagnose_tumor()
    
    def build_main_app(self) -> None:
        with self.main_container:
            self.diagnosis_container = st.container(border=True)
            with self.diagnosis_container:
                st.header("Diagnosis")
                self.diagnosis_display = st.empty()
                self.probability_display = st.empty()

                self.display_prediction(
                    self.prediction,
                    self.probability_malignant
                )
            self.plot_container = st.container(border=True)
            with self.plot_container:
                self.plot = st.pyplot(self.pca_plot)
    
    def display_prediction(self, diagnosis:str|None, probability) -> None:

        color = self.diagnosis_colors[diagnosis]
        if diagnosis is None:
            background_color = "#e8f1fb"
            legend = "Awaiting prediction"
        elif diagnosis == "W":
            background_color = "#FFFDD0"
            legend = "Tumor values cannot be 0"
        else:
            background_color = f"{color}20"
            legend = diagnosis
        
        self.diagnosis_display.markdown(
                        f'<div style="background-color: {background_color}; color: {color}; '
                        'padding: 12px; border-radius: 8px; font-weight: bold;">'
                        f'{legend}'
                        '</div>',
                        unsafe_allow_html=True
                        )
        self.probability_display.markdown(
                        f'<p style="color: #000000; font-size: 20px;">'
                        f'Probability of malignancy: --'
                        f'</p>',
                        unsafe_allow_html=True
                        )

        if probability is not None:
            self.probability_display.markdown(
                f'<p style="color: {color}; font-size: 20px;">'
                f'Probability of malignancy: {probability:.2%}'
                f'</p>',
                unsafe_allow_html=True
            )

    def diagnose_tumor(self) -> None:
        tumor_data = {}
        for feature in self.tumor_features:
            if feature == "date_diagnosis":
                tumor_data[feature] = st.session_state[feature].isoformat()
            else:
                tumor_data[feature] = st.session_state[feature]
                if tumor_data[feature] == 0:
                    self.prediction = "W"
                    return
        try:
            response  = requests.post("http://127.0.0.1:8000/predict", json=tumor_data)
            response.raise_for_status()
        except requests.HTTPError as err:
            print(err)
        else:
            prediction = response.json()
            self.prediction = prediction["diagnosis"]
            self.probability_malignant = prediction["probability_malignant"]

if __name__ == "__main__":

    CancerDoc()
