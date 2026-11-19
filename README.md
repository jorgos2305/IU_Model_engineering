# IU International University of Applied Sciences

## Course: Model Engineering

Welcome to the repository of the Model Engineering course.  
The task for this project consists in developing a classification model that predicts a tumor to be malignant or benign.

### Objectives
- The model should achieve a F1 score > 0.95
- The model should be interpretable
- Propose how the model can be integrated into doctors' workflow, e.g., through a GUI

## Dataset information
- Name: Diagnostic Wisconsin Breast Cancer Database
- Source: [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

## Project structure

```bash
.
├── data                                         # Contains original datasets
│   ├── Data_for_Task_1.csv
│   └── baseline_data.csv
├── notebooks                                    # Notebooks describing data exploration and modeling
│   ├── 1_EDA.ipynb
│   └── 2_Modeling.ipynb
└── src                                          # Complete application
    ├── api                                      # REST API for serving predictions
    │   ├── __init__.py
    │   ├── dataschemas.py
    │   └── server.py
    ├── app.py                                   # Application entry point
    ├── artifacts                                # Work items produced during exploration and modeling
    │   ├── __init__.py
    │   └── models
    ├── components                               # Utilities used by the applications
    │   ├── __init__.py
    │   ├── plots.py
    │   └── transformations.py
    └── training                                 # Training service - creates new models and registers them in mlflow
        └── mlflow_utils.py
```

## How to run the system locally
### Conda environment

This project was implemented using a conda environment. To replicate it run the following commands:
```bash
git clone git@github.com:jorgos2305/IU_Model_engineering.git
cd IU_Model_engineering
conda env create -f environment.yml
conda activate modeleng
```
### Environment variables
Copy the ```.env.example``` file and modify the values according to your environment variables.
```bash
cp .env.example .env
```

### MLflow
Open a new terminal make sure you are in the correct conda environment and navigate to the folder where you want to store the MLflow database.

Once there, run:
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
```

### Streamlit
In a new terminal windows, navigate to the ```IU_Model_engineering``` folder and run:
```bash
streamlit run src/app.py
```