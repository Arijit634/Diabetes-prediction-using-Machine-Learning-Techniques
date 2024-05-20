# Diabetes Prediction System

This repository contains code for a diabetes prediction system. The system is designed to predict whether a patient is likely to have diabetes based on various medical features.

## Introduction

Diabetes is a chronic disease that affects millions of people worldwide. Early detection and management of diabetes are crucial for preventing complications and improving health outcomes. Machine learning techniques offer a promising approach for predicting diabetes risk based on patient data.

## Features

The features used in this prediction system include:

- **Pregnancies:** Number of times pregnant. Reflects the pregnancy history of the patient.
- **Glucose:** Plasma glucose concentration measured 2 hours after an oral glucose tolerance test.
- **BloodPressure:** Diastolic blood pressure (mm Hg).
- **SkinThickness:** Triceps skinfold thickness (mm).
- **Insulin:** 2-hour serum insulin level (mu U/ml).
- **BMI:** Body mass index (weight in kg/(height in m)^2).
- **DiabetesPedigreeFunction:** A measure of genetic predisposition to diabetes based on family history.
- **Age:** Age of the patient in years.

## Diabetes Pedigree Function (DPF)

The Diabetes Pedigree Function (DPF) is a feature included in the prediction system to quantify the genetic predisposition to diabetes based on family history. The DPF value is calculated using the following formula:

DPF = (numerator_sum + 20) / (denominator_sum + 50)

Where:
- `numerator_sum` is the sum of shared genes with relatives diagnosed with diabetes multiplied by the difference between their age at diagnosis and 88.
- `denominator_sum` is the sum of shared genes with relatives without diabetes multiplied by the difference between their age at the last examination and 14.

The original formula for DPF was proposed in a paper published in 1984 and has been used in various studies to assess the familial aggregation of diabetes. (Reference:(https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2245318/))

The DPF value provides insight into the likelihood of developing diabetes based on family history. Higher DPF values indicate a stronger genetic predisposition to diabetes.

## Data Preprocessing and Model Training

The dataset used for training the model is loaded from a CSV file (`diabetes - DS.csv`). The following preprocessing steps are applied to the data:

- Missing values are replaced by 0.
- Zero values in certain columns (Glucose, BloodPressure, SkinThickness, Insulin, BMI) are replaced with NaN.
- Missing values are imputed using mean or median values.
- Standardization of features is performed using `StandardScaler`.
- Class imbalance is addressed using Random Over Sampling.

Custom implementation of two classifiers, Gradient Bossting (GB) and K-Nearest Neighbors (KNN), are trained separately and combined into an ensemble model using a custom Voting Classifier.

## Running the Application

The main application (`diabetesApp.py`) is built using Streamlit. Users can input patient details through the sidebar and get predictions on whether the patient is likely to have diabetes or not. The application loads the trained model and the scaler to make predictions based on user input.

## Usage

To use the application, follow these steps:

1. Install the required dependencies listed in `requirements.txt`.
2. Run the Streamlit application using the command `streamlit run diabetesApp.py`.
3. Input patient details in the sidebar.
4. Obtain the prediction result displayed on the app.

## License

This project is licensed under the [MIT License](./LICENSE).
