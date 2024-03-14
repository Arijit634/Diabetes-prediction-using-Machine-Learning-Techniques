import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.write("""
# Diabetes Prediction App

#### This app predicts whether the patient is Diabetic or not

""")

st.divider()
st.write("Below are features used to predict the prediction")
st.write("""<table>
        <tr><th> Features</th><th> Description</th></tr>
        <tr><th> Glucose</th><td>Plasma glucose concentration a 2 hours in an oral glucose tolerance test</td></tr>
        <tr><th> BloodPressure </th><td>Diastolic blood pressure (mm Hg)</td></tr>
        <tr><th> SkinThickness </th><td>Triceps skin fold thickness (mm)</td></tr>
        <tr><th> Insulin </th><td>2-Hour serum insulin (mu U/ml)</td></tr>
        <tr><th> BMI </th><td>Body mass index (weight in kg/(height in m)^2)</td></tr>
        <tr><th> DiabetesPedigreeFunction </th><td>A higher value implies a stronger diabetes familial link
            <br><br>If your family members have a low likelihood of diabetes, you may find values in the range of 0.078 to 0.2.
            <br><br>Values between 0.2 and 0.4 may indicate a moderate likelihood.
            <br><br>Values beyond 0.4 suggest a relatively higher likelihood of diabetes.</td></tr>
        <tr><th> Age</th><td>Age in years</td></tr>
</table><br>""", unsafe_allow_html=True)
st.write(':blue[_Click the left side bar to insert information_]')
st.divider()

st.sidebar.header('Please enter patient details')

from sklearn.impute import SimpleImputer

def calculate_diabetes_pedigree(has_diabetic_relatives):
    if not has_diabetic_relatives:
        dpf_value = 0.078
        st.sidebar.write(f"No relatives with diabetes. DPF set to the minimum value ({dpf_value}).")
        return dpf_value

    relatives_with_diabetes = []
    relatives_without_diabetes = []

    with_diabetes_input = st.sidebar.text_area("Enter relatives with diabetes (relationship,age separated by comma):", height=150)
    for relative_info in with_diabetes_input.strip().split("\n"):
        if relative_info:
            try:
                relative_type, age_diagnosed = relative_info.split(",")
                age_diagnosed = int(age_diagnosed)
                if age_diagnosed > 88 or age_diagnosed < 14:
                    st.sidebar.warning("'adm' for relatives with diabetes must be between 14 and 88.")
                    continue
                relatives_with_diabetes.append((relative_type.strip().lower(), age_diagnosed))
            except ValueError:
                st.sidebar.warning("Invalid input format. Please enter 'relationship,age' for each relative with diabetes.")

    without_diabetes_input = st.sidebar.text_area("Enter relatives without diabetes (relationship,age separated by comma):", height=150)
    for relative_info in without_diabetes_input.strip().split("\n"):
        if relative_info:
            try:
                relative_type, age_last_exam = relative_info.split(",")
                age_last_exam = int(age_last_exam)
                if age_last_exam <= 14:
                    st.sidebar.warning("'acl' for relatives without diabetes must be greater than 14.")
                    continue
                relatives_without_diabetes.append((relative_type.strip().lower(), age_last_exam))
            except ValueError:
                st.sidebar.warning("Invalid input format. Please enter 'relationship,age' for each relative without diabetes.")

    numerator_sum = 0
    denominator_sum = 0

    for relative_type, age in relatives_with_diabetes:
        shared_genes = get_shared_genes(relative_type)
        numerator_sum += shared_genes * (88 - age)

    for relative_type, age in relatives_without_diabetes:
        shared_genes = get_shared_genes(relative_type)
        denominator_sum += shared_genes * (age - 14)

    if denominator_sum == 0:
        dpf_value = 0.078
        st.sidebar.error("Dividing by zero error: No relatives without diabetes have valid 'acl' values. DPF set to the minimum value (0.078).")
    else:
        diabetes_pedigree = (numerator_sum + 20) / (denominator_sum + 50)
        dpf_value = round(diabetes_pedigree, 3)

    st.sidebar.write(f"Diabetes Pedigree Function (DPF) value: {dpf_value}")
    return dpf_value

def get_shared_genes(relative_type):
    shared_genes = {
        "parent": 0.5,
        "sibling": 0.5,
        "half-sibling": 0.25,
        "grandparent": 0.25,
        "aunt": 0.25,
        "uncle": 0.25,
        "half-aunt": 0.125,
        "half-uncle": 0.125,
        "cousin": 0.125
    }
    return shared_genes.get(relative_type, 0)

def user_input_features():
    pregnancies = 0  # Fixed value for pregnancies
    Glucose = st.sidebar.number_input('Glucose', 0, 200, 120)
    BloodPressure = st.sidebar.number_input('BloodPressure (mm Hg)', 0.0, 150.0, 60.0)
    SkinThickness = st.sidebar.number_input('SkinThickness (mm)', 0.1, 100.0, 29.0)
    Insulin = st.sidebar.number_input('Insulin (mu U/ml)', 0.0, 1000.0, 125.0)
    BMI = st.sidebar.number_input('BMI', 0.0, 70.0, 30.1)
    has_diabetic_relatives = st.sidebar.radio("Do any of your family members have diabetes?", (True, False), index=1)  # Set initial value to False
    DiabetesPedigreeFunction = calculate_diabetes_pedigree(has_diabetic_relatives)
    Age = st.sidebar.number_input('Age', 0, 120, 30)
    data = {
        'Pregnancies': pregnancies,
        'Glucose': Glucose,
        'BloodPressure': BloodPressure,
        'SkinThickness': SkinThickness,
        'Insulin': Insulin,
        'BMI': BMI,
        'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
        'Age': Age
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Load the StandardScaler from pickle
with open('standard_scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Function to scale user input features
def scale_user_input(input_df, scaler):
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    input_df_imputed = pd.DataFrame(imputer.fit_transform(input_df), columns=input_df.columns)
    scaled_input = scaler.transform(input_df_imputed)
    return scaled_input

input_df = user_input_features()

# Scale the user input features
scaled_input_df = scale_user_input(input_df, scaler)

# Reads in saved classification model
load_clf = pickle.load(open('SVC_KNN.pkl', 'rb'))

# Use the model to make predictions
prediction = load_clf.predict(scaled_input_df)

# Display the prediction result
st.subheader('Prediction')
if prediction[0] == 1:
    st.error('The patient is likely to have diabetes.')
else:
    st.success('The patient is not likely to have diabetes.')


# # Calculate metrics
# accuracy = "83%"
# precision = "82%"
# recall = "85%"
# f1 = "83%"
# conf_matrix = np.array([[78, 21], [5, 50]])

# # Display metrics
# st.subheader('Model Performance Metrics')
# st.write(f'Accuracy: {accuracy}')
# st.write(f'Precision: {precision}')
# st.write(f'Recall: {recall}')
# st.write(f'F1-Score: {f1}')

# # Display confusion matrix
# st.subheader('Confusion Matrix')
# st.table(pd.DataFrame(conf_matrix, columns=['Predicted Negative', 'Predicted Positive'], index=['Actual Negative', 'Actual Positive']))
