import streamlit as st
import pandas as pd
import numpy as np
import pickle
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
st.write("""
# Diabetes Prediction App

#### This app predicts whether the patient is Diabetic or not

""")

st.write(':blue[_Click the left side bar to insert information_]')

class GradientBoostingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=200, learning_rate=0.1, max_depth=2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.models = []

    def fit(self, X, y):
        # Initialize residuals to y
        residuals = np.copy(y).astype(float)  # Convert residuals to float

        # Fit base learner
        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.models.append(tree)

            # Update residuals
            predictions = tree.predict(X)
            residuals -= self.learning_rate * predictions

    def predict(self, X):
        predictions = np.zeros(len(X))
        for tree in self.models:
            predictions += self.learning_rate * tree.predict(X)
        return (predictions > 0.5).astype(int)

    def predict_proba(self, X):
        # Calculate the sum of predictions across all trees
        sum_predictions = np.zeros(len(X))
        for tree in self.models:
            sum_predictions += self.learning_rate * tree.predict(X)

        # Convert to probabilities
        proba_positive_class = 1 / (1 + np.exp(-sum_predictions))
        proba_negative_class = 1 - proba_positive_class

        return np.column_stack((proba_negative_class, proba_positive_class))

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

# Define CustomKNN class
class CustomKNN(BaseEstimator, ClassifierMixin):
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        distances = [np.sqrt(np.sum((x - x_train)**2)) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        return Counter(k_nearest_labels).most_common(1)[0][0]

    def predict_proba(self, X):
        probabilities = []
        for x in X:
            distances = [np.sqrt(np.sum((x - x_train)**2)) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            class_counts = Counter(k_nearest_labels)
            prob_class_0 = class_counts[0] / self.k
            prob_class_1 = class_counts[1] / self.k
            probabilities.append([prob_class_0, prob_class_1])
        return np.array(probabilities)

class CustomVotingClassifier:
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights
        if weights is None:
            # initializes with equal weights
            self.weights = [1/len(models)] * len(models)  

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
        
        # Calculating weights based on using accuracy meaning:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        for i, model in enumerate(self.models):
            y_pred_val = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred_val)
            self.weights[i] = accuracy

    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.models])
        weighted_votes = np.average(predictions, axis=0, weights=self.weights)
        return (weighted_votes > 0.5).astype(int)

st.sidebar.header('Please enter patient details')

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
                if age_last_exam < 14 or age_last_exam > 88:
                    st.sidebar.warning("'acl' for relatives without diabetes must be between 14 and 88.")
                    continue
                relatives_without_diabetes.append((relative_type.strip().lower(), age_last_exam))
            except ValueError:
                st.sidebar.warning("Invalid input format. Please enter 'relationship,age' for each relative without diabetes.")

    numerator_sum = 0.0  # Initialize as float
    denominator_sum = 0.0  # Initialize as float

    for relative_type, age in relatives_with_diabetes:
        shared_genes = get_shared_genes(relative_type)
        numerator_sum += shared_genes * (88 - age)

    for relative_type, age in relatives_without_diabetes:
        shared_genes = get_shared_genes(relative_type)
        denominator_sum += shared_genes * (age - 14)

    # Prevent division by zero
    if denominator_sum == 0:
        diabetes_pedigree = 0.078 # Default value when no relatives without diabetes are entered
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
    Pregnancy = st.sidebar.slider('Pregnancies', 0, 17, 3)
    Glucose = st.sidebar.number_input('Glucose', 0, 200, 120)
    BloodPressure = st.sidebar.number_input('BloodPressure (mm Hg)', 0.0, 150.0, 60.0)
    SkinThickness = st.sidebar.number_input('SkinThickness (mm)', 0.1, 100.0, 29.0)
    Insulin = st.sidebar.number_input('Insulin (mu U/ml)', 0.0, 1000.0, 125.0)
    BMI = st.sidebar.number_input('BMI', 0.0, 70.0, 30.1)
    has_diabetic_relatives = st.sidebar.radio("Do any of your family members have diabetes?", (True, False), index=1)  # Set initial value to False
    DiabetesPedigreeFunction = calculate_diabetes_pedigree(has_diabetic_relatives)
    # DiabetesPedigreeFunction = st.sidebar.number_input('DiabetesPedigreeFunction', 0.0, 3.0, 0.349)

    Age = st.sidebar.number_input('Age', 0, 100, 30)
    data = {
        'Pregnancies': Pregnancy,
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

input_df = user_input_features()

# Preprocess the input data
input_data = scaler.transform(input_df)

# Reads in saved classification model
with open('hybrid_model.pkl', 'rb') as f:
    voting_clf = pickle.load(f)

# Use the model to make predictions
prediction = voting_clf.predict(input_data)

# Display the prediction result
st.subheader('Prediction')
if prediction[0] == 1:
    st.error('The patient is likely to have diabetes.')
else:
    st.success('The patient is not likely to have diabetes.')

st.write("Below are features used to predict the prediction")
st.write("""
#### Features and Description
- **Pregnancies:** Number of times pregnant. Reflects the pregnancy history of the patient.
- **Glucose:** Plasma glucose concentration after a 2-hour oral glucose tolerance test (OGTT) in mg/dl. Reflects how well the body processes glucose.
  - **Normal Range:**
    - 2-hour post-OGTT: Less than 140 mg/dL
  - **Abnormal Range:**
    - Prediabetes (Impaired Glucose Tolerance): 140-199 mg/dL
    - Diabetes: 200 mg/dL or higher
- **Blood Pressure:** Diastolic blood pressure (mm Hg). Measures the pressure in the arteries when the heart rests between beats.
  - **Normal Range:** Less than 80 mmHg
  - **Abnormal Range:**
    - Elevated: 80-89 mmHg
    - Hypertension Stage 1: 90-99 mmHg
    - Hypertension Stage 2: 100 mmHg or higher
- **Skin Thickness:** Triceps skin fold thickness (mm). Indicator of subcutaneous fat.
  - **Normal Range:** Varies widely depending on age, sex, and body composition.
- **Insulin:** 2-hour serum insulin (mu U/ml). Reflects the body's insulin response to a glucose challenge.
  - **Normal Range:** 16-166 mIU/L
  - **Abnormal Range:**
    - Low: Below normal range, indicating problems with insulin production.
    - High: Above normal range, may indicate insulin resistance.
- **BMI:** Body Mass Index (kg/m2). Measure of body fat based on height and weight.
  - **Normal Range:** 18.5 - 24.9 kg/m2
  - **Abnormal Range:**
    - Underweight: Less than 18.5 kg/m2
    - Overweight: 25 - 29.9 kg/m2
    - Obesity Class I: 30 - 34.9 kg/m2
    - Obesity Class II: 35 - 39.9 kg/m2
    - Obesity Class III (Severe Obesity): 40 kg/m2 or higher
- **Diabetes Pedigree Function (DPF):** Estimates the likelihood of developing diabetes based on family history.
- **Age:** Age in years.
\n
These ranges and insights were obtained from the Pima Indians Diabetes Dataset study available at https://github.com/npradaschnor/Pima-Indians-Diabetes-Dataset/tree/master
         
#### How to input data for Diabetes Pedigree Function (DPF) estimation:
- For each relative with diabetes, enter their relationship and age separated by a comma.
    - Example: parent,50
- For each relative without diabetes, enter their relationship and age separated by a comma.
    - Example: sibling,30

#### Relatives' Information and Shared Genes:
- **Parent (e.g., mother or father):** Carries a higher weight (0.5) in determining DPF.
- **Sibling:** Also carries a higher weight (0.5) similar to parents.
- **Half-Sibling:** Carries a moderate weight (0.25) in determining DPF.
- **Grandparent:** Shares genes with you but to a lesser extent (0.25).
- **Aunt/Uncle:** Shares genes with you but to a lesser extent (0.25).
- **Half-Aunt/Half-Uncle:** Shares genes with you, but the genetic influence is less (0.125).
- **Cousin:** Genetic influence is present but minimal (0.125).
""")