import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in range(np.max(y) + 1)]
        predicted_class = np.argmax(num_samples_per_class)
        node = {'predicted_class': predicted_class}

        if depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] <= thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node['threshold'] = thr
                node['feature_index'] = idx
                node['left'] = self._grow_tree(X_left, y_left, depth + 1)
                node['right'] = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def _best_split(self, X, y):
        m, n = X.shape
        if m <= 1:
            return None, None

        num_parent = [np.sum(y == c) for c in range(np.max(y) + 1)]
        best_gini = 1.0 - sum((num / m) ** 2 for num in num_parent)
        best_idx, best_thr = None, None

        for idx in range(n):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            num_left = [0] * (np.max(y) + 1)
            num_right = num_parent.copy()

            for i in range(1, m):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in range(np.max(y) + 1))
                gini_right = 1.0 - sum((num_right[x] / (m - i)) ** 2 for x in range(np.max(y) + 1))
                gini = (i * gini_left + (m - i) * gini_right) / m
                if thresholds[i] == thresholds[i - 1]:
                    continue
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2

        return best_idx, best_thr

    def predict(self, X):
        return [self._predict_tree(x) for x in X]

    def _predict_tree(self, x, node=None):
        node = self.tree if node is None else node
        if 'threshold' in node:
            if x[node['feature_index']] <= node['threshold']:
                return self._predict_tree(x, node['left'])
            else:
                return self._predict_tree(x, node['right'])
        else:
            return node['predicted_class']


class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.forest = [DecisionTree(max_depth=self.max_depth) for _ in range(self.n_estimators)]

    def fit(self, X, y):
        for tree in self.forest:
            indices = np.random.choice(X.shape[0], X.shape[0], replace=True)
            tree.fit(X[indices], y[indices])

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.forest])
        return np.array([np.argmax(np.bincount(predictions[:, i])) for i in range(predictions.shape[1])])
    
    
import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        # Calculate distances between x and all examples in the training set
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]

        # Get indices of k-nearest training data points
        k_neighbors_indices = np.argsort(distances)[:self.k]

        # Get the labels of the k-nearest training data points
        k_neighbor_labels = [self.y_train[i] for i in k_neighbors_indices]

        # Perform majority voting to find the most common class label
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]
    
class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0

        for _ in range(self.n_iterations):
            model = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(model)

            dw = (1 / m) * np.dot(X.T, (predictions - y))
            db = (1 / m) * np.sum(predictions - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        model = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(model)
        return (predictions >= 0.5).astype(int)
    
class GaussianNaiveBayes:
    def fit(self, X, y):
        self.classes, counts = np.unique(y, return_counts=True)
        self.class_probabilities = counts / len(y)
        
        self.means = {}
        self.stds = {}

        for c in self.classes:
            X_c = X[y == c]
            self.means[c] = np.mean(X_c, axis=0)
            self.stds[c] = np.std(X_c, axis=0)

    def _calculate_probability(self, x, mean, std):
        exponent = np.exp(-((x - mean) ** 2) / (2 * (std ** 2)))
        return (1 / (np.sqrt(2 * np.pi) * std)) * exponent

    def predict(self, X):
        predictions = []

        for x in X:
            class_probs = []

            for c in self.classes:
                prior = np.log(self.class_probabilities[c])
                likelihood = np.sum(np.log(self._calculate_probability(x, self.means[c], self.stds[c])))
                posterior = prior + likelihood
                class_probs.append(posterior)

            predicted_class = self.classes[np.argmax(class_probs)]
            predictions.append(predicted_class)

        return np.array(predictions)   
import numpy as np

class SVM:
    def __init__(self, C=0.5, max_iter=100, tol=1e-3):
        self.C = C  # regularization parameter
        self.max_iter = max_iter  # maximum number of iterations
        self.tol = tol  # tolerance for stopping criterion
        self.b = 0  # bias term
        self.alphas = None  # Lagrange multipliers
        self.w = None  # weights

    def fit(self, X, y):
        m, n = X.shape
        self.alphas = np.zeros(m)
        self.b = 0
        self.w = np.zeros(n)

        for _ in range(self.max_iter):
            for i in range(m):
                E_i = self.decision_function(X[i]) - y[i]

                if (y[i] * E_i < -self.tol and self.alphas[i] < self.C) or (y[i] * E_i > self.tol and self.alphas[i] > 0):
                    j = self._select_random_index(i, m)
                    E_j = self.decision_function(X[j]) - y[j]

                    alpha_i_old, alpha_j_old = self.alphas[i], self.alphas[j]

                    if y[i] != y[j]:
                        L = max(0, self.alphas[j] - self.alphas[i])
                        H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
                    else:
                        L = max(0, self.alphas[i] + self.alphas[j] - self.C)
                        H = min(self.C, self.alphas[i] + self.alphas[j])

                    if L == H:
                        continue

                    eta = 2 * np.dot(X[i], X[j]) - np.dot(X[i], X[i]) - np.dot(X[j], X[j])
                    if eta >= 0:
                        continue

                    self.alphas[j] -= y[j] * (E_i - E_j) / eta
                    self.alphas[j] = max(L, min(self.alphas[j], H))

                    if np.abs(self.alphas[j] - alpha_j_old) < 1e-5:
                        continue

                    self.alphas[i] += y[i] * y[j] * (alpha_j_old - self.alphas[j])

                    b1 = self.b - E_i - y[i] * (self.alphas[i] - alpha_i_old) * np.dot(X[i], X[i]) - y[j] * (
                            self.alphas[j] - alpha_j_old) * np.dot(X[i], X[j])
                    b2 = self.b - E_j - y[i] * (self.alphas[i] - alpha_i_old) * np.dot(X[i], X[j]) - y[j] * (
                            self.alphas[j] - alpha_j_old) * np.dot(X[j], X[j])

                    if 0 < self.alphas[i] < self.C:
                        self.b = b1
                    elif 0 < self.alphas[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2

        self.w = np.dot((self.alphas * y).T, X)

    def decision_function(self, X):
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        return np.sign(self.decision_function(X))

    def _select_random_index(self, i, m):
        j = i
        while j == i:
            j = np.random.randint(0, m)
        return j

import numpy as np

class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y, depth=0)

    def _grow_tree(self, X, y, depth):
        num_samples_per_class = [np.sum(y == i) for i in np.unique(y)]
        predicted_class = np.argmax(num_samples_per_class)

        node = {'predicted_class': predicted_class}

        if depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] <= thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node['threshold'] = thr
                node['feature_index'] = idx
                node['left'] = self._grow_tree(X_left, y_left, depth + 1)
                node['right'] = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def _best_split(self, X, y):
        m, n = X.shape
        if m <= 1:
            return None, None

        num_parent = [np.sum(y == c) for c in np.unique(y)]
        best_gini = 1.0 - sum((num / m) ** 2 for num in num_parent)
        best_idx, best_thr = None, None

        for idx in range(n):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            num_left = [0] * len(np.unique(y))
            num_right = num_parent.copy()

            for i in range(1, m):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in np.unique(y))
                gini_right = 1.0 - sum((num_right[x] / (m - i)) ** 2 for x in np.unique(y))
                gini = (i * gini_left + (m - i) * gini_right) / m
                if thresholds[i] == thresholds[i - 1]:
                    continue
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2

        return best_idx, best_thr

    def _predict_tree(self, x, node):
        if 'threshold' in node:
            if x[node['feature_index']] <= node['threshold']:
                return self._predict_tree(x, node['left'])
            else:
                return self._predict_tree(x, node['right'])
        else:
            return node['predicted_class']

    def predict(self, X):
        return [self._predict_tree(x, self.tree) for x in X]

import numpy as np

class AdaBoostClassifier:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.alphas = []
        self.stumps = []

    def fit(self, X, y):
        m, n = X.shape
        weights = np.ones(m) / m  # Initialize weights

        for _ in range(self.n_estimators):
            stump = DecisionStump()
            stump.fit(X, y, weights)
            predictions = stump.predict(X)

            err = np.sum(weights * (predictions != y))

            # Prevent division by zero
            alpha = 0.5 * np.log((1 - err) / max(err, 1e-10))
            self.alphas.append(alpha)
            self.stumps.append(stump)

            weights *= np.exp(-alpha * y * predictions)
            weights /= np.sum(weights)

    def predict(self, X):
        stump_predictions = np.array([stump.predict(X) for stump in self.stumps])
        weighted_sum = np.dot(self.alphas, stump_predictions)
        return np.sign(weighted_sum)

class DecisionStump:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.prediction = None

    def fit(self, X, y, weights):
        m, n = X.shape
        min_error = float('inf')

        for i in range(n):
            thresholds = np.unique(X[:, i])

            for threshold in thresholds:
                predictions = np.ones(m)
                predictions[X[:, i] <= threshold] = -1

                error = np.sum(weights * (predictions != y))

                if error < min_error:
                    min_error = error
                    self.feature_index = i
                    self.threshold = threshold
                    self.prediction = 1 if np.sum(weights * predictions) > 0 else -1

    def predict(self, X):
        predictions = np.ones(X.shape[0])
        predictions[X[:, self.feature_index] <= self.threshold] = -1
        return self.prediction * predictions

st.write("""
# Diabetes Prediction App

#### This app predicts whether the patient is Diabetic or not

""")

st.divider()
st.write("Below are features used to predict the prediction")
st.write("""<table>
        <tr><th> Features</th><th> Description</th></tr>
        <tr><th> Pregnancies</th><td> Number of pregnancies</td></tr>
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

# Collects user input features into dataframe
def user_input_features():
    Pregnancies = st.sidebar.number_input('Pregnancies', 0 ,60, 1)
    Glucose = st.sidebar.number_input('Glucose', 0, 200, 120)
    BloodPressure = st.sidebar.number_input('BloodPressure (mm Hg)', 0.0, 150.0, 60.0)
    SkinThickness = st.sidebar.number_input('SkinThickness (mm)', 0.1, 100.0, 29.0)
    Insulin = st.sidebar.number_input('Insulin (mu U/ml)', 0.0, 1000.0, 125.0)
    BMI = st.sidebar.number_input('BMI', 0.0, 70.0, 30.1)
    DiabetesPedigreeFunction = st.sidebar.number_input('DiabetesPedigreeFunction', 0.0, 3.0, 0.349)
    Age = st.sidebar.number_input('Age',0, 120, 30)
    data = {
            'Pregnancies': Pregnancies,
            'Glucose': Glucose,
            'BloodPressure': BloodPressure,
            'SkinThickness': SkinThickness,
            'Insulin': Insulin,
            'BMI': BMI,
            'DiabetesPedigreeFunction':DiabetesPedigreeFunction,
            'Age':Age
            }
    features = pd.DataFrame(data, index=[0])
    return features

# Load the StandardScaler from pickle
with open('standard_scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Function to scale user input features
def scale_user_input(input_df, scaler):
    scaled_input = scaler.transform(input_df)
    return scaled_input

input_df = user_input_features()

# Scale the user input features
scaled_input_df = scale_user_input(input_df, scaler)

# Reads in saved classification model
load_clf = pickle.load(open('SVC_model.pkl', 'rb'))

# Use the model to make predictions
prediction = load_clf.predict(scaled_input_df)

# Display the prediction result
st.subheader('Prediction')
if prediction[0] == 1:
    st.error('The patient is likely to have diabetes.')
else:
    st.success('The patient is likely to not have diabetes.')

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
