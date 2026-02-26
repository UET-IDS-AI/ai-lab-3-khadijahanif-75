"""
Linear & Logistic Regression Lab

Follow the instructions in each function carefully.
DO NOT change function names.
Use random_state=42 everywhere required.
"""

import numpy as np

from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# =========================================================
# QUESTION 1 – Linear Regression Pipeline (Diabetes)
# =========================================================

def diabetes_linear_pipeline():
    """
    STEP 1: Load diabetes dataset.
    STEP 2: Split into train and test (80-20).
            Use random_state=42.
    STEP 3: Standardize features using StandardScaler.
            IMPORTANT:
            - Fit scaler only on X_train
            - Transform both X_train and X_test
    STEP 4: Train LinearRegression model.
    STEP 5: Compute:
            - train_mse
            - test_mse
            - train_r2
            - test_r2
    STEP 6: Identify indices of top 3 features
            with largest absolute coefficients.

    RETURN:
        train_mse,
        test_mse,
        train_r2,
        test_r2,
        top_3_feature_indices (list length 3)
    """
    # STEP 1
    X, y = load_diabetes(return_X_y=True)

    # STEP 2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # STEP 3 (Fit only on training data)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # STEP 4
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # STEP 5
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)

    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)

    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    # STEP 6 – Feature importance via coefficient magnitude
    importance = np.abs(model.coef_)
    top_3_feature_indices = np.argsort(importance)[-3:].tolist()

    return train_mse, test_mse, train_r2, test_r2, top_3_feature_indices



# =========================================================
# QUESTION 2 – Cross-Validation (Linear Regression)
# =========================================================

def diabetes_cross_validation():
    """
    STEP 1: Load diabetes dataset.
    STEP 2: Standardize entire dataset (after splitting is NOT needed for CV,
            but use pipeline logic manually).
    STEP 3: Perform 5-fold cross-validation
            using LinearRegression.
            Use scoring='r2'.

    STEP 4: Compute:
            - mean_r2
            - std_r2

    RETURN:
        mean_r2,
        std_r2
    """
    # STEP 1
    X, y = load_diabetes(return_X_y=True)

    # STEP 2 (Standardize full dataset for CV)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # STEP 3
    model = LinearRegression()
    scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')

    # STEP 4
    mean_r2 = np.mean(scores)
    std_r2 = np.std(scores)

    return mean_r2, std_r2



# =========================================================
# QUESTION 3 – Logistic Regression Pipeline (Cancer)
# =========================================================

def cancer_logistic_pipeline():
    """
    STEP 1: Load breast cancer dataset.
    STEP 2: Split into train-test (80-20).
            Use random_state=42.
    STEP 3: Standardize features.
    STEP 4: Train LogisticRegression(max_iter=5000).
    STEP 5: Compute:
            - train_accuracy
            - test_accuracy
            - precision
            - recall
            - f1
            - confusion matrix (optional to compute but not return)

    In comments:
        Explain what a False Negative represents medically.

    RETURN:
        train_accuracy,
        test_accuracy,
        precision,
        recall,
        f1
    """
    # STEP 1
    X, y = load_breast_cancer(return_X_y=True)

    # STEP 2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # STEP 3
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # STEP 4
    model = LogisticRegression(max_iter=5000, solver="liblinear")
    model.fit(X_train_scaled, y_train)

    # STEP 5
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)

    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)

    precision = precision_score(y_test, test_pred)
    recall = recall_score(y_test, test_pred)
    f1 = f1_score(y_test, test_pred)

    # Confusion matrix (not returned, but useful)
    cm = confusion_matrix(y_test, test_pred)

    """
    False Negative (Medical Meaning):
    Patient actually HAS cancer,
    but model predicts NO cancer.
    This is extremely dangerous because
    treatment may be delayed.
    """

    return train_accuracy, test_accuracy, precision, recall, f1



# =========================================================
# QUESTION 4 – Logistic Regularization Path
# =========================================================

def cancer_logistic_regularization():
    """
    STEP 1: Load breast cancer dataset.
    STEP 2: Split into train-test (80-20).
    STEP 3: Standardize features.
    STEP 4: For C in [0.01, 0.1, 1, 10, 100]:
            - Train LogisticRegression(max_iter=5000, C=value)
            - Compute train accuracy
            - Compute test accuracy

    STEP 5: Store results in dictionary:
            {
                C_value: (train_accuracy, test_accuracy)
            }

    In comments:
        - What happens when C is very small?
        - What happens when C is very large?
        - Which case causes overfitting?

    RETURN:
        results_dictionary
    """
    # STEP 1
    X, y = load_breast_cancer(return_X_y=True)

    # STEP 2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # STEP 3
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # STEP 4
    for C in [0.01, 0.1, 1, 10, 100]:

        model = LogisticRegression(max_iter=5000, solver="liblinear")
        model.fit(X_train_scaled, y_train)

        train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
        test_acc = accuracy_score(y_test, model.predict(X_test_scaled))

        results[C] = (train_acc, test_acc)

    """
    Small C → Strong regularization → Underfitting
    Large C → Weak regularization → Overfitting
    Overfitting happens when C is very large
    """

    return results


# =========================================================
# QUESTION 5 – Cross-Validation (Logistic Regression)
# =========================================================

def cancer_cross_validation():
    """
    STEP 1: Load breast cancer dataset.
    STEP 2: Standardize entire dataset.
    STEP 3: Perform 5-fold cross-validation
            using LogisticRegression(C=1, max_iter=5000).
            Use scoring='accuracy'.

    STEP 4: Compute:
            - mean_accuracy
            - std_accuracy

    In comments:
        Explain why cross-validation is especially
        important in medical diagnosis problems.

    RETURN:
        mean_accuracy,
        std_accuracy
    """
    # STEP 1
    X, y = load_breast_cancer(return_X_y=True)

    # STEP 2
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # STEP 3
    model = LogisticRegression(C=1, max_iter=5000, solver="liblinear")
    scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')

    # STEP 4
    mean_accuracy = np.mean(scores)
    std_accuracy = np.std(scores)

    """
    Cross-validation is especially important in medical diagnosis
    because we must ensure the model performs consistently
    across different patient groups, not just one data split.
    """

    return mean_accuracy, std_accuracy
