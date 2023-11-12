# Required installations:
# pip install streamlit scikit-learn matplotlib numpy

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss

# Function to plot calibration curves
def plot_calibration_curve(est, name, fig_index, X_train, X_test, y_train, y_test):
    # Calibrated with sigmoid calibration
    calibrated_clf = CalibratedClassifierCV(est, method='sigmoid', cv='prefit')
    calibrated_clf.fit(X_train, y_train)

    # Probability predictions
    prob_pos_clf = est.predict_proba(X_test)[:, 1]
    prob_pos_calibrated = calibrated_clf.predict_proba(X_test)[:, 1]

    # Reliability curve
    fraction_of_positives_clf, mean_predicted_value_clf = calibration_curve(y_test, prob_pos_clf, n_bins=10)
    fraction_of_positives_calibrated, mean_predicted_value_calibrated = calibration_curve(y_test, prob_pos_calibrated, n_bins=10)

    # Plot
    ax1 = plt.subplot2grid((3, 1), (fig_index, 0), rowspan=1, colspan=1)
    ax1.plot(mean_predicted_value_clf, fraction_of_positives_clf, "s-", label="%s" % (name, ))
    ax1.plot(mean_predicted_value_calibrated, fraction_of_positives_calibrated, "s-", label="%s + Calibration" % (name, ))
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title(f'Calibration plots (reliability curve) for {name}')

    return calibrated_clf

# Streamlit app
st.title("Probability Calibration Demo")

# # Sidebar for dataset selection
# dataset_option = st.sidebar.selectbox('Select Dataset', ('Dataset 1', 'Dataset 2', 'Dataset 3'))

# # Generate synthetic datasets
# if dataset_option == 'Dataset 1':
#     X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=10, random_state=42)
# elif dataset_option == 'Dataset 2':
#     X, y = make_classification(n_samples=1000, n_features=20, n_clusters_per_class=1, random_state=42)
# else:
#     X, y = make_classification(n_samples=1000, n_features=20, n_clusters_per_class=2, random_state=42)

# Sidebar for dataset selection
dataset_option = st.sidebar.selectbox(
    'Select Dataset', 
    ('Two Informative Features, High Redundancy', 
     'Single Cluster Per Class', 
     'Multiple Clusters Per Class')
)

# Generate synthetic datasets based on the selection
if dataset_option == 'Two Informative Features, High Redundancy':
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=10, random_state=42)
elif dataset_option == 'Single Cluster Per Class':
    X, y = make_classification(n_samples=1000, n_features=20, n_clusters_per_class=1, random_state=42)
else:  # 'Multiple Clusters Per Class'
    X, y = make_classification(n_samples=1000, n_features=20, n_clusters_per_class=2, random_state=42)


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Classifier options
classifiers = {'Logistic Regression': LogisticRegression(max_iter=1000),
               'Naive Bayes': GaussianNB(),
               'Random Forest': RandomForestClassifier(n_estimators=100)}

plt.figure(figsize=(10, 10))

# Plot calibration curves for each classifier
for i, (name, classifier) in enumerate(classifiers.items()):
    classifier.fit(X_train, y_train)
    plot_calibration_curve(classifier, name, i, X_train, X_test, y_train, y_test)

st.pyplot(plt)
