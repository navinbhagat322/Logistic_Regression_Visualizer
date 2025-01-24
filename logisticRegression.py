import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def load_initial_graph(dataset, ax):
    if dataset == "Binary":
        X, y = make_blobs(n_features=2, centers=2, random_state=6)
        ax.scatter(X.T[0], X.T[1], c=y, cmap='rainbow')
        return X, y
    elif dataset == "Multiclass":
        X, y = make_blobs(n_features=2, centers=3, random_state=2)
        ax.scatter(X.T[0], X.T[1], c=y, cmap='rainbow')
        return X, y

def draw_meshgrid():
    a = np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.01)
    b = np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.01)
    XX, YY = np.meshgrid(a, b)
    input_array = np.array([XX.ravel(), YY.ravel()]).T
    return XX, YY, input_array

plt.style.use('fivethirtyeight')

st.sidebar.markdown("# Logistic Regression Classifier")

# Sidebar options for Logistic Regression hyperparameters
dataset = st.sidebar.selectbox(
    'Select Dataset',
    ('Binary', 'Multiclass')
)

penalty = st.sidebar.selectbox(
    'Regularization',
    ('l2', 'l1', 'elasticnet', 'none')
)

c_input = float(st.sidebar.number_input('C (Inverse Regularization Strength)', value=1.0))

solver = st.sidebar.selectbox(
    'Solver',
    ('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga')
)

max_iter = int(st.sidebar.number_input('Max Iterations', value=100))

multi_class = st.sidebar.selectbox(
    'Multi Class',
    ('auto', 'ovr', 'multinomial')
)

l1_ratio = st.sidebar.number_input('l1 Ratio (For ElasticNet)', value=0.5)

tol = st.sidebar.number_input('Tolerance for Convergence (tol)', value=1e-4)

fit_intercept = st.sidebar.selectbox('Fit Intercept', (True, False))

intercept_scaling = st.sidebar.number_input('Intercept Scaling (For liblinear)', value=1.0)

class_weight = st.sidebar.selectbox(
    'Class Weight',
    ('None', 'balanced')
)

dual = st.sidebar.selectbox(
    'Dual (For liblinear with l2 penalty only)', 
    (False, True)
)

warm_start = st.sidebar.selectbox('Warm Start (Reuse Previous Solution)', (False, True))

n_jobs = st.sidebar.number_input('Number of CPU Cores (n_jobs)', value=1, step=1, min_value=1)

# Load initial graph
fig, ax = plt.subplots()

# Plot initial graph
X, y = load_initial_graph(dataset, ax)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
orig = st.pyplot(fig)

if st.sidebar.button('Run Algorithm'):
    orig.empty()
    
    # Class weight handling
    class_weight_param = None if class_weight == 'None' else 'balanced'
    
    # Logistic Regression with all hyperparameters
    clf = LogisticRegression(
        penalty=penalty,
        C=c_input,
        solver=solver,
        max_iter=max_iter,
        multi_class=multi_class,
        l1_ratio=None if penalty != 'elasticnet' else l1_ratio,
        tol=tol,
        fit_intercept=fit_intercept,
        intercept_scaling=intercept_scaling,
        class_weight=class_weight_param,
        dual=dual,
        warm_start=warm_start,
        n_jobs=n_jobs
    )
    
    # Train the classifier
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)

    # Plot decision boundary
    XX, YY, input_array = draw_meshgrid()
    labels = clf.predict(input_array)

    ax.contourf(XX, YY, labels.reshape(XX.shape), alpha=0.5, cmap='rainbow')
    plt.xlabel("Col1")
    plt.ylabel("Col2")
    orig = st.pyplot(fig)

    # Show accuracy
    st.subheader("Accuracy for Logistic Regression: " + str(round(accuracy_score(y_test, y_pred), 2)))
