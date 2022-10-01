import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# problem_score = "accuracy"
problem_score = "precision"

# Load data
dataObj = load_digits()
X = dataObj.data
y = dataObj.target

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.20, random_state=1
)

# Estimator
pipe_svc = Pipeline(
    [
        ("scl", StandardScaler()),
        ("pca", PCA(n_components=10)),
        ("clf", SVC(random_state=1)),
    ]
)

# Parameter range for grid search
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
set1 = {"clf__C": param_range, "clf__kernel": ["linear"]}
set2 = {"clf__C": param_range, "clf__gamma": param_range, "clf__kernel": ["rbf"]}
param_grid = [set1, set2]


if problem_score == "accuracy":
    gs = GridSearchCV(
        estimator=pipe_svc, param_grid=param_grid, scoring="accuracy", cv=10, n_jobs=-1
    )
elif problem_score == "precision":
    precision_scorer = make_scorer(precision_score, average="micro")
    gs = GridSearchCV(
        estimator=pipe_svc,
        param_grid=param_grid,
        scoring=precision_scorer,
        cv=10,
        n_jobs=-1,
    )

gs.fit(X_train, y_train)

print(f"Optimized by {problem_score}")
print(f"Best score:{gs.best_score_:6.3f}")
print("Best parameters")
print(gs.best_params_)

df = pd.DataFrame(gs.cv_results_)
df = df.sort_values(by=["rank_test_score"]).head()
display(df.head())
