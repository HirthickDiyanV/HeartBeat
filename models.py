"""The 14 classifier configurations of Table 4, built from scikit-learn / imbalanced-learn.

Every model sits in Pipeline(median imputer -> StandardScaler -> classifier) so scaling
parameters are learned from training folds only.
MATLAB-style names are used ("Fine KNN", "Fine RBF" ...) as in the paper.
"""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


def _inv_sq(d):
    return 1.0 / (np.asarray(d) ** 2 + 1e-12)       # inverse-squared distance weights


def make_models(n_features, seed=42):
    P = n_features
    # MATLAB kernel scale s: kernel = exp(-||x-y||^2 / s^2)  =>  gamma = 1/s^2
    # Fine RBF: s = sqrt(P)/4 ; Medium RBF: s = sqrt(P)
    models = {
        "KNN Fine":      KNeighborsClassifier(n_neighbors=1),
        "KNN Weighted":  KNeighborsClassifier(n_neighbors=10, weights=_inv_sq),
        "KNN Medium":    KNeighborsClassifier(n_neighbors=10),
        "KNN Cosine":    KNeighborsClassifier(n_neighbors=10, metric="cosine", algorithm="brute"),
        "KNN Coarse":    KNeighborsClassifier(n_neighbors=100),
        "RF Bagged":     RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                                                n_jobs=-1, random_state=seed),
        "Bagging KNN":   BaggingClassifier(KNeighborsClassifier(), n_estimators=50,
                                           bootstrap_features=True, n_jobs=-1, random_state=seed),
        "AdaBoost":      AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200,
                                            learning_rate=0.1, random_state=seed),
        "RUSBoosted":    None,   # filled below (needs imbalanced-learn)
        "SVM Linear":    SVC(kernel="linear", random_state=seed),
        "SVM Quad":      SVC(kernel="poly", degree=2, gamma="scale", random_state=seed),
        "SVM Cubic":     SVC(kernel="poly", degree=3, gamma="scale", random_state=seed),
        "SVM FineRBF":   SVC(kernel="rbf", gamma=16.0 / P, random_state=seed),
        "SVM MedRBF":    SVC(kernel="rbf", gamma=1.0 / P, random_state=seed),
    }
    try:
        from imblearn.ensemble import RUSBoostClassifier
        models["RUSBoosted"] = RUSBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1), n_estimators=200,
            learning_rate=0.1, random_state=seed)
    except Exception as e:                           # pragma: no cover
        print("RUSBoost unavailable:", e)
        models.pop("RUSBoosted")
    return {name: Pipeline([("imp", SimpleImputer(strategy="median")),
                            ("sc", StandardScaler()), ("clf", clf)])
            for name, clf in models.items()}


def scores(pipe, X):
    """Continuous score for the positive (Abnormal) class, used for ROC/PR/Brier."""
    if hasattr(pipe, "predict_proba"):
        try:
            return pipe.predict_proba(X)[:, 1]
        except Exception:
            pass
    return pipe.decision_function(X)
