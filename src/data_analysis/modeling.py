import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import shap

def train_and_evaluate(df):
    """
    Train models (Logistic Regression, Random Forest, and XGBoost) and evaluate their performance.
    Use SHAP for model interpretability.
    """
    # Separate features and target
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    
    # Train-test split (70/30 stratified)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    print("Training samples:", X_train.shape[0], "Test samples:", X_test.shape[0])
    
    # Define cross-validation strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 1. Logistic Regression with hyperparameter tuning
    log_reg = LogisticRegression(max_iter=1000, solver='lbfgs')
    param_grid_lr = {'C': [0.01, 0.1, 1, 10]}
    grid_lr = GridSearchCV(log_reg, param_grid_lr, cv=cv, scoring='f1')
    grid_lr.fit(X_train, y_train)
    best_lr = grid_lr.best_estimator_
    print("Best Logistic Regression parameters:", grid_lr.best_params_)
    
    # 2. Random Forest with hyperparameter tuning
    rf_clf = RandomForestClassifier(random_state=42)
    param_grid_rf = {'n_estimators': [100, 200],
                     'max_depth': [None, 5, 10],
                     'min_samples_leaf': [1, 4]}
    grid_rf = GridSearchCV(rf_clf, param_grid_rf, cv=cv, scoring='f1')
    grid_rf.fit(X_train, y_train)
    best_rf = grid_rf.best_estimator_
    print("Best Random Forest parameters:", grid_rf.best_params_)
    
    # 3. XGBoost with hyperparameter tuning
    xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    param_grid_xgb = {'n_estimators': [100, 200],
                      'max_depth': [3, 6],
                      'learning_rate': [0.1, 0.3]}
    grid_xgb = GridSearchCV(xgb_clf, param_grid_xgb, cv=cv, scoring='f1')
    grid_xgb.fit(X_train, y_train)
    best_xgb = grid_xgb.best_estimator_
    print("Best XGBoost parameters:", grid_xgb.best_params_)
    
    # Evaluate each model on the test set
    models = {"Logistic Regression": best_lr,
              "Random Forest": best_rf,
              "XGBoost": best_xgb}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        print(f"{name} - Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}")
    
    # SHAP analysis using the best XGBoost model
    explainer = shap.TreeExplainer(best_xgb)
    shap_values = explainer.shap_values(X_test)
    # SHAP summary plot (save as PNG)
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig("shap_summary.png")
    plt.close()
    print("SHAP summary plot saved as shap_summary.png")