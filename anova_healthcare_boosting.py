# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 22:11:34 2025

@author: avina
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_hc = pd.read_csv('P:/My Documents/Books & Research/Analytics Vidya Blackbelt program/Advanced ML Algorithms/Healthcare_Dataset_Preprocessednew2.csv')
df_hc.columns
df_hc.dtypes
df_hc.shape
df_hc.head()

missing_values = df_hc.isnull().sum()
print(missing_values)

for col in ['Diet_Type__Vegan', 'Diet_Type__Vegetarian', 'Blood_Group_AB', 'Blood_Group_B','Blood_Group_O']: 
    if df_hc[col].dtype == 'bool':
        df_hc[col] = df_hc[col].astype(int)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb

X = df_hc.drop(columns=['Target'])
y = df_hc['Target']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Gradient Boosting

gb_param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

gb = GradientBoostingClassifier(random_state=42)
gb_grid = GridSearchCV(estimator=gb, param_grid=gb_param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
gb_grid.fit(X_train, y_train)

gb_best = gb_grid.best_estimator_
gb_train_score = gb_best.score(X_train, y_train)
gb_test_score = gb_best.score(X_test, y_test)


# XGBoost

xgb_param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 2, 4],
    'subsample': [0.8, 1.0]
}

xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_grid = GridSearchCV(estimator=xgb_model, param_grid=xgb_param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
xgb_grid.fit(X_train, y_train)


xgb_best = xgb_grid.best_estimator_
xgb_train_score = xgb_best.score(X_train, y_train)
xgb_test_score = xgb_best.score(X_test, y_test)


# LightGBM

lgb_param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'num_leaves': [15, 31, 63],
    'subsample': [0.8, 1.0]
}

lgb_model = lgb.LGBMClassifier(random_state=42)
lgb_grid = GridSearchCV(estimator=lgb_model, param_grid=lgb_param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
lgb_grid.fit(X_train, y_train)


lgb_best = lgb_grid.best_estimator_
lgb_train_score = lgb_best.score(X_train, y_train)
lgb_test_score = lgb_best.score(X_test, y_test)


# Print Results
print("\n--- Boosting Model Comparison ---")
print(f"Gradient Boosting - Best Params: {gb_grid.best_params_}, Train Score: {gb_train_score:.4f}, Test Score: {gb_test_score:.4f}")
print(f"XGBoost - Best Params: {xgb_grid.best_params_}, Train Score: {xgb_train_score:.4f}, Test Score: {xgb_test_score:.4f}")
print(f"LightGBM - Best Params: {lgb_grid.best_params_}, Train Score: {lgb_train_score:.4f}, Test Score: {lgb_test_score:.4f}")


# Visualization
models = ['Gradient Boosting', 'XGBoost', 'LightGBM']
train_scores = [gb_train_score, xgb_train_score, lgb_train_score]
test_scores = [gb_test_score, xgb_test_score, lgb_test_score]

plt.figure(figsize=(10, 6))
x = range(len(models))
plt.bar(x, train_scores, width=0.4, label='Train Score', align='center')
plt.bar(x, test_scores, width=0.4, label='Test Score', align='edge')
plt.xticks(x, models)
plt.ylim(0.7, 1.0)
plt.ylabel('Accuracy Score')
plt.title('Boosting Model Comparison: Train vs Test Accuracy')
plt.legend()
plt.grid(True)
plt.show()