## Librerías
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import seaborn as sns


## Carga de dataset

df = pd.read_csv(r"C:\Users\malobato\Desktop\Fer\tfi\cs-training.csv")
df = df.drop(df.columns[0], axis=1)
df['SeriousDlqin2yrs'] = df['SeriousDlqin2yrs'].astype(int)
print(df.head()) 

##División train/test
X = df.drop(columns="SeriousDlqin2yrs")
y = df["SeriousDlqin2yrs"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

##SMOTE e imputación en train
# Imputación en train, el resto en pipeline
imputer = SimpleImputer(strategy='median')

# Balanceo solo en train
smote = SMOTE(random_state=42)
X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

X_train_sm, y_train_sm = smote.fit_resample(X_train_imp, y_train)

## MODELOS
#Definición de los pasos de preprocesamiento que se van a aplicar sobre las variables numéricas del data set
numeric_preprocessing = [('imputer', SimpleImputer(strategy='median')),
                         ('scaler', StandardScaler())]

# Regresión Logística
log_pipe = ImbPipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('clf', LogisticRegression(solver='liblinear', random_state=42))
])

# Random Forest
rf_pipe = ImbPipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('smote', SMOTE(random_state=42)),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

# XGBoost
xgb_pipe = ImbPipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('smote', SMOTE(random_state=42)),
    ('clf', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
])

# Grillas para GridSearch
rf_grid = {'clf__max_features': [2, 3, 4, 5]}
xgb_grid = {
    'clf__max_depth': [4, 6, 8],
    'clf__n_estimators': [100, 150],
    'clf__learning_rate': [0.01, 0.1],
    'clf__subsample': [0.8],
    'clf__colsample_bytree': [0.8],
    'clf__gamma': [0],
    'clf__min_child_weight': [1]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

## Entrenamiento con Grid Search (ROC-AUC como score principal)
print("Entrenando modelos...")

log_clf = GridSearchCV(log_pipe, {}, scoring='roc_auc', cv=cv, refit=True, verbose=1)
rf_clf = GridSearchCV(rf_pipe, rf_grid, scoring='roc_auc', cv=cv, refit=True, verbose=1)
xgb_clf = GridSearchCV(xgb_pipe, xgb_grid, scoring='roc_auc', cv=cv, refit=True, verbose=1)

log_clf.fit(X_train, y_train)
rf_clf.fit(X_train, y_train)
xgb_clf.fit(X_train, y_train)

print("Mejores RF parámetros:", rf_clf.best_params_)
print("Mejores XGB parámetos:", xgb_clf.best_params_)


############################################################

##Evaluación en TEST
modelos = {
    "Regresión Logística": log_clf.best_estimator_,
    "Random Forest": rf_clf.best_estimator_,
    "XGBoost": xgb_clf.best_estimator_
}

resultados = []

for nombre, modelo in modelos.items():
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:,1]
    
    prec = precision_score(y_test, y_pred, pos_label=1)
    rec = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred, labels=[0,1])
    acc = accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    
    resultados.append({
        "Modelo": nombre,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "Specificity": specificity,
        "F1": f1,
        "AUC": auc
    })

resultados_df = pd.DataFrame(resultados)
print(resultados_df.round(4))
resultados_df.to_csv(r"C:\Users\malobato\Desktop\Fer\tfi\metricas_modelos_python.csv", index=False)
