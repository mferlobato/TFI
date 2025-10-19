## Librerías
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier # Si tienes problemas, comenta esta línea y el pipeline XGBoost.
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score,
                            f1_score, roc_auc_score)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import seaborn as sns


## Carga de dataset

df = pd.read_csv("/Users/mariafernandalobato/Documents/Fernanda/ITBA/TFI/data/raw/cs-training.csv")
# El dataset limpio debe tener 11 columnas (10 predictoras + 1 objetivo)
print(f"Dataset Limpio cargado. Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
df['SeriousDlqin2yrs'] = df['SeriousDlqin2yrs'].astype(int)

## División train/test
# =================================================================
# X ahora usa el dataset limpio, con MonthlyIncome_log en lugar de MonthlyIncome
X = df.drop(columns="SeriousDlqin2yrs")
y = df["SeriousDlqin2yrs"]

# Stratify se asegura de que la proporción 93.3%/6.7% se mantenga en ambos sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

## SECCIÓN DE SMOTE MANUAL Y IMPUTACIÓN ANTERIOR ELIMINADA (Es redundante)
# =================================================================

## MODELOS y PIPELINES (Incluyen SMOTE, Escalado y una salvaguarda de Imputer)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) 




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
    ('smote', SMOTE(random_state=42)), # Los modelos de árbol no necesitan Scaler
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

# XGBoost
xgb_pipe = ImbPipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('smote', SMOTE(random_state=42)), # Los modelos de árbol no necesitan Scaler
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

## Entrenamiento con Grid Search (ROC-AUC como score principal)
print("Entrenando modelos... Esto puede tardar varios minutos.")

# NOTA: log_clf no necesita una grilla de hiperparámetros complejos
log_clf = GridSearchCV(log_pipe, {}, scoring='roc_auc', cv=cv, refit=True, verbose=0) 
rf_clf = GridSearchCV(rf_pipe, rf_grid, scoring='roc_auc', cv=cv, refit=True, verbose=0)
xgb_clf = GridSearchCV(xgb_pipe, xgb_grid, scoring='roc_auc', cv=cv, refit=True, verbose=0)

log_clf.fit(X_train, y_train)
rf_clf.fit(X_train, y_train)
xgb_clf.fit(X_train, y_train)

print("Mejores RF parámetros:", rf_clf.best_params_)
print("Mejores XGB parámetos:", xgb_clf.best_params_)

# =================================================================
## Evaluación en TEST
# =================================================================
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
print("\nTabla de Métricas Final:")
print(resultados_df.round(4))
# nuevo CSV con las métricas corregidas.
resultados_df.to_csv(r"C:\Users\malobato\Desktop\Fer\tfi\metricas_modelos_final.csv", index=False)

# Muestra los mejores parámetros para Random Forest
print("Mejores RF parámetros:", rf_clf.best_params_) 

# Muestra los mejores parámetros para XGBoost
print("Mejores XGB parámetros:", xgb_clf.best_params_)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Matriz de ccorrelacion: Regresión logistica
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

cm_log = np.array([[21328, 3382],
                   [650, 1114]])

plt.figure(figsize=(6, 6))
sns.heatmap(cm_log, annot=True, fmt='d', cmap='pink', cbar=False, 
            xticklabels=['No Moroso (0)', 'Moroso (1)'], 
            yticklabels=['No Moroso (0)', 'Moroso (1)'])
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.title('Matriz de Confusión: Regresión Logística', fontsize=14)
plt.show()


# Matriz de ccorrelacion: Random Forest
cm_rf = np.array([[23817, 893],
                  [1202, 562]])

plt.figure(figsize=(6, 6))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='pink', cbar=False, 
            xticklabels=['No Moroso (0)', 'Moroso (1)'], 
            yticklabels=['No Moroso (0)', 'Moroso (1)'])
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.title('Matriz de Confusión: Random Forest', fontsize=14)
plt.show()

# Matriz de ccorrelacion: CGboost
cm_xgb = np.array([[22026, 2684],
                   [688, 1076]])

plt.figure(figsize=(6, 6))
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='pink', cbar=False, 
            xticklabels=['No Moroso (0)', 'Moroso (1)'], 
            yticklabels=['No Moroso (0)', 'Moroso (1)'])
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.title('Matriz de Confusión: XGBoost (Modelo Final)', fontsize=14)
plt.show()

##CURVA ROCimport pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split

modelos = {
    "Regresión Logística (AUC: 0.8202)": log_clf.best_estimator_,
    "Random Forest (AUC: 0.8294)": rf_clf.best_estimator_,
    "XGBoost (AUC: 0.8447)": xgb_clf.best_estimator_ 
}

plt.figure(figsize=(10, 8))

# Línea base (Clasificador aleatorio)
plt.plot([0, 1], [0, 1], 'k--', label='Aleatorio (AUC = 0.50)')

for nombre, modelo in modelos.items():
    # Obtener las probabilidades para la clase positiva (1) en el conjunto de prueba
    y_prob = modelo.predict_proba(X_test)[:, 1]
    
    # Calcular la curva ROC: Tasa de Falsos Positivos (FPR) y Tasa de Verdaderos Positivos (TPR)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)
    
    # Graficar la curva
    # El nombre se construye con el AUC real obtenido
    plt.plot(fpr, tpr, label=f'{nombre.split(" (AUC: ")[0]} (AUC = {auc_score:.4f})', linewidth=2)

plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR) / Recall')
plt.title('Figura 21. Curva ROC Comparativa de Modelos', fontsize=16)
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

print("Se generó la Figura 21 (Curva ROC) para los modelos.")