
# TFI - Predicción de Morosidad Crediticia

Este repositorio contiene el código fuente, datasets y recursos complementarios para el Trabajo Final Integrador (TFI) de la Especialización en Ciencia de Datos realizado por María Fernanda Lobato.  
El objetivo es predecir la probabilidad de morosidad utilizando modelos de machine learning aplicados al dataset “Give Me Some Credit” (Kaggle).

## Estructura del repositorio
├── data/
│   └── cs-training.csv         # Dataset principal utilizado para el modelado
├── notebooks/
│   └── TFI_analisis_modelos.ipynb  # Notebook Jupyter con todo el workflow reproducible
├── scripts/
│   ├── limpieza y preparacion de datos.py     # Script de limpieza y transformación de datos
│   ├── Análisis Exploratorio.py        #Script con el EDA        
│   ├── Balanceo de clases.py            #Balanceo de clases                     
│   └── Experimentación y resultados.py   #Obtención de hiperparámetros, Entrenamiento y validación de modelos, cálculo de métricas y generación de gráficos  
├── outputs/
│   ├── metricas_modelos_python.csv  # Resultados de evaluación final
│   └── figuras/                # Carpeta con imágenes de matrices de confusión y curvas ROC
├── requirements.txt            # Listado de librerías necesarias
└── README.md                   # Este archivo

## Descripción

El pipeline desarrollado incluye:
- Preprocesamiento de datos (imputación, balanceo con SMOTE, normalización)
- Entrenamiento y ajuste de hiperparámetros de modelos:
  - Regresión Logística
  - Random Forest
  - XGBoost
- Validación cruzada estratificada (k-fold, k=5)
- Evaluación de desempeño mediante métricas: Accuracy, Precision, Recall, F1-score, Specificity, AUC-ROC
- Visualización de resultados (matrices de confusión, curvas ROC)
- Documentación y análisis de resultados empleados en la presentación final del TFI

## Requisitos

Para ejecutar los notebooks o scripts es necesario tener instalado:

- Python 3.9 o superior
- Las siguientes librerías:
  
´´´bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn

## EJECUCIÓN:

1. Clonar el repositorio
git clone https://github.com/mferlobato/TFI.git
cd TFI
2. Instalar las dependencias indicadas en requirements.txt.
3. Ejecutar los scripts de la carpeta /scripts para reproducir los resultados.
4. Los outputs (métricas y gráficos) se almacenan en la carpeta /outputs.

## Dataset
El dataset cs-training.csv proviene de la competencia Give Me Some Credit de Kaggle. Se provee para fines académicos exclusivamente.

Variable objetivo: SeriousDlqin2yrs (1 si el cliente estuvo en mora severa en los próximos 2 años, 0 en caso contrario).
