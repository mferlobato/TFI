import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Cargar dataset
df = pd.read_csv("/Users/mariafernandalobato/Documents/Fernanda/ITBA/TFI/data/raw/cs-training.csv")

# Separar variables predictoras y objetivo
X = df.drop('SeriousDlqin2yrs', axis=1)
y = df['SeriousDlqin2yrs']

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# Imputar valores faltantes con la mediana
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)  # Ajustar e imputar en X_train
X_test_imputed = imputer.transform(X_test)        # Imputar en X_test (sin ajustar)

# Aplicar SMOTE al conjunto de entrenamiento
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_imputed, y_train)

# Verificar nuevo balance
print("Distribución original en y_train:\n", y_train.value_counts())
print("\nDistribución tras SMOTE en y_train_bal:\n", y_train_bal.value_counts())

import matplotlib.pyplot as plt
import seaborn as sns

# Gráfico de la distribución original
plt.figure(figsize=(10, 5))
sns.countplot(x=y_train, palette='Set2')
plt.title('Distribución de la variable objetivo antes de SMOTE')
plt.xlabel('SeriousDlqin2yrs')
plt.ylabel('Cantidad de registros')
plt.show()

# Gráfico de la distribución tras SMOTE
plt.figure(figsize=(10, 5))
sns.countplot(x=y_train_bal, palette='Set1')
plt.title('Distribución de la variable objetivo después de SMOTE')
plt.xlabel('SeriousDlqin2yrs')
plt.ylabel('Cantidad de registros')
plt.show()


# Tamaño del conjunto balanceado
print("Tamaño de X_train_bal:", X_train_bal.shape)
