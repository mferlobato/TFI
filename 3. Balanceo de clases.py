import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# 1. Cargar el dataset LIMPIO y PREPROCESADO
df = pd.read_csv("C:/Users/malobato/Desktop/Fer/tfi/cs-training-limpio-v2.csv")

# Revisar si hay una columna de índice no nombrada
if df.columns[0].startswith('Unnamed'):
    df = df.drop(df.columns[0], axis=1)

# 2. Separar variables predictoras (X) y objetivo (y)
# X ahora contendrá 'MonthlyIncome_log' y las demás variables limpias.
X = df.drop('SeriousDlqin2yrs', axis=1)
y = df['SeriousDlqin2yrs']

# Verificar el tamaño del dataset limpio antes de la división
print(f"Tamaño del Dataset Limpio: {df.shape}")

# 3. Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# 4. **ELIMINAR** la imputación con SimpleImputer. 
# Los NaNs fueron tratados en la fase de limpieza.
# Si quedan NaNs, el problema está en la fase de limpieza.
# X_train_imputed = X_train
# X_test_imputed = X_test
# Para evitar errores si SMOTE encuentra NaNs inesperados, puedes usar la siguiente línea:
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

# 5. Aplicar SMOTE al conjunto de entrenamiento
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# 6. Verificar nuevo balance
print("Distribución original en y_train:\n", y_train.value_counts())
print("\nDistribución tras SMOTE en y_train_bal:\n", y_train_bal.value_counts())

# 7. Generación de Gráficos
import matplotlib.pyplot as plt
import seaborn as sns

# Gráfico de la distribución original
plt.figure(figsize=(10, 5))
sns.countplot(x=y_train,  palette='pastel')
plt.title('Distribución de la variable objetivo antes de SMOTE (Train Set)')
plt.xlabel('SeriousDlqin2yrs')
plt.ylabel('Cantidad de registros')
plt.show()

# Gráfico de la distribución tras SMOTE (Figura 16)
plt.figure(figsize=(10, 5))
sns.countplot(x=y_train_bal, palette='pastel')
plt.title('Figura 16. Distribución de la variable objetivo después de SMOTE (Train Set)')
plt.xlabel('SeriousDlqin2yrs')
plt.ylabel('Cantidad de registros')
plt.show()

print("Se generó la Figura 16 que muestra el balanceo.")
print("Tamaño del conjunto de entrenamiento balanceado:", X_train_bal.shape)