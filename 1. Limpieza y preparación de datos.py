#TFI - Trabajo Final Integrador

#Carga del Dataset.
import pandas as pd
#df = pd.read_csv("/Users/mariafernandalobato/Documents/Fernanda/ITBA/TFI/data/raw/cs-training.csv")
df = pd.read_csv('C:/Users/malobato/Desktop/Fer/tfi/cs-training.csv')
print(df.shape)
df.head()

#Evaluación de datos nulos
df.info()
df.isnull().sum()

# Eliminar la columna de índice no nombrada si existe
if df.columns[0].startswith('Unnamed'):
    df = df.drop(df.columns[0], axis=1)

print("Registros iniciales:", df.shape[0])

#Tratamiento de valores faltantes
# Imputación de MonthlyIncome (con la mediana de los datos originales)
df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())
# Imputación de NumberOfDependents con 0 (como en tu versión original)
df['NumberOfDependents'] = df['NumberOfDependents'].fillna(0)

#Detección de duplicados
print("Duplicados:", df.duplicated().sum())

#Outliers
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



# 2. Eliminación de Outliers
# ====================================================================

# Se eliminan las filas que contienen valores claramente erróneos o extremadamente raros (outliers funcionales).

# a) RevolvingUtilizationOfUnsecuredLines
# Se eliminan los registros con valores de utilización irrazonablemente altos (ej. > 1000).
# Valores muy por encima de 10-15 se consideran errores de carga en este dataset (e.g., 98321.0, 2340.0)
df = df[df['RevolvingUtilizationOfUnsecuredLines'] < 1000]

# b) Variables de atraso (NumberOfTime30-59DaysPastDueNotWorse, NumberOfTime60-89DaysPastDueNotWorse, NumberOfTimes90DaysLate)
# Se eliminan los valores simbólicos de error (96, 98, 99) o valores máximos que exceden un umbral razonable (ej. > 13).
# Usamos un filtro para eliminar todas las filas donde *cualquiera* de estas columnas tenga un valor >= 90.
delinquency_cols = [
    'NumberOfTimes90DaysLate',
    'NumberOfTime60-89DaysPastDueNotWorse',
    'NumberOfTime30-59DaysPastDueNotWorse'
]
df = df[~((df[delinquency_cols] >= 90).any(axis=1))]
# También aplicamos un cap estricto para otros outliers en los conteos, por ejemplo, > 13
df = df[~((df[delinquency_cols] >= 13).any(axis=1))]

# c) DebtRatio
# Se eliminan los valores de DebtRatio extremadamente altos (ej. > 1000).
df = df[df['DebtRatio'] <= 1000]

# d) NumberOfOpenCreditLinesAndLoans
# Se aplica un límite superior de 30 para eliminar los casos inusuales.
df = df[df['NumberOfOpenCreditLinesAndLoans'] <= 30]

# e) NumberRealEstateLoansOrLines
# Se aplica un límite superior de 20 para eliminar los casos inusuales cercanos a 50.
df = df[df['NumberRealEstateLoansOrLines'] <= 20]

# 3. Transformación Logarítmica (MonthlyIncome)
# ====================================================================

# Aplicar la transformación log(V+1) a MonthlyIncome para estabilizar la varianza
# y reducir el impacto de la cola de la distribución.
df['MonthlyIncome_log'] = np.log1p(df['MonthlyIncome'])

# Eliminar la columna de ingresos original y el logaritmo de edad (si se incluyó)
df = df.drop(columns=['MonthlyIncome']) # La mantuviste en el EDA original, pero la transformamos
# df = df.drop(columns=['age'], axis=1, errors='ignore') # Si usaste age_log, eliminas age

# 4. Resumen y Guardado
# ====================================================================
print("Registros finales:", df.shape[0])
print(f"Registros eliminados: {150000 - df.shape[0]}") # Asumiendo 150k iniciales

# Guardar el dataset limpio
df.to_csv("C:/Users/malobato/Desktop/Fer/tfi/cs-training-limpio-v2.csv", index=False)

