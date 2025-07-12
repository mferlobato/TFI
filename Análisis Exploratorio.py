import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar datos
df = pd.read_csv("/Users/mariafernandalobato/Documents/Fernanda/ITBA/TFI/data/raw/cs-training.csv")

# Imputación si no se hizo antes
df['MonthlyIncome'].fillna(df['MonthlyIncome'].median(), inplace=True)
df['NumberOfDependents'].fillna(0, inplace=True)

# Estadísticas descriptivas generales
print(df.describe())

# Distribución variable objetivo
sns.countplot(x='SeriousDlqin2yrs', data=df, color='pink')
plt.title('Distribución de la variable objetivo')
plt.show()

# Histograma de edades
plt.figure(figsize=(8,4))
sns.histplot(df['age'], bins=30, kde=True, color='pink')
plt.title('Distribución de la Edad')
plt.show()

# Correlación entre variables numéricas
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Mapa de correlación')
plt.show()

# Boxplot ingreso mensual según variable objetivo
plt.figure(figsize=(8,5))
sns.boxplot(x='SeriousDlqin2yrs', y='MonthlyIncome', data=df, color='pink')
plt.yscale('log')  # Por la gran dispersión en ingresos
plt.title('Ingreso Mensual según morosidad')
plt.show()

