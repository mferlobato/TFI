#TFI - Trabajo Final Integrador

#Carga del Dataset.
import pandas as pd
df = pd.read_csv("/Users/mariafernandalobato/Documents/Fernanda/ITBA/TFI/data/raw/cs-training.csv")

print(df.shape)
df.head()

#Evaluación de datos nulos
df.info()
df.isnull().sum()

#Tratamiento de valores faltantes
df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())
df['NumberOfDependents'] = df['NumberOfDependents'].fillna(0)

#Detección de duplicados
print("Duplicados:", df.duplicated().sum())

#Outliers
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

numeric_cols = df.select_dtypes(include='number').columns.drop('Unnamed: 0') #Todas las columnas numéricas excepto ID.

palette = sns.color_palette("Pastel1", len(numeric_cols)) #paleta de colores para los graficos

import os
os.makedirs("figures", exist_ok=True)  # Crear carpeta si no existe


#INGRESO MENSUAL
# Boxplot con escala logarítmica en Ximport pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10, 5))
sns.boxplot(x=df["MonthlyIncome"], color="skyblue")
plt.title("Boxplot de MonthlyIncome", fontsize=14, fontweight="bold")
plt.xlabel("MonthlyIncome", fontsize=12)
plt.xscale('log')  # Escala logarítmica para comprimir los outliers
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.savefig("figures/boxplot_MonthlyIncome_log.png", dpi=300, bbox_inches='tight')
plt.show()
#Grafico de densidad
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(8, 5))
sns.kdeplot(df["MonthlyIncome"].dropna(), fill=True, color="coral", alpha=0.6)
plt.title("Densidad de MonthlyIncome", fontsize=14, fontweight="bold")
plt.xlabel("MonthlyIncome", fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.savefig("figures/kde_MonthlyIncome.png", dpi=300, bbox_inches='tight')
plt.show()


#LINEA DE CREDITOS ABIERTA
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10, 5))
sns.boxplot(x=df["NumberOfOpenCreditLinesAndLoans"], color="lightgreen")
plt.title("Boxplot de NumberOfOpenCreditLinesAndLoans", fontsize=14, fontweight="bold")
plt.xlabel("NumberOfOpenCreditLinesAndLoans", fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.savefig("figures/boxplot_NumberOfOpenCreditLinesAndLoans.png", dpi=300, bbox_inches='tight')
plt.show()

#Grafico de días en mora
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10, 5))
sns.boxplot(x=df["NumberOfTimes90DaysLate"], color="salmon")
plt.title("Boxplot de NumberOfTimes90DaysLate", fontsize=14, fontweight="bold")
plt.xlabel("NumberOfTimes90DaysLate", fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.savefig("figures/boxplot_NumberOfTimes90DaysLate.png", dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 5))
sns.countplot(x=df["NumberOfTimes90DaysLate"], color="coral")
plt.title("Conteo de NumberOfTimes90DaysLate", fontsize=14, fontweight="bold")
plt.xlabel("NumberOfTimes90DaysLate", fontsize=12)
plt.ylabel("Frecuencia")
plt.xticks(rotation=45)
plt.show()


#Grafico para número de dependientes
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10, 5))
sns.boxplot(x=df["NumberOfDependents"], color="gold")
plt.title("Boxplot de NumberOfDependents", fontsize=14, fontweight="bold")
plt.xlabel("NumberOfDependents", fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.savefig("figures/boxplot_NumberOfDependents.png", dpi=300, bbox_inches='tight')
plt.show()

#Grafico para edad del cliente
#BOXPLOT
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10, 5))
sns.boxplot(x=df["age"], color="mediumseagreen")
plt.title("Boxplot de age", fontsize=14, fontweight="bold")
plt.xlabel("age", fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.savefig("figures/boxplot_age.png", dpi=300, bbox_inches='tight')
plt.show()
#

#Número de posibles outliers por columna
Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1

outlier_mask = (df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))
outlier_counts = outlier_mask.sum().sort_values(ascending=False)

print("Número de posibles outliers por columna:")
print(outlier_counts)

