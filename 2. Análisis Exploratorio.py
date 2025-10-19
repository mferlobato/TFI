import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar datos
#df_clean = pd.read_csv("/Users/mariafernandalobato/Documents/Fernanda/ITBA/TFI/data/raw/cs-training.csv")

df_clean = pd.read_csv("C:/Users/malobato/Desktop/Fer/tfi/cs-training-limpio-v2.csv")

# Imputación si no se hizo antes
df_clean['MonthlyIncome'].fillna(df_clean['MonthlyIncome'].median(), inplace=True)
df_clean['NumberOfDependents'].fillna(0, inplace=True)

# Estadísticas descriptivas generales
print(df_clean.describe())
print(df.shape)

##VARIABLE OBJETIVO: SeriousDlqin2yrs
# Distribución variable objetivo
sns.countplot(x='SeriousDlqin2yrs', data=df_clean, color='pink')
plt.title('Distribución de la variable objetivo')
plt.show()

#Violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(x='SeriousDlqin2yrs', y='MonthlyIncome_log', data=df_clean, color='pink')
plt.title('Distribución del Ingreso Mensual (Log) según Morosidad', fontsize=14)
plt.xlabel('Morosidad (0: No Moroso, 1: Moroso)', fontsize=12)
plt.ylabel('Ingreso Mensual Transformado (Log(V+1))', fontsize=12)
plt.xticks([0, 1], ['0 - No Moroso', '1 - Moroso'])
plt.show()

##VARIABLE AGE
# Histograma de edades
plt.figure(figsize=(8,4))
sns.histplot(df_clean['age'], bins=30, kde=True, color='pink')
plt.title('Distribución de la Edad')
plt.show()
#boxplot
plt.figure(figsize=(10, 5))
sns.boxplot(x=df["age"], color="pink")
plt.title("Boxplot de age", fontsize=14, fontweight="bold")
plt.xlabel("age", fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.savefig("figures/boxplot_age.png", dpi=300, bbox_inches='tight')
plt.show()


##VARIABLE : uso de líneas de crédito no aseguradas
variable = 'RevolvingUtilizationOfUnsecuredLines'

#  Histograma
plt.figure(figsize=(10, 5))
sns.histplot(df_clean[variable], bins=50, kde=True, color='pink', log_scale=False)
plt.title(f'Histograma de {variable} ', fontsize=14)
plt.xlabel(variable, fontsize=12)
plt.ylabel('Frecuencia', fontsize=12)
plt.xlim(0, 1.1) 
plt.show() 
# Boxplot
plt.figure(figsize=(10, 5))
sns.boxplot(x=df_clean[variable], color='pink')
plt.title(f'Boxplot de {variable} (Post-limpieza)', fontsize=14)
plt.xlabel(variable, fontsize=12)
plt.xlim(-0.1, 2.0) 
plt.show()


## Variable Días de mora
#Histogramas de distribución
delinquency_cols = [
    'NumberOfTimes90DaysLate',
    'NumberOfTime30-59DaysPastDueNotWorse',
    'NumberOfTime60-89DaysPastDueNotWorse'
]

plt.figure(figsize=(15, 5))
colors = ['#87CEEB', '#FF9999', '#99FF99'] 

for i, col in enumerate(delinquency_cols):
    plt.subplot(1, 3, i + 1)
    sns.countplot(x=df_clean[col], color=colors[i])
    

    plt.yscale('log') 
    
    plt.title(f'Conteo de {col}', fontsize=12)
    plt.xlabel('Número de Veces Atraso', fontsize=10)
    plt.ylabel('Frecuencia (Log)', fontsize=10)
    plt.xticks(rotation=0)
    
plt.tight_layout(pad=3.0)
plt.suptitle('Figura 6. Distribución de Frecuencia de Atrasos (Post-limpieza)', fontsize=16, y=1.05)
plt.show() 

# Boxplots
delinquency_cols = [
    'NumberOfTimes90DaysLate',
    'NumberOfTime30-59DaysPastDueNotWorse',
    'NumberOfTime60-89DaysPastDueNotWorse'
]
plt.figure(figsize=(15, 5))
colors = ['#87CEEB', '#FF9999', '#99FF99'] 
for i, col in enumerate(delinquency_cols):
    plt.subplot(1, 3, i + 1)
    sns.boxplot(x=df_clean[col], color=colors[i])
    
    plt.title(f'Boxplot de {col}', fontsize=12)
    plt.xlabel('Número de Veces Atraso', fontsize=10)
    
    plt.xlim(-0.5, 13) 
    plt.tight_layout(pad=3.0)
plt.suptitle('Figura 7. Boxplots de Frecuencia de Atrasos (Post-limpieza)', fontsize=16, y=1.05)
plt.show() 

## Variable número de dependientes
plt.figure(figsize=(10, 5))
sns.boxplot(x=df["NumberOfDependents"], color="pink")
plt.title("Boxplot de NumberOfDependents", fontsize=14, fontweight="bold")
plt.xlabel("NumberOfDependents", fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()

plt.figure(figsize=(10, 5))
sns.countplot(x=df_clean['NumberOfDependents'], color='pink')
plt.title('Distribución de Número de Dependientes (Post-limpieza)', fontsize=14)
plt.xlabel('Número de Dependientes', fontsize=12)
plt.ylabel('Frecuencia', fontsize=12)
plt.xlim(-0.5, df_clean['NumberOfDependents'].max() + 0.5)
plt.show()

##VARIALBE INGRESO MENSUAL
variable = 'MonthlyIncome_log'
# Plot 1: Histograma
plt.figure(figsize=(10, 5))
sns.histplot(df_clean[variable], bins=50, kde=True, color='pink')
plt.title(f'Histograma de {variable} (Post-transformación)', fontsize=14)
plt.xlabel('Ingreso Mensual Transformado (Log(V+1))', fontsize=12)
plt.ylabel('Frecuencia / Densidad', fontsize=12)
plt.show()
# Plot 2: Boxplot
plt.figure(figsize=(10, 5))
sns.boxplot(x=df_clean[variable], color='pink')
plt.title(f'Boxplot de {variable} (Post-transformación)', fontsize=14)
plt.xlabel('Ingreso Mensual Transformado (Log(V+1))', fontsize=12)
plt.show()


##VARIALBE lineas de créditos abiertas
variable = 'NumberOfOpenCreditLinesAndLoans'

# Plot 1: Histograma/Countplot
plt.figure(figsize=(10, 5))
sns.countplot(x=df_clean[variable], color='pink')
plt.title(f'Histograma de {variable} (Post-limpieza)', fontsize=14)
plt.xlabel('Número de Líneas de Crédito Abiertas', fontsize=12)
plt.ylabel('Frecuencia (Log)', fontsize=12)
plt.yscale('log')
plt.xlim(-0.5, 30.5) 
plt.xticks(range(0, 31, 5)) 
plt.show()

# Plot 2: Boxplot
plt.figure(figsize=(10, 5))
sns.boxplot(x=df_clean[variable], color='pink')
plt.title(f'Boxplot de {variable} (Post-limpieza)', fontsize=14)
plt.xlabel('Número de Líneas de Crédito Abiertas', fontsize=12)
plt.xlim(-0.5, 30.5)
plt.xticks(range(0, 31, 5)) 
plt.show()

##VARIALBE lineas de créditos abiertas
variable = 'DebtRatio'

# Plot 1: Histograma (Foco en el rango 0 a 1)
plt.figure(figsize=(10, 5))
sns.histplot(df_clean[variable], bins=100, kde=True, color='pink')
plt.title(f'Histograma de {variable} (Post-limpieza)', fontsize=14)
plt.xlabel('Ratio de Deuda', fontsize=12)
plt.ylabel('Frecuencia', fontsize=12)
plt.xlim(0, 1.0) 
plt.show()

# Plot 2: Boxplot (Foco en el rango 0 a 2.0)
plt.figure(figsize=(10, 5))
sns.boxplot(x=df_clean[variable], color='pink')
plt.title(f'Boxplot de {variable} (Post-limpieza) - Foco en el Rango Central', fontsize=14)
plt.xlabel('Ratio de Deuda', fontsize=12)
plt.xlim(-0.1, 2.0)
plt.show()

##Variable NumberRealEstateLoansOrLines
variable = 'NumberRealEstateLoansOrLines'
# Plot 1: Histograma/Countplot
plt.figure(figsize=(10, 5))
sns.countplot(x=df_clean[variable], color='pink')
plt.title(f'Histograma de {variable} (con Escala Logarítmica)', fontsize=14)
plt.xlabel('Número de Préstamos Inmobiliarios', fontsize=12)
plt.ylabel('Frecuencia (Log)', fontsize=12)
plt.yscale('log') 
plt.xlim(-0.5, 10.5) # Limitar a 10 para una visualización más limpia
plt.xticks(range(0, 11, 1)) 
plt.show()
# Plot 2: Boxplot
plt.figure(figsize=(10, 5))
sns.boxplot(x=df_clean[variable], color='pink')
plt.title(f'Boxplot de {variable} (Post-limpieza)', fontsize=14)
plt.xlabel('Número de Préstamos Inmobiliarios', fontsize=12)
plt.xlim(-0.5, 20.5)
plt.xticks(range(0, 21, 2)) 
plt.show()

##Matriz de correlacion de variables
# Calcular la matriz de correlación
correlation_matrix = df_clean.corr()

plt.figure(figsize=(12, 10))

sns.heatmap(
    correlation_matrix, 
    annot=True, 
    fmt=".2f", 
    cmap='RdPu', 
    linewidths=.5, 
    linecolor='black',
    cbar_kws={'label': 'Coeficiente de Correlación'}
)
plt.title('Matriz de Correlación de Variables (Tonos Rosados)', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show() 

## boxplot comparativo del ingreso mensual vs morisidad
plt.figure(figsize=(10, 6))
sns.boxplot(x='SeriousDlqin2yrs', y='MonthlyIncome_log', data=df_clean, palette='pastel')
plt.title('Figura 13. Boxplot Comparativo: Ingreso Mensual (Log) vs. Morosidad', fontsize=14)
plt.xlabel('Morosidad (0: No Moroso, 1: Moroso)', fontsize=12)
plt.ylabel('Ingreso Mensual Transformado (Log(V+1))', fontsize=12)
plt.xticks([0, 1], ['0 - No Moroso', '1 - Moroso'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show() 

