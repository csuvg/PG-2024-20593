# %%
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np

# %%
pd.set_option('display.max_columns', 2)
#pd.set_option('display.max_rows', None)

# %%
file_path = '../data/BASE_ENTRENAMIENTO_VALIDACION_GT_NIv2.xlsx'


entrenamiento_gt_df = pd.read_excel(file_path, sheet_name='entrenamiento_gt')
entrenamiento_gt_df.columns

# %%
entrenamiento_gt_df['TCH'].hist()

# %%
# Ordenar los datos
sorted_df = entrenamiento_gt_df.sort_values(by='TCH', ascending=False)

# Calcular el número de filas que corresponden al 10% superior
n_drop = int(len(sorted_df) * 0.01)

# Eliminar el 10% superior
filtered_df = sorted_df.iloc[n_drop:-n_drop]

# Crear el histograma con los datos filtrados
plt.figure(figsize=(10, 6))
filtered_df['TCH'].hist(bins=30)
plt.title('Histograma de TCH sin el 10% superior')
plt.xlabel('TCH')
plt.ylabel('Frecuencia')
plt.show()

# %%


rendimiento = entrenamiento_gt_df['rendimiento']

# Transformaciones
log_transformed = np.log(rendimiento)
sqrt_transformed = np.sqrt(rendimiento)

# Crear figura y ejes
fig, axs = plt.subplots(2, 2, figsize=(12, 12))

# Histograma de datos originales
sns.histplot(rendimiento, bins=30, kde=True, ax=axs[0, 0])
axs[0, 0].set_title('Original')

# Q-Q plot de datos originales
stats.probplot(rendimiento, dist="norm", plot=axs[0, 1])
axs[0, 1].set_title('Original: Q-Q plot')

# Histograma de datos transformados con logaritmo
sns.histplot(log_transformed, bins=30, kde=True, ax=axs[1, 0])
axs[1, 0].set_title('Log Transformation')

# Histograma de datos transformados con raíz cuadrada
sns.histplot(sqrt_transformed, bins=30, kde=True, ax=axs[1, 1])
axs[1, 1].set_title('Sqrt Transformation')

plt.suptitle('Normality Diagnosis Plot (rendimiento)')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# %%
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np

rendimiento = filtered_df['TCH']

# Transformaciones
log_transformed = np.log(rendimiento)
sqrt_transformed = np.sqrt(rendimiento)

# Crear figura y ejes
fig, axs = plt.subplots(2, 2, figsize=(12, 12))

# Histograma de datos originales
sns.histplot(rendimiento, bins=30, kde=True, ax=axs[0, 0])
axs[0, 0].set_title('Original')

# Q-Q plot de datos originales
stats.probplot(rendimiento, dist="norm", plot=axs[0, 1])
axs[0, 1].set_title('Original: Q-Q plot')

# Histograma de datos transformados con logaritmo
sns.histplot(log_transformed, bins=30, kde=True, ax=axs[1, 0])
axs[1, 0].set_title('Log Transformation')

# Histograma de datos transformados con raíz cuadrada
sns.histplot(sqrt_transformed, bins=30, kde=True, ax=axs[1, 1])
axs[1, 1].set_title('Sqrt Transformation')

plt.suptitle('Normality Diagnosis Plot (rendimiento)')
plt.tight_layout(rect=[0, 0, 1, 0.95])
#plt.show()

# %%
#filtered_df

# %%
file_path_INDICES = 'data/INDICES/GT_INDICES_NAX.xlsx'

indices_gt = pd.read_excel(file_path_INDICES, sheet_name='INDICES')


# %%
file_path_CLIMA = 'data/CLIMA/GT_CLIMA.xlsx'


clima_gt = pd.read_excel(file_path_CLIMA, sheet_name='Reporte')




# %% [markdown]
# # ANÁLISIS DE DATOS DE MUESTRAS DE CAMPO DE CAÑA DE AZÚCAR
# 
# ## Analisis Exploratorio 
# 
# ### Tipo de cada Variable de los Datasets
# 
# #### Datos de Cosecha
# 
# * **ID:** (ZAFRA + TERRENO) 
# * **TIPO_MUESTRA:** (AUN NO EXPLICADO) Cualitativa Nominal
# * **ZAFRA:** (AÑO DE LA ZAFRA) Cualitativa Ordinal
# * **TERRENO:** (TERRENO DE LA ZAFRA ) Cualitativa Nominal
# * **FECHA:** (FECHA DE LA MUESTRA) Cualitativa Ordinal
# * **FI:** (FECHA DE INICIO DE LA ZAFRA) Cualitativa Ordinal
# * **FF:** (FECHA DE FIN DE LA ZAFRA) Cualitativa Ordinal
# * **AREA:** (AREA DEL TERRENO) Cuantitativa Continua
# * **PROD_MAD:** (TIPO DE MADURACION USADA) Cualitativa Nominal
# * **FECHA_MAD:** (FECHA DE MADURACION) Cuantitativa Continua | NaN si no tiene tipo de maduración
# * **ESTACION:** (ESTACION METEOROLOGICA) Cualitativa 
# * **VARIEDAD:** (VARIEDAD DE CAÑA) Cualitativa Nominal
# * **NUMERO_CORTE:** (NUMERO DE CORTE) Cuantitativa Discreta
# * **SIST_RIEGO:** (SISTEMA DE RIEGO) Cualitativa Nominal
# * **TIPO_COSECHA:** (TIPO DE COSECHA) Cualitativa Nominal
# * **REGION:** (REGION DEL TERRENO) Cualitativa Nominal
# * **ESTRATO:** (ESTRATO DEL TERRENO) Cualitativa Nominal
# * **CUADRANTE:** (CUADRANTE DEL TERRENO) Cualitativa Nominal
# * **MES_MUESTRA:** (AUN NO EXPLICADO) Cualitativa Ordinal
# * **MES_CORTE:** (MES DE CORTE) Cualitativa Ordinal
# * **EDAD_PROYECTADA:** (DIAS DE VIDA DE LA ZAFRA AL FINAL DEL CORTE) Cuantitativa Discreta
# * **EDAD_ACT:** (DIAS DE VIDA DE LA ZAFRA AL INICIO DEL CORTE) Cuantitativa Discreta 
# * **DIAS_CORTE:** (DIAS DE DIFERENCIA ENTRE EL CORTE Y LA MUESTRA) Cuantitativa Discreta
# * **SEMANA_CORTE:** (SEMANA DE CORTE) Cuantitativa Discreta
# * **PRODUCTO_ACTUAL:** (TIPO DE MADURACION FINAL) Cualitativa Nominal
# * **ESTATUS_MAD:** (ESTADO DE MADURACION FINAL) Cualitativa Nominal
# * **RENDIMIENTO:** (TONELADAS DE CAÑA POR HECTAREA) Cuantitativa Continua
# 
# 
# #### Datos de Indices de la Caña
# 
# * **IDCOMP:** (PANTONE) Cualitativa Nominal
# * **ABS_IDCOMP:** (TERRENO) Cualitativa Nominal
# * **FECHA:** (FECHA DE LA MUESTRA) Cualitativa Ordinal
# * **NDVI_POND:** (Índice de Vegetación de Diferencia Normalizada (NDVI) ponderado) Cuantitativa Continua
# * **AGUA_POND:** (Índice de contenido de agua ponderado) Cuantitativa Continua
# * **LAI_POND:** (Índice de Área Foliar ponderado) Cuantitativa Continua
# * **PRODUCCION_PON:** (Índice de Producción ponderado) Cuantitativa Continua
# * **ANOMALIAS_POND:** (Índice de Anomalías ponderado) Cuantitativa Continua
# * **NITROGENADO_POND:** (Índice de Nitrógeno ponderado) Cuantitativa Continua
# * **MADURACION_POND:** (Índice de Maduración ponderado) Cuantitativa Continua
# * **HUMEDAD_POND:** (Índice de Humedad ponderado) Cuantitativa Continua
# * **ESTATUS_COSECHA:** (ESTADO DE LA COSECHA) Cualitativa Nominal
# 
# 
# #### Datos de Clima
# 
# * **Zafra:** (AÑO DE LA ZAFRA) Cualitativa Ordinal
# * **Año:** (AÑO DE LA MUESTRA) Cualitativa Ordinal
# * **Mes:** (MES DE LA MUESTRA) Cualitativa Ordinal
# * **Día:** (DIA DE LA MUESTRA) Cualitativa Ordinal
# * **Cuadrante:** (CUADRANTE DEL TERRENO) Cualitativa Nominal
# * **Estrato:** (ESTRATO DEL TERRENO) Cualitativa Nominal
# * **Región:** (REGION DEL TERRENO) Cualitativa Nominal
# * **estatus:** (AUN NO EXPLICADO) Cualitativa Nominal
# * **ETP:** (EVAPOTRANSPIRACION) Cuantitativa Continua
# * **Radiacion (MJ/m2):** (RADIACION ?) Cuantitativa Continua
# * **Amplitud Térmica:** (AMPLITU DE TEMPERATURA) Cuantitativa Continua
# * **R0:** (RADIACION SOLAR) Cuantitativa Continua
# * **Estacion:** (ESTACION METEOROLOGICA) Cualitativa Nominal
# * **Fecha:** (FECHA DE LA MUESTRA) Cualitativa Ordinal
# * **temperatura:** (TEMPERATURA EN CELCIUS) Cuantitativa Continua
# * **temperatura minima:** (TEMPERATURA MINIMA EN CELCIUS) Cuantitativa Continua
# * **temperatura maxima:** (TEMPERATURA MAXIMA EN CELCIUS) Cuantitativa Continua
# * **radiacion:** (RADIACION) Cuantitativa Continua
# * **radiacion promedio:** (RADIACION PROMEDIO) Cuantitativa Continua
# * **humedad relativa:** (HUMEDAD RELATIVA) Cuantitativa Continua
# * **humedad relativa minima:** (HUMEDAD RELATIVA MINIMA) Cuantitativa Continua
# * **humedad relativa maxima:** (HUMEDAD RELATIVA MAXIMA) Cuantitativa Continua
# * **precipitacion:** (PRECIPITACION) Cuantitativa Continua
# * **velocidad viento:** (VELOCIDAD DEL VIENTO) Cuantitativa Continua
# * **velocidad viento minima:** (VELOCIDAD DEL VIENTO MINIMA) Cuantitativa Continua
# * **velocidad viento maxima:** (VELOCIDAD DEL VIENTO MAXIMA) Cuantitativa Continua
# * **mojadura:** (HUMEDAD DE LA TIERRA) Cuantitativa Continua
# * **presion atmosferica:** (PRESION ATMOSFERICA DEL TERRENO) Cuantitativa Continua
# * **presion atmosferica minima:** (PRESION ATMOSFERICA MINIMA DEL TERRENO) Cuantitativa Continua
# * **presion atmosferica maxima:** (PRESION ATMOSFERICA MAXIMA DEL TERRENO) Cuantitativa Continua
# * **direccion viento:** (DIRECCION DEL VIENTO) Cuantitativa Continua
# 
# 
# 
# 

# %%
# GUARDAR EN CSV LOS DATOS
clima_gt.to_csv('./CLIMA/GT_CLIMA.csv', index=False)
indices_gt.to_csv('./INDICES/GT_INDICES_NAX.csv', index=False)
entrenamiento_gt_df.to_csv('./BASE_ENTRENAMIENTO_VALIDACION_GT_NI.csv', index=False)

# %%
filtered_df.to_csv('./data/BASE_ENTRENAMIENTO_VALIDACION_GT_NIv2.csv', index=False)

# %% [markdown]
# ### Analisis de NA y Limpieza de Datos

# %%
import pandas as pd
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

pd.set_option('display.max_columns',2)
#pd.set_option('display.max_rows', None)

# %%
# LEER CSV
weather_data  = pd.read_csv('../data/CLIMA/GT_CLIMA.csv', parse_dates=['Fecha'])
index_data = pd.read_csv('../data/INDICES/GT_INDICES_NAX.csv', parse_dates=['FECHA'], low_memory=False)
harvest_data    = pd.read_csv('../data/BASE_ENTRENAMIENTO_VALIDACION_GT_NIv2.csv', parse_dates=['fecha', 'fi', 'ff'])

# %%
harvest_data.columns

# %%
harvest_data['numero_corte'].hist()

# %%
# SAMPLES
# Guardar los primeros 100 datos de cada dataset
# clima_gt_sample = clima_gt.head(100)
# indices_gt_sample = indices_gt.head(100)
# entrenamiento_gt_sample = entrenamiento_gt_df.head(100)

# # Guardar los datasets de muestra como nuevos archivos CSV
# clima_gt_sample.to_csv('./SAMPLE/CLIMA.csv', index=False)
# indices_gt_sample.to_csv('./SAMPLE/INDICES.csv', index=False)
# entrenamiento_gt_sample.to_csv('./SAMPLE/BASE.csv', index=False)


# %%
# Variables numéricas
numeric_vars = ['area', 'numero_corte', 'edad_proyectada', 'edad_act', 'rendimiento','TCH']

# Calcular estadísticas descriptivas
descriptive_stats = harvest_data[numeric_vars].describe()
#print(descriptive_stats)

# %%
numeric_vars_indice = ['NDVI_POND','AGUA_POND','LAI_POND','PRODUCCION_PON','ANOMALIAS_POND','NITROGENADO_POND','MADURACION_POND','HUMEDAD_POND']

descriptive_stats_indice = index_data[numeric_vars_indice].describe()
#print(descriptive_stats_indice)

# %%
numeric_vars_clima = ['ETP','Radiacion (MJ/m2)','Amplitud Térmica',
'R0','temperatura','temperatura minima', 'temperatura maxima','radiacion', 'radiacion promedio','humedad relativa', 
'humedad relativa minima', 'humedad relativa maxima','precipitacion', 'velocidad viento','velocidad viento minima',
'velocidad viento maxima','mojadura', 'presion atmosferica', 'presion atmosferica minima',
'presion atmosferica maxima', 'direccion viento']

descriptive_stats_clima = weather_data[numeric_vars_clima].describe()
#print(descriptive_stats_clima)

# %%
base_data = pd.read_csv('data/PROCESS/processed_tch_prediction_datav2.csv')
base_data_sample = base_data.head(100)

# Guardar los datasets de muestra como nuevos archivos CSV
base_data_sample.to_csv('./SAMPLE/FIN.csv', index=False)

# %%
# Calculate correlation matrices
weather_corr = weather_data[numeric_vars_clima].corr()
index_corr = index_data[numeric_vars_indice].corr()
harvest_corr = harvest_data[numeric_vars].corr()

# %%


import matplotlib.pyplot as plt
import seaborn as sns

# Function to plot the correlation matrix
def plot_correlation_matrix(corr_matrix, title):
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True)
    plt.title(title)
    plt.show()

# Plotting correlation matrices
plot_correlation_matrix(weather_corr, "Weather Data Correlation Matrix")
plot_correlation_matrix(index_corr, "Index Data Correlation Matrix")
plot_correlation_matrix(harvest_corr, "Harvest Data Correlation Matrix")

# %%
# Calcula el porcentaje de NA en cada dataset
weather_na_percent = (weather_data.isna().sum() / len(weather_data)) * 100
index_na_percent = (index_data.isna().sum() / len(index_data)) * 100
harvest_na_percent = (harvest_data.isna().sum() / len(harvest_data)) * 100

# Ordena por porcentaje de NA de mayor a menor
weather_na_percent_sorted = weather_na_percent.sort_values(ascending=False)
index_na_percent_sorted = index_na_percent.sort_values(ascending=False)
harvest_na_percent_sorted = harvest_na_percent.sort_values(ascending=False)

# Muestra los resultados
# print("Weather NA Percent Sorted:\n", weather_na_percent_sorted)
# print("\nIndex NA Percent Sorted:\n", index_na_percent_sorted)
# print("\nHarvest NA Percent Sorted:\n", harvest_na_percent_sorted)


# %%

# Count the number of NA values in each dataset
weather_na_count = weather_data.isna().sum()
index_na_count = index_data.isna().sum()
harvest_na_count = harvest_data.isna().sum()

#weather_na_count

# %%
#index_na_count

# %%
#harvest_na_count

# %%
#weather_data

# %%
#weather_data


