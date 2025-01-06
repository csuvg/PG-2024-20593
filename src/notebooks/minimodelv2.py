# Importar librerías necesarias
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

# Cargar los datos (asumiendo que tienes el archivo en la ruta correcta)
data = pd.read_csv('../data/PROCESS/encoded_tch_prediction_data_zafrav3.2.csv')

# Crear una máscara para ZAFRA 23-24
mask_23_24 = data['ZAFRA'] == '23-24'

# Dividir los datos en entrenamiento y prueba
X_train = data[~mask_23_24].drop('TCH', axis=1)
y_train = data[~mask_23_24]['TCH']

# Eliminar columnas innecesarias
X_train = X_train.drop(columns=['ABS_IDCOMP', 'ZAFRA'])

# Imputar valores faltantes con la media
imputer = SimpleImputer(strategy='mean')
X_train_imputed = pd.DataFrame(
    imputer.fit_transform(X_train), columns=X_train.columns)

# Crear y ajustar el TruncatedSVD
svd = TruncatedSVD(n_components=14, random_state=0)
X_train_svd = svd.fit_transform(X_train_imputed)

# Obtener la matriz de componentes
components = pd.DataFrame(
    svd.components_.T,
    columns=[f'Componente_{i+1}' for i in range(svd.n_components)],
    index=X_train.columns
)

# Calcular la importancia relativa de cada variable en cada componente
importance = np.abs(components.values)
importance = importance / importance.sum(axis=0)

# Crear un DataFrame con la importancia de las variables
importance_df = pd.DataFrame(
    importance,
    columns=[f'Componente_{i+1}' for i in range(svd.n_components)],
    index=X_train.columns
)

# Mostrar las 5 variables más importantes para cada componente
for i in range(svd.n_components):
    print(f"\nComponente_{i+1} - Top 5 variables:")
    print(importance_df[f'Componente_{i+1}'].nlargest(5))

# Visualizar la importancia de las variables en los primeros 3 componentes
plt.figure(figsize=(15, 10))
for i in range(3):
    plt.subplot(3, 1, i+1)
    importance_df[f'Componente_{i+1}'].sort_values(
        ascending=False).plot(kind='bar')
    plt.title(f'Importancia de las variables en Componente_{i+1}')
    plt.tight_layout()
plt.show()

# Calcular y mostrar la varianza explicada por cada componente
explained_variance_ratio = svd.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

plt.figure(figsize=(10, 5))
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
plt.step(range(1, len(cumulative_variance_ratio) + 1),
         cumulative_variance_ratio, where='mid', label='Acumulada')
plt.xlabel('Número de componentes')
plt.ylabel('Proporción de varianza explicada')
plt.title('Varianza explicada por componente')
plt.legend()
plt.tight_layout()
plt.show()
