# %%
import pandas as pd
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# %%
data = pd.read_csv('../data/PROCESS/processed_tch_prediction_datav6.csv')



# %%

# Supongamos que 'df' es tu DataFrame procesado
suffixes = ['_sum', '_std', '_integral','Q1_mean','Q2_mean']

# Lista de columnas base sin los sufijos especificados
base_columns = [col for col in data.columns if not any(col.endswith(suffix) for suffix in suffixes)]



# %%
midata = data[base_columns]


# %%

# List of categorical columns to encode
cat_columns = ['prod_mad', 'estacion', 'variedad', 'sist_riego', 'tipo_cosecha', 'region', 'estrato', 'cuadrante', 'PRODUCTO_ACTUAL']

# 1. One-Hot Encoding for nominal variables with few categories
ohe_columns = ['prod_mad', 'sist_riego', 'tipo_cosecha', 'PRODUCTO_ACTUAL']
ohe = OneHotEncoder(handle_unknown='ignore')
ohe_encoded = ohe.fit_transform(data[ohe_columns])
ohe_feature_names = ohe.get_feature_names_out(ohe_columns)
ohe_df = pd.DataFrame(ohe_encoded.toarray(), columns=ohe_feature_names, index=data.index)

# Guardar diccionario de One-Hot Encoding
ohe_mapping = {col: list(ohe.categories_[i]) for i, col in enumerate(ohe_columns)}
ohe_mapping_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in ohe_mapping.items()]))
ohe_mapping_df.to_csv('../data/ENCODING_DICTIONARIES/one_hot_encoding_mapping.csv', index=False)

# 2. Ordinal Encoding for ordinal variables
ordinal_columns = ['region', 'estrato']
oe = OrdinalEncoder()
data[ordinal_columns] = oe.fit_transform(data[ordinal_columns])

# Guardar diccionario de Ordinal Encoding
ordinal_mapping = {col: list(oe.categories_[i]) for i, col in enumerate(ordinal_columns)}
ordinal_mapping_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in ordinal_mapping.items()]))
ordinal_mapping_df.to_csv('../data/ENCODING_DICTIONARIES/ordinal_encoding_mapping.csv', index=False)


# %%

# 3. Label Encoding for nominal variables with many categories
le_columns = ['estacion', 'variedad', 'cuadrante']
label_mappings = {}
for col in le_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_mappings[col] = dict(enumerate(le.classes_))

# Guardar diccionario de Label Encoding
for col, mapping in label_mappings.items():
    mapping_df = pd.DataFrame(list(mapping.items()), columns=[f'Encoded_Value_{col}', f'Original_Value_{col}'])
    mapping_df.to_csv(f'../data/ENCODING_DICTIONARIES/label_encoding_mapping_{col}.csv', index=False)

# Combine the encoded data
encoded_data = pd.concat([
    data[['ABS_IDCOMP', 'ZAFRA']],  # Keep these columns without encoding
    data.drop(columns=cat_columns + ['ABS_IDCOMP', 'ZAFRA']),  # Numeric columns
    ohe_df,  # One-hot encoded columns
    data[ordinal_columns],  # Ordinal encoded columns
    data[le_columns]  # Label encoded columns
], axis=1)

# Handle missing values
numeric_columns = encoded_data.select_dtypes(include=[np.number]).columns
imputer = SimpleImputer(strategy='mean')
encoded_data[numeric_columns] = imputer.fit_transform(encoded_data[numeric_columns])
encoded_data = encoded_data.drop('rendimiento', axis=1)

# Save the encoded data
encoded_data.to_csv('../data/PROCESS/encoded_tch_prediction_data_zafrav3.csv', index=False)

mask_22_23 = encoded_data['ZAFRA'] == '23-24'
encoded_data[~mask_22_23].to_csv('../data/UP/TRAINC.csv', index=False)
encoded_data[mask_22_23].to_csv('../data/UP/TESTC.csv', index=False)


# %%

encoded_data.dtypes

# %%

import seaborn as sns

adata = encoded_data.drop(['ABS_IDCOMP', 'ZAFRA', 'fecha'], axis=1)

adata.dtypes

# %%

plt.figure(figsize=(12, 10))
sns.heatmap(adata.corr(), annot=False, cmap='coolwarm', square=True)
plt.title('TOTAL')
plt.show()

# %%

# Supongamos que df es tu DataFrame original
correlation_matrix = adata.corr().abs()

# Selecciona el triángulo superior de la matriz de correlación
upper = correlation_matrix.where(
    np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
)

# Encuentra las columnas con una correlación mayor a 0.9
to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]

# Elimina las columnas que tienen una correlación mayor a 0.9
df_reduced = adata.drop(columns=to_drop)


df_reduced

# %%
# Supongamos que 'df' es tu DataFrame procesado
suffixes = [ '_sum', '_std', '_integral']

# Lista de columnas base sin los sufijos especificados
base_columns = [col for col in df_reduced.columns if not any(col.endswith(suffix) for suffix in suffixes)]
base_columns


# %%
# Combine the encoded data
encoded_data = pd.concat([
    data[['ABS_IDCOMP', 'ZAFRA']],  # Keep these columns without encoding
    df_reduced
], axis=1)



print(f"Columnas eliminadas: {to_drop}")
encoded_data.to_csv('../data/PROCESS/encoded_tch_prediction_data_zafrav3.2.csv', index=False)


# %%

# Supongamos que 'encoded_data' es tu DataFrame
columnas_totales = df_reduced.columns.tolist()

# Vamos a hacer bloques de 50 columnas
tamaño_bloque = 51
bloques = [columnas_totales[i:i + tamaño_bloque] for i in range(0, len(columnas_totales), tamaño_bloque)]

# Ahora, por cada bloque, puedes hacer algo con esas columnas.
# Visualizar el primer bloque de 50 columnas
primer_bloque = df_reduced[bloques[0]]  # Primeras 50 columnas

# Si quieres mostrar las primeras filas de esas columnas
print(primer_bloque.head())
for idx, bloque in enumerate(bloques):
    bloque_data = df_reduced[bloque]
    print(f"Bloque {idx + 1}")
    plt.figure(figsize=(35, 25))
    sns.heatmap(bloque_data.corr(), annot=True, cmap='coolwarm', square=True)
    plt.title(f"Bloque {idx + 1}")
    plt.show() # Aquí puedes generar una imagen o gráfico si es necesario.


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Supongamos que df es tu DataFrame original
# Excluye la variable objetivo 'TCH'
X = df_reduced.drop(columns=['TCH'])

# Estandarizar las variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Crear un DataFrame con las cargas de las variables en cada componente
pca_components = pd.DataFrame(pca.components_, columns=X.columns, index=[f'PC{i+1}' for i in range(len(pca.components_))])

# Graficar un heatmap para visualizar cuánto contribuye cada variable en cada componente
plt.figure(figsize=(12, 8))
sns.heatmap(pca_components, cmap='coolwarm', annot=False)
plt.title('Cargas de las Variables en cada Componente Principal')
plt.show()


# %%
explained_variance_ratio = pca.explained_variance_ratio_

cumulative_variance = np.cumsum(explained_variance_ratio)

# Determinar el número de componentes necesarios para explicar el 80% de la varianza
num_components = np.argmax(cumulative_variance >= 0.80) + 1

print(f"Necesitas {num_components} componentes principales para explicar al menos el 80% de la varianza.")
# Graficar la varianza acumulada
plt.figure(figsize=(8, 6))
plt.plot(cumulative_variance, marker='o', linestyle='--')
plt.axhline(y=0.80, color='r', linestyle='--')
plt.title('Varianza Acumulada por Componentes Principales')
plt.xlabel('Número de Componentes')
plt.ylabel('Varianza Acumulada')
plt.grid(True)
plt.show()

# %%
# Aplicar PCA con el número de componentes óptimos (num_components)
pca_optimal = PCA(n_components=num_components)
X_pca_optimal = pca_optimal.fit_transform(X_scaled)

# Crear un DataFrame con los componentes principales
pca_columns = [f'PC{i+1}' for i in range(num_components)]
df_pca_optimal = pd.DataFrame(X_pca_optimal, columns=pca_columns)

# Agregar las columnas 'TCH', 'ABS_IDCOMP', y 'ZAFRA' del DataFrame original
df_pca_optimal['TCH'] = data['TCH'].values
df_pca_optimal[['ABS_IDCOMP', 'ZAFRA']] = data[['ABS_IDCOMP', 'ZAFRA']].values

# Mostrar el DataFrame resultante
df_pca_optimal.head()


# %%
df_pca_optimal.to_csv('../data/PROCESS/encoded_tch_prediction_data_zafrav3pca.csv', index=False)


# %%
print("Categorical variables encoded. Encoded data saved to 'encoded_tch_prediction_data.csv'.")
print(f"Shape of encoded data: {encoded_data.shape}")
print("\nEncoded columns:")
print(encoded_data.columns.tolist())

# Display some statistics about the encoded data
print("\nDataset Statistics:")
print(encoded_data.describe())

# Check for any remaining missing values
print("\nMissing values after encoding and imputation:")
print(encoded_data.isnull().sum())

# %%
encoded_data.head()


