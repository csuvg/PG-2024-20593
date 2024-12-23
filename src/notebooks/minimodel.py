# Importar librerías necesarias
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score

import xgboost as xgb
import shap
import matplotlib.pyplot as plt

# Cargar los datos
data = pd.read_csv('../data/PROCESS/encoded_tch_prediction_data_zafrav3.2.csv')

# Crear una máscara para ZAFRA 23-24
mask_23_24 = data['ZAFRA'] == '23-24'

# Dividir los datos en entrenamiento y prueba
X_train = data[~mask_23_24].drop('TCH', axis=1)
y_train = data[~mask_23_24]['TCH']
X_test = data[mask_23_24].drop('TCH', axis=1)
y_test = data[mask_23_24]['TCH']

# Eliminar columnas innecesarias
X_train = X_train.drop(columns=['ABS_IDCOMP', 'ZAFRA'])
X_test = X_test.drop(columns=['ABS_IDCOMP', 'ZAFRA'])

print("\nZAFRA Split:")
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Asegurarse de que todos los datos sean numéricos
# Si hay columnas categóricas, deben ser codificadas antes de este paso

# Imputar valores faltantes con la media
imputer = SimpleImputer(strategy='mean')

X_train_imputed = pd.DataFrame(
    imputer.fit_transform(X_train), columns=X_train.columns)
X_test_imputed = pd.DataFrame(
    imputer.transform(X_test), columns=X_train.columns)

# Definir el primer pipeline con MaxAbsScaler y XGBRegressor
pipeline_0 = Pipeline([
    ('scaler', MaxAbsScaler()),
    ('model', xgb.XGBRegressor(
        objective='reg:squarederror',
        learning_rate=0.3,
        max_depth=6,
        n_estimators=100,
        verbosity=0,
        random_state=0
    ))
])

# Definir el segundo pipeline con TruncatedSVD y XGBRegressor con diferentes hiperparámetros
pipeline_1 = Pipeline([
    ('svd', TruncatedSVD(n_components=14, random_state=0)),
    ('model', xgb.XGBRegressor(
        objective='reg:squarederror',
        learning_rate=0.2,
        max_depth=10,
        n_estimators=400,
        colsample_bytree=0.8,
        subsample=0.6,
        gamma=10,
        reg_lambda=0.625,
        verbosity=0,
        random_state=0
    ))
])

# Crear el VotingRegressor con los dos pipelines
voting_regressor = VotingRegressor([
    ('pipeline_0', pipeline_0),
    ('pipeline_1', pipeline_1)
])

# Entrenar el modelo ensamble
voting_regressor.fit(X_train_imputed, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = voting_regressor.predict(X_test_imputed)

# Calcular métricas de evaluación
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"MSE: {mse}")
print(f"R^2 Score: {r2}")

# Entrenar pipeline_0 individualmente para asegurarnos de que esté entrenado
pipeline_0.fit(X_train_imputed, y_train)

# Crear el objeto explainer de SHAP para el modelo XGBoost
explainer = shap.Explainer(pipeline_0.named_steps['model'])

# Obtener los valores SHAP para el conjunto de prueba
# Aplicamos el escalado a X_test_imputed
X_test_scaled = pipeline_0.named_steps['scaler'].transform(X_test_imputed)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test_imputed.columns)

# Calcular los valores SHAP
shap_values = explainer(X_test_scaled)

# Realizar el resumen gráfico de SHAP
shap.summary_plot(shap_values, X_test_scaled)

# Si deseas guardar el gráfico
# plt.savefig('shap_summary_plot.png')

# También puedes visualizar el impacto de una característica específica
# Por ejemplo, la característica más importante según SHAP
shap.plots.bar(shap_values)

# Dependence plot para una característica específica
# Reemplaza 'feature_name' con el nombre de la característica que deseas analizar
feature_name = 'area'  # Ejemplo
shap.dependence_plot(feature_name, shap_values.values, X_test_scaled)

# Si deseas guardar el gráfico
# plt.savefig('shap_dependence_plot.png')
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue', label='Predicciones')
plt.scatter(y_test, y_test, alpha=0.5, color='red', label='Valores reales')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r', lw=2)
plt.title('Predicted vs Actual Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.show()  # Utilizar SHAP para explicar el pipeline 1
# Entrenar pipeline_1 individualmente para asegurarnos de que esté entrenado
pipeline_1.fit(X_train_imputed, y_train)

# Crear el objeto explainer de SHAP para el modelo XGBoost del pipeline 1
explainer_pipeline_1 = shap.Explainer(pipeline_1.named_steps['model'])

# Aplicar la reducción de dimensionalidad con TruncatedSVD al conjunto de prueba
X_test_svd = pipeline_1.named_steps['svd'].transform(X_test_imputed)
X_test_svd = pd.DataFrame(
    X_test_svd, columns=[f'Componente_{i+1}' for i in range(X_test_svd.shape[1])])

# Calcular los valores SHAP para el conjunto de prueba reducido
shap_values_pipeline_1 = explainer_pipeline_1(X_test_svd)

# Realizar el resumen gráfico de SHAP para el pipeline 1
shap.summary_plot(shap_values_pipeline_1, X_test_svd)

# También puedes visualizar el impacto de una característica específica en el pipeline 1
# Dependence plot para una característica específica en el espacio reducido
# Ejemplo, reemplaza con el índice de la característica que deseas analizar
feature_index = 0
shap.dependence_plot(feature_index, shap_values_pipeline_1.values, X_test_svd)
