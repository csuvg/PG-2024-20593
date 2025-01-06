# Importar librerías necesarias
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim
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

X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_train.columns)

# Definir el primer pipeline con MaxAbsScaler y una Red Neuronal con PyTorch
class NeuralNetwork_0(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNetwork_0, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_model_0(model, X_train, y_train, epochs=100, batch_size=32, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

# Definir el segundo pipeline con TruncatedSVD y una Red Neuronal con PyTorch
class NeuralNetwork_1(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNetwork_1, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_model_1(model, X_train, y_train, epochs=150, batch_size=16, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

# Escalar los datos y entrenar los modelos
scaler = MaxAbsScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Entrenar el primer modelo
model_0 = NeuralNetwork_0(input_dim=X_train_scaled.shape[1])
train_model_0(model_0, X_train_scaled, y_train)

# Entrenar el segundo modelo con SVD
svd = TruncatedSVD(n_components=14, random_state=0)
X_train_svd = svd.fit_transform(X_train_scaled)
X_test_svd = svd.transform(X_test_scaled)
model_1 = NeuralNetwork_1(input_dim=X_train_svd.shape[1])
train_model_1(model_1, X_train_svd, y_train)

# Crear el VotingRegressor con los dos modelos entrenados
class VotingRegressorEnsemble:
    def __init__(self, models):
        self.models = models

    def predict(self, X):
        predictions = []
        for model in self.models:
            if isinstance(model, NeuralNetwork_1):
                X_input = X[:, :14]  # Para el modelo 1, usar solo las primeras 14 componentes
            else:
                X_input = X
            predictions.append(model(X_input))
        return torch.mean(torch.stack(predictions), dim=0).detach().numpy()

voting_regressor = VotingRegressorEnsemble([model_0, model_1])

# Hacer predicciones en el conjunto de prueba
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_pred = voting_regressor.predict(X_test_tensor)

# Calcular métricas de evaluación
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"MSE: {mse}")
print(f"R^2 Score: {r2}")

# Utilizar SHAP para explicar el modelo
# Debido a que VotingRegressor no es compatible directamente con SHAP,
# explicaremos uno de los modelos individuales, por ejemplo, model_0

explainer = shap.Explainer(lambda x: model_0(torch.tensor(x, dtype=torch.float32)).detach().numpy(), X_train_scaled)

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
plt.show()