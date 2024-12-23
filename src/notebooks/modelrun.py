import random
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
# Realizar Grid Search
from itertools import product

# Seed all possible
seed_ = 2023
random.seed(seed_)
np.random.seed(seed_)
torch.manual_seed(seed_)

# If using CUDA, you can set the seed for CUDA devices as well
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_)
    torch.cuda.manual_seed_all(seed_)

import torch.backends.cudnn as cudnn

cudnn.deterministic = True
cudnn.benchmark = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

data = pd.read_csv('../data/UP/encoded_tch_prediction_data_zafrav3.csv')

# Create a mask for ZAFRA 22-23
mask_22_23 = data['ZAFRA'] == '23-24'

# Split the data
X_train = data[~mask_22_23].drop('TCH', axis=1)
y_train = data[~mask_22_23]['TCH']
X_test = data[mask_22_23].drop('TCH', axis=1)
y_test = data[mask_22_23]['TCH']

X_train = X_train.drop(columns=['ABS_IDCOMP','ZAFRA','fecha'])
y_train = y_train.drop(columns=['ABS_IDCOMP','ZAFRA','fecha'])
X_test = X_test.drop(columns=['ABS_IDCOMP','ZAFRA','fecha'])
y_test = y_test.drop(columns=['ABS_IDCOMP','ZAFRA','fecha'])

print("\nZAFRA Split:")
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test.values).unsqueeze(1)

X_train = X_train_tensor
y_train = y_train_tensor
X_test = X_test_tensor
y_test = y_test_tensor

# Dividir X_test en X_val y X_test
X_val_tensor, X_test_tensor, y_val_tensor, y_test_tensor = train_test_split(
    X_test_tensor, y_test_tensor, test_size=0.5, random_state=42)


# Definir la función para calcular R^2
def calculate_r2(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    return r2.item()


# Clase Net (la misma que proporcioné antes)
class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()

        output_size = 1  # Para regresión

        N = 16  # Número total de capas lineales

        # Definir el tamaño máximo como un múltiplo del tamaño de entrada
        peak_size = input_size * 4  # Ajustar el multiplicador según sea necesario

        # Inicializar la lista de tamaños de capas
        layer_sizes = []

        # Capas ascendentes
        for i in range(N // 2):
            size = input_size + int((peak_size - input_size) * (i + 1) / (N // 2))
            layer_sizes.append(size)

        # Capas descendentes
        for i in range(N // 2):
            size = peak_size - int((peak_size - output_size) * (i + 1) / (N // 2))
            layer_sizes.append(size)

        # Definir las capas
        self.layers = nn.ModuleList()
        prev_size = input_size
        for size in layer_sizes:
            self.layers.append(nn.Linear(prev_size, size))
            prev_size = size
        self.layers.append(nn.Linear(prev_size, output_size))  # Capa de salida

        # Activaciones
        self.sigmoid = nn.Sigmoid()
        self.leaky_relu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        N = len(self.layers)
        for i, layer in enumerate(self.layers[:-1]):  # Excluir la última capa
            x = layer(x)

            # Activaciones: sigmoid al principio, luego leaky ReLU, luego ReLU
            if i < N // 3:
                x = self.sigmoid(x)
            elif i < 2 * N // 3:
                x = self.leaky_relu(x)
            else:
                x = self.relu(x)

            # Dropout cada dos capas ocultas
            if (i + 1) % 2 == 0:
                x = self.dropout(x)

        # Capa final sin activación (para regresión)
        x = self.layers[-1](x)
        return x

    # Función de pérdida L1
    def l1_loss(self):
        l1_loss = 0
        for param in self.parameters():
            l1_loss += torch.sum(torch.abs(param))
        return l1_loss

    # Función de pérdida L2
    def l2_loss(self):
        l2_loss = 0
        for param in self.parameters():
            l2_loss += torch.sum(param.pow(2))
        return l2_loss


# Definir una función para entrenar el modelo con hiperparámetros específicos
def train_model(model, criterion, optimizer, scheduler, X_train, y_train, X_val, y_val, num_epochs=100,
                l1_lambda=0.01, l2_lambda=0.01, early_stopping_patience=10, batch_size=32, writer=None):
    # Inicializar variables para early stopping
    global best_model_state
    best_val_loss = np.inf
    patience_counter = 0

    # Crear DataLoader para manejo de batches
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        # Entrenamiento
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            l1_reg = l1_lambda * model.l1_loss()
            l2_reg = l2_lambda * model.l2_loss()
            loss = loss + l1_reg + l2_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Actualizar el learning rate
        scheduler.step()

        # Evaluación en validación
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            val_r2 = calculate_r2(y_val, val_outputs)

        # Registrar en TensorBoard
        if writer:
            writer.add_scalar('Loss/train', epoch_loss / len(train_loader), epoch)
            writer.add_scalar('Loss/val', val_loss.item(), epoch)
            writer.add_scalar('R2/val', val_r2, epoch)
            writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)

        # Early Stopping
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            patience_counter = 0
            # Guardar el mejor modelo
            best_model_state = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break

        # Mostrar métricas cada 50 epochs
        if (epoch + 1) % 50 == 0:
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], LR: {scheduler.get_last_lr()[0]:.6f}, Train Loss: {epoch_loss / len(train_loader):.4f}, Val Loss: {val_loss.item():.4f}, Val R^2: {val_r2:.4f}')

    # Cargar el mejor modelo
    model.load_state_dict(best_model_state)

    return model


# Definir los hiperparámetros para el Grid Search
param_grid = {
    'learning_rate': [0.01, 0.001],
    'l1_lambda': [0.0, 0.001],
    'l2_lambda': [0.0, 0.001],
    'batch_size': [1024],
    'num_epochs': [200],
    'optimizer_type': ['Adam', 'SGD']
}
# Obtener todas las combinaciones de hiperparámetros
keys = param_grid.keys()
values = (param_grid[key] for key in keys)
param_combinations = [dict(zip(keys, combination)) for combination in product(*values)]

best_model = None
best_val_r2 = -np.inf
best_params = None


for params in param_combinations:
    print(f"Evaluating combination: {params}")

    # Crear nuevo modelo
    input_size = X_train_tensor.shape[1]
    model = Net(input_size)

    # Definir criterio y optimizador según los parámetros
    criterion = nn.MSELoss()

    if params['optimizer_type'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    elif params['optimizer_type'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'], momentum=0.9)

    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # Inicializar TensorBoard
    writer = SummaryWriter(
        comment=f"LR_{params['learning_rate']}_L1_{params['l1_lambda']}_L2_{params['l2_lambda']}_BS_{params['batch_size']}_OPT_{params['optimizer_type']}")

    # Entrenar el modelo
    trained_model = train_model(
        model, criterion, optimizer, scheduler,
        X_train_tensor, y_train_tensor,
        X_val_tensor, y_val_tensor,
        num_epochs=params['num_epochs'],
        l1_lambda=params['l1_lambda'],
        l2_lambda=params['l2_lambda'],
        batch_size=params['batch_size'],
        writer=writer
    )

    # Evaluar en el conjunto de validación
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_r2 = calculate_r2(y_val_tensor, val_outputs)

    print(f"Validation R^2: {val_r2:.4f}")

    # Cerrar el escritor de TensorBoard
    writer.close()

    # Guardar el mejor modelo
    if val_r2 > best_val_r2:
        best_val_r2 = val_r2
        best_model = trained_model
        best_params = params

print(f"Best Validation R^2: {best_val_r2:.4f} with parameters {best_params}")

# Evaluar el mejor modelo en el conjunto de prueba
best_model.eval()
with torch.no_grad():
    test_outputs = best_model(X_test_tensor)
    test_r2 = calculate_r2(y_test_tensor, test_outputs)
    test_loss = nn.MSELoss()(test_outputs, y_test_tensor)

print(f"Test Loss: {test_loss.item():.4f}, Test R^2: {test_r2:.4f}")
