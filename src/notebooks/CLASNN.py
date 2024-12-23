# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.utils.data as data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import numpy as np


import random


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

# %%

# Load the data
data = pd.read_csv('../data/PROCESS/encoded_tch_prediction_data_zafrav3.csv')

# %%
# Create a mask for ZAFRA 22-23
mask_22_23 = data['ZAFRA'] == '23-24'

# Split the data
X_train_zafra = data[~mask_22_23].drop('TCH', axis=1)
y_train_zafra = data[~mask_22_23]['TCH']
X_test_zafra = data[mask_22_23].drop('TCH', axis=1)
y_test_zafra = data[mask_22_23]['TCH']

X_train_zafra = X_train_zafra.drop(columns=['ABS_IDCOMP','ZAFRA','fecha'])
y_train_zafra = y_train_zafra.drop(columns=['ABS_IDCOMP','ZAFRA','fecha'])
X_test_zafra = X_test_zafra.drop(columns=['ABS_IDCOMP','ZAFRA','fecha'])
y_test_zafra = y_test_zafra.drop(columns=['ABS_IDCOMP','ZAFRA','fecha'])

print("\nZAFRA Split:")
print(f"Training set shape: {X_train_zafra.shape}")
print(f"Testing set shape: {X_test_zafra.shape}")

# %%

# Crear grupos de TCH
min_tch = data['TCH'].min()
max_tch = data['TCH'].max()
bins = np.arange(min_tch, max_tch + 2, 2)
labels = [f'{i}-{i+2}' for i in bins[:-1]]
data['TCH_grupo'] = pd.cut(data['TCH'], bins=bins, labels=labels, include_lowest=True)

# Imprimir los grupos
print("Grupos de TCH:")
print(data['TCH_grupo'].value_counts().sort_index())
print('TOTAL GRUPOS: ', data['TCH_grupo'].value_counts().count())
# Create a mask for ZAFRA 22-23
mask_22_23 = data['ZAFRA'] == '23-24'

# Split the data
X_train_zafra = data[~mask_22_23].drop(['TCH', 'TCH_grupo'], axis=1)
y_train_zafra = data[~mask_22_23]['TCH_grupo']
X_test_zafra = data[mask_22_23].drop(['TCH', 'TCH_grupo'], axis=1)
y_test_zafra = data[mask_22_23]['TCH_grupo']

X_train_zafra = X_train_zafra.drop(columns=['ABS_IDCOMP','ZAFRA'])
X_test_zafra = X_test_zafra.drop(columns=['ABS_IDCOMP','ZAFRA'])

print("\nZAFRA Split:")
print(f"Training set shape: {X_train_zafra.shape}")
print(f"Testing set shape: {X_test_zafra.shape}")

print(len(data['TCH_grupo'].unique()))

# %%
# Define the neural network
class ClasNet(nn.Module):
    def __init__(self, input_size,output_size):
        super(ClasNet, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, 360)
        # self.fc3 = nn.Linear(256, 256)
        # self.fc4 = nn.Linear(256, 360)
        self.fc5 = nn.Linear(360, 360)
        self.fc6 = nn.Linear(360, 512)
        self.fc7 = nn.Linear(512, 512)
        self.fc8 = nn.Linear(512, 512)
        self.fc9 = nn.Linear(512, 360)
        self.fc10 = nn.Linear(360, 360)
        self.fc11 = nn.Linear(360, 256)
        self.fc12 = nn.Linear(256, 128)
        self.fc13 = nn.Linear(128, 64)
        self.fc14 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.leak = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.sig(self.fc1(x))
        x = self.dropout(x)
        x = self.sig(self.fc2(x))
        x = self.dropout(x)
        # x = self.sig(self.fc3(x))
        # x = self.dropout(x)
        # x = self.sig(self.fc4(x))
        # x = self.dropout(x)
        x = self.sig(self.fc5(x)) 
        x = self.dropout(x)
        x = self.sig(self.fc6(x))
        x = self.dropout(x)
        x = self.leak(self.fc7(x))
        x = self.dropout(x)
        x = self.leak(self.fc8(x))
        x = self.dropout(x)
        x = self.relu(self.fc9(x))
        x = self.relu(self.fc10(x))     
        x = self.dropout(x)
        x = self.relu(self.fc11(x))
        x = self.relu(self.fc12(x))
        x = self.relu(self.fc13(x))
        x = self.fc14(x)
        return x
    
    def l1_loss(self):
        l1_loss = 0
        for param in self.parameters():
            l1_loss += torch.sum(torch.abs(param))
        return l1_loss

    def l2_loss(self):
        l2_loss = 0
        for param in self.parameters():
            l2_loss += torch.sum(param.pow(2))
        return l2_loss


# %%
from torch.utils.data import TensorDataset

X_train = X_train_zafra
y_train = y_train_zafra
X_test = X_test_zafra
y_test = y_test_zafra

# Inicializar el modelo, el criterio y el optimizador
input_size = X_train.shape[1]
output_size = len(data['TCH_grupo'].unique())
model = ClasNet(input_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Preprocesamiento
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convertir a tensores de PyTorch
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.LongTensor(y_train.cat.codes.values)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.LongTensor(y_test.cat.codes.values)

# Crear DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)



# %%

def train_model(model, criterion, optimizer, train_loader, test_loader, num_epochs=100):
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader)
        train_accuracy = train_correct / train_total

        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        test_loss /= len(test_loader)
        test_accuracy = test_correct / test_total

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')

    return train_losses, test_losses, train_accuracies, test_accuracies


# %%

# Entrenar el modelo
num_epochs = 300
train_losses, test_losses, train_accuracies, test_accuracies = train_model(model, criterion, optimizer, train_loader, test_loader, num_epochs=num_epochs)

# Guardar las pérdidas y precisiones en un archivo
losses_df = pd.DataFrame({
    'Epoch': range(1, num_epochs + 1),
    'Train Loss': train_losses,
    'Test Loss': test_losses,
    'Train Accuracy': train_accuracies,
    'Test Accuracy': test_accuracies
})
losses_df.to_csv('training_losses_accuracies.csv', index=False)

print('Training complete. Losses and accuracies saved to training_losses_accuracies.csv')


# %%

# Graficar las pérdidas a través de las épocas
plt.figure(figsize=(12, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over Epochs')
plt.show()

# Graficar las precisiones a través de las épocas
plt.figure(figsize=(12, 6))
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, num_epochs + 1), test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy over Epochs')
plt.show()

# Evaluar el modelo en el conjunto de prueba
model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.numpy())
        y_true.extend(labels.numpy())

# Convertir índices de vuelta a etiquetas originales
y_pred_labels = data['TCH_grupo'].cat.categories[y_pred]
y_true_labels = data['TCH_grupo'].cat.categories[y_true]

# # Imprimir el informe de clasificación
# print("\nClassification Report:")
# print(classification_report(y_true_labels, y_pred_labels))

# %%
torch.save(model.state_dict(), '6monthCLNN.pth')



