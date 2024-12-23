import torch
import torch.nn as nn


class ClassNet(nn.Module):
    def __init__(self, input_size, output_size, N=16):
        super(ClassNet, self).__init__()

        # Número total de capas lineales

        # Definir el tamaño máximo como un múltiplo del tamaño de entrada
        # Ajustar el multiplicador según sea necesario
        peak_size = input_size * 4

        # Inicializar la lista de tamaños de capas
        layer_sizes = []

        # Capas ascendentes
        for i in range(N // 2):
            size = input_size + \
                int((peak_size - input_size) * (i + 1) / (N // 2))
            layer_sizes.append(size)

        # Capas descendentes
        for i in range(N // 2):
            size = peak_size - \
                int((peak_size - output_size) * (i + 1) / (N // 2))
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
