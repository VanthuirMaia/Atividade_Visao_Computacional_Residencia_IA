# -*- coding: utf-8 -*-
"""
Definições de modelos CNN
"""

import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    CNN simples sem transfer learning

    Arquitetura:
    - 3 camadas convolucionais (32, 64, 128 filtros)
    - MaxPooling após cada convolução
    - Adaptive pooling para garantir tamanho fixo
    - 2 camadas fully connected
    - Dropout para regularização

    Args:
        num_classes: Número de classes para classificação
        dropout_rate: Taxa de dropout (padrão: 0.5)
        hidden_units: Unidades na camada oculta (padrão: 512)
    """

    def __init__(self, num_classes, dropout_rate=0.5, hidden_units=512):
        super(SimpleCNN, self).__init__()

        # Camadas convolucionais
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        # Regularização
        self.dropout = nn.Dropout(dropout_rate)

        # Camadas fully connected
        self.fc1 = nn.Linear(128 * 7 * 7, hidden_units)
        self.fc2 = nn.Linear(hidden_units, num_classes)

        # Ativação
        self.relu = nn.ReLU()

    def forward(self, x):
        # Bloco 1
        x = self.pool(self.relu(self.conv1(x)))

        # Bloco 2
        x = self.pool(self.relu(self.conv2(x)))

        # Bloco 3
        x = self.pool(self.relu(self.conv3(x)))

        # Adaptive pooling para garantir tamanho fixo
        x = self.adaptive_pool(x)

        # Flatten
        x = x.view(-1, 128 * 7 * 7)

        # Fully connected com dropout
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x
