# -*- coding: utf-8 -*-
"""
Configurações do Projeto - Classificação de Imagens
"""

import os

# Configuração de dispositivo (CPU ou GPU)
USE_GPU = True  # Altere para False para usar CPU

# Configurações da base de dados
DATA_DIR = 'data'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
VAL_DIR = os.path.join(DATA_DIR, 'val')

# Configurações do dataset Kaggle
KAGGLE_DATASET = "hassnainzaidi/ai-art-vs-human-art"
USE_KAGGLE_DATASET = True  # Se True, usa dataset do Kaggle
TRAIN_SPLIT = 0.7  # Proporção de dados para treinamento
TEST_SPLIT = 0.3   # Proporção de dados para teste

# Tamanho das imagens
IMG_SIZE = (224, 224)  # Tamanho padrão para modelos de deep learning
IMG_CHANNELS = 3  # RGB

# Configurações de treinamento
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Configurações de data augmentation
USE_AUGMENTATION = True
AUGMENTATION_PARAMS = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'horizontal_flip': True,
    'zoom_range': 0.2,
    'fill_mode': 'nearest'
}

# Configurações de otimização de hiperparâmetros
USE_HYPEROPT = True
HYPEROPT_TRIALS = 20  # Número de tentativas para otimização

# Diretórios de saída
OUTPUT_DIR = 'outputs'
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')

# Criar diretórios se não existirem
for dir_path in [OUTPUT_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR]:
    os.makedirs(dir_path, exist_ok=True)

