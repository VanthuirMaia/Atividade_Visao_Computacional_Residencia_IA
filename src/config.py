# -*- coding: utf-8 -*-
"""
Configurações do Projeto - Classificação de Imagens
"""

import os
from pathlib import Path

# Diretório raiz do projeto
ROOT_DIR = Path(__file__).parent.parent.absolute()

# Configuração de dispositivo (CPU ou GPU)
USE_GPU = True  # Altere para False para usar CPU

# Configurações da base de dados
DATA_DIR = ROOT_DIR / 'data'
TRAIN_DIR = DATA_DIR / 'train'
TEST_DIR = DATA_DIR / 'test'
VAL_DIR = DATA_DIR / 'val'  # Opcional: diretório de validação separado (não usado por padrão)

# Configurações do dataset Kaggle
KAGGLE_DATASET = "hassnainzaidi/ai-art-vs-human-art"
USE_KAGGLE_DATASET = True  # Se True, usa dataset do Kaggle
TRAIN_SPLIT = 0.7  # Proporção de dados para treinamento
TEST_SPLIT = 0.3   # Proporção de dados para teste

# Tamanho das imagens
IMG_SIZE = (224, 224)  # Tamanho padrão para modelos de deep learning
IMG_SIZE_CLASSIC = (64, 64)  # Tamanho menor para modelos clássicos (economiza memória)
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
# Nota: O projeto usa RandomizedSearchCV do scikit-learn, não hyperopt
# USE_HYPEROPT = True  # Não implementado - removido
# HYPEROPT_TRIALS = 20  # Não implementado - removido

# ============================================
# CONFIGURAÇÕES DE GERENCIAMENTO DE MEMÓRIA
# ============================================

# Usar lazy loading (carrega imagens sob demanda)
USE_LAZY_LOADING = True

# Tamanho do cache de imagens (0 = sem cache)
IMAGE_CACHE_SIZE = 100

# Batch size mínimo (para adaptive batch size)
MIN_BATCH_SIZE = 4

# Limites de memória para alertas
MEMORY_WARNING_THRESHOLD = 0.8   # 80% de uso
MEMORY_CRITICAL_THRESHOLD = 0.9  # 90% de uso

# Fator de segurança para cálculo de batch size
MEMORY_SAFETY_FACTOR = 0.7

# Limpar memória a cada N batches durante treinamento
CLEAR_MEMORY_EVERY_N_BATCHES = 50

# Usar mixed precision (float16) para economizar memória GPU
# TODO: Implementar mixed precision training no pipeline deep_learning.py
USE_MIXED_PRECISION = False  # Não implementado ainda

# ============================================
# CONFIGURAÇÕES ESPECÍFICAS PARA RESNET50
# ============================================

# Batch sizes para Random Search do ResNet50 (reduzidos para evitar estouro de memória)
# ResNet50 é um modelo muito grande (25M+ parâmetros) e precisa de batch sizes menores
RESNET50_BATCH_SIZES = [8, 16, 32]  # Reduzido de [16, 32, 64] para economizar memória

# Batch size padrão para ResNet50 (se não usar Random Search)
RESNET50_DEFAULT_BATCH_SIZE = 16  # Reduzido de 32

# Épocas para Random Search do ResNet50 (reduzidas para economizar memória)
RESNET50_SEARCH_EPOCHS = 10  # Número máximo de épocas durante Random Search

# Limpar memória entre iterações do Random Search do ResNet50
RESNET50_CLEAR_MEMORY_BETWEEN_ITERATIONS = True  # IMPORTANTE: Limpar entre iterações

# ============================================
# CONFIGURAÇÕES ESPECÍFICAS PARA MODELOS CLÁSSICOS
# ============================================

# Configurações de memória para modelos clássicos (SVM, Random Forest)
CLASSIC_USE_PCA = True  # Usar PCA para redução de dimensionalidade antes do SVM
CLASSIC_PCA_COMPONENTS = 500  # Número de componentes PCA (None = auto, reduz para 95% variância)
CLASSIC_USE_LINEAR_SVM = False  # Usar LinearSVC ao invés de SVC (mais eficiente em memória, mas apenas kernel linear)
CLASSIC_MAX_SAMPLES = None  # Limitar número de amostras para treinamento (None = usar todas)
CLASSIC_SVM_N_JOBS = 1  # Jobs paralelos para SVM (1 = sem paralelização para economizar memória)
CLASSIC_RF_N_JOBS = -1  # Jobs paralelos para Random Forest (-1 = todos os cores, Random Forest usa memória de forma mais eficiente)
CLASSIC_CV_FOLDS = 2  # Número de folds para validação cruzada (2 ao invés de 3 para economizar memória) - aplica-se a TODOS os modelos clássicos

# Diretórios de saída
OUTPUT_DIR = ROOT_DIR / 'outputs'
MODELS_DIR = OUTPUT_DIR / 'models'
RESULTS_DIR = OUTPUT_DIR / 'results'
FIGURES_DIR = OUTPUT_DIR / 'figures'

# Criar diretórios se não existirem
for dir_path in [DATA_DIR, OUTPUT_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
