# -*- coding: utf-8 -*-
"""
Script de diagnóstico para verificar estrutura dos dados
"""

import os
from pathlib import Path

# Diretórios
DATA_DIR = Path('data')
TRAIN_DIR = DATA_DIR / 'train'
TEST_DIR = DATA_DIR / 'test'

print("="*60)
print("DIAGNÓSTICO DA ESTRUTURA DE DADOS")
print("="*60)

# Verificar se diretórios existem
print(f"\n1. Verificando diretórios:")
print(f"   DATA_DIR: {DATA_DIR} -> {DATA_DIR.exists()}")
print(f"   TRAIN_DIR: {TRAIN_DIR} -> {TRAIN_DIR.exists()}")
print(f"   TEST_DIR: {TEST_DIR} -> {TEST_DIR.exists()}")

if not TRAIN_DIR.exists():
    print(f"\n❌ ERRO: Diretório de treinamento não encontrado: {TRAIN_DIR}")
    print(f"   Execute: python scripts/download_dataset.py")
    exit(1)

# Verificar classes em treinamento
print(f"\n2. Verificando classes em {TRAIN_DIR}:")
train_classes = [d.name for d in TRAIN_DIR.iterdir() if d.is_dir()]
print(f"   Classes encontradas: {len(train_classes)}")
for class_name in train_classes:
    class_dir = TRAIN_DIR / class_name
    # Contar imagens
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    image_files = [f for f in class_dir.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    print(f"   - {class_name}: {len(image_files)} imagens")
    
    if len(image_files) == 0:
        print(f"      AVISO: Nenhuma imagem encontrada nesta classe!")

# Verificar classes em teste
print(f"\n3. Verificando classes em {TEST_DIR}:")
if TEST_DIR.exists():
    test_classes = [d.name for d in TEST_DIR.iterdir() if d.is_dir()]
    print(f"   Classes encontradas: {len(test_classes)}")
    for class_name in test_classes:
        class_dir = TEST_DIR / class_name
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        image_files = [f for f in class_dir.iterdir() 
                       if f.is_file() and f.suffix.lower() in image_extensions]
        print(f"   - {class_name}: {len(image_files)} imagens")
else:
    print(f"   ERRO: Diretório de teste não encontrado!")

# Resumo
print(f"\n4. RESUMO:")
if len(train_classes) < 2:
    print(f"   ERRO: Apenas {len(train_classes)} classe(s) encontrada(s)!")
    print(f"      São necessárias pelo menos 2 classes para classificação.")
    print(f"\n   Possíveis soluções:")
    print(f"   1. Verifique se o dataset foi baixado corretamente")
    print(f"   2. Execute: python scripts/download_dataset.py")
    print(f"   3. Verifique se há subpastas em {TRAIN_DIR} com nomes de classes")
    print(f"   4. Cada subpasta deve conter imagens válidas (JPG, PNG, JPEG)")
else:
    print(f"   ✅ {len(train_classes)} classes encontradas (OK)")
    
    # Verificar se todas as classes têm imagens
    classes_without_images = []
    for class_name in train_classes:
        class_dir = TRAIN_DIR / class_name
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        image_files = [f for f in class_dir.iterdir() 
                       if f.is_file() and f.suffix.lower() in image_extensions]
        if len(image_files) == 0:
            classes_without_images.append(class_name)
    
    if classes_without_images:
        print(f"   AVISO: {len(classes_without_images)} classe(s) sem imagens:")
        for class_name in classes_without_images:
            print(f"      - {class_name}")
        print(f"   Isso pode causar erro de 'apenas 1 classe' se todas as imagens")
        print(f"   dessa classe forem rejeitadas durante o carregamento.")
    else:
        print(f"   OK: Todas as classes têm imagens.")

print("\n" + "="*60)
