# -*- coding: utf-8 -*-
"""
Script para criar um subset pequeno do dataset para testes rápidos
Pega 10 imagens por classe para treinamento e teste
"""

import os
import sys
import shutil
import random
from pathlib import Path

# Adicionar diretório raiz ao path
ROOT_DIR = Path(__file__).parent.parent.absolute()

# Definir diretórios diretamente (sem importar config)
DATA_DIR = ROOT_DIR / 'data'
TRAIN_DIR = DATA_DIR / 'train'
TEST_DIR = DATA_DIR / 'test'

# Definir seed para reprodutibilidade
random.seed(42)

# Configurações
TRAIN_IMAGES_PER_CLASS = 10
TEST_IMAGES_PER_CLASS = 10

# Diretórios do subset
TRAIN_SUBSET_DIR = DATA_DIR / 'train_subset'
TEST_SUBSET_DIR = DATA_DIR / 'test_subset'


def create_subset():
    """
    Cria um subset pequeno do dataset para testes
    """
    print("="*60)
    print("CRIANDO SUBSET DO DATASET PARA TESTES")
    print("="*60)
    
    # Verificar se os diretórios originais existem
    if not TRAIN_DIR.exists():
        print(f"\nERRO: Diretório de treinamento não encontrado: {TRAIN_DIR}")
        print("Execute primeiro: python scripts/download_dataset.py")
        return False
    
    # Criar diretórios do subset
    TRAIN_SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    TEST_SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    
    # Limpar diretórios se já existirem
    if TRAIN_SUBSET_DIR.exists():
        print(f"\nLimpando diretório existente: {TRAIN_SUBSET_DIR}")
        shutil.rmtree(TRAIN_SUBSET_DIR)
    if TEST_SUBSET_DIR.exists():
        print(f"Limpando diretório existente: {TEST_SUBSET_DIR}")
        shutil.rmtree(TEST_SUBSET_DIR)
    
    TRAIN_SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    TEST_SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    
    # Encontrar classes
    train_classes = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir()])
    
    # Se houver apenas uma classe, dividir em duas para teste
    if len(train_classes) < 2:
        print(f"\nAVISO: Apenas {len(train_classes)} classe(s) encontrada(s)!")
        print("Dividindo as imagens em duas classes artificiais para teste...")
        print("(Isso é apenas para testar o código, não representa classificação real)")
        
        # Pegar a primeira (e única) classe
        original_class = train_classes[0]
        train_class_dir = TRAIN_DIR / original_class
        test_class_dir = TEST_DIR / original_class if (TEST_DIR / original_class).exists() else None
        
        # Coletar todas as imagens
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        all_train_images = [
            f for f in train_class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        
        all_test_images = []
        if test_class_dir and test_class_dir.exists():
            all_test_images = [
                f for f in test_class_dir.iterdir()
                if f.is_file() and f.suffix.lower() in image_extensions
            ]
        
        # Verificar se temos imagens suficientes
        total_need = (TRAIN_IMAGES_PER_CLASS * 2) + (TEST_IMAGES_PER_CLASS * 2)
        total_available = len(all_train_images) + len(all_test_images)
        
        if total_available < total_need:
            print(f"\nERRO: Não há imagens suficientes!")
            print(f"Necessário: {total_need} imagens (10 treino + 10 teste por classe)")
            print(f"Disponível: {total_available} imagens")
            print("\nSoluções:")
            print("1. Execute: python scripts/download_dataset.py para baixar o dataset completo")
            print("2. Ou adicione mais imagens ao diretório")
            return False
        
        # Criar duas classes artificiais: "classe_a" e "classe_b"
        train_classes = ['classe_a', 'classe_b']
        
        # Embaralhar e dividir
        random.shuffle(all_train_images)
        random.shuffle(all_test_images)
        
        # Dividir treinamento
        train_classe_a = all_train_images[:TRAIN_IMAGES_PER_CLASS]
        train_classe_b = all_train_images[TRAIN_IMAGES_PER_CLASS:TRAIN_IMAGES_PER_CLASS*2]
        
        # Dividir teste
        test_classe_a = all_test_images[:TEST_IMAGES_PER_CLASS] if len(all_test_images) >= TEST_IMAGES_PER_CLASS else []
        test_classe_b = all_test_images[TEST_IMAGES_PER_CLASS:TEST_IMAGES_PER_CLASS*2] if len(all_test_images) >= TEST_IMAGES_PER_CLASS*2 else []
        
        # Copiar para classes artificiais
        for class_name, images in [('classe_a', train_classe_a), ('classe_b', train_classe_b)]:
            train_subset_class_dir = TRAIN_SUBSET_DIR / class_name
            train_subset_class_dir.mkdir(parents=True, exist_ok=True)
            
            for img_path in images:
                dst_path = train_subset_class_dir / img_path.name
                shutil.copy2(img_path, dst_path)
            
            print(f"  {class_name} (treino): {len(images)} imagens")
        
        for class_name, images in [('classe_a', test_classe_a), ('classe_b', test_classe_b)]:
            if images:
                test_subset_class_dir = TEST_SUBSET_DIR / class_name
                test_subset_class_dir.mkdir(parents=True, exist_ok=True)
                
                for img_path in images:
                    dst_path = test_subset_class_dir / img_path.name
                    shutil.copy2(img_path, dst_path)
                
                print(f"  {class_name} (teste): {len(images)} imagens")
        
        total_train = len(train_classe_a) + len(train_classe_b)
        total_test = len(test_classe_a) + len(test_classe_b)
        
        print("\n" + "="*60)
        print("SUBSET CRIADO COM SUCESSO!")
        print("="*60)
        print("AVISO: Classes artificiais criadas apenas para teste do código")
        print("="*60)
        print(f"Total de imagens copiadas:")
        print(f"  Treinamento: {total_train} imagens ({len(train_classes)} classes)")
        print(f"  Teste: {total_test} imagens ({len(train_classes)} classes)")
        print(f"\nDiretórios criados:")
        print(f"  {TRAIN_SUBSET_DIR}")
        print(f"  {TEST_SUBSET_DIR}")
        print("\nPara usar o subset, execute:")
        print("  python main_subset.py")
        print("="*60)
        
        return True
    
    print(f"\nClasses encontradas: {train_classes}")
    print(f"Imagens por classe - Treino: {TRAIN_IMAGES_PER_CLASS}, Teste: {TEST_IMAGES_PER_CLASS}")
    
    total_train = 0
    total_test = 0
    
    # Processar cada classe
    for class_name in train_classes:
        print(f"\nProcessando classe: {class_name}")
        
        train_class_dir = TRAIN_DIR / class_name
        test_class_dir = TEST_DIR / class_name
        
        # Criar diretórios do subset
        train_subset_class_dir = TRAIN_SUBSET_DIR / class_name
        test_subset_class_dir = TEST_SUBSET_DIR / class_name
        train_subset_class_dir.mkdir(parents=True, exist_ok=True)
        test_subset_class_dir.mkdir(parents=True, exist_ok=True)
        
        # Coletar imagens de treinamento
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        train_images = [
            f for f in train_class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        
        test_images = []
        if test_class_dir.exists():
            test_images = [
                f for f in test_class_dir.iterdir()
                if f.is_file() and f.suffix.lower() in image_extensions
            ]
        
        # Embaralhar e pegar subset
        random.shuffle(train_images)
        random.shuffle(test_images)
        
        # Copiar imagens de treinamento
        train_subset = train_images[:TRAIN_IMAGES_PER_CLASS]
        for img_path in train_subset:
            dst_path = train_subset_class_dir / img_path.name
            shutil.copy2(img_path, dst_path)
        
        # Copiar imagens de teste
        test_subset = test_images[:TEST_IMAGES_PER_CLASS]
        for img_path in test_subset:
            dst_path = test_subset_class_dir / img_path.name
            shutil.copy2(img_path, dst_path)
        
        print(f"  Treino: {len(train_subset)}/{len(train_images)} imagens copiadas")
        print(f"  Teste: {len(test_subset)}/{len(test_images)} imagens copiadas")
        
        total_train += len(train_subset)
        total_test += len(test_subset)
        
        # Se não tiver imagens suficientes, avisar
        if len(train_subset) < TRAIN_IMAGES_PER_CLASS:
            print(f"  AVISO: Apenas {len(train_subset)} imagens disponíveis para treino")
        if len(test_subset) < TEST_IMAGES_PER_CLASS:
            print(f"  AVISO: Apenas {len(test_subset)} imagens disponíveis para teste")
    
    print("\n" + "="*60)
    print("SUBSET CRIADO COM SUCESSO!")
    print("="*60)
    print(f"Total de imagens copiadas:")
    print(f"  Treinamento: {total_train} imagens")
    print(f"  Teste: {total_test} imagens")
    print(f"\nDiretórios criados:")
    print(f"  {TRAIN_SUBSET_DIR}")
    print(f"  {TEST_SUBSET_DIR}")
    print("\nPara usar o subset, execute:")
    print("  python main_subset.py")
    print("\nOu modifique src/config.py temporariamente:")
    print(f"  TRAIN_DIR = ROOT_DIR / 'data' / 'train_subset'")
    print(f"  TEST_DIR = ROOT_DIR / 'data' / 'test_subset'")
    print("="*60)
    
    return True


def backup_and_switch_to_subset():
    """
    Faz backup dos diretórios originais e troca para o subset
    """
    print("\nDeseja fazer backup e trocar para o subset automaticamente?")
    resposta = input("Isso vai renomear os diretórios originais (s/n): ").strip().lower()
    
    if resposta != 's':
        print("Operação cancelada. Use os diretórios train_subset e test_subset manualmente.")
        return
    
    # Fazer backup
    if TRAIN_DIR.exists() and TRAIN_DIR.name == 'train':
        backup_train = DATA_DIR / 'train_backup'
        if backup_train.exists():
            shutil.rmtree(backup_train)
        TRAIN_DIR.rename(backup_train)
        print(f"Backup criado: {backup_train}")
    
    if TEST_DIR.exists() and TEST_DIR.name == 'test':
        backup_test = DATA_DIR / 'test_backup'
        if backup_test.exists():
            shutil.rmtree(backup_test)
        TEST_DIR.rename(backup_test)
        print(f"Backup criado: {backup_test}")
    
    # Renomear subset
    TRAIN_SUBSET_DIR.rename(TRAIN_DIR)
    TEST_SUBSET_DIR.rename(TEST_DIR)
    
    print("\nTroca realizada! Os diretórios originais foram renomeados para *_backup")
    print("Para restaurar, renomeie os diretórios de volta manualmente.")


if __name__ == "__main__":
    success = create_subset()
    
    if success:
        print("\nPara usar o subset, você tem duas opções:")
        print("1. Execute: python main_subset.py (usa subset sem modificar originais)")
        print("2. Ou modifique temporariamente src/config.py para usar os diretórios *_subset")
        print("\nSe quiser fazer backup e trocar automaticamente, descomente a linha abaixo")
        # backup_and_switch_to_subset()
