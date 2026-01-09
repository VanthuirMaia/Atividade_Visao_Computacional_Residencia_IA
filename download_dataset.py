# -*- coding: utf-8 -*-
"""
Script para baixar e organizar dataset do Kaggle
Dataset: AI Art vs Human Art
"""

import os
import shutil
import kagglehub
from pathlib import Path
from config import DATA_DIR, TRAIN_DIR, TEST_DIR
import random

# Definir seed para reprodutibilidade
random.seed(42)

def download_kaggle_dataset(dataset_name="hassnainzaidi/ai-art-vs-human-art"):
    """
    Baixa o dataset do Kaggle usando kagglehub
    
    Args:
        dataset_name: Nome do dataset no formato "usuario/dataset"
    
    Returns:
        path: Caminho onde o dataset foi baixado
    """
    print("="*80)
    print("BAIXANDO DATASET DO KAGGLE")
    print("="*80)
    print(f"Dataset: {dataset_name}")
    
    try:
        # Baixar a versão mais recente do dataset
        print("\nIniciando download...")
        path = kagglehub.dataset_download(dataset_name)
        print(f"Dataset baixado com sucesso!")
        print(f"Caminho: {path}")
        return path
    except Exception as e:
        print(f"ERRO ao baixar dataset: {e}")
        print("\nCertifique-se de que:")
        print("1. Você tem uma conta no Kaggle")
        print("2. Suas credenciais do Kaggle estão configuradas")
        print("3. Você aceitou os termos de uso do dataset no Kaggle")
        raise

def explore_dataset_structure(dataset_path):
    """
    Explora a estrutura do dataset baixado
    
    Args:
        dataset_path: Caminho do dataset baixado
    
    Returns:
        structure: Dicionário com informações da estrutura
    """
    print("\n" + "="*80)
    print("EXPLORANDO ESTRUTURA DO DATASET")
    print("="*80)
    
    structure = {
        'root': dataset_path,
        'directories': [],
        'files': []
    }
    
    # Listar estrutura
    for root, dirs, files in os.walk(dataset_path):
        level = root.replace(dataset_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # Mostrar apenas 5 primeiros arquivos
            print(f"{subindent}{file}")
        if len(files) > 5:
            print(f"{subindent}... e mais {len(files) - 5} arquivos")
        
        structure['directories'].append(root)
        structure['files'].extend([os.path.join(root, f) for f in files])
    
    return structure

def find_class_directories(dataset_path):
    """
    Encontra diretórios que podem conter as classes
    
    Args:
        dataset_path: Caminho do dataset
    
    Returns:
        class_dirs: Lista de diretórios de classes encontrados
    """
    class_dirs = []
    possible_class_names = ['ai', 'human', 'ai-art', 'human-art', 'ai_art', 'human_art', 
                           'AI', 'Human', 'AI-Art', 'Human-Art']
    
    # Procurar por diretórios que possam ser classes
    for root, dirs, files in os.walk(dataset_path):
        # Verificar se há imagens neste diretório
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        if image_files:
            # Verificar se o nome do diretório parece ser uma classe
            dir_name = os.path.basename(root).lower()
            for class_name in possible_class_names:
                if class_name.lower() in dir_name:
                    class_dirs.append(root)
                    break
    
    # Se não encontrou, procurar por padrões comuns
    if not class_dirs:
        # Listar todos os diretórios que contêm imagens
        for root, dirs, files in os.walk(dataset_path):
            image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            if image_files and root != dataset_path:
                class_dirs.append(root)
    
    return class_dirs

def organize_dataset(dataset_path, train_split=None, test_split=None):
    """
    Organiza o dataset baixado na estrutura esperada pelo projeto
    
    Args:
        dataset_path: Caminho do dataset baixado
        train_split: Proporção de dados para treinamento (padrão: usa config.py)
        test_split: Proporção de dados para teste (padrão: usa config.py)
    """
    # Usar valores do config se não fornecidos
    if train_split is None or test_split is None:
        from config import TRAIN_SPLIT, TEST_SPLIT
        if train_split is None:
            train_split = TRAIN_SPLIT
        if test_split is None:
            test_split = TEST_SPLIT
    print("\n" + "="*80)
    print("ORGANIZANDO DATASET")
    print("="*80)
    
    # Criar diretórios de destino
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    
    # Encontrar diretórios de classes
    class_dirs = find_class_directories(dataset_path)
    
    if not class_dirs:
        print("AVISO: Não foi possível identificar automaticamente as classes.")
        print("Tentando estrutura alternativa...")
        
        # Tentar estrutura alternativa: procurar por subdiretórios diretos
        for item in os.listdir(dataset_path):
            item_path = os.path.join(dataset_path, item)
            if os.path.isdir(item_path):
                # Verificar se contém imagens
                image_files = [f for f in os.listdir(item_path) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                if image_files:
                    class_dirs.append(item_path)
    
    if not class_dirs:
        raise ValueError("Não foi possível encontrar classes no dataset. Verifique a estrutura manualmente.")
    
    print(f"\nClasses encontradas: {len(class_dirs)}")
    
    # Processar cada classe
    for class_dir in class_dirs:
        class_name = os.path.basename(class_dir)
        
        # Limpar nome da classe (remover caracteres especiais)
        class_name_clean = class_name.replace(' ', '_').replace('-', '_').lower()
        
        # Criar diretórios de destino
        train_class_dir = os.path.join(TRAIN_DIR, class_name_clean)
        test_class_dir = os.path.join(TEST_DIR, class_name_clean)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)
        
        # Coletar todas as imagens
        image_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        if not image_files:
            # Se não há imagens diretas, procurar em subdiretórios
            for root, dirs, files in os.walk(class_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        image_files.append(os.path.join(root, file))
        
        print(f"\nClasse: {class_name_clean}")
        print(f"  Total de imagens: {len(image_files)}")
        
        # Embaralhar e dividir
        random.shuffle(image_files)
        split_idx = int(len(image_files) * train_split)
        train_files = image_files[:split_idx]
        test_files = image_files[split_idx:]
        
        print(f"  Treinamento: {len(train_files)} imagens")
        print(f"  Teste: {len(test_files)} imagens")
        
        # Copiar arquivos de treinamento
        for img_file in train_files:
            if isinstance(img_file, str):
                src_path = os.path.join(class_dir, img_file) if not os.path.isabs(img_file) else img_file
            else:
                src_path = img_file
            
            if os.path.exists(src_path):
                dst_path = os.path.join(train_class_dir, os.path.basename(src_path))
                shutil.copy2(src_path, dst_path)
        
        # Copiar arquivos de teste
        for img_file in test_files:
            if isinstance(img_file, str):
                src_path = os.path.join(class_dir, img_file) if not os.path.isabs(img_file) else img_file
            else:
                src_path = img_file
            
            if os.path.exists(src_path):
                dst_path = os.path.join(test_class_dir, os.path.basename(src_path))
                shutil.copy2(src_path, dst_path)
    
    print("\n" + "="*80)
    print("DATASET ORGANIZADO COM SUCESSO!")
    print("="*80)
    print(f"Treinamento: {TRAIN_DIR}")
    print(f"Teste: {TEST_DIR}")

def main():
    """
    Função principal para baixar e organizar o dataset
    """
    dataset_name = "hassnainzaidi/ai-art-vs-human-art"
    
    # Verificar se os dados já existem
    if os.path.exists(TRAIN_DIR) and os.listdir(TRAIN_DIR):
        print(f"AVISO: Diretório de treinamento já existe: {TRAIN_DIR}")
        resposta = input("Deseja baixar e reorganizar o dataset? (s/n): ").strip().lower()
        if resposta != 's':
            print("Operação cancelada.")
            return
    
    try:
        # Baixar dataset
        dataset_path = download_kaggle_dataset(dataset_name)
        
        # Explorar estrutura
        structure = explore_dataset_structure(dataset_path)
        
        # Organizar dataset (usa valores do config.py)
        organize_dataset(dataset_path)
        
        print("\nPróximos passos:")
        print("1. Execute: python main.py")
        print("2. Escolha o pipeline desejado")
        
    except Exception as e:
        print(f"\nERRO: {e}")
        print("\nSolução alternativa:")
        print("1. Baixe o dataset manualmente do Kaggle")
        print("2. Organize os dados em:")
        print(f"   {TRAIN_DIR}/classe1/")
        print(f"   {TRAIN_DIR}/classe2/")
        print(f"   {TEST_DIR}/classe1/")
        print(f"   {TEST_DIR}/classe2/")

if __name__ == "__main__":
    main()
