# -*- coding: utf-8 -*-
"""
Script para baixar e organizar dataset do Kaggle
Dataset: AI Art vs Human Art
"""

import os
import sys
import shutil
import random
from pathlib import Path

# Adicionar diretório raiz ao path
ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(ROOT_DIR))

import kagglehub
from src.config import DATA_DIR, TRAIN_DIR, TEST_DIR, TRAIN_SPLIT, TEST_SPLIT

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
    """
    print("\n" + "="*80)
    print("EXPLORANDO ESTRUTURA DO DATASET")
    print("="*80)

    structure = {
        'root': dataset_path,
        'directories': [],
        'files': []
    }

    for root, dirs, files in os.walk(dataset_path):
        level = root.replace(dataset_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")

        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:
            print(f"{subindent}{file}")
        if len(files) > 5:
            print(f"{subindent}... e mais {len(files) - 5} arquivos")

        structure['directories'].append(root)
        structure['files'].extend([os.path.join(root, f) for f in files])

    return structure


def find_class_directories(dataset_path):
    """
    Encontra diretórios que podem conter as classes
    """
    class_dirs = []
    possible_class_names = ['ai', 'human', 'ai-art', 'human-art', 'ai_art', 'human_art',
                           'AI', 'Human', 'AI-Art', 'Human-Art', 'aiartdata', 'realart',
                           'AiArtData', 'RealArt', 'aiart', 'real']
    
    # Primeiro, verificar estrutura comum: subdiretórios diretos
    dataset_path = Path(dataset_path)
    if dataset_path.exists():
        for item in dataset_path.iterdir():
            if item.is_dir():
                # Verificar se contém imagens diretamente ou em subpastas
                image_count = 0
                for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']:
                    image_count += len(list(item.rglob(f'*{ext}')))
                    image_count += len(list(item.rglob(f'*{ext.upper()}')))
                
                if image_count > 0:
                    # Verificar nome da classe
                    dir_name_lower = item.name.lower()
                    is_class_dir = any(
                        class_name.lower() in dir_name_lower or dir_name_lower in class_name.lower()
                        for class_name in possible_class_names
                    ) or len([d for d in dataset_path.iterdir() if d.is_dir()]) <= 5
                    
                    if is_class_dir and item not in class_dirs:
                        class_dirs.append(item)
    
    # Se não encontrou, procurar recursivamente
    if not class_dirs:
        for root, dirs, files in os.walk(dataset_path):
            image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
            if image_files and Path(root) != dataset_path:
                root_path = Path(root)
                # Evitar adicionar subpastas de classes já encontradas
                is_subfolder = False
                for cd in class_dirs:
                    if isinstance(cd, Path):
                        try:
                            is_subfolder = root_path.is_relative_to(cd)
                        except AttributeError:
                            # Python < 3.9 não tem is_relative_to
                            is_subfolder = str(root_path).startswith(str(cd))
                        if is_subfolder:
                            break
                
                if not is_subfolder and root_path not in class_dirs:
                    class_dirs.append(root_path)
    
    # Remover duplicatas e ordenar
    class_dirs = list(set(class_dirs))
    class_dirs = sorted([Path(cd) for cd in class_dirs])
    
    return class_dirs


def organize_dataset(dataset_path, train_split=TRAIN_SPLIT, test_split=TEST_SPLIT):
    """
    Organiza o dataset baixado na estrutura esperada pelo projeto
    """
    print("\n" + "="*80)
    print("ORGANIZANDO DATASET")
    print("="*80)

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    class_dirs = find_class_directories(dataset_path)

    if not class_dirs:
        print("AVISO: Não foi possível identificar automaticamente as classes.")
        print("Tentando estrutura alternativa...")

        for item in os.listdir(dataset_path):
            item_path = os.path.join(dataset_path, item)
            if os.path.isdir(item_path):
                image_files = [f for f in os.listdir(item_path)
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                if image_files:
                    class_dirs.append(item_path)

    if not class_dirs:
        raise ValueError("Não foi possível encontrar classes no dataset.")
    
    if len(class_dirs) < 2:
        print(f"\nAVISO: Apenas {len(class_dirs)} classe(s) encontrada(s)!")
        print("Estrutura do dataset:")
        for item in os.listdir(dataset_path):
            item_path = os.path.join(dataset_path, item)
            if os.path.isdir(item_path):
                image_count = sum(1 for f in os.listdir(item_path) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')))
                print(f"  {item}: {image_count} imagens")
        
        print("\nTentando estrutura alternativa...")
        # Tentar estrutura plana: todas as imagens em um diretório
        # Ou estrutura com subpastas diferentes
        
        # Verificar se há subpastas que podem ser classes
        all_subdirs = []
        for root, dirs, files in os.walk(dataset_path):
            for d in dirs:
                subdir = Path(root) / d
                image_files = [f for f in subdir.iterdir() 
                              if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']]
                if image_files and subdir not in class_dirs:
                    all_subdirs.append(subdir)
        
        if len(all_subdirs) >= 2:
            print(f"Encontradas {len(all_subdirs)} subpastas com imagens, usando-as como classes.")
            class_dirs = all_subdirs[:2]  # Usar as primeiras 2
        else:
            raise ValueError(
                f"Apenas {len(class_dirs)} classe(s) encontrada(s). "
                f"O dataset deve ter pelo menos 2 classes para classificação. "
                f"Verifique a estrutura do dataset baixado."
            )

    print(f"\nClasses encontradas: {len(class_dirs)}")
    for i, cd in enumerate(class_dirs):
        print(f"  {i+1}. {cd.name}")

    for class_dir in class_dirs:
        class_name = os.path.basename(class_dir)
        class_name_clean = class_name.replace(' ', '_').replace('-', '_').lower()

        train_class_dir = TRAIN_DIR / class_name_clean
        test_class_dir = TEST_DIR / class_name_clean
        train_class_dir.mkdir(parents=True, exist_ok=True)
        test_class_dir.mkdir(parents=True, exist_ok=True)

        image_files = [f for f in os.listdir(class_dir)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

        if not image_files:
            for root, dirs, files in os.walk(class_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        image_files.append(os.path.join(root, file))

        print(f"\nClasse: {class_name_clean}")
        print(f"  Total de imagens: {len(image_files)}")

        random.shuffle(image_files)
        split_idx = int(len(image_files) * train_split)
        train_files = image_files[:split_idx]
        test_files = image_files[split_idx:]

        print(f"  Treinamento: {len(train_files)} imagens")
        print(f"  Teste: {len(test_files)} imagens")

        for img_file in train_files:
            if isinstance(img_file, str):
                src_path = os.path.join(class_dir, img_file) if not os.path.isabs(img_file) else img_file
            else:
                src_path = img_file

            if os.path.exists(src_path):
                dst_path = train_class_dir / os.path.basename(src_path)
                shutil.copy2(src_path, dst_path)

        for img_file in test_files:
            if isinstance(img_file, str):
                src_path = os.path.join(class_dir, img_file) if not os.path.isabs(img_file) else img_file
            else:
                src_path = img_file

            if os.path.exists(src_path):
                dst_path = test_class_dir / os.path.basename(src_path)
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

    if TRAIN_DIR.exists() and any(TRAIN_DIR.iterdir()):
        print(f"AVISO: Diretório de treinamento já existe: {TRAIN_DIR}")
        resposta = input("Deseja baixar e reorganizar o dataset? (s/n): ").strip().lower()
        if resposta != 's':
            print("Operação cancelada.")
            return

    try:
        dataset_path = download_kaggle_dataset(dataset_name)
        structure = explore_dataset_structure(dataset_path)
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
