# -*- coding: utf-8 -*-
"""
Script para adicionar imagens de pastas externas ao dataset
Permite adicionar imagens criadas por IA e por humanos para complementar a base de dados
"""

import os
import sys
import shutil
import random
from pathlib import Path

# Adicionar diretório raiz ao path
ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(ROOT_DIR))

# Definir diretórios diretamente (sem importar config para evitar dependências)
DATA_DIR = ROOT_DIR / 'data'
TRAIN_DIR = DATA_DIR / 'train'
TEST_DIR = DATA_DIR / 'test'
TRAIN_SPLIT = 0.7  # Proporção padrão de dados para treinamento

# Definir seed para reprodutibilidade
random.seed(42)

# Nomes das classes esperadas
CLASS_AI = 'aiartdata'  # Arte gerada por IA
CLASS_HUMAN = 'realart'  # Arte criada por humanos

# Extensões de imagem suportadas
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.jfif']


def find_images_in_directory(directory):
    """
    Encontra todas as imagens em um diretório (incluindo subdiretórios)
    
    Args:
        directory: Caminho do diretório
        
    Returns:
        Lista de caminhos das imagens
    """
    directory = Path(directory)
    images = []
    
    if not directory.exists():
        return images
    
    # Procurar em todos os subdiretórios
    for root, dirs, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in IMAGE_EXTENSIONS:
                images.append(Path(root) / file)
    
    return images


def add_images_to_dataset(ai_folder=None, human_folder=None, train_split=TRAIN_SPLIT, 
                          class_ai=CLASS_AI, class_human=CLASS_HUMAN):
    """
    Adiciona imagens de pastas externas ao dataset
    
    Args:
        ai_folder: Caminho da pasta com imagens criadas por IA (None se não fornecido)
        human_folder: Caminho da pasta com imagens criadas por humanos (None se não fornecido)
        train_split: Proporção de imagens para treinamento (padrão: 0.7)
        class_ai: Nome da classe para imagens de IA (padrão: 'aiartdata')
        class_human: Nome da classe para imagens de humanos (padrão: 'realart')
    """
    print("="*80)
    print("ADICIONANDO IMAGENS AO DATASET")
    print("="*80)
    
    # Verificar se pelo menos uma pasta foi fornecida
    if not ai_folder and not human_folder:
        print("\n❌ ERRO: Pelo menos uma pasta deve ser fornecida!")
        print("\nUso:")
        print("  python scripts/add_images_to_dataset.py <pasta_IA> [pasta_humanos]")
        print("\nOu chamar a função diretamente:")
        print("  from scripts.add_images_to_dataset import add_images_to_dataset")
        print("  add_images_to_dataset(ai_folder='caminho/para/IA', human_folder='caminho/para/humanos')")
        return False
    
    # Criar diretórios se não existirem
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    
    train_class_ai_dir = TRAIN_DIR / class_ai
    test_class_ai_dir = TEST_DIR / class_ai
    train_class_human_dir = TRAIN_DIR / class_human
    test_class_human_dir = TEST_DIR / class_human
    
    train_class_ai_dir.mkdir(parents=True, exist_ok=True)
    test_class_ai_dir.mkdir(parents=True, exist_ok=True)
    train_class_human_dir.mkdir(parents=True, exist_ok=True)
    test_class_human_dir.mkdir(parents=True, exist_ok=True)
    
    total_added_train = 0
    total_added_test = 0
    
    # Processar pasta de IA
    if ai_folder:
        ai_folder = Path(ai_folder)
        if not ai_folder.exists():
            print(f"\n⚠️  AVISO: Pasta de IA não encontrada: {ai_folder}")
        else:
            print(f"\n📁 Processando pasta de IA: {ai_folder}")
            ai_images = find_images_in_directory(ai_folder)
            print(f"   Total de imagens encontradas: {len(ai_images)}")
            
            if ai_images:
                # Embaralhar e dividir
                random.shuffle(ai_images)
                split_idx = int(len(ai_images) * train_split)
                train_images = ai_images[:split_idx]
                test_images = ai_images[split_idx:]
                
                # Copiar para treinamento
                added_train = 0
                for img_path in train_images:
                    # Gerar nome único para evitar conflitos
                    dst_name = img_path.name
                    dst_path = train_class_ai_dir / dst_name
                    
                    # Se já existe, adicionar número
                    counter = 1
                    while dst_path.exists():
                        stem = img_path.stem
                        suffix = img_path.suffix
                        dst_name = f"{stem}_{counter}{suffix}"
                        dst_path = train_class_ai_dir / dst_name
                        counter += 1
                    
                    shutil.copy2(img_path, dst_path)
                    added_train += 1
                
                # Copiar para teste
                added_test = 0
                for img_path in test_images:
                    # Gerar nome único para evitar conflitos
                    dst_name = img_path.name
                    dst_path = test_class_ai_dir / dst_name
                    
                    # Se já existe, adicionar número
                    counter = 1
                    while dst_path.exists():
                        stem = img_path.stem
                        suffix = img_path.suffix
                        dst_name = f"{stem}_{counter}{suffix}"
                        dst_path = test_class_ai_dir / dst_name
                        counter += 1
                    
                    shutil.copy2(img_path, dst_path)
                    added_test += 1
                
                print(f"   ✅ Adicionadas {added_train} imagens para treinamento")
                print(f"   ✅ Adicionadas {added_test} imagens para teste")
                total_added_train += added_train
                total_added_test += added_test
            else:
                print(f"   ⚠️  Nenhuma imagem válida encontrada nesta pasta")
    
    # Processar pasta de humanos
    if human_folder:
        human_folder = Path(human_folder)
        if not human_folder.exists():
            print(f"\n⚠️  AVISO: Pasta de humanos não encontrada: {human_folder}")
        else:
            print(f"\n📁 Processando pasta de humanos: {human_folder}")
            human_images = find_images_in_directory(human_folder)
            print(f"   Total de imagens encontradas: {len(human_images)}")
            
            if human_images:
                # Embaralhar e dividir
                random.shuffle(human_images)
                split_idx = int(len(human_images) * train_split)
                train_images = human_images[:split_idx]
                test_images = human_images[split_idx:]
                
                # Copiar para treinamento
                added_train = 0
                for img_path in train_images:
                    # Gerar nome único para evitar conflitos
                    dst_name = img_path.name
                    dst_path = train_class_human_dir / dst_name
                    
                    # Se já existe, adicionar número
                    counter = 1
                    while dst_path.exists():
                        stem = img_path.stem
                        suffix = img_path.suffix
                        dst_name = f"{stem}_{counter}{suffix}"
                        dst_path = train_class_human_dir / dst_name
                        counter += 1
                    
                    shutil.copy2(img_path, dst_path)
                    added_train += 1
                
                # Copiar para teste
                added_test = 0
                for img_path in test_images:
                    # Gerar nome único para evitar conflitos
                    dst_name = img_path.name
                    dst_path = test_class_human_dir / dst_name
                    
                    # Se já existe, adicionar número
                    counter = 1
                    while dst_path.exists():
                        stem = img_path.stem
                        suffix = img_path.suffix
                        dst_name = f"{stem}_{counter}{suffix}"
                        dst_path = test_class_human_dir / dst_name
                        counter += 1
                    
                    shutil.copy2(img_path, dst_path)
                    added_test += 1
                
                print(f"   ✅ Adicionadas {added_train} imagens para treinamento")
                print(f"   ✅ Adicionadas {added_test} imagens para teste")
                total_added_train += added_train
                total_added_test += added_test
            else:
                print(f"   ⚠️  Nenhuma imagem válida encontrada nesta pasta")
    
    # Resumo final
    print("\n" + "="*80)
    print("RESUMO")
    print("="*80)
    print(f"Total de imagens adicionadas para treinamento: {total_added_train}")
    print(f"Total de imagens adicionadas para teste: {total_added_test}")
    print(f"Total geral: {total_added_train + total_added_test}")
    print(f"\nDiretórios:")
    print(f"  Treinamento: {TRAIN_DIR}")
    print(f"  Teste: {TEST_DIR}")
    print("="*80)
    
    return True


def main():
    """
    Função principal - permite usar o script via linha de comando
    """
    if len(sys.argv) < 2:
        print("\n❌ ERRO: Pelo menos uma pasta deve ser fornecida!")
        print("\nUso:")
        print("  python scripts/add_images_to_dataset.py <pasta_IA> [pasta_humanos]")
        print("\nExemplos:")
        print("  python scripts/add_images_to_dataset.py C:/Users/Usuario/imagens_IA")
        print("  python scripts/add_images_to_dataset.py C:/Users/Usuario/imagens_IA C:/Users/Usuario/imagens_humanos")
        sys.exit(1)
    
    ai_folder = sys.argv[1] if len(sys.argv) > 1 else None
    human_folder = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = add_images_to_dataset(ai_folder=ai_folder, human_folder=human_folder)
    
    if success:
        print("\n✅ Imagens adicionadas com sucesso!")
    else:
        print("\n❌ Erro ao adicionar imagens!")
        sys.exit(1)


if __name__ == "__main__":
    main()
