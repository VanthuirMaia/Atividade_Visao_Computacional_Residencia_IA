# -*- coding: utf-8 -*-
"""
Projeto de Classificação de Imagens - Visão Computacional

Ponto de entrada principal do projeto.
Permite executar os pipelines clássico e de deep learning.
"""

import sys
from pathlib import Path

# Adicionar diretório raiz ao path
ROOT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(ROOT_DIR))

from src.config import TRAIN_DIR, TEST_DIR, DATA_DIR
import src.config


def check_data_structure(train_dir=None, test_dir=None):
    """
    Verifica se a estrutura de dados existe
    
    Args:
        train_dir: Diretório de treinamento (usa config se None)
        test_dir: Diretório de teste (usa config se None)
    """
    train_dir = train_dir or src.config.TRAIN_DIR
    test_dir = test_dir or src.config.TEST_DIR
    
    if not train_dir.exists() or not any(train_dir.iterdir()):
        return False
    if not test_dir.exists() or not any(test_dir.iterdir()):
        return False
    return True


def count_images(train_dir=None, test_dir=None):
    """
    Conta imagens em cada diretório
    
    Args:
        train_dir: Diretório de treinamento (usa config se None)
        test_dir: Diretório de teste (usa config se None)
    """
    train_dir = train_dir or src.config.TRAIN_DIR
    test_dir = test_dir or src.config.TEST_DIR
    
    stats = {'train': {}, 'test': {}}

    if train_dir.exists():
        for class_dir in train_dir.iterdir():
            if class_dir.is_dir():
                count = len(list(class_dir.glob('*.[jJ][pP][gG]')) +
                           list(class_dir.glob('*.[jJ][pP][eE][gG]')) +
                           list(class_dir.glob('*.[pP][nN][gG]')))
                stats['train'][class_dir.name] = count

    if test_dir.exists():
        for class_dir in test_dir.iterdir():
            if class_dir.is_dir():
                count = len(list(class_dir.glob('*.[jJ][pP][gG]')) +
                           list(class_dir.glob('*.[jJ][pP][eE][gG]')) +
                           list(class_dir.glob('*.[pP][nN][gG]')))
                stats['test'][class_dir.name] = count

    return stats


def print_data_info(train_dir=None, test_dir=None):
    """
    Imprime informações sobre os dados
    
    Args:
        train_dir: Diretório de treinamento (usa config se None)
        test_dir: Diretório de teste (usa config se None)
    """
    stats = count_images(train_dir, test_dir)

    print("\n" + "="*60)
    print("INFORMAÇÕES DO DATASET")
    print("="*60)

    if stats['train']:
        print("\nDados de Treinamento:")
        total_train = 0
        for class_name, count in stats['train'].items():
            print(f"  {class_name}: {count} imagens")
            total_train += count
        print(f"  Total: {total_train} imagens")

    if stats['test']:
        print("\nDados de Teste:")
        total_test = 0
        for class_name, count in stats['test'].items():
            print(f"  {class_name}: {count} imagens")
            total_test += count
        print(f"  Total: {total_test} imagens")

    print("="*60)


def run_classic_pipeline(train_dir=None, test_dir=None):
    """
    Executa o pipeline clássico
    
    Args:
        train_dir: Diretório de treinamento (usa config se None)
        test_dir: Diretório de teste (usa config se None)
    """
    print("\n" + "="*60)
    print("EXECUTANDO PIPELINE CLÁSSICO")
    print("="*60)

    from src.pipelines.classic import ClassicPipeline

    train_dir = train_dir or src.config.TRAIN_DIR
    test_dir = test_dir or src.config.TEST_DIR
    
    pipeline = ClassicPipeline(train_dir, test_dir)
    pipeline.load_data()
    pipeline.train_svm(use_random_search=True, n_iter=50)
    pipeline.train_random_forest(use_random_search=True, n_iter=50)
    pipeline.save_results()

    print("\nPipeline clássico concluído!")


def run_deep_learning_pipeline(train_dir=None, test_dir=None):
    """
    Executa o pipeline de deep learning
    
    Args:
        train_dir: Diretório de treinamento (usa config se None)
        test_dir: Diretório de teste (usa config se None)
    """
    print("\n" + "="*60)
    print("EXECUTANDO PIPELINE DEEP LEARNING")
    print("="*60)

    from src.config import USE_GPU, EPOCHS
    from src.pipelines.deep_learning import DeepLearningPipeline

    train_dir = train_dir or src.config.TRAIN_DIR
    test_dir = test_dir or src.config.TEST_DIR

    pipeline = DeepLearningPipeline(train_dir, test_dir, use_gpu=USE_GPU)
    pipeline.load_data()
    pipeline.train_simple_cnn(use_random_search=True, n_iter=10, final_epochs=EPOCHS)
    pipeline.train_resnet_transfer(use_random_search=True, n_iter=10, final_epochs=EPOCHS)
    pipeline.save_results()

    print("\nPipeline de deep learning concluído!")


def download_dataset():
    """
    Baixa o dataset do Kaggle
    """
    print("\nBaixando dataset do Kaggle...")
    from scripts.download_dataset import main as download_main
    download_main()


def main():
    """
    Função principal
    """
    print("="*60)
    print("PROJETO DE CLASSIFICAÇÃO DE IMAGENS")
    print("Visão Computacional")
    print("="*60)

    # Variáveis para controlar qual dataset usar
    current_train_dir = src.config.TRAIN_DIR
    current_test_dir = src.config.TEST_DIR
    using_subset = False
    
    # Verificar estrutura de dados
    if not check_data_structure(current_train_dir, current_test_dir):
        print("\nAVISO: Dados não encontrados!")
        print(f"Diretório esperado: {DATA_DIR}")
        print("\nOpções:")
        print("1. Baixar dataset do Kaggle automaticamente")
        print("2. Usar subset pequeno para testes (se existir)")
        print("3. Sair e organizar dados manualmente")

        opcao = input("\nEscolha uma opção (1-3): ").strip()

        if opcao == '1':
            download_dataset()
            if not check_data_structure(current_train_dir, current_test_dir):
                print("\nERRO: Não foi possível organizar os dados.")
                return
        elif opcao == '2':
            # Tentar usar subset
            TRAIN_SUBSET = DATA_DIR / 'train_subset'
            TEST_SUBSET = DATA_DIR / 'test_subset'
            if TRAIN_SUBSET.exists() and TEST_SUBSET.exists():
                print(f"\nUsando subset encontrado em:")
                print(f"  {TRAIN_SUBSET}")
                print(f"  {TEST_SUBSET}")
                print("\nAVISO: Este é um subset pequeno para testes!")
                print("Os resultados não serão representativos.\n")
                current_train_dir = TRAIN_SUBSET
                current_test_dir = TEST_SUBSET
                using_subset = True
            else:
                print("\nERRO: Subset não encontrado!")
                print("Execute primeiro: python scripts/create_subset.py")
                return
        else:
            print("\nOrganize seus dados na seguinte estrutura:")
            print(f"  {src.config.TRAIN_DIR}/classe1/")
            print(f"  {src.config.TRAIN_DIR}/classe2/")
            print(f"  {src.config.TEST_DIR}/classe1/")
            print(f"  {src.config.TEST_DIR}/classe2/")
            return
    
    # Verificar se há pelo menos 2 classes nos diretórios
    train_dirs = [d for d in current_train_dir.iterdir() if d.is_dir()] if current_train_dir.exists() else []
    
    if len(train_dirs) < 2 and not using_subset:
        print(f"\n{'='*60}")
        print("AVISO: Apenas 1 classe encontrada no dataset principal!")
        print("="*60)
        
        # Verificar se existe subset
        TRAIN_SUBSET = DATA_DIR / 'train_subset'
        TEST_SUBSET = DATA_DIR / 'test_subset'
        
        if TRAIN_SUBSET.exists() and TEST_SUBSET.exists():
            print("\nOpções:")
            print("1. Usar subset pequeno para testes (recomendado para validar o código)")
            print("2. Baixar/reorganizar dataset completo do Kaggle")
            print("3. Sair")
            
            opcao = input("\nEscolha uma opção (1-3): ").strip()
            
            if opcao == '1':
                print(f"\nTrocando para subset: {TRAIN_SUBSET}")
                current_train_dir = TRAIN_SUBSET
                current_test_dir = TEST_SUBSET
                using_subset = True
                print("AVISO: Usando subset pequeno - resultados não representativos!\n")
                # Continuar execução
            elif opcao == '2':
                download_dataset()
                # Recarregar diretórios após download
                current_train_dir = src.config.TRAIN_DIR
                current_test_dir = src.config.TEST_DIR
                if not check_data_structure(current_train_dir, current_test_dir):
                    print("\nERRO: Não foi possível organizar os dados.")
                    return
                # Verificar novamente quantas classes há
                train_dirs = [d for d in current_train_dir.iterdir() if d.is_dir()] if current_train_dir.exists() else []
                if len(train_dirs) < 2:
                    print("\nERRO: Dataset ainda não tem 2 classes após download.")
                    print("Verifique a estrutura do dataset baixado.")
                    print(f"Classes encontradas: {[d.name for d in train_dirs]}")
                    return
            else:
                return
        else:
            print("\nSoluções:")
            print("1. Execute: python scripts/create_subset.py (cria subset para testes)")
            print("2. Execute: python scripts/download_dataset.py (baixa dataset completo)")
            print("3. Organize manualmente 2 classes em data/train/ e data/test/")
            return

    # Mostrar informações dos dados
    print_data_info(current_train_dir, current_test_dir)

    # Menu principal
    print("\nEscolha o pipeline a ser executado:")
    print("1. Pipeline Clássico (SVM + Random Forest)")
    print("2. Pipeline Deep Learning (CNN + ResNet)")
    print("3. Ambos os pipelines")
    print("4. Sair")

    opcao = input("\nEscolha uma opção (1-4): ").strip()

    if opcao == '1':
        run_classic_pipeline(current_train_dir, current_test_dir)
    elif opcao == '2':
        run_deep_learning_pipeline(current_train_dir, current_test_dir)
    elif opcao == '3':
        run_classic_pipeline(current_train_dir, current_test_dir)
        run_deep_learning_pipeline(current_train_dir, current_test_dir)
    elif opcao == '4':
        print("\nSaindo...")
    else:
        print("\nOpção inválida!")

    print("\n" + "="*60)
    print("PROJETO CONCLUÍDO!")
    print("="*60)
    print("Resultados salvos em:")
    print("  - outputs/results/")
    print("  - outputs/models/")
    print("  - outputs/figures/")


if __name__ == "__main__":
    main()
