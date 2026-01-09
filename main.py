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


def check_data_structure():
    """
    Verifica se a estrutura de dados existe
    """
    if not TRAIN_DIR.exists() or not any(TRAIN_DIR.iterdir()):
        return False
    if not TEST_DIR.exists() or not any(TEST_DIR.iterdir()):
        return False
    return True


def count_images():
    """
    Conta imagens em cada diretório
    """
    stats = {'train': {}, 'test': {}}

    if TRAIN_DIR.exists():
        for class_dir in TRAIN_DIR.iterdir():
            if class_dir.is_dir():
                count = len(list(class_dir.glob('*.[jJ][pP][gG]')) +
                           list(class_dir.glob('*.[jJ][pP][eE][gG]')) +
                           list(class_dir.glob('*.[pP][nN][gG]')))
                stats['train'][class_dir.name] = count

    if TEST_DIR.exists():
        for class_dir in TEST_DIR.iterdir():
            if class_dir.is_dir():
                count = len(list(class_dir.glob('*.[jJ][pP][gG]')) +
                           list(class_dir.glob('*.[jJ][pP][eE][gG]')) +
                           list(class_dir.glob('*.[pP][nN][gG]')))
                stats['test'][class_dir.name] = count

    return stats


def print_data_info():
    """
    Imprime informações sobre os dados
    """
    stats = count_images()

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


def run_classic_pipeline():
    """
    Executa o pipeline clássico
    """
    print("\n" + "="*60)
    print("EXECUTANDO PIPELINE CLÁSSICO")
    print("="*60)

    from src.pipelines.classic import ClassicPipeline

    pipeline = ClassicPipeline(TRAIN_DIR, TEST_DIR)
    pipeline.load_data()
    pipeline.train_svm(use_random_search=True, n_iter=50)
    pipeline.train_random_forest(use_random_search=True, n_iter=50)
    pipeline.save_results()

    print("\nPipeline clássico concluído!")


def run_deep_learning_pipeline():
    """
    Executa o pipeline de deep learning
    """
    print("\n" + "="*60)
    print("EXECUTANDO PIPELINE DEEP LEARNING")
    print("="*60)

    from src.config import USE_GPU, EPOCHS
    from src.pipelines.deep_learning import DeepLearningPipeline

    pipeline = DeepLearningPipeline(TRAIN_DIR, TEST_DIR, use_gpu=USE_GPU)
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

    # Verificar estrutura de dados
    if not check_data_structure():
        print("\nAVISO: Dados não encontrados!")
        print(f"Diretório esperado: {DATA_DIR}")
        print("\nOpções:")
        print("1. Baixar dataset do Kaggle automaticamente")
        print("2. Sair e organizar dados manualmente")

        opcao = input("\nEscolha uma opção (1-2): ").strip()

        if opcao == '1':
            download_dataset()
            if not check_data_structure():
                print("\nERRO: Não foi possível organizar os dados.")
                return
        else:
            print("\nOrganize seus dados na seguinte estrutura:")
            print(f"  {TRAIN_DIR}/classe1/")
            print(f"  {TRAIN_DIR}/classe2/")
            print(f"  {TEST_DIR}/classe1/")
            print(f"  {TEST_DIR}/classe2/")
            return

    # Mostrar informações dos dados
    print_data_info()

    # Menu principal
    print("\nEscolha o pipeline a ser executado:")
    print("1. Pipeline Clássico (SVM + Random Forest)")
    print("2. Pipeline Deep Learning (CNN + ResNet)")
    print("3. Ambos os pipelines")
    print("4. Sair")

    opcao = input("\nEscolha uma opção (1-4): ").strip()

    if opcao == '1':
        run_classic_pipeline()
    elif opcao == '2':
        run_deep_learning_pipeline()
    elif opcao == '3':
        run_classic_pipeline()
        run_deep_learning_pipeline()
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
