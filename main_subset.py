# -*- coding: utf-8 -*-
"""
Versão do main.py que usa o subset pequeno do dataset
Útil para testes rápidos do projeto
"""

import sys
from pathlib import Path

# Adicionar diretório raiz ao path
ROOT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(ROOT_DIR))

# Modificar temporariamente os diretórios no config antes de importar
import src.config

# Salvar originais
original_train_dir = src.config.TRAIN_DIR
original_test_dir = src.config.TEST_DIR

# Redefinir para subset
src.config.TRAIN_DIR = src.config.DATA_DIR / 'train_subset'
src.config.TEST_DIR = src.config.DATA_DIR / 'test_subset'

# Verificar se subset existe
if not src.config.TRAIN_DIR.exists():
    print(f"ERRO: Subset não encontrado: {src.config.TRAIN_DIR}")
    print("Execute primeiro: python scripts/create_subset.py")
    sys.exit(1)

# Importar apenas funções auxiliares do main
from main import check_data_structure, count_images, print_data_info


def main():
    """
    Função principal usando subset
    """
    print("="*60)
    print("PROJETO DE CLASSIFICAÇÃO DE IMAGENS - MODO SUBSET")
    print("Usando dataset reduzido para testes rápidos")
    print("="*60)
    
    # Verificar estrutura de dados
    if not check_data_structure():
        print("\nERRO: Subset não encontrado!")
        print(f"Diretório esperado: {src.config.TRAIN_DIR}")
        print("\nExecute primeiro: python scripts/create_subset.py")
        return
    
    # Mostrar informações dos dados
    print_data_info()
    
    print("\n" + "="*60)
    print("AVISO: ESTE É UM MODO DE TESTE COM DADOS REDUZIDOS")
    print("Usando apenas 10 imagens por classe (total: 20 treino + 20 teste)")
    print("Os resultados NÃO serão representativos do dataset completo")
    print("Use apenas para validar que o código está funcionando")
    print("="*60)
    
    # Menu principal
    print("\nEscolha o pipeline a ser executado:")
    print("1. Pipeline Clássico (SVM + Random Forest) - Mais rápido")
    print("2. Pipeline Deep Learning (CNN + ResNet) - Mais lento")
    print("3. Ambos os pipelines")
    print("4. Sair")
    
    opcao = input("\nEscolha uma opção (1-4): ").strip()
    
    if opcao == '1':
        # Modificar temporariamente para usar menos iterações (testes rápidos)
        from src.pipelines.classic import ClassicPipeline
        pipeline = ClassicPipeline(src.config.TRAIN_DIR, src.config.TEST_DIR)
        pipeline.load_data()
        # Reduzir iterações para teste rápido
        pipeline.train_svm(use_random_search=True, n_iter=10)  # Reduzido de 50 para 10
        pipeline.train_random_forest(use_random_search=True, n_iter=10)  # Reduzido de 50 para 10
        pipeline.save_results()
    elif opcao == '2':
        from src.config import USE_GPU
        from src.pipelines.deep_learning import DeepLearningPipeline
        # Reduzir épocas para teste rápido
        pipeline = DeepLearningPipeline(src.config.TRAIN_DIR, src.config.TEST_DIR, use_gpu=USE_GPU)
        pipeline.load_data()
        pipeline.train_simple_cnn(use_random_search=True, n_iter=3, final_epochs=5)  # Muito reduzido para teste
        pipeline.train_resnet_transfer(use_random_search=True, n_iter=3, final_epochs=5)  # Muito reduzido para teste
        pipeline.save_results()
    elif opcao == '3':
        print("\nExecutando ambos os pipelines (pode demorar mesmo com subset reduzido)...")
        # Pipeline clássico
        from src.pipelines.classic import ClassicPipeline
        pipeline_classic = ClassicPipeline(src.config.TRAIN_DIR, src.config.TEST_DIR)
        pipeline_classic.load_data()
        pipeline_classic.train_svm(use_random_search=True, n_iter=10)
        pipeline_classic.train_random_forest(use_random_search=True, n_iter=10)
        pipeline_classic.save_results()
        # Pipeline deep learning
        from src.config import USE_GPU
        from src.pipelines.deep_learning import DeepLearningPipeline
        pipeline_dl = DeepLearningPipeline(src.config.TRAIN_DIR, src.config.TEST_DIR, use_gpu=USE_GPU)
        pipeline_dl.load_data()
        pipeline_dl.train_simple_cnn(use_random_search=True, n_iter=3, final_epochs=5)
        pipeline_dl.train_resnet_transfer(use_random_search=True, n_iter=3, final_epochs=5)
        pipeline_dl.save_results()
    elif opcao == '4':
        print("\nSaindo...")
    else:
        print("\nOpção inválida!")
    
    print("\n" + "="*60)
    print("TESTE CONCLUÍDO!")
    print("="*60)
    print("Resultados salvos em:")
    print("  - outputs/results/")
    print("  - outputs/models/")
    print("  - outputs/figures/")
    print("\nNota: Estes resultados são baseados em um subset pequeno.")
    print("Para resultados completos, use o dataset completo com: python main.py")


if __name__ == "__main__":
    try:
        main()
    finally:
        # Restaurar diretórios originais
        src.config.TRAIN_DIR = original_train_dir
        src.config.TEST_DIR = original_test_dir
