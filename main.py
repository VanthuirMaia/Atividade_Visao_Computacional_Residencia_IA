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

    # Extensões de imagem suportadas (mesmas do utils.py e add_images_to_dataset.py)
    IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.jfif']
    
    if train_dir.exists():
        for class_dir in train_dir.iterdir():
            if class_dir.is_dir():
                # Contar todas as imagens com extensões suportadas
                count = 0
                for img_path in class_dir.iterdir():
                    if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
                        count += 1
                stats['train'][class_dir.name] = count

    if test_dir.exists():
        for class_dir in test_dir.iterdir():
            if class_dir.is_dir():
                # Contar todas as imagens com extensões suportadas
                count = 0
                for img_path in class_dir.iterdir():
                    if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
                        count += 1
                stats['test'][class_dir.name] = count

    return stats


def classify_new_image():
    """
    Função para classificar uma nova imagem usando os modelos treinados
    """
    try:
        # Tentar importar funções do classify_image
        import sys
        from pathlib import Path
        
        # Adicionar caminho do classify_image ao path se necessário
        classify_image_path = Path(__file__).parent / 'classify_image.py'
        if not classify_image_path.exists():
            print("\n❌ ERRO: Arquivo classify_image.py não encontrado!")
            return
        
        # Importar função de seleção de imagem
        try:
            import tkinter as tk
            from tkinter import filedialog
            TKINTER_AVAILABLE = True
        except ImportError:
            TKINTER_AVAILABLE = False
        
        print("\n" + "="*60)
        print("CLASSIFICAR NOVA IMAGEM")
        print("="*60)
        
        # Selecionar imagem
        if TKINTER_AVAILABLE:
            print("\n📁 Abrindo seletor de arquivo...")
            root = tk.Tk()
            root.withdraw()  # Esconder janela principal
            root.attributes('-topmost', True)  # Trazer para frente
            
            image_path = filedialog.askopenfilename(
                title="Selecione uma imagem para classificar",
                filetypes=[
                    ("Imagens", "*.jpg *.jpeg *.png *.bmp *.gif *.jfif"),
                    ("JPEG", "*.jpg *.jpeg"),
                    ("PNG", "*.png"),
                    ("Todos os arquivos", "*.*")
                ]
            )
            
            root.destroy()
            
            if not image_path:
                print("\n⚠️  Nenhuma imagem selecionada. Cancelando...")
                return
            
            image_path = Path(image_path)
        else:
            # Fallback: pedir caminho via input
            print("\nDigite o caminho completo da imagem:")
            print("(Exemplo: C:/Users/Usuario/Desktop/imagem.jpg)")
            image_path_str = input("Caminho: ").strip().strip('"').strip("'")
            
            if not image_path_str:
                print("\n⚠️  Caminho vazio. Cancelando...")
                return
            
            image_path = Path(image_path_str)
        
        # Verificar se arquivo existe
        if not image_path.exists():
            print(f"\n❌ ERRO: Arquivo não encontrado: {image_path}")
            return
        
        if not image_path.is_file():
            print(f"\n❌ ERRO: O caminho especificado não é um arquivo: {image_path}")
            return
        
        # Verificar extensão
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.jfif']
        if image_path.suffix.lower() not in valid_extensions:
            print(f"\n⚠️  AVISO: Extensão '{image_path.suffix}' pode não ser suportada.")
            print(f"Extensões suportadas: {', '.join(valid_extensions)}")
            resposta = input("\nDeseja continuar mesmo assim? (s/n): ").strip().lower()
            if resposta != 's':
                return
        
        # Importar e executar classificação
        print(f"\n✅ Imagem selecionada: {image_path}")
        print("\nClassificando imagem...")
        print("="*60)
        
        # Importar funções do classify_image
        sys.path.insert(0, str(Path(__file__).parent))
        from classify_image import classify_with_all_models, print_all_models_results
        
        # Classificar usando todos os modelos
        try:
            final_prediction, final_confidence, results = classify_with_all_models(str(image_path))
            print_all_models_results(final_prediction, final_confidence, results)
        except FileNotFoundError as e:
            print(f"\n❌ ERRO: {e}")
            print("\n💡 Dica: Execute o pipeline de treinamento primeiro para gerar os modelos.")
            print("   Escolha a opção 1, 2 ou 3 do menu principal.")
        except Exception as e:
            print(f"\n❌ ERRO ao classificar imagem: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Operação cancelada pelo usuário.")
    except Exception as e:
        print(f"\n❌ ERRO inesperado: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


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
    pipeline.train_svm(use_random_search=True, n_iter=100)  # Aumentado de 50 para 100
    pipeline.train_random_forest(use_random_search=True, n_iter=100)  # Aumentado de 50 para 100
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

    # Menu principal em loop
    while True:
        print("\n" + "="*60)
        print("MENU PRINCIPAL")
        print("="*60)
        print("Escolha uma opção:")
        print("1. Pipeline Clássico (SVM + Random Forest)")
        print("2. Pipeline Deep Learning (CNN + ResNet)")
        print("3. Ambos os pipelines")
        print("4. Classificar Nova Imagem")
        print("5. Sair")

        opcao = input("\nEscolha uma opção (1-5): ").strip()

        if opcao == '1':
            print("\n" + "="*60)
            run_classic_pipeline(current_train_dir, current_test_dir)
            print("\n" + "="*60)
            print("Pipeline Clássico concluído!")
            print("Resultados salvos em:")
            print("  - outputs/results/classic_pipeline_results.csv")
            print("  - outputs/models/")
            print("  - outputs/figures/")
            print("="*60)
        elif opcao == '2':
            print("\n" + "="*60)
            run_deep_learning_pipeline(current_train_dir, current_test_dir)
            print("\n" + "="*60)
            print("Pipeline Deep Learning concluído!")
            print("Resultados salvos em:")
            print("  - outputs/results/deep_learning_results.csv")
            print("  - outputs/models/")
            print("  - outputs/figures/")
            print("="*60)
        elif opcao == '3':
            print("\n" + "="*60)
            run_classic_pipeline(current_train_dir, current_test_dir)
            print("\n" + "="*60)
            run_deep_learning_pipeline(current_train_dir, current_test_dir)
            print("\n" + "="*60)
            print("Todos os pipelines concluídos!")
            print("Resultados salvos em:")
            print("  - outputs/results/")
            print("  - outputs/models/")
            print("  - outputs/figures/")
            print("="*60)
        elif opcao == '4':
            # Classificar nova imagem
            classify_new_image()
        elif opcao == '5':
            print("\n" + "="*60)
            print("Saindo...")
            print("Obrigado por usar o sistema de classificação!")
            print("="*60)
            break
        else:
            print("\n❌ Opção inválida! Escolha uma opção entre 1 e 5.")


if __name__ == "__main__":
    main()
