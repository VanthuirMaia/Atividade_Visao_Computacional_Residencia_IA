# -*- coding: utf-8 -*-
"""
Script Principal - Classificação de Imagens
Executa pipelines clássico e de deep learning
"""

import os
import sys
from config import TRAIN_DIR, TEST_DIR, USE_KAGGLE_DATASET

def check_data_structure():
    """
    Verifica se a estrutura de dados está correta
    """
    if not os.path.exists(TRAIN_DIR):
        print(f"ERRO: Diretório de treinamento não encontrado: {TRAIN_DIR}")
        return False
    
    if not os.path.exists(TEST_DIR):
        print(f"ERRO: Diretório de teste não encontrado: {TEST_DIR}")
        return False
    
    # Verificar se há imagens nos diretórios
    train_classes = [d for d in os.listdir(TRAIN_DIR) 
                     if os.path.isdir(os.path.join(TRAIN_DIR, d))] if os.path.exists(TRAIN_DIR) else []
    
    if not train_classes:
        return False
    
    return True

def offer_dataset_download():
    """
    Oferece opção de baixar o dataset do Kaggle
    """
    print("\n" + "="*80)
    print("DATASET NÃO ENCONTRADO")
    print("="*80)
    print("O projeto está configurado para usar o dataset 'AI Art vs Human Art' do Kaggle.")
    print("\nDeseja baixar e organizar o dataset agora?")
    print("1. Sim, baixar do Kaggle")
    print("2. Não, vou organizar manualmente")
    
    choice = input("\nDigite sua escolha (1/2): ").strip()
    
    if choice == "1":
        try:
            print("\nIniciando download do dataset...")
            from download_dataset import main as download_main
            download_main()
            return True
        except ImportError:
            print("\nERRO: Módulo download_dataset não encontrado.")
            print("Execute: python download_dataset.py")
            return False
        except Exception as e:
            print(f"\nERRO ao baixar dataset: {e}")
            return False
    else:
        print("\nOrganize os dados manualmente no formato:")
        print("  data/train/classe1/")
        print("  data/train/classe2/")
        print("  data/test/classe1/")
        print("  data/test/classe2/")
        return False

def main():
    """
    Função principal
    """
    print("="*80)
    print("PROJETO DE CLASSIFICAÇÃO DE IMAGENS")
    print("="*80)
    
    # Verificar estrutura de dados
    if not check_data_structure():
        if USE_KAGGLE_DATASET:
            if not offer_dataset_download():
                print("\nExecute o script de download manualmente:")
                print("  python download_dataset.py")
                sys.exit(1)
            # Verificar novamente após download
            if not check_data_structure():
                print("\nERRO: Dataset não foi organizado corretamente.")
                sys.exit(1)
        else:
            print("\nPor favor, organize os dados no formato:")
            print("  data/train/classe1/")
            print("  data/train/classe2/")
            print("  data/test/classe1/")
            print("  data/test/classe2/")
            sys.exit(1)
    
    print("\nEscolha o pipeline a executar:")
    print("1. Pipeline Clássico")
    print("2. Pipeline Deep Learning")
    print("3. Ambos os pipelines")
    
    choice = input("\nDigite sua escolha (1/2/3): ").strip()
    
    if choice == "1":
        print("\nExecutando Pipeline Clássico...")
        from pipeline_classico import main as classic_main
        classic_main()
    
    elif choice == "2":
        print("\nExecutando Pipeline Deep Learning...")
        from pipeline_deep_learning import main as dl_main
        dl_main()
    
    elif choice == "3":
        print("\nExecutando Pipeline Clássico...")
        from pipeline_classico import main as classic_main
        classic_main()
        
        print("\n" + "="*80)
        print("Executando Pipeline Deep Learning...")
        print("="*80)
        from pipeline_deep_learning import main as dl_main
        dl_main()
    
    else:
        print("Opção inválida!")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("PROJETO CONCLUÍDO!")
    print("="*80)
    print("Resultados salvos em:")
    print("  - outputs/results/")
    print("  - outputs/models/")
    print("  - outputs/figures/")

if __name__ == "__main__":
    main()

