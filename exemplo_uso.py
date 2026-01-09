# -*- coding: utf-8 -*-
"""
Exemplo de uso do projeto
Este script demonstra como usar os pipelines
"""

import os
from config import TRAIN_DIR, TEST_DIR, DATA_DIR

def criar_estrutura_exemplo():
    """
    Cria estrutura de diretórios de exemplo
    """
    print("Criando estrutura de diretórios de exemplo...")
    
    # Criar diretórios
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    
    # Exemplo de classes
    classes_exemplo = ['classe1', 'classe2', 'classe3']
    
    for classe in classes_exemplo:
        train_class_dir = os.path.join(TRAIN_DIR, classe)
        test_class_dir = os.path.join(TEST_DIR, classe)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)
        print(f"  Criado: {train_class_dir}")
        print(f"  Criado: {test_class_dir}")
    
    print("\nEstrutura criada!")
    print("Agora adicione suas imagens nos diretórios apropriados:")
    print(f"  - {TRAIN_DIR}/classe1/")
    print(f"  - {TRAIN_DIR}/classe2/")
    print(f"  - {TEST_DIR}/classe1/")
    print(f"  - {TEST_DIR}/classe2/")

def verificar_estrutura():
    """
    Verifica se a estrutura de dados está correta
    """
    print("Verificando estrutura de dados...")
    
    if not os.path.exists(TRAIN_DIR):
        print(f"ERRO: {TRAIN_DIR} não existe")
        return False
    
    if not os.path.exists(TEST_DIR):
        print(f"ERRO: {TEST_DIR} não existe")
        return False
    
    # Verificar classes
    train_classes = [d for d in os.listdir(TRAIN_DIR) 
                     if os.path.isdir(os.path.join(TRAIN_DIR, d))]
    test_classes = [d for d in os.listdir(TEST_DIR) 
                    if os.path.isdir(os.path.join(TEST_DIR, d))]
    
    print(f"Classes encontradas em treinamento: {train_classes}")
    print(f"Classes encontradas em teste: {test_classes}")
    
    if set(train_classes) != set(test_classes):
        print("AVISO: Classes de treinamento e teste não coincidem!")
    
    # Contar imagens
    total_train = 0
    total_test = 0
    
    for classe in train_classes:
        train_dir = os.path.join(TRAIN_DIR, classe)
        train_imgs = [f for f in os.listdir(train_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        total_train += len(train_imgs)
        print(f"  {classe}: {len(train_imgs)} imagens (treino)")
    
    for classe in test_classes:
        test_dir = os.path.join(TEST_DIR, classe)
        test_imgs = [f for f in os.listdir(test_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        total_test += len(test_imgs)
        print(f"  {classe}: {len(test_imgs)} imagens (teste)")
    
    print(f"\nTotal de imagens de treinamento: {total_train}")
    print(f"Total de imagens de teste: {total_test}")
    
    if total_train == 0 or total_test == 0:
        print("AVISO: Nenhuma imagem encontrada!")
        return False
    
    return True

if __name__ == "__main__":
    print("="*80)
    print("EXEMPLO DE USO DO PROJETO")
    print("="*80)
    
    print("\n1. Criar estrutura de diretórios")
    print("2. Verificar estrutura existente")
    
    escolha = input("\nEscolha (1/2): ").strip()
    
    if escolha == "1":
        criar_estrutura_exemplo()
    elif escolha == "2":
        if verificar_estrutura():
            print("\nEstrutura OK! Pode executar main.py")
        else:
            print("\nEstrutura com problemas. Verifique os diretórios.")
    else:
        print("Opção inválida!")

