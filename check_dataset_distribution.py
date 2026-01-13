# -*- coding: utf-8 -*-
"""
Script para verificar a distribuição da base de dados
"""

from pathlib import Path

# Configurações (valores do config.py)
TRAIN_SPLIT = 0.7  # 70% para treinamento
TEST_SPLIT = 0.3   # 30% para teste

def check_distribution():
    """Verifica a distribuição real da base de dados"""
    
    # Definir diretórios
    DATA_DIR = Path(__file__).parent / 'data'
    TRAIN_DIR = DATA_DIR / 'train'
    TEST_DIR = DATA_DIR / 'test'
    
    print("="*70)
    print("DISTRIBUIÇÃO DA BASE DE DADOS")
    print("="*70)
    
    # Contar imagens
    train_counts = {}
    test_counts = {}
    
    total_train = 0
    total_test = 0
    
    IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.jfif']
    
    if TRAIN_DIR.exists():
        print(f"\nTREINAMENTO ({TRAIN_DIR}):")
        for class_dir in sorted(TRAIN_DIR.iterdir()):
            if class_dir.is_dir():
                # Contar arquivos de imagem (incluindo subdiretórios se houver)
                count = 0
                for img_path in class_dir.rglob('*'):
                    if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
                        count += 1
                train_counts[class_dir.name] = count
                total_train += count
                print(f"  {class_dir.name:20s}: {count:4d} imagens")
    
    if TEST_DIR.exists():
        print(f"\nTESTE ({TEST_DIR}):")
        for class_dir in sorted(TEST_DIR.iterdir()):
            if class_dir.is_dir():
                # Contar arquivos de imagem (incluindo subdiretórios se houver)
                count = 0
                for img_path in class_dir.rglob('*'):
                    if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
                        count += 1
                test_counts[class_dir.name] = count
                total_test += count
                print(f"  {class_dir.name:20s}: {count:4d} imagens")
    
    total = total_train + total_test
    
    if total > 0:
        train_pct = (total_train / total) * 100
        test_pct = (total_test / total) * 100
        
        print("\n" + "="*70)
        print("RESUMO GERAL")
        print("="*70)
        print(f"\nTotal de imagens: {total:,}")
        print(f"Treinamento: {total_train:,} ({train_pct:.1f}%)")
        print(f"Teste:        {total_test:,} ({test_pct:.1f}%)")
        
        print(f"\nConfiguração esperada (config.py):")
        print(f"  TRAIN_SPLIT = {TRAIN_SPLIT} ({TRAIN_SPLIT*100:.0f}%)")
        print(f"  TEST_SPLIT  = {TEST_SPLIT} ({TEST_SPLIT*100:.0f}%)")
        
        # Verificar se corresponde à configuração
        expected_train = total * TRAIN_SPLIT
        expected_test = total * TEST_SPLIT
        diff_train = abs(total_train - expected_train) / total * 100
        diff_test = abs(total_test - expected_test) / total * 100
        
        if diff_train < 1 and diff_test < 1:
            print(f"\n[OK] Distribuicao corresponde a configuracao (diferenca < 1%)")
        else:
            print(f"\n[AVISO] Diferenca da configuracao:")
            print(f"   Treino: {diff_train:.1f}% de diferença")
            print(f"   Teste:  {diff_test:.1f}% de diferença")
        
        # Distribuição por classe
        print("\n" + "="*70)
        print("DISTRIBUIÇÃO POR CLASSE")
        print("="*70)
        
        all_classes = set(list(train_counts.keys()) + list(test_counts.keys()))
        for class_name in sorted(all_classes):
            train_count = train_counts.get(class_name, 0)
            test_count = test_counts.get(class_name, 0)
            class_total = train_count + test_count
            if class_total > 0:
                train_class_pct = (train_count / class_total) * 100
                test_class_pct = (test_count / class_total) * 100
                print(f"\n{class_name}:")
                print(f"  Treino: {train_count:4d} ({train_class_pct:5.1f}%)")
                print(f"  Teste:  {test_count:4d} ({test_class_pct:5.1f}%)")
                print(f"  Total:  {class_total:4d}")
    else:
        print("\n[AVISO] Nenhuma imagem encontrada nos diretorios!")
        print(f"   Verifique se os diretórios existem:")
        print(f"   - {TRAIN_DIR}")
        print(f"   - {TEST_DIR}")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    check_distribution()
